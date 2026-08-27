"""ccut.blocks.heads — lm_head / embed 投影 + logprobs（§3.2 逐行 gather 协同）。

特殊点（R10 协同）：
- ``lm_head``: [vocab=248320, hidden=2048] ≈ 509MB（BF16）——**不进 ring buffer**
  （每步都要、太大），而是「逐行 gather」：decode 时只读被采样行？不行——
  需要全部 vocab logits。实际策略：**mmap 常驻 + 每步全量读**（509MB/步，
  NVMe 4.5GB/s ≈ 113ms 不可接受）→ 改为**页缓存友好**：lm_head 首次读后
  留页缓存，后续步命中（Ornith tie_word_embeddings=false，独立权重）；
  若内存预算极紧 → 在线量化 int8 减半（255MB，P8 评估项）。
- ``embed_tokens``: 同 lm_head 对称（vocab×hidden），prefill 时按 token id
  **逐行 gather**（只读用到的行 → 字节数 = seq×2KB，极小）——这是 §3.2
  「embed/lm_head 逐行 gather」的主场景。
"""

from __future__ import annotations

import numpy as np

__all__ = ["EmbedTable", "LmHead", "gather_rows", "logprobs_from_logits"]


def gather_rows(table: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """按行号 gather（embed 场景：``rows`` 为 token id）。

    ``table``: [vocab, hidden]（mmap 视图或常驻数组）；``rows``: [n] int；
    返回 [n, hidden]。mmap 视图下每行 2KB 顺序/随机读都走页缓存（P0-3 协同）。
    """
    return table[rows]


class EmbedTable:
    """embedding 表（逐行 gather）。"""

    def __init__(self, weight: np.ndarray):
        self.weight = weight  # [vocab, hidden]

    @property
    def vocab_size(self) -> int:
        return self.weight.shape[0]

    @property
    def hidden(self) -> int:
        return self.weight.shape[1]

    def __call__(self, token_ids: np.ndarray) -> np.ndarray:
        return gather_rows(self.weight, token_ids).astype(np.float32)


class LmHead:
    """输出投影（全 vocab logits）。

    ``weight``: [vocab, hidden]（dequant 后 float32 或 BF16→f32 视图）。
    每步 ``hidden @ W^T``（[seq, hidden] → [seq, vocab]）。
    """

    def __init__(self, weight: np.ndarray):
        self.weight = weight

    def __call__(self, hidden: np.ndarray) -> np.ndarray:
        return hidden @ self.weight.T


def logprobs_from_logits(logits: np.ndarray, top_k: int | None = None) -> np.ndarray:
    """logits → logprobs（[rows, vocab]；top_k 时只保留 top-k 行）。"""
    l = logits - np.log(np.maximum(np.exp(logits - logits.max(axis=1, keepdims=True)).sum(axis=1, keepdims=True), 1e-30))
    if top_k is not None and top_k > 0:
        idx = np.argpartition(l, -top_k, axis=1)[:, -top_k:]
        out = np.full(l.shape, -1e30, dtype=np.float32)
        for r in range(l.shape[0]):
            for c in idx[r]:
                out[r, c] = l[r, c]
        return out
    return l


def temperature_topk(
    logits: np.ndarray,
    temperature: float,
    top_k: int = 0,
    top_p: float = 1.0,
    min_p: float = 0.0,
    typical_p: float = 1.0,
) -> np.ndarray:
    """采样前变换（温度 + top-k + nucleus + min-p + typical）→ 概率分布。

    顺序对齐 vLLM：温度 → 各约束（作用于 logits）→ softmax。
    ``top_k=0`` 禁用 top-k；约束可叠加（vLLM 语义：top-k 优先于 nucleus）。
    """
    x = logits / temperature if temperature > 0 else logits
    if top_k and 0 < top_k < x.shape[1]:
        # top_k 阈值 = 每行第 k 大的值（标量/行广播）
        thr = np.partition(x, -top_k, axis=1)[:, -top_k:]
        thr = thr.min(axis=1, keepdims=True)
        x = np.where(x < thr, -1e30, x)
    if min_p and 0 < min_p < 1.0:
        # min-p：保留 p ≥ min_p × p_max
        m = x.max(axis=1, keepdims=True)
        e = np.exp(x - m)
        p = e / np.maximum(e.sum(axis=1, keepdims=True), 1e-30)
        pmax = p.max(axis=1, keepdims=True)
        x = np.where(p >= min_p * pmax, x, -1e30)
    if top_p and 0 < top_p < 1.0:
        order = np.argsort(x, axis=1)[:, ::-1]
        sx = np.sort(x, axis=1)[:, ::-1]
        csum = np.cumsum(np.exp(sx - sx[:, :1]), axis=1)
        csum = csum / np.maximum(csum[:, -1:], 1e-30)
        # 找到 csum ≥ top_p 的首位，其后屏蔽（保留 top_p 之前的）
        cond = csum > top_p
        # 每行：cond 首个 True 的前一位为止保留
        first = np.argmax(cond, axis=1)
        for r in range(x.shape[0]):
            if first[r] > 0:
                x[r, order[r, first[r] :]] = -1e30
            elif not cond[r].any():
                pass
    if typical_p and 0 < typical_p < 1.0:
        m = x.max(axis=1, keepdims=True)
        e = np.exp(x - m)
        p = e / np.maximum(e.sum(axis=1, keepdims=True), 1e-30)
        entropy = -np.sum(p * np.log(np.maximum(p, 1e-30)), axis=1, keepdims=True)
        surpr = -np.log(np.maximum(p, 1e-30))
        # 保留 surpr ≤ 熵分位数（typical_p）
        thr = np.quantile(surpr, typical_p, axis=1, keepdims=True)
        x = np.where(surpr <= thr, x, -1e30)
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / np.maximum(e.sum(axis=1, keepdims=True), 1e-30)
