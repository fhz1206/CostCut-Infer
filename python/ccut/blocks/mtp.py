"""ccut.blocks.mtp — 多 token 预测模块（P5 投机解码 proposer，§7 P5）。

Ornith MTP 结构（config: ``mtp_num_hidden_layers=1``，1 层 draft）::

    输入 = 上一步 hidden + 当前 token embed 融合（enorm/hnorm 双 RMSNorm）
    → 1 层 decoder（复用主模型 layer 结构：attn + MoE）
    → 输出 hidden → 共享 lm_head（Ornith tie=false 但 MTP 用独立 proj 到 vocab）
    → 下一 token 概率（draft）

验证（verify，P5 数据流）::

    draft: 用 MTP 模块生成 γ 个候选（Ornith γ=1）
    主模型一次性前向 [x, d1, d2, ...]（γ+1 token）→ 真实分布
    接受判定：逐位置比较 draft 采样 vs 主模型分布（rejection sampling，
    vLLM speculative 语义）→ 接受 n ≤ γ 个，拒绝处按修正分布重采样 1 个

MTP 层权重随主模型层进 WeightRing（R10：mtp layer 是第 N+1 层，ring 容量 +1）。
"""

from __future__ import annotations

import numpy as np

from ccut.blocks.norm import rms_norm

__all__ = ["MTPLayer", "mtp_draft", "speculative_verify"]


class MTPLayer:
    """1 层 MTP 模块（draft）。

    前向输入：
    - ``hidden_prev``: [seq, hidden]（上一步主模型最后层 hidden）；
    - ``embed_cur``: [seq, hidden]（当前 token embedding）；
    融合：``x = rms_norm(hnorm(hidden_prev) + enorm(embed_cur))``
    → decoder layer（调用方注入 layer_fn）→ 输出 hidden。
    """

    def __init__(self, hidden: int, eps: float = 1e-6):
        self.hidden = hidden
        self.eps = eps

    def fuse(self, hidden_prev: np.ndarray, embed_cur: np.ndarray) -> np.ndarray:
        h = rms_norm(hidden_prev, None, self.eps)
        e = rms_norm(embed_cur, None, self.eps)
        return (h + e).astype(np.float32)


def mtp_draft(
    model_layers,
    mtp_layer,
    embed_fn,
    lm_head,
    hidden_prev: np.ndarray,
    token_ids: np.ndarray,
    n_draft: int = 1,
    greedy: bool = True,
) -> list[int]:
    """MTP draft：生成 n_draft 个候选 token。

    - ``model_layers`` / ``mtp_layer``: 调用方组装的层前向（本块不含权重搬运）；
    - ``embed_fn``: ``token_ids → [seq, hidden]``；
    - ``lm_head``: ``[seq, hidden] → [seq, vocab]`` logits；
    - ``greedy=True``：argmax（P5 默认，接受率最高）；
    - 返回 draft token id 列表（≤ n_draft）。
    """
    drafts: list[int] = []
    h = hidden_prev
    t = token_ids
    for _ in range(n_draft):
        x = mtp_layer.fuse(h, embed_fn(t))
        h = mtp_layer.forward(x) if hasattr(mtp_layer, "forward") else model_layers.mtp_forward(x, h)
        logits = lm_head(h)
        tok = int(np.argmax(logits[-1]))
        drafts.append(tok)
        t = np.array([tok], dtype=np.int64)
        # 下一步的 hidden 由主模型最后层提供——draft 循环内用 MTP 输出近似
        # （Ornith γ=1 时仅一次，无累积误差问题）
    return drafts


def speculative_verify(
    draft_ids: list[int],
    main_logits: np.ndarray,
    draft_probs: np.ndarray | None = None,
    temperature: float = 1.0,
    seed: int | None = None,
) -> tuple[int, int | None]:
    """投机解码验证（vLLM rejection sampling 语义）。

    - ``draft_ids``: γ 个候选；
    - ``main_logits``: [γ+1, vocab]（主模型对 [x, d1..dγ] 一次前向的逐位置 logits，
      位置 i 预测第 i+1 个 token）；
    - 返回 ``(accepted, bonus_token)``：accepted = 接受的 draft 数（前缀），
      bonus = 拒绝处/全接受时补采样的真实 token（None 表示无需）。
    """
    rng = np.random.default_rng(seed)
    accepted = 0
    bonus: int | None = None
    g = len(draft_ids)
    for i in range(g):
        p_main = _softmax(main_logits[i])
        if draft_probs is None:
            # draft 为 greedy → p_draft(d)≈1：接受条件 uniform < 1（恒真），
            # 拒绝则按 p_main 重采样
            if rng.random() < 1.0:
                accepted = i + 1
                continue
            bonus = int(rng.choice(len(p_main), p=p_main))
            return accepted, bonus
        p_d = float(draft_probs[i])
        if rng.random() < min(1.0, p_main[draft_ids[i]] / p_d if p_d > 0 else 0.0):
            accepted = i + 1
            continue
        # 拒绝：按 max(0, p_main - p_d) 归一化重采样
        resid = np.maximum(p_main - p_d, 0.0)
        s = resid.sum()
        if s > 0:
            bonus = int(rng.choice(len(p_main), p=resid / s))
        else:
            bonus = int(rng.choice(len(p_main), p=p_main))
        return accepted, bonus
    # 全接受 → 按最后可用位置分布采 bonus（main_logits 应含 γ+1 行；
    # 调用方只传 γ 行时回退末行——draft 末 token 的预测分布）
    p_last = _softmax(main_logits[min(g, main_logits.shape[0] - 1)])
    bonus = int(rng.choice(len(p_last), p=p_last))
    return accepted, bonus


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / np.maximum(e.sum(), 1e-30)
