"""ccut.blocks.attn_mla — DeepSeek-V3 MLA 压缩注意力（L0 家族积木）。

DeepSeek-V3.2/V4 MLA 数据流（已从 vLLM registry 家族模块确认）::

    kv_a = kv_a_proj_with_mqa(x)      # [hidden → kv_lora_rank + qk_rope_head_dim]
    kv_a 切分 → c [kv_lora_rank], k_pe [qk_rope_head_dim]
    q = q_proj(x)                     # [hidden → num_heads*(qk_nope_head_dim + qk_rope_head_dim)]
    q 切分 → q_nope [qk_nope], q_pe [qk_rope]
    RoPE 仅作用于 k_pe / q_pe（qk_rope_head_dim 维，Ornith 式 64）
    c → 每头 kv 缓存（低秩共享）：kv 块池按 rank 存（字节 = rank×4B/token/head 组）
    decode: c_t 展开 → k = k_up_proj(c), v = v_up_proj(c)
    prefill: 同上；c 进 L2 下沉（MLA 的 KV 块天然小 → L1 容量放大）
    score = (q_nope·k_nope + q_pe·k_pe)/√(d_nope+d_rope)
    o = softmax(score + mask) · v

CPU 实现：float32；KV 缓存布局 ``[batch, rank]`` 每 token（c 向量）+
``k_pe [batch, d_rope]``——**MLA 的 KV 字节 = (rank + d_rope)×4B/token**，
比 GQA 的 ``2×kv_heads×d×2B`` 小一个量级（§3.4-7 KV 预算参数化的 MLA 分支）。
"""

from __future__ import annotations

import numpy as np

__all__ = ["MLAState", "mla_update_state", "mla_decode", "mla_prefill"]


class MLAState:
    """MLA 压缩状态（每请求）：c 向量序列 + k_pe 序列。

    与 GDN 状态不同：MLA 状态随序列**线性增长** → 进 KV 块池（R1），
    块字节 = ``(kv_lora_rank + qk_rope_head_dim) × 4B / token``。
    """

    def __init__(self, kv_lora_rank: int, qk_rope_head_dim: int, max_tokens: int = 32768):
        self.c = np.zeros((max_tokens, kv_lora_rank), dtype=np.float32)
        self.k_pe = np.zeros((max_tokens, qk_rope_head_dim), dtype=np.float32)
        self.n = 0
        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim

    def append(self, c: np.ndarray, k_pe: np.ndarray) -> int:
        """追加本步 c/k_pe，返回写入位置（decode 单 token / prefill 批量）。"""
        n_new = c.shape[0]
        if self.n + n_new > self.c.shape[0]:
            raise MemoryError(f"MLAState 容量不足：{self.n}+{n_new} > {self.c.shape[0]}")
        self.c[self.n : self.n + n_new] = c
        self.k_pe[self.n : self.n + n_new] = k_pe
        pos = self.n
        self.n += n_new
        return pos

    def reset(self) -> None:
        self.n = 0

    def bytes(self) -> int:
        return (self.kv_lora_rank + self.qk_rope_head_dim) * 4 * self.n


def mla_update_state(
    state: MLAState,
    x: np.ndarray,
    kv_a_proj: np.ndarray,
    kv_a_scale: np.ndarray | None,
    q_pe: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """MLA 前投影：x → c, k_pe（decode 单 token 或 prefill 批量）。

    - ``x``: [1, hidden]（decode）或 [seq, hidden]（prefill）float32；
    - ``kv_a_proj``: [hidden, kv_lora_rank + d_rope]（已 dequant）；
    - ``q_pe``: [seq, d_rope]（已 RoPE）；
    - 返回 (c [seq, rank], k_pe_stored [seq, d_rope]) 并写入 state。
    """
    kv_a = x @ kv_a_proj
    c = kv_a[..., : state.kv_lora_rank]
    k_pe = kv_a[..., state.kv_lora_rank :]
    # k_pe 的 RoPE 由调用方在 q/k 切分阶段施加；此处存**未旋转**的 k_pe 低秩部分
    # （V3 语义：k_pe 进缓存前做 RoPE；本实现约定调用方传已旋转值）
    state.append(c, q_pe)
    return c, q_pe


def mla_decode(
    state: MLAState,
    q_nope: np.ndarray,
    q_pe_rot: np.ndarray,
    k_up_proj: np.ndarray,
    v_up_proj: np.ndarray,
    scale: float,
    start_pos: int,
) -> np.ndarray:
    """MLA decode（单 token，逐头）：c 历史 → k/v 展开 → 注意力。

    - ``q_nope``: [num_heads, d_nope]，``q_pe_rot``: [num_heads, d_rope]（已 RoPE）；
    - ``k_up_proj``: [rank, num_heads×d_nope]，``v_up_proj``: [rank, num_heads×d_v]；
    - ``start_pos``: 当前 token 绝对位置（prefix 偏移）；
    - 返回 [num_heads, d_v] float32。
    """
    kv_len = state.n
    c_all = state.c[:kv_len]
    k_pe_all = state.k_pe[:kv_len]
    # 展开：K [kv_len, heads*d_nope]，V [kv_len, heads*d_v]
    k_expanded = c_all @ k_up_proj
    v_expanded = c_all @ v_up_proj
    num_heads = q_nope.shape[0]
    d_nope = q_nope.shape[1]
    d_rope = q_pe_rot.shape[1]
    d_v = v_expanded.shape[1] // num_heads
    scores = np.empty((num_heads, kv_len), dtype=np.float32)
    for h in range(num_heads):
        qn = q_nope[h]
        qp = q_pe_rot[h]
        kh = k_expanded[:, h * d_nope : (h + 1) * d_nope]
        s = kh @ qn
        s += k_pe_all @ qp
        scores[h] = s * scale
    # causal：decode 单 token 可见全部历史（含本步——本步已 append）
    probs = scores - scores.max(axis=1, keepdims=True)
    exp = np.exp(probs)
    probs = exp / np.maximum(exp.sum(axis=1, keepdims=True), 1e-30)
    out = np.empty((num_heads, d_v), dtype=np.float32)
    for h in range(num_heads):
        out[h] = probs[h] @ v_expanded[:, h * d_v : (h + 1) * d_v]
    return out


def mla_prefill(
    state: MLAState,
    q_nope: np.ndarray,
    q_pe_rot: np.ndarray,
    k_up_proj: np.ndarray,
    v_up_proj: np.ndarray,
    scale: float,
    start_pos: int,
) -> np.ndarray:
    """MLA prefill（causal 批量）：返回 [seq, num_heads, d_v]。

    ``q_nope``: [seq, heads, d_nope]，``q_pe_rot``: [seq, heads, d_rope]。
    """
    kv_len = state.n
    c_all = state.c[:kv_len]
    k_pe_all = state.k_pe[:kv_len]
    k_expanded = c_all @ k_up_proj
    v_expanded = c_all @ v_up_proj
    seq, heads, d_nope = q_nope.shape
    d_rope = q_pe_rot.shape[2]
    d_v = v_expanded.shape[1] // heads
    scores = np.empty((seq, heads, kv_len), dtype=np.float32)
    for s in range(seq):
        for h in range(heads):
            kh = k_expanded[:, h * d_nope : (h + 1) * d_nope]
            sc = kh @ q_nope[s, h]
            sc += k_pe_all @ q_pe_rot[s, h]
            # causal mask：可见 = 全部 prefix（start_pos）+ 本批已 append 的
            # s+1 个 token（含自身）；state 尾部即当前批 token（append 序）。
            max_vis = min(start_pos + s + 1, kv_len)
            for c in range(max_vis, kv_len):
                sc[c] = -1e30
            scores[s, h] = sc * scale
    scores -= scores.max(axis=-1, keepdims=True)
    exp = np.exp(scores)
    exp[scores < -1e29] = 0.0
    probs = exp / np.maximum(exp.sum(axis=-1, keepdims=True), 1e-30)
    out = np.empty((seq, heads, d_v), dtype=np.float32)
    for h in range(heads):
        out[:, h] = probs[:, h] @ v_expanded[:, h * d_v : (h + 1) * d_v]
    return out
