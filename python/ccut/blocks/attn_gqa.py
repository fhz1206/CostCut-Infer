"""ccut.blocks.attn_gqa — GQA/MQA 全注意力（full_attn 层通用积木）。

数据流（Ornith full_attn 层，每 4 层一次）::

    q,k,v = qkv_proj(x)                     # [seq, heads*d] 切分
    q,k = apply_rope(q, k)                  # RoPE（KV 头独立施加）
    kv_cache.update(k, v)                   # R1 块池写（offset 对齐 block）
    k_all, v_all = kv_cache.gather(...)     # 读含前缀缓存的历史
    o = softmax(q·k^T/√d + mask + bias)·v   # numba 内核（bf16 输入/float32 累加）
    y = o_proj(o)

CPU 实现：float32 softmax + numba prange；causal mask 按位置构造（支持
prefix 偏移：``start_pos`` 为当前批第一个 token 的绝对位置）。
"""

from __future__ import annotations

import numpy as np

try:
    from numba import njit, prange
except ImportError:
    def njit(*args, **kwargs):
        def wrap(fn):
            return fn

        if args and callable(args[0]):
            return args[0]
        return wrap

    def prange(n):
        return range(n)


__all__ = ["gqa_attention", "build_causal_mask", "repetition_penalty_logits"]


@njit(cache=True, fastmath=True)
def _softmax_kernel(scores: np.ndarray, mask: np.ndarray | None, out: np.ndarray) -> None:
    """行 softmax（[rows, cols]，mask 为 -inf 屏蔽位）。float32 稳定路径。"""
    rows, cols = scores.shape
    for r in prange(rows):
        m = -1e30
        for c in range(cols):
            v = scores[r, c]
            if mask is not None and mask[r, c] < -1e29:
                continue
            if v > m:
                m = v
        s = 0.0
        for c in range(cols):
            v = scores[r, c]
            if mask is not None and mask[r, c] < -1e29:
                out[r, c] = 0.0
                continue
            e = np.exp(v - m)
            out[r, c] = e
            s += e
        inv = 1.0 / s if s > 0 else 0.0
        for c in range(cols):
            out[r, c] *= inv


def build_causal_mask(seq_len: int, start_pos: int) -> np.ndarray:
    """causal mask [seq_len, start_pos + seq_len]：位置 i 可见 < start_pos+i。

    返回 float32，可见=0 / 屏蔽=-1e30（与 softmax 内核约定）。
    """
    q_idx = np.arange(start_pos, start_pos + seq_len)[:, None]
    kv_idx = np.arange(start_pos + seq_len)[None, :]
    mask = np.zeros((seq_len, start_pos + seq_len), dtype=np.float32)
    mask[kv_idx > q_idx] = -1e30
    return mask


def gqa_attention(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    mask: np.ndarray | None = None,
    scale: float | None = None,
    softcap: float | None = None,
) -> np.ndarray:
    """GQA 注意力（float32 累加）。

    - ``q``: [batch, heads, seq, d]；``k``/``v``: [batch, kv_heads, kv_len, d]
      （GQA：``heads % kv_heads == 0``，按组共享）；
    - ``mask``: [batch, heads, seq, kv_len] 或 None（causal 由调用方 build_causal_mask 广播）；
    - ``softcap``: attn_logit_softcapping（Kimi 用）：``tanh(scores/softcap)*softcap``；
    - 返回 [batch, heads, seq, d] float32。
    """
    batch, heads, seq, d = q.shape
    kv_heads = k.shape[1]
    if heads % kv_heads != 0:
        raise ValueError(f"GQA 要求 heads({heads}) % kv_heads({kv_heads}) == 0")
    g = heads // kv_heads
    q32 = q.astype(np.float32, copy=False)
    k32 = k.astype(np.float32, copy=False)
    v32 = v.astype(np.float32, copy=False)
    scale = scale or 1.0 / np.sqrt(d)
    out = np.empty((batch, heads, seq, d), dtype=np.float32)
    # 按 (batch, group) 分块计算，kv_len 维 numba 内循环
    scores = np.empty((seq, k32.shape[2]), dtype=np.float32)
    probs = np.empty((seq, k32.shape[2]), dtype=np.float32)
    for b in range(batch):
        for h in range(heads):
            hk = h // g
            for s in range(seq):
                # q_s · k^T / scale（[kv_len]）
                np.dot(q32[b, h, s], k32[b, hk], out=scores[s, : k32.shape[2]])
                scores[s, : k32.shape[2]] *= scale
                if softcap is not None:
                    scores[s, : k32.shape[2]] = np.tanh(scores[s, : k32.shape[2]] / softcap) * softcap
                if mask is not None:
                    scores[s, : k32.shape[2]] += mask[b, h, s, : k32.shape[2]]
                _softmax_kernel(scores[s : s + 1], None, probs[s : s + 1])
                # p · v（[d]）
                np.dot(probs[s, : k32.shape[2]], v32[b, hk], out=out[b, h, s])
    return out


def gqa_attention_fast(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    mask: np.ndarray | None = None,
    scale: float | None = None,
) -> np.ndarray:
    """批量 GQA（numpy 向量化版，prefill 大 seq 用；decode 单 token 用上面逐 token 版）。

    语义与 :func:`gqa_attention` 一致（float32 累加）。
    """
    batch, heads, seq, d = q.shape
    kv_heads = k.shape[1]
    g = heads // kv_heads
    q32 = q.astype(np.float32, copy=False)
    k32 = k.astype(np.float32, copy=False)
    v32 = v.astype(np.float32, copy=False)
    scale = scale or 1.0 / np.sqrt(d)
    # q [b,h,s,d] → 组共享 kv：k [b,h,kv,d] 展开
    if g > 1:
        k32 = np.repeat(k32, g, axis=1)
        v32 = np.repeat(v32, g, axis=1)
    scores = np.einsum("bhqd,bhkd->bhqk", q32, k32) * scale
    if mask is not None:
        scores = scores + mask
    scores -= scores.max(axis=-1, keepdims=True)
    exp = np.exp(scores)
    exp[scores < -1e29] = 0.0
    probs = exp / np.maximum(exp.sum(axis=-1, keepdims=True), 1e-30)
    return np.einsum("bhqk,bhkd->bhqd", probs, v32)


def repetition_penalty_logits(
    logits: np.ndarray,
    generated: np.ndarray,
    penalty: float,
) -> np.ndarray:
    """repetition penalty（HuggingFace 语义）：已生成 token 的 logit 除以惩罚因子。

    ``logits``: [batch, vocab] float32；``generated``: [batch] token 集合（ndarray of sets
    不友好——用 [batch, max_gen] int 数组，-1 填充）；``penalty>1`` 惩罚正 logit，
    ``<1`` 奖励负 logit。
    """
    if penalty == 1.0 or generated.size == 0:
        return logits
    out = logits.copy()
    batch = out.shape[0]
    for b in range(batch):
        for tok in generated[b]:
            t = int(tok)
            if t < 0:
                continue
            v = out[b, t]
            if v > 0:
                out[b, t] = v / penalty
            else:
                out[b, t] = v * penalty
    return out


def apply_presence_frequency_penalties(
    logits: np.ndarray,
    history_counts: np.ndarray,
    presence_penalty: float,
    frequency_penalty: float,
    length_penalty: float = 1.0,
    length: int = 1,
) -> np.ndarray:
    """presence/frequency/length penalty（对齐 vLLM SamplingParams 语义）。

    - presence: ``- pp * 1[seen]``
    - frequency: ``- fp * count``
    - length (ITL): ``- lp * ln(length)``（仅当 lp != 1，作用于全部 logit 的常数偏移，
      不影响采样分布——保留以对齐接口）。
    """
    out = logits - presence_penalty * (history_counts > 0).astype(np.float32)
    out = out - frequency_penalty * history_counts.astype(np.float32)
    if length_penalty != 1.0 and length > 1:
        out = out - length_penalty * np.log(float(length))
    return out
