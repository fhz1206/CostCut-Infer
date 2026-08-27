"""ccut.blocks.moe — MoE 路由与专家前向数据流（需求 2 计算侧，§2 数据流）。

路由数据流（Ornith：256 专家 / top-8 / norm_topk_prob）::

    gate_logits = router(x)                    # [seq, 256]（ignore 层 → BF16 直通）
    topk = topk_softmax(gate_logits, k)        # 值 + 专家 id
    若 norm_topk_prob：topk 归一化（组内和=1）
    专家前向：每 token 只路由 top-k 专家（§3.3 第 3 点：路由决策只依赖当前 token
    的 gate 向量 → 可投机预取）
        h_e = down_e( silu(gate_e(x)) * up_e(x) )   # 每专家独立
    y = Σ_e topk_val[e] · h_e                      # 加权求和（numba 融合）

专家权重**零驻留**（R2）：本块只接收「已 dequant 的激活矩阵」（ExpertReader
从 ring buffer 读出并经 quant method apply），路由与融合在 CPU 完成。
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


__all__ = [
    "topk_softmax",
    "topk",
    "route_experts",
    "expert_ffn",
    "moe_combine",
    "shared_expert_add",
]


@njit(cache=True)
def topk(x: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """每行 top-k（值降序 + 原始下标）。``x``: [rows, cols] → (vals [rows,k], idx [rows,k])。"""
    rows = x.shape[0]
    cols = x.shape[1]
    vals = np.empty((rows, k), dtype=x.dtype)
    idx = np.empty((rows, k), dtype=np.int64)
    for r in prange(rows):
        # 部分选择（cols 小，直接选 k 次）
        used = np.zeros(cols, dtype=np.int64)
        for j in range(k):
            best = -1.0e30
            bi = -1
            for c in range(cols):
                if used[c]:
                    continue
                if x[r, c] > best:
                    best = x[r, c]
                    bi = c
            used[bi] = 1
            vals[r, j] = best
            idx[r, j] = bi
    return vals, idx


def topk_softmax(logits: np.ndarray, k: int, norm_topk_prob: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """top-k 后 softmax（路由权重）。

    - ``norm_topk_prob=True``（Ornith）：仅在 top-k 组内归一化（组内和=1）；
    - False：全局 softmax 后取 top-k（DeepSeek 早期语义）。
    返回 (weights [rows,k] float32, expert_ids [rows,k] int64)。
    """
    rows = logits.shape[0]
    if not norm_topk_prob:
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        p = e / np.maximum(e.sum(axis=1, keepdims=True), 1e-30)
        vals, idx = topk(p, k)
        return vals.astype(np.float32), idx
    vals, idx = topk(logits, k)
    # 组内 softmax
    v = vals - vals.max(axis=1, keepdims=True)
    exp = np.exp(v)
    w = exp / np.maximum(exp.sum(axis=1, keepdims=True), 1e-30)
    return w.astype(np.float32), idx


@njit(cache=True)
def route_experts(gate_logits: np.ndarray, top_k: int, norm_topk: int) -> tuple[np.ndarray, np.ndarray]:
    """路由融合核（top-k + 组内 softmax 一次完成；``norm_topk``: 0/1）。"""
    rows = gate_logits.shape[0]
    cols = gate_logits.shape[1]
    vals, idx = topk(gate_logits, top_k)
    weights = np.empty((rows, top_k), dtype=np.float32)
    for r in prange(rows):
        if norm_topk:
            v = vals[r] - vals[r, 0] if top_k > 0 else 0.0
            m = -1.0e30
            for j in range(top_k):
                if vals[r, j] > m:
                    m = vals[r, j]
            s = 0.0
            e = np.empty(top_k, dtype=np.float32)
            for j in range(top_k):
                e[j] = np.exp(vals[r, j] - m)
                s += e[j]
            for j in range(top_k):
                weights[r, j] = e[j] / s if s > 0 else 0.0
        else:
            s = 0.0
            for j in range(top_k):
                s += np.exp(vals[r, j])
            for j in range(top_k):
                weights[r, j] = np.exp(vals[r, j]) / s if s > 0 else 0.0
    return weights, idx


def expert_ffn(
    x: np.ndarray,
    w_gate: np.ndarray,
    w_up: np.ndarray,
    w_down: np.ndarray,
) -> np.ndarray:
    """单专家 SwiGLU 前向：``down( silu(x·gate) · (x·up) )``。

    - ``x``: [rows, hidden]（路由到该专家的 token 子集）；
    - ``w_gate``: [hidden, inter]，``w_up``: [hidden, inter]，``w_down``: [inter, hidden]
      （均已 dequant，float32）；
    - 返回 [rows, hidden]。融合核 :func:`silu_mul` 在 quant/kernels.py。
    """
    from ccut.quant import kernels

    gate = x @ w_gate
    up = x @ w_up
    inter = np.empty(gate.shape, dtype=np.float32)
    kernels.silu_mul_fused(gate, up, inter)
    return inter @ w_down


def moe_combine(
    x: np.ndarray,
    expert_ids: np.ndarray,
    weights: np.ndarray,
    expert_outputs: dict[int, np.ndarray] | None = None,
    expert_fn=expert_ffn,
    w_lookup=None,
) -> np.ndarray:
    """MoE 加权融合：``y[t] = Σ_e w[t,e] · expert_e(x[t])``。

    - ``x``: [seq, hidden]；``expert_ids``: [seq, k]；``weights``: [seq, k]；
    - ``w_lookup``: ``expert_id → (w_gate, w_up, w_down)`` 可调用（权重按 id 现取，
      R2 ring buffer 命中）；
    - 同专家 token 合并成一批计算（减少小 GEMM 次数）；
    - 返回 [seq, hidden] float32。
    """
    seq, hidden = x.shape
    k = expert_ids.shape[1]
    out = np.zeros((seq, hidden), dtype=np.float32)
    # 按专家聚合 token
    by_expert: dict[int, list[int]] = {}
    for t in range(seq):
        for e in range(k):
            by_expert.setdefault(int(expert_ids[t, e]), []).append(t)
    for eid, toks in by_expert.items():
        wg, wu, wd = w_lookup(eid)
        batch = x[np.array(toks, dtype=np.int64)]
        y = expert_fn(batch, wg, wu, wd)
        for i, t in enumerate(toks):
            out[t] += weights[t, list(expert_ids[t]).index(eid)] * y[i]
    return out


def shared_expert_add(y: np.ndarray, shared_out: np.ndarray, gate_value: float) -> None:
    """共享专家残差（就地）：``y += gate_value · shared_out``（Ornith shared_expert_gate）。"""
    y += gate_value * shared_out
