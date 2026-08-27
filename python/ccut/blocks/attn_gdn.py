"""ccut.blocks.attn_gdn — Gated DeltaNet 线性注意力（Ornith linear_attn 层，§3.4-6）。

Ornith linear_attn 结构（Qwen3-Next 同源 GDN，已从 checkpoint 张量名确认）::

    in_proj_qkv  → q [16 heads, k_dim=128], k [16 heads, 128], v [32 heads, 128]
    in_proj_z    → z [32 heads, 128]（门控）
    in_proj_b    → beta [32 heads]（delta 学习率）
    in_proj_a    → a [32 heads]（衰减）
    conv1d       → 短卷积（kernel=4，causal）
    递归核       → GDN 状态更新（delta rule + 门控衰减）
    out_proj     → [32*128 → 2048]
    门控         → silu(z) 融合（attn_output_gate=true）

GDN 递推（Gated DeltaNet，Yang et al. 2025 / Qwen3-Next 实现）::

    每时间步 t、每 v-head h（kv 头按 g=2 映射到 16 个 k 头）：
        k_t, v_t: 本步输入（conv 后）
        a_t = -exp(A_log) * softplus(a_t)        # 衰减率 ∈ (-∞, 0)
        b_t = sigmoid(beta_t)                    # delta 门
        β_t = 1 - b_t * ...                      # GDN 的 delta 更新系数
        状态 S [k_dim, v_dim] 更新（delta rule）：
            v_hat = S @ k_t                       # 预测
            δ = v_t - v_hat * b_t                 # 误差
            S = S * exp(a_t) + δ ⊗ k_t            # 衰减 + 写入
        o_t = S @ q_t

CPU 实现：float32 递推（numba，序列维单核不可并行 → 逐层时间步循环，
prefill 用 chunked 并行版——本文件提供**逐步递推核** + chunked 组装）。
"""

from __future__ import annotations

import numpy as np

try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):
        def wrap(fn):
            return fn

        if args and callable(args[0]):
            return args[0]
        return wrap


__all__ = ["gdn_step", "gdn_prefill", "GDNState"]


@njit(cache=True, fastmath=True)
def _softplus(x: float) -> float:
    if x > 20.0:
        return x
    if x < -20.0:
        return 0.0
    return np.log(1.0 + np.exp(x))


@njit(cache=True, fastmath=True)
def gdn_step(
    state: np.ndarray,
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    a: float,
    beta: float,
    a_log: float,
    out: np.ndarray,
) -> None:
    """单步 GDN 递推（一个 v-head）。

    - ``state``: [k_dim, v_dim] float32（**就地更新**，跨 token 持久化）；
    - ``q``: [k_dim]，``k``: [k_dim]，``v``: [v_dim]（本步，conv 后、已门控）；
    - ``a``: 衰减原始值（softplus 前），``beta``: sigmoid 前门控值，
      ``a_log``: 每头衰减参数（A_log，负值越大衰减越快）；
    - ``out``: [v_dim] 输出（``S @ q``）。
    """
    kd = q.shape[0]
    vd = v.shape[0]
    decay = np.exp(-_softplus(a) + a_log)  # ∈ (0, 1]：衰减因子
    b = 1.0 / (1.0 + np.exp(-beta))  # beta ∈ (0,1)
    # v_hat = S @ k（预测）
    for j in range(vd):
        s = 0.0
        for i in range(kd):
            s += state[i, j] * k[i]
        v_hat = s
        # δ = (v_t - v_hat) * b
        delta = (v[j] - v_hat) * b
        # S[:, j] = S[:, j] * decay + k * delta
        for i in range(kd):
            state[i, j] = state[i, j] * decay + k[i] * delta
    # out = S @ q
    for j in range(vd):
        s = 0.0
        for i in range(kd):
            s += state[i, j] * q[i]
        out[j] = s


class GDNState:
    """GDN 递归状态容器（每 v-head 一个 [k_dim, v_dim] 矩阵）。

    与 KV 块池（R1）正交：GDN 状态**每请求独立**、大小固定
    （``num_v_heads × k_dim × v_dim × 4B``，Ornith = 32×128×128×4 = 2MB/请求），
    不随序列增长 → 不进 L2 下沉路径，由 coordinator 单独管理（随请求生命周期）。
    """

    def __init__(self, num_v_heads: int, k_dim: int, v_dim: int):
        self.states = np.zeros((num_v_heads, k_dim, v_dim), dtype=np.float32)
        self.k_dim = k_dim
        self.v_dim = v_dim

    def reset(self, head: int | None = None) -> None:
        if head is None:
            self.states[:] = 0.0
        else:
            self.states[head] = 0.0

    def bytes(self) -> int:
        return self.states.nbytes


def gdn_prefill(
    state: GDNState,
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    a: np.ndarray,
    beta: np.ndarray,
    a_log: np.ndarray,
) -> np.ndarray:
    """prefill：沿时间步逐步递推（seq 维不可并行，numba 内循环已足够）。

    - ``q``: [seq, num_k_heads, k_dim]，``k``: [seq, num_k_heads, k_dim]；
    - ``v``: [seq, num_v_heads, v_dim]，``a``/``beta``: [seq, num_v_heads]；
    - ``a_log``: [num_v_heads]；
    - 每个 v-head 共享 k/q 头（``num_v_heads // num_k_heads`` 个 v-head 共用 1 个 k 头）；
    - 返回 [seq, num_v_heads, v_dim]，**并更新 state**（prefill 结束后状态即前缀状态，
      可序列化进 KV 块做 prefix cache——§3.4-6 GDN 状态与 KV 块协同）。
    """
    seq, num_k_heads, kd = q.shape
    num_v_heads = v.shape[1]
    vd = v.shape[2]
    g = num_v_heads // num_k_heads
    out = np.empty((seq, num_v_heads, vd), dtype=np.float32)
    q32 = q.astype(np.float32, copy=False)
    k32 = k.astype(np.float32, copy=False)
    v32 = v.astype(np.float32, copy=False)
    for t in range(seq):
        for h in range(num_v_heads):
            hk = h // g
            gdn_step(
                state.states[h],
                q32[t, hk],
                k32[t, hk],
                v32[t, h],
                float(a[t, h]),
                float(beta[t, h]),
                float(a_log[h]),
                out[t, h],
            )
    return out


def short_conv1d(x: np.ndarray, weight: np.ndarray, kernel_size: int) -> np.ndarray:
    """causal 短卷积（GDN 的 conv1d，kernel=4）。

    ``x``: [seq, channels]（或 [seq, heads, dim]，按最后两维广播）；
    ``weight``: [channels, kernel_size]（每通道独立核）。
    """
    if x.ndim == 3:
        seq, heads, dim = x.shape
        flat = x.reshape(seq, heads * dim)
        out = _conv1d_2d(flat, weight, kernel_size).reshape(seq, heads, dim)
        return out
    return _conv1d_2d(x, weight, kernel_size)


@njit(cache=True, fastmath=True)
def _conv1d_2d(x: np.ndarray, weight: np.ndarray, kernel_size: int) -> np.ndarray:
    """causal conv1d：out[t, c] = Σ_j w[c, j] · x[t - j, c]（j=0..K-1，t-j<0 补 0）。

    ``x``: [seq, C]；``weight``: [C, K]。
    """
    seq, c = x.shape
    out = np.zeros((seq, c), dtype=np.float32)
    for t in range(seq):
        for ch in range(c):
            s = 0.0
            for j in range(kernel_size):
                tt = t - j
                if tt < 0:
                    break
                s += weight[ch, j] * x[tt, ch]
            out[t, ch] = s
    return out
