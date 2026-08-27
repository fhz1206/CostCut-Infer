"""ccut.blocks.attn_kimi_linear — Kimi K2/K3 线性注意力（KDA，L0 家族积木）。

Kimi Linear Attention（KDA）与 GDN 同属 Gated Delta 家族（kimi_linear_attn 层）：
- 无独立 k 头组：每 token 一个**门控衰减 + delta 更新**的递归状态；
- 状态维度 ``[d_k, d_v]`` 每头，与 GDN 的 ``GDNState`` 接口兼容——
  本模块复用 :func:`ccut.blocks.attn_gdn.gdn_step` 递推核（数值语义一致：
  GDN 是 KDA 的特定参数化，delta rule + 门控衰减）。

差异点（vs GDN）：
- Kimi K2 的 linear 层**每层都有**（无 full_attn 间隔），KV 字节 = 0（纯递归状态）；
- ``conv1d kernel`` 通常为 4（与 GDN 相同）；
- 输出门控：``silu(z) ⊙ o``（与 GDN attn_output_gate 一致）。

因此本模块是**薄适配层**：把 Kimi linear_attn 的投影布局（qkv 融合方式）
映射到 GDN 递推核，避免重复实现数值路径。
"""

from __future__ import annotations

import numpy as np

from ccut.blocks.attn_gdn import GDNState, gdn_prefill, gdn_step, short_conv1d

__all__ = ["KimiLinearState", "kimi_linear_prefill", "kimi_linear_step"]


class KimiLinearState(GDNState):
    """Kimi 线性注意力状态（GDNState 别名，接口兼容）。"""


def kimi_linear_prefill(
    state: KimiLinearState,
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    a: np.ndarray,
    beta: np.ndarray,
    a_log: np.ndarray,
    conv_weight: np.ndarray | None = None,
    kernel_size: int = 4,
) -> np.ndarray:
    """Kimi linear 层 prefill（短卷积 + GDN 递推核）。

    参数布局与 :func:`gdn_prefill` 相同（Kimi 与 GDN 的 q/k/v 头布局一致：
    v-heads = 2× k-heads）。``conv_weight`` 非 None 时对 q/k/v 先做 causal conv1d。
    """
    if conv_weight is not None:
        q = short_conv1d(q, conv_weight["q"], kernel_size)
        k = short_conv1d(k, conv_weight["k"], kernel_size)
        v = short_conv1d(v, conv_weight["v"], kernel_size)
    return gdn_prefill(state, q, k, v, a, beta, a_log)


def kimi_linear_step(
    state: KimiLinearState,
    head: int,
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    a: float,
    beta: float,
    a_log: float,
) -> np.ndarray:
    """Kimi linear 层 decode 单步（一个 v-head）。返回 [v_dim]。"""
    out = np.empty(state.v_dim, dtype=np.float32)
    gdn_step(state.states[head], q, k, v, a, beta, a_log, out)
    return out
