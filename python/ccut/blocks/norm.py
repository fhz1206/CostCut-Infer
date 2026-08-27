"""ccut.blocks.norm — RMSNorm（所有家族通用）。

Ornith/Qwen3/Kimi：``y = x * rsqrt(mean(x^2) + eps) * w``，
数值路径用 float32 累加（bf16 输入输出，防下溢）。
"""

from __future__ import annotations

import numpy as np

__all__ = ["rms_norm"]


def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """RMSNorm。``x``: [batch, dim] 或 [seq, dim] float32；``weight``: [dim]（可选 None=无仿射）。

    返回与 ``x`` 同 shape 的 float32。数值按 vLLM fused 语义：
    float32 求均方 → rsqrt → 乘 weight（weight 缺省全 1）。
    """
    x32 = x.astype(np.float32, copy=False)
    var = np.mean(x32 * x32, axis=-1, keepdims=True)
    y = x32 * (var + eps) ** -0.5
    if weight is not None:
        y = y * weight.astype(np.float32)
    return y


def rms_norm_inplace(x: np.ndarray, weight: np.ndarray, eps: float = 1e-6, out: np.ndarray | None = None) -> np.ndarray:
    """RMSNorm（可选写回 ``out``，供 WeightRing 复用 buffer）。"""
    y = rms_norm(x, weight, eps)
    if out is not None:
        np.copyto(out, y)
        return out
    return y
