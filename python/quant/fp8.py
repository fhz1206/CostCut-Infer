"""FP8 反量化（NVFP8：E4M3 / E5M2 浮点格式）。

- E4M3：1 符号 + 4 指数（偏置 7）+ 3 尾数，最大 448.0，e=15 为 NaN（无 inf）
- E5M2：1 符号 + 5 指数（偏置 15）+ 2 尾数，最大 57344.0，e=31 为 inf/NaN
- 权重存储：uint8 字节（F8_E4M3 / F8_E5M2）+ 每张量/每轴 fp32 缩放
"""
from __future__ import annotations

import numpy as np
from numpy import float32, ndarray, uint8

from quant.config import QuantConfig
from engine.registry import register_quant_method

__all__ = ["e4m3_to_f32", "e5m2_to_f32", "dequantize_fp8"]


def e4m3_to_f32(b: int) -> float:
    """FP8 E4M3 字节 → fp32。"""
    s = (b >> 7) & 1
    e = (b >> 3) & 0xF
    m = b & 0x7
    if e == 0:
        v = m * 2.0 ** -6                 # 次正规：2^(1-7) × m/8
    elif e == 15:
        return float("nan")
    else:
        v = (1.0 + m / 8.0) * 2.0 ** (e - 7)
    return -v if s else v


def e5m2_to_f32(b: int) -> float:
    """FP8 E5M2 字节 → fp32。"""
    s = (b >> 7) & 1
    e = (b >> 2) & 0x1F
    m = b & 0x3
    if e == 0:
        v = m * 2.0 ** -14
    elif e == 31:
        return float("inf") if m == 0 else float("nan")
    else:
        v = (1.0 + m / 4.0) * 2.0 ** (e - 15)
    return -v if s else v


def dequantize_fp8(qweight: ndarray, scales, e4m3: bool = True,
                   dtype: str = "float32") -> ndarray:
    """FP8 权重反量化：qweight 为 uint8 字节，乘缩放（vLLM 约定——灵活广播）。

    缩放支持：per-tensor（标量）/ per-channel（[out] 或 [in] 向量）/ 2D [out, in]。
    """
    conv = e4m3_to_f32 if e4m3 else e5m2_to_f32
    w = np.array([conv(int(b)) for b in qweight.reshape(-1)], dtype=float32)
    w = w.reshape(qweight.shape)
    sc = np.asarray(scales, dtype=float32)
    # per-channel（out 维）缩放：[out] → (out, 1) 广播（vLLM fp8 per-channel 约定）
    if sc.ndim == 1 and w.ndim >= 2 and sc.shape[0] == w.shape[0]:
        sc = sc[:, None]
    out = w * sc
    return out.astype(float32 if dtype == "float32" else np.float16)


@register_quant_method("fp8")
@register_quant_method("e4m3")
@register_quant_method("e5m2")
def _handle_fp8(qweight, qzeros, scales, cfg: QuantConfig, dtype: str = "float32") -> ndarray:
    """注册表处理器：quant_method == fp8/e4m3/e5m2。"""
    e4m3 = cfg.dtype.lower() not in ("e5m2", "fp8_e5m2")
    return dequantize_fp8(qweight, scales, e4m3=e4m3, dtype=dtype)
