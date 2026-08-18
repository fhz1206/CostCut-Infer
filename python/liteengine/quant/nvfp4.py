"""NVFP4 反量化（NVIDIA E2M1 + 分级块缩放）。

- E2M1：1 符号 + 2 指数（偏置 1）+ 1 尾数，值集 {0, 0.5, 1, 1.5, 2, 3, 4, 6} ±
- 分级缩放：``x = e2m1(q) * s_block * s_global``
  - ``s_block``：每 16 个连续元素一个 FP8 E4M3 缩放（已转换为 f32 传入）
  - ``s_global``：作用于整个张量的 FP32 缩放
- 权重存储：uint8 打包（每字节 2 个 4-bit E2M1，低 4 位在前）
"""
from __future__ import annotations

import numpy as np
from numpy import float32, ndarray

from liteengine.quant.config import QuantConfig
from liteengine.quant.fp8 import e4m3_to_f32
from liteengine.registry import register_quant_method

__all__ = ["e2m1_to_f32", "dequantize_nvfp4"]


def e2m1_to_f32(q: int) -> float:
    """NVFP4 E2M1 4 位值 → fp32。"""
    s = (q >> 3) & 1
    e = (q >> 1) & 0x3
    m = q & 0x1
    if e == 0:
        v = m * 0.5                       # 次正规：2^0 × m/2
    else:
        v = (1.0 + m / 2.0) * 2.0 ** (e - 1)
    return -v if s else v


def dequantize_nvfp4(qweight: ndarray, s_block: ndarray, s_global: float,
                     block_size: int = 16, dtype: str = "float32") -> ndarray:
    """NVFP4 权重反量化：``x = e2m1(qweight) * s_block * s_global``。

    - qweight：uint8 打包（每字节 2 个 4-bit E2M1，低 4 位在前）
    - s_block：每 block_size 个元素一个 E4M3 块缩放（已转 f32），长度 = 元素数 / block_size
    - s_global：全局 FP32 缩放
    """
    flat_len = int(qweight.size) * 2
    q = np.zeros(flat_len, dtype=np.uint8)
    q[0::2] = qweight.reshape(-1) & 0xF
    q[1::2] = (qweight.reshape(-1) >> 4) & 0xF
    w = np.array([e2m1_to_f32(int(x)) for x in q], dtype=float32)
    blocks = w.reshape(-1, block_size)
    sb = np.asarray(s_block.reshape(-1), dtype=float32)
    out = (blocks * sb[:, None]).reshape(-1) * float(s_global)
    return out.astype(float32 if dtype == "float32" else np.float16)


@register_quant_method("nvfp4")
def _handle_nvfp4(qweight, qzeros, scales, cfg: QuantConfig, dtype: str = "float32") -> ndarray:
    """注册表处理器：quant_method == "nvfp4"。

    约定：``scales`` 为 [num_blocks, 2]——第 0 列为 E4M3 块缩放（转 f32），
    第 1 列为全局缩放（同值广播）。
    """
    sb = np.asarray(scales[:, 0], dtype=float32)
    sg = float(np.asarray(scales[:, 1], dtype=float32).mean())
    return dequantize_nvfp4(qweight, sb, sg, dtype=dtype)
