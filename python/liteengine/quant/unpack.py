"""量化解包工具：查表 LUT、列打包解包（AWQ 列序 / 通用线性）、共享反量化公式。"""
from __future__ import annotations

from numpy import (asarray, ascontiguousarray, empty, float16, float32, int16,
                   int32, ndarray, repeat, stack, uint8, zeros)

__all__ = ["_LO_LUT", "_HI_LUT", "_REVERSE_AWQ_PACK_ORDER", "_OUT_DTYPES",
           "_unpack_int4_colwise", "_unpack_colwise", "_dequant_formula"]

# 预计算查表：字节(0..255) → 低 4 位 / 高 4 位 两个 int4 值（0..15）
_LO_LUT: ndarray = zeros(256, dtype=uint8)
_HI_LUT: ndarray = zeros(256, dtype=uint8)
for _b in range(256):
    _LO_LUT[_b] = _b & 0x0F
    _HI_LUT[_b] = (_b >> 4) & 0x0F

# AWQ 非标准打包列序（vLLM _REVERSE_AWQ_PACK_ORDER / gptqmodel reverse_awq_order）：
# int32 内 8 个 int4 按 [0,2,4,6,1,3,5,7]（AWQ_ORDER）存储，线性解包后须按逆序重排还原真实列序。
_REVERSE_AWQ_PACK_ORDER = (0, 4, 1, 5, 2, 6, 3, 7)

# 可选输出 dtype
_OUT_DTYPES = {"float16": float16, "float32": float32}


def _unpack_int4_colwise(packed: ndarray) -> ndarray:
    """AWQ 列打包解包：把 [*, in//8] 的 int32 按列展开为 [*, in]（含 AWQ 列序还原）。

    小端字节序下，int32 的第 k 个字节对应 bits [8k, 8k+8)：
        低 4 位 = int4 #(2k)，高 4 位 = int4 #(2k+1)
    一次查表 + 交错 reshape 完成展开，再按 AWQ 逆序重排还原真实列序。
    """
    packed = ascontiguousarray(packed)
    out_dim = packed.shape[-1] * 8
    byte_view: ndarray = packed.view(uint8)          # [*, in//2]
    lo: ndarray = _LO_LUT[byte_view]                 # 每字节低 4 位 → int4 #(2k)
    hi: ndarray = _HI_LUT[byte_view]                 # 高 4 位 → int4 #(2k+1)
    interleaved = stack((lo, hi), axis=-1)        # [*, in//2, 2]
    out = interleaved.reshape(packed.shape[:-1] + (out_dim,))   # [*, in]
    # AWQ 列序还原：每个 int32 的 8 个槽位按逆序重排（否则专家权重列错乱 → 输出乱码）
    return out.reshape(out.shape[0], -1, 8)[:, :, _REVERSE_AWQ_PACK_ORDER].reshape(out.shape)


def _unpack_colwise(packed: ndarray, bits: int = 4) -> ndarray:
    """通用列打包解包（线性序，无 AWQ 列序重排）：[*, in//(32//bits)] → [*, in]。"""
    packed = ascontiguousarray(packed)
    factor = 32 // bits
    out_dim = packed.shape[-1] * factor
    if bits == 4:
        byte_view: ndarray = packed.view(uint8)
        lo: ndarray = _LO_LUT[byte_view]
        hi: ndarray = _HI_LUT[byte_view]
        return stack((lo, hi), axis=-1).reshape(packed.shape[:-1] + (out_dim,))
    mask = (1 << bits) - 1
    out = empty(packed.shape[:-1] + (out_dim,), dtype=int32)
    for i in range(factor):
        out[..., i::factor] = (packed >> (bits * i)) & mask
    return out


def _dequant_formula(w: ndarray, z: ndarray | None, scales: ndarray, group_size: int,
                     sym: bool, out_dtype) -> ndarray:
    """共享反量化公式：``(w - z) * s``（``sym`` 对称时无 z），group 沿 out 维 repeat。"""
    s = repeat(asarray(scales, dtype=float32), group_size, axis=0)
    if sym:
        dequant = w.astype(float32) * s
    else:
        z = repeat(z, group_size, axis=0)
        dequant = (w.astype(int16) - z.astype(int16)).astype(float32) * s
    return dequant.astype(out_dtype)
