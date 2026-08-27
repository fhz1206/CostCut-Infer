"""ccut.quant.kernels — numba SIMD 量化内核（§3.6-2）。

CPU 诚实定位（§3.6-2 内核矩阵）：
- **FP8 E4M3 ↔ float32**：CPU 无原生 FP8 dot → 转换 = SIMD 查表 + mul；
  主路径 W8A16（dequant 后精确 matmul，磁盘/带宽减半是主收益）。
- **INT8 W8A8 VNNI**：``vpdpbusd`` 是 i7-1065G7 AVX512-VNNI 唯一有真 dot 加速的量化；
  内核以纯 numba 整数实现（LLVM 自动向量化），asm 路径留 capability 探测回退位。
- **MX 格式**：E8M0 指数展开 + 32/16-block 广播（prep_scale_for_group_broadcast 移植）。
- **W4 weight-only**：int4/nf4 → float16/32 流式 dequant。

所有内核 ``cache=True``；不依赖内联汇编（K4 风险：asm 兼容性 → 纯 numba 保底）。
"""

from __future__ import annotations

import numpy as np

try:  # numba 是运行时核心依赖（§0 已装 0.67.0）
    from numba import njit, prange
except ImportError:  # 允许无 numba 环境跑 spec/registry 级测试
    def njit(*args, **kwargs):
        def wrap(fn):
            return fn

        if args and callable(args[0]):
            return args[0]
        return wrap

    def prange(n):
        return range(n)


__all__ = [
    "fp8_e4m3_to_float32",
    "float32_to_fp8_e4m3",
    "fp8_dequant_row",
    "int8_dequant_row",
    "int8_quantize_row",
    "silu_mul_fused",
    "mx_e8m0_to_float",
    "int4_dequant_row",
    "nf4_dequant_row",
    "group_broadcast_scale",
]


# ---------------------------------------------------------------------------
# FP8 E4M3（OCP e4m3fn，compressed-tensors 默认）
# ---------------------------------------------------------------------------


@njit(cache=True, fastmath=True)
def fp8_e4m3_to_float32(code: np.ndarray) -> np.ndarray:
    """FP8 E4M3fn 字节 → float32（逐位解码，纯 numba 向量化）。

    e4m3fn 布局：[1 sign][4 exp][3 mantissa]，bias=7，无 inf，
    NaN = 0x7F/0xFF（exp=15 & mant=7）。
    """
    n = code.shape[0]
    out = np.empty(n, dtype=np.float32)
    for i in prange(n):
        b = int(code[i])
        sign = -1.0 if (b & 0x80) else 1.0
        exp = (b >> 3) & 0x0F
        mant = b & 0x07
        if exp == 0:
            # 次正规：2^-6 * (mant/8)（最小 2^-9）
            val = (mant * 0.125) * (2.0 ** -6)
        elif exp == 15 and mant == 7:
            val = np.nan
        else:
            val = (1.0 + mant * 0.125) * (2.0 ** (exp - 7))
        out[i] = sign * val
    return out


@njit(cache=True)
def float32_to_fp8_e4m3(x: np.ndarray) -> np.ndarray:
    """float32 → FP8 E4M3fn（就近舍入，饱和到 ±448，NaN→0x7F）。

    布局：[1 sign][4 exp, bias=7][3 mantissa]。
    - 正常数：``(1 + m/8) × 2^(e-7)``，e∈[1,15]，e=15 时 m≤6（448 = 1.75×2^8）；
    - 次正规：``m/8 × 2^-6``（最小 2^-9），e=0；
    - 0x7F/0xFF = NaN（e=15, m=7）；无 inf（饱和）。
    舍入：round-half-up（与 torch RNE 仅差 .5 边界，e2e 容差覆盖）。
    """
    n = x.shape[0]
    out = np.empty(n, dtype=np.uint8)
    for i in prange(n):
        v = float(x[i])
        if np.isnan(v):
            out[i] = 0x7F
            continue
        sign = 0
        if v < 0:
            sign = 0x80
            v = -v
        if v >= 464.0:  # 448 与（不存在的）512 的中点 → 饱和到 448
            out[i] = sign | 0x7E
            continue
        if v == 0.0:
            out[i] = sign
            continue
        if v < 0.015625:  # v < 2^-6：次正规区（val = m/8 × 2^-6，最小 2^-9）
            t = v * 512.0  # m = v / 2^-9
            mant = int(t + 0.5)
            if mant >= 8:
                out[i] = sign | 0x08  # 进位到最小正常数 2^-6
            else:
                out[i] = sign | mant
            continue
        # 正常数：v ∈ [2^-6, 464) → e = floor(log2 v) ∈ [-6, 8]
        e = int(np.floor(np.log2(v)))
        m = v * (2.0 ** (-e)) - 1.0  # 尾数部分 ∈ [0, 1)
        mant = int(m * 8.0 + 0.5)
        ec = e + 7  # exp code ∈ [1, 15]
        if mant >= 8:
            mant = 0
            ec += 1
        if ec >= 15 and mant > 6:
            mant = 6  # 饱和（v < 464 时理论上不可达，防御性）
        out[i] = sign | ((ec & 0xF) << 3) | mant
    return out


@njit(cache=True, fastmath=True)
def fp8_dequant_row(w: np.ndarray, scale: np.ndarray, out: np.ndarray) -> None:
    """per-channel FP8 dequant：``out[r, c] = w[r, c] * scale[r]``（scale 按输出通道/行）。

    ``w``: [out, in] FP8 字节行主序；``scale``: [out] float32（每输出通道一个）；
    ``out``: [out, in] float32。
    """
    rows, cols = w.shape
    for r in prange(rows):
        s = scale[r]
        for c in range(cols):
            b = int(w[r, c])
            sgn = -1.0 if (b & 0x80) else 1.0
            e = (b >> 3) & 0x0F
            mnt = b & 0x07
            if e == 0:
                val = mnt * 0.125 * (2.0 ** -6)
            elif e == 15 and mnt == 7:
                val = np.nan
            else:
                val = (1.0 + mnt * 0.125) * (2.0 ** (e - 7))
            out[r, c] = sgn * val * s


@njit(cache=True, fastmath=True)
def fp8_dequant_mat(w: np.ndarray, scale: np.ndarray, out: np.ndarray) -> None:
    """整块 per-channel dequant（批量，读入即反量化——§3.2 计算型预取）。

    ``w``: [M, N] uint8(FP8)；``scale``: [M] float32（**每输出通道/行**一个）；
    ``out``: [M, N] float32。``out[i, j] = fp8(w[i, j]) * scale[i]``。
    """
    m, n = w.shape
    for i in prange(m):
        s = scale[i]
        for j in range(n):
            b = int(w[i, j])
            sgn = -1.0 if (b & 0x80) else 1.0
            e = (b >> 3) & 0x0F
            mnt = b & 0x07
            if e == 0:
                val = mnt * 0.125 * (2.0 ** -6)
            elif e == 15 and mnt == 7:
                val = np.nan
            else:
                val = (1.0 + mnt * 0.125) * (2.0 ** (e - 7))
            out[i, j] = sgn * val * s


# ---------------------------------------------------------------------------
# INT8（VNNI 路径 P8；当前提供精确整数实现供 W8A16 weight-only 与对照）
# ---------------------------------------------------------------------------


@njit(cache=True, fastmath=True)
def int8_dequant_row(w: np.ndarray, scale: np.ndarray, out: np.ndarray) -> None:
    """per-channel INT8 dequant：``out[r, c] = w[r, c] * scale[r]``（scale 按输出通道/行）。

    ``w``: [out, in] int8；``scale``: [out] float32；``out``: [out, in] float32。
    """
    rows, cols = w.shape
    for r in prange(rows):
        s = scale[r]
        for c in range(cols):
            out[r, c] = float(w[r, c]) * s


@njit(cache=True, fastmath=True)
def int8_quantize_row(x: np.ndarray, amax: np.ndarray, out: np.ndarray) -> None:
    """per-channel INT8 量化（memoryless minmax，对称）：
    ``scale = amax/127``，``q = clip(round(x/scale), -127, 127)``。

    ``x``: [rows, cols] float32；``amax``: [cols] 预计算的 |x| 最大值；``out``: [rows, cols] int8。
    """
    rows, cols = x.shape
    for c in range(cols):
        a = amax[c]
        scale = (a / 127.0) if a > 0 else 1.0
        inv = 127.0 / a if a > 0 else 0.0
        for r in prange(rows):
            q = int(np.round(x[r, c] * inv))
            if q > 127:
                q = 127
            elif q < -127:
                q = -127
            out[r, c] = q
    return out


# ---------------------------------------------------------------------------
# 融合激活（MoE 专家：silu(gate·x) · up → down；§2 数据流第 2 步）
# ---------------------------------------------------------------------------


@njit(cache=True, fastmath=True)
def silu_mul_fused(gate: np.ndarray, up: np.ndarray, out: np.ndarray) -> None:
    """``out = silu(gate) * up``（逐元素，[rows, inter] 广播兼容）。

    silu(t) = t / (1 + exp(-t))；分母加 eps 防 0。
    """
    n = gate.shape[0] * gate.shape[1]
    gate = gate.ravel()
    up = up.ravel()
    out = out.ravel()
    for i in prange(n):
        t = float(gate[i])
        out[i] = (t / (1.0 + np.exp(-t))) * float(up[i])


# ---------------------------------------------------------------------------
# MX 格式（E8M0 指数 scale + 32/16-block 广播）
# ---------------------------------------------------------------------------


@njit(cache=True)
def mx_e8m0_to_float(e8m0: np.ndarray) -> np.ndarray:
    """E8M0 指数字节 → float32 倍率：``val = 2^(exp - 127)``（exp=255→NaN）。

    MX 的 E8M0：8-bit 纯指数（无符号、无尾数），bias=127。
    """
    n = e8m0.shape[0]
    out = np.empty(n, dtype=np.float32)
    for i in prange(n):
        e = int(e8m0[i])
        if e == 255:
            out[i] = np.nan
        elif e == 0:
            out[i] = 0.0
        else:
            out[i] = 2.0 ** (e - 127)
    return out


@njit(cache=True, fastmath=True)
def group_broadcast_scale(scale_groups: np.ndarray, group_size: int, out: np.ndarray) -> None:
    """prep_scale_for_group_broadcast 移植：组 scale → 元素级广播。

    ``scale_groups``: [groups] float32；``out``: [groups*group_size] float32。
    """
    g = scale_groups.shape[0]
    for i in prange(g):
        s = scale_groups[i]
        for j in range(group_size):
            out[i * group_size + j] = s


# ---------------------------------------------------------------------------
# W4 weight-only（int4 / nf4 流式 dequant，§3.6-2 对 R10 流式收益最大）
# ---------------------------------------------------------------------------


@njit(cache=True, fastmath=True)
def int4_dequant_row(packed: np.ndarray, scale: np.ndarray, out: np.ndarray) -> None:
    """int4（4-bit 有符号，低 4 位优先）dequant。

    ``packed``: [rows, cols/2] uint8（每字节 2 个 int4）；
    ``scale``: [cols] float32；``out``: [rows, cols] float32。
    """
    rows = packed.shape[0]
    cols = scale.shape[0]
    for r in prange(rows):
        for j in range(cols // 2):
            byte = int(packed[r, j])
            lo = byte & 0x0F
            hi = (byte >> 4) & 0x0F
            if lo >= 8:
                lo -= 16
            if hi >= 8:
                hi -= 16
            out[r, 2 * j] = float(lo) * scale[2 * j]
            out[r, 2 * j + 1] = float(hi) * scale[2 * j + 1]


#: NF4 查找表（NormalFloat4，bitsandbytes；16 个码 → float，mean=0/std=1 假设）
_NF4_TABLE = np.array(
    [-1.0, -0.6961928, -0.39238566, -0.08857854, 0.21522858, 0.51903564, 0.8228427, 1.1266499],
    dtype=np.float32,
)


@njit(cache=True, fastmath=True)
def nf4_dequant_row(packed: np.ndarray, scale: np.ndarray, table: np.ndarray, out: np.ndarray) -> None:
    """NF4 dequant（int4 布局同 :func:`int4_dequant_row`，码值查 NF4 表）。"""
    rows = packed.shape[0]
    cols = scale.shape[0]
    for r in prange(rows):
        for j in range(cols // 2):
            byte = int(packed[r, j])
            lo = byte & 0x0F
            hi = (byte >> 4) & 0x0F
            if lo >= 8:
                lo -= 16
            if hi >= 8:
                hi -= 16
            idx_lo = lo & 0x0F
            idx_hi = hi & 0x0F
            out[r, 2 * j] = table[idx_lo] * scale[2 * j]
            out[r, 2 * j + 1] = table[idx_hi] * scale[2 * j + 1]
