"""ccut.quant.int8 — Int8Method（§3.6-2：W8A8 VNNI 目标路径 / W8A16 weight-only）。

CPU 内核矩阵（§3.6-2）：INT8 W8A8 VNNI（``vpdpbusd``）是 i7-1065G7 AVX512
**唯一有真 dot 加速**的量化：int8 dot 吞吐 = BF16 的 2×（8 字节/32B 向量），
且权重磁盘减半。P8 目标路径；当前提供精确整数实现（LLVM 自动向量化），
VNNI asm 路径留 capability 探测回退位（K4：无编译器 → 纯 numba 保底）。
"""

from __future__ import annotations

import numpy as np

from ccut.quant import kernels
from ccut.quant.method import QuantizeMethodBase
from ccut.quant.spec import LayerQuantSpec, QuantDType

__all__ = ["Int8LinearMethod"]


class Int8LinearMethod(QuantizeMethodBase):
    """INT8 线性层：per-channel 权重 + per-token 激活（W8A8 语义）。

    - ``compute_path=w8a16``：dequant 权重 → float matmul（weight-only 对照）；
    - ``compute_path=w8a8``：激活 per-token 动态量化 → 整数累加 → 反量化
      （VNNI 语义；纯 numba 精确实现，与 torch int8 参考对拍）。
    """

    def __init__(self, spec: LayerQuantSpec, compute_mode: str = "w8a16"):
        self.spec = spec
        self.compute_mode = compute_mode.casefold()
        key = spec.effective_key()
        self.compute_path = "w8a8" if self.compute_mode == "w8a8" and key.act_scale != "none" else "w8a16"
        self.scale_name = spec.scales[0].name if spec.scales else None

    def create_weights(self, spec: LayerQuantSpec) -> None:
        QuantizeMethodBase.create_weights(self, spec)

    def apply(self, weight_bytes, scales: dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
        scale = scales.get(self.scale_name) if self.scale_name else None
        in_features = x.shape[1]
        w = np.frombuffer(
            weight_bytes if isinstance(weight_bytes, (bytes, bytearray)) else weight_bytes,
            dtype=np.int8,
        ).reshape(-1, in_features)
        if self.compute_path == "w8a8":
            return self._w8a8(w, scale, x)
        out = np.empty(w.shape, dtype=np.float32)
        kernels.int8_dequant_row(w, scale, out)
        return out @ x

    # -- W8A8（VNNI 语义精确实现） ------------------------------------------
    def _w8a8(self, w: np.ndarray, scale: np.ndarray, x: np.ndarray) -> np.ndarray:
        """激活 per-token 对称量化 → int8 点积 → 双重反量化。

        数值语义与 torch ``_weight_int8pack_q`` + ``_dynamic_matmul_inplace`` 对齐：
        ``y[b, o] = (Σ_i xq[b, i] · w[i, o]) · x_scale[b] · scale[o]``。
        """
        xq, x_scale = _quantize_per_token(x)
        # 整数 GEMM（numba 精确；VNNI 硬件加速位：asm vpdpbusd 回退点）
        acc = _int8_matmul(xq, w.T)  # [batch, out] int64
        y = acc.astype(np.float32) * x_scale[:, None] * scale[None, :]
        return y


def _quantize_per_token(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """per-token 对称 INT8 量化（memoryless amax，scale=amax/127）。

    计算顺序与 torch 参考（``_dynamic_matmul_inplace`` 前置量化）逐位一致：
    ``sc = amax/127``（float32）→ ``factor = 127/sc``（float32 除法）→
    ``round(x*factor)``（half-even）→ clamp ±127。
    """
    amax = np.abs(x).max(axis=1).astype(np.float32)
    amax = np.where(amax > 0, amax, 1.0).astype(np.float32)
    x_scale = (amax / 127.0).astype(np.float32)
    factor = (127.0 / x_scale).astype(np.float32)
    xq = np.clip(np.round(x * factor[:, None]), -127, 127).astype(np.int8)
    return xq, x_scale


def _int8_matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """int8 × int8 → int64（numba prange 分片；LLVM 自动向量化）。

    a: [M, K] int8；b: [K, N] int8；返回 [M, N] int64。
    """
    from numba import njit, prange

    @njit(cache=True, fastmath=True)
    def _core(a, b):
        m, k = a.shape
        n = b.shape[1]
        out = np.zeros((m, n), dtype=np.int64)
        for i in prange(m):
            for j in range(n):
                s = 0
                for t in range(k):
                    s += a[i, t] * b[t, j]
                out[i, j] = s
        return out

    return _core(a, b)
