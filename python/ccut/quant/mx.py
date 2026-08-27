"""ccut.quant.mx — MxMethod（mxfp8/mxfp4/nvfp4：E8M0 指数 scale + 分组广播）。

MX 格式（OCP Microscaling，§3.6-2）：
- 数据元素 = FP8 E4M3（mxfp8）/ FP4（mxfp4/nvfp4）；
- 每 32 元素（nvfp4 为 16）共享一个 **E8M0 指数 scale**（8-bit 纯指数，bias 127）；
- nvfp4 另有 per-tensor FP8 E4M3 二次 scale。
CPU 无 MX dot → dequant 内核 = E8M0 指数展开 + group 广播（P8 🟡 兼容）。
"""

from __future__ import annotations

import numpy as np

from ccut.quant import kernels
from ccut.quant.method import QuantizeMethodBase
from ccut.quant.spec import LayerQuantSpec, QuantDType

__all__ = ["MxLinearMethod"]


class MxLinearMethod(QuantizeMethodBase):
    """MX 格式线性层：dequant → float matmul（checkpoint 兼容 + 在线路径）。"""

    compute_path = "w8a16"

    def __init__(self, spec: LayerQuantSpec):
        self.spec = spec
        key = spec.effective_key()
        self.weight_dtype = key.weight_dtype
        self.group_size = key.group_shape.group_size or 32
        self.scale_name = spec.scales[0].name if spec.scales else None
        self.scale2_name = spec.scales[1].name if len(spec.scales) > 1 else None

    def create_weights(self, spec: LayerQuantSpec) -> None:
        QuantizeMethodBase.create_weights(self, spec)

    def apply(self, weight_bytes, scales: dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
        raw = np.frombuffer(
            weight_bytes if isinstance(weight_bytes, (bytes, bytearray)) else weight_bytes,
            dtype=np.uint8,
        )
        in_features = x.shape[1]
        if self.weight_dtype == QuantDType.MXFP8:
            w_fp8 = raw.reshape(-1, in_features)
            scale_e8m0 = scales[self.scale_name].astype(np.float32) if self.scale_name else None
            if scale_e8m0 is None:
                raise ValueError(f"MXFP8 层缺 E8M0 scale（{self.scale_name}）")
            w = self._dequant_mx(w_fp8, scale_e8m0, self.group_size)
            return x @ w.T
        if self.weight_dtype in (QuantDType.MXFP4, QuantDType.NVFP4):
            packed = raw.reshape(-1, in_features // 2)
            scale_e8m0 = scales[self.scale_name].astype(np.float32) if self.scale_name else None
            w = self._dequant_mx4(packed, scale_e8m0, self.group_size)
            if self.weight_dtype == QuantDType.NVFP4 and self.scale2_name:
                w = w * float(np.float32(scales[self.scale2_name]).mean())
            return x @ w.T
        raise ValueError(f"MX 方法不支持权重 dtype {self.weight_dtype}")

    # -- dequant 内核 ---------------------------------------------------------
    def _dequant_mx(self, w_fp8: np.ndarray, scale_e8m0: np.ndarray, group_size: int) -> np.ndarray:
        """MXFP8 dequant：``w[r, c] = fp8(w) · 2^(e8m0[rg] - 127)``。"""
        m, n = w_fp8.shape
        groups_per_row = (n + group_size - 1) // group_size
        scale = kernels.mx_e8m0_to_float(scale_e8m0.reshape(-1))
        out = np.empty((m, n), dtype=np.float32)
        vals = kernels.fp8_e4m3_to_float32(w_fp8.reshape(-1).astype(np.uint8))
        vals = vals.reshape(m, n)
        for r in range(m):
            for g in range(groups_per_row):
                lo = g * group_size
                hi = min(lo + group_size, n)
                out[r, lo:hi] = vals[r, lo:hi] * scale[r * groups_per_row + g]
        return out

    def _dequant_mx4(self, packed: np.ndarray, scale_e8m0: np.ndarray, group_size: int) -> np.ndarray:
        """MXFP4/NVFP4 dequant：4-bit 码 → 对称 ±1/±0.5/±1.5（E2M1）× E8M0 组 scale。"""
        m, n2 = packed.shape
        n = n2 * 2
        scale = kernels.mx_e8m0_to_float(scale_e8m0.reshape(-1)) if scale_e8m0 is not None else None
        # E2M1 码表：0→0, 1→0.5, 2→1.0, 3→1.5（有符号偏移码）
        lut = np.array([0.0, 0.5, 1.0, 1.5, -0.0, -0.5, -1.0, -1.5, 0.0, 0.5, 1.0, 1.5, -0.0, -0.5, -1.0, -1.5], dtype=np.float32)
        lo = packed & 0x0F
        hi = (packed >> 4) & 0x0F
        w = np.empty((m, n), dtype=np.float32)
        w[:, 0::2] = lut[lo]
        w[:, 1::2] = lut[hi]
        if scale is not None:
            groups_per_row = (n + group_size - 1) // group_size
            for r in range(m):
                for g in range(groups_per_row):
                    lo_ = g * group_size
                    hi_ = min(lo_ + group_size, n)
                    w[r, lo_:hi_] = w[r, lo_:hi_] * scale[r * groups_per_row + g]
        return w
