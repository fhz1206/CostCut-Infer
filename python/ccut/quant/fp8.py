"""ccut.quant.fp8 — Fp8Method（§3.6-2：W8A16 主路径 / W8A8 对照）。

CPU 诚实定位（D5）：FP8 在 CPU 无原生 dot——
- **W8A16（默认）**：dequant 后精确 BF16/FP32 matmul，磁盘/带宽减半是主收益；
- **W8A8**：激活动态量化路径，数值等价 W8A16 + 量化误差，作对照基准（`fp8_compute_mode=w8a8`）。
"""

from __future__ import annotations

import numpy as np

from ccut.quant import kernels
from ccut.quant.method import LinearMethod, QuantizeMethodBase
from ccut.quant.spec import LayerQuantSpec, QuantDType, ScaleStrategy

__all__ = ["Fp8LinearMethod", "Fp8Config"]


class Fp8LinearMethod(QuantizeMethodBase):
    """FP8 线性层方法：读 FP8 段 + per-channel scale → dequant → matmul。"""

    def __init__(self, spec: LayerQuantSpec, compute_mode: str = "w8a16"):
        self.spec = spec
        self.compute_mode = compute_mode.casefold()
        key = spec.effective_key()
        self.compute_path = "w8a8" if self.compute_mode == "w8a8" else "w8a16"
        self._inner = LinearMethod(key, scale_name=spec.scales[0].name if spec.scales else None,
                                   compute_mode=self.compute_mode)

    def create_weights(self, spec: LayerQuantSpec) -> None:
        super().create_weights(spec)

    def apply(self, weight_bytes, scales: dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
        return self._inner.apply(weight_bytes, scales, x)


class Fp8Config:
    """占位：FP8 独立 checkpoint 格式（非 compressed-tensors 包装）走此路径。

    当前 Ornith 与全部已测 checkpoint 均为 compressed-tensors 包装，本类保留给
    裸 FP8 格式（quant_method="fp8"）；未实现 from_config → 显式 NotImplementedError。
    """

    name = "fp8"

    @classmethod
    def from_config(cls, quant_cfg: dict, model_dir=None):  # type: ignore[no-untyped-def]
        raise NotImplementedError("裸 fp8 格式暂未实现，请检查 checkpoint 是否 compressed-tensors 包装")

    def get_layer_spec(self, layer_name: str):  # type: ignore[no-untyped-def]
        raise NotImplementedError
