"""ccut.quant.weight_only — GPTQ/AWQ/bnbf int4/nf4 weight-only dequant（§3.6-2）。

CPU 等价路径（vLLM marlin kernel 为 GPU 专用，⬜ 不适用）：
- int4/nf4 码流式 dequant → float matmul（W4A16）；
- **对 R10 层流式收益最大**：磁盘体积再减半 → 流式带宽需求减半。
"""

from __future__ import annotations

import numpy as np

from ccut.quant import kernels
from ccut.quant.method import QuantizeMethodBase
from ccut.quant.spec import LayerQuantSpec, QuantDType

__all__ = ["WeightOnlyMethod"]

_NF4_LUT = np.array(
    [-1.0, -0.6961928, -0.39238566, -0.08857854, 0.21522858, 0.51903564, 0.8228427, 1.1266499],
    dtype=np.float32,
)


class WeightOnlyMethod(QuantizeMethodBase):
    """W4 weight-only（GPTQ/AWQ/bnbf 等价）：流式 dequant + float matmul。"""

    compute_path = "w4a16"

    def __init__(self, spec: LayerQuantSpec):
        self.spec = spec
        self.weight_dtype = spec.effective_key().weight_dtype
        self.scale_name = spec.scales[0].name if spec.scales else None

    def create_weights(self, spec: LayerQuantSpec) -> None:
        QuantizeMethodBase.create_weights(self, spec)

    def apply(self, weight_bytes, scales: dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
        raw = np.frombuffer(
            weight_bytes if isinstance(weight_bytes, (bytes, bytearray)) else weight_bytes,
            dtype=np.uint8,
        )
        in_features = x.shape[1]
        packed = raw.reshape(-1, in_features // 2)
        scale = scales.get(self.scale_name) if self.scale_name else None
        if scale is None:
            raise ValueError(f"W4 层缺 scale 张量（{self.scale_name}）")
        out = np.empty((packed.shape[0], in_features), dtype=np.float32)
        if self.weight_dtype == QuantDType.INT4:
            kernels.int4_dequant_row(packed, scale, out)
        elif self.weight_dtype == QuantDType.NF4:
            kernels.nf4_dequant_row(packed, scale, _NF4_LUT, out)
        else:
            raise ValueError(f"weight-only 不支持 {self.weight_dtype}")
        return x @ out.T
