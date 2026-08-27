"""ccut.quant.online — 在线量化（§3.6-3，对齐 vLLM _ONLINE_SHORTHANDS 子集）。

BF16/FP16 checkpoint 走**加载期量化**：权重流经 ExpertReader/WeightRing 时
用 numba ``scaled_quantize``（per-channel，memoryless_minmax observer 在线算 amax）
就地量化进 ring buffer——磁盘仍存 BF16，运行时按量化路径走（W8A16 精确）。

与 checkpoint 自带量化**互斥**（registry.resolve_checkpoint_quant 校验）。
在线简写（大小写不敏感）：``fp8_per_token`` / ``int8_per_channel_weight_only`` /
``int8_per_token`` / ``mxfp8`` / ``nvfp4_per_token``。
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import numpy as np

from ccut.quant import kernels
from ccut.quant.registry import QuantizationConfig
from ccut.quant.spec import (
    K_FP8_STATIC_CHANNEL_SYM,
    K_INT8_STATIC_CHANNEL_SYM,
    K_MXFP8_DYNAMIC,
    K_NVFP4_DYNAMIC,
    K_BF16,
    LayerQuantSpec,
    QuantDType,
    ScaleDesc,
    ScaleStrategy,
    get_quant_key,
)

__all__ = ["OnlineQuantConfig", "quantize_buffer_inplace"]


def quantize_buffer_inplace(
    raw_bf16: bytes | np.ndarray,
    scale: np.ndarray | None,
    out: np.ndarray,
    in_features: int | None = None,
) -> None:
    """加载期量化核：BF16 字节段 → FP8 码（per-channel，memoryless minmax）。

    ``raw_bf16``: ``[out, in]`` BF16 字节（2 字节/元素）；
    ``scale``: [out] float32（每输出通道；None 时本函数就地按 amax/448 算）；
    ``out``: [out, in] uint8（FP8 码，写回 ring buffer）。
    """
    w16 = np.frombuffer(
        raw_bf16 if isinstance(raw_bf16, (bytes, bytearray)) else raw_bf16,
        dtype=np.uint16,
    )
    if in_features is None:
        in_features = out.shape[1]
    w16 = w16.reshape(-1, in_features)
    w32 = (w16.astype(np.uint32) << 16).view(np.float32)
    o, _n = w32.shape
    # per-channel amax（memoryless：流式单次 pass 即得，无存储 observer）
    amax = np.abs(w32).max(axis=1)
    for r in range(o):
        s = float(scale[r]) if scale is not None else (float(amax[r]) / 448.0 or 1.0)
        s = s if s > 0 else 1.0
        q = np.clip(w32[r] / s, -448.0, 448.0)
        out[r] = kernels.float32_to_fp8_e4m3(q)


class OnlineQuantConfig(QuantizationConfig):
    """在线量化配置：所有层统一走简写 spec（ignore 正则除外）。"""

    name = "online"
    _instances: ClassVar[list["OnlineQuantConfig"]] = []

    def __init__(self, shorthand: dict, weight_key_name: str, ignore_patterns: list[str]):
        self.shorthand = shorthand
        self.weight_key_name = weight_key_name
        self.ignore_patterns = [__import__("re").compile(p) for p in (ignore_patterns or [])]
        self._specs: dict[str, LayerQuantSpec] = {}

    @classmethod
    def from_config(
        cls,
        shorthand: dict,
        model_dir: str | Path | None = None,
        ignore_patterns: list[str] | None = None,
    ) -> "OnlineQuantConfig":
        key_name = shorthand["weight"]
        cls._instances.append(cls.__new__(cls))
        return cls(shorthand, key_name, ignore_patterns or [])

    def is_layer_skipped(self, layer_name: str) -> bool:
        for rx in self.ignore_patterns:
            if rx.search(layer_name):
                return True
        return False

    def get_layer_spec(self, layer_name: str) -> LayerQuantSpec:
        if layer_name in self._specs:
            return self._specs[layer_name]
        if self.is_layer_skipped(layer_name):
            spec = LayerQuantSpec(
                layer_name=layer_name,
                weight_key=get_quant_key(K_BF16),
                skipped=True,
                quant_method=self.name,
            )
            self._specs[layer_name] = spec
            return spec
        base = get_quant_key(self.weight_key_name)
        # 在线量化：scale 运行时算（不落盘）→ ScaleDesc 仅占位（offset/length=0）
        scales = (ScaleDesc(name=f"{layer_name}.online_scale", dtype="F32", shape=()),)
        spec = LayerQuantSpec(
            layer_name=layer_name,
            weight_key=base,
            scales=scales,
            skipped=False,
            quant_method=self.name,
        )
        self._specs[layer_name] = spec
        return spec

    def validate(self) -> None:
        return None
