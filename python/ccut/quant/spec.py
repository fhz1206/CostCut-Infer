"""ccut.quant.spec — 声明式量化规格（移植 vLLM QuantKey 语义）。

设计原则（§3.6）：量化格式 = **数据声明**，kernel 按 key 派发——
``QuantKey(dtype, scale, scale2, symmetric, group_shape)`` 唯一描述一层权重的
量化布局；method 层只负责 create_weights（解析 scale 布局）/ apply（选 kernel）。

CPU 定位（诚实标注）：
- FP8 在 CPU 无原生 dot → 默认 **W8A16**（dequant 后 BF16/FP32 matmul，磁盘/带宽减半）；
- INT8 W8A8 VNNI（vpdpbusd）是本机 AVX512 唯一有真 dot 加速的量化（P8 目标路径）；
- MX 格式 = E8M0 指数 scale + 128/32 组广播。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "QuantDType",
    "ScaleStrategy",
    "GroupShape",
    "QuantKey",
    "ScaleDesc",
    "LayerQuantSpec",
]


class QuantDType:
    """量化后存储的数值类型（对齐 vLLM QuantKey.dtype）。"""

    BF16 = "bf16"
    FP16 = "fp16"
    FP32 = "fp32"
    INT8 = "int8"
    FP8_E4M3 = "fp8_e4m3"
    FP8_E5M2 = "fp8_e5m2"
    INT4 = "int4"
    NF4 = "nf4"
    MXFP8 = "mxfp8"
    MXFP4 = "mxfp4"
    NVFP4 = "nvfp4"

    ALL = (BF16, FP16, FP32, INT8, FP8_E4M3, FP8_E5M2, INT4, NF4, MXFP8, MXFP4, NVFP4)


class ScaleStrategy:
    """scale 粒度（对齐 compressed-tensors strategy）。"""

    PER_TENSOR = "tensor"
    PER_CHANNEL = "channel"
    PER_TOKEN = "token"
    PER_GROUP = "group"
    NO_SCALE = "none"  # 无量化（BF16 直通）


@dataclass(frozen=True)
class GroupShape:
    """分组粒度（group strategy 用）：行分组 / 列分组大小，None=不分组。"""

    rows: int | None = None  # group_size（沿输入维，如 128）
    cols: int | None = None  # 沿输出维分组（少见）

    def is_grouped(self) -> bool:
        return self.rows is not None and self.rows > 0

    @property
    def group_size(self) -> int:
        return self.rows if self.rows else 0


#: 预定义 QuantKey（对齐 vLLM QuantKey 命名子集，§3.6）
K_FP8_DYNAMIC_TOKEN_SYM = "fp8_dynamic_token_sym"
K_FP8_STATIC_CHANNEL_SYM = "fp8_static_channel_sym"
K_FP8_STATIC_TENSOR_SYM = "fp8_static_tensor_sym"
K_INT8_STATIC_CHANNEL_SYM = "int8_static_channel_sym"
K_INT8_DYNAMIC_TOKEN_SYM = "int8_dynamic_token_sym"
K_INT8_W4A16_SYM = "int4_w4a16_sym"
K_NF4_W4A16 = "nf4_w4a16"
K_MXFP8_DYNAMIC = "mxfp8_dynamic"
K_MXFP4_DYNAMIC = "mxfp4_dynamic"
K_NVFP4_DYNAMIC = "nvfp4_dynamic"
K_BF16 = "bf16"

_KEY_REGISTRY: dict[str, "QuantKey"] = {}


def _reg(name: str, key: "QuantKey") -> "QuantKey":
    _KEY_REGISTRY[name] = key
    return key


def get_quant_key(name: str) -> QuantKey:
    """按预定义名取 QuantKey（大小写不敏感）；未注册显式报错。"""
    key = _KEY_REGISTRY.get(name.casefold())
    if key is None:
        raise KeyError(f"未知 QuantKey {name!r}，已注册: {sorted(_KEY_REGISTRY)}")
    return key


@dataclass(frozen=True)
class QuantKey:
    """一层权重/激活的量化声明（移植 vLLM QuantKey）。

    - ``weight_dtype`` / ``weight_scale``：权重存储 dtype 与 scale 策略；
    - ``act_dtype`` / ``act_scale``：激活 dtype 与 scale 策略；
    - ``symmetric``：对称量化（无 zero-point）；
    - ``group_shape``：group 粒度。
    """

    weight_dtype: str = QuantDType.BF16
    weight_scale: str = ScaleStrategy.NO_SCALE
    act_dtype: str = QuantDType.BF16
    act_scale: str = ScaleStrategy.NO_SCALE
    symmetric: bool = True
    group_shape: GroupShape = field(default_factory=GroupShape)
    name: str = ""

    def is_quantized(self) -> bool:
        return self.weight_scale != ScaleStrategy.NO_SCALE

    def is_weight_only(self) -> bool:
        """权重量化但激活不量化（W4A16 / W8A16）。"""
        return self.is_quantized() and self.act_scale == ScaleStrategy.NO_SCALE

    def is_w8a8(self) -> bool:
        """权重与激活都量化（W8A8，VNNI 路径）。"""
        return self.is_quantized() and self.act_scale != ScaleStrategy.NO_SCALE

    def compute_path(self) -> str:
        """CPU 计算路径：``bf16`` / ``w8a16`` / ``w8a8`` / ``w4a16``。"""
        if not self.is_quantized():
            return "bf16"
        if self.is_weight_only():
            return "w4a16" if self.weight_dtype in (QuantDType.INT4, QuantDType.NF4) else "w8a16"
        return "w8a8"


_reg(K_BF16, QuantKey(QuantDType.BF16, ScaleStrategy.NO_SCALE, QuantDType.BF16, ScaleStrategy.NO_SCALE, True))
_reg(K_FP8_DYNAMIC_TOKEN_SYM, QuantKey(QuantDType.FP8_E4M3, ScaleStrategy.PER_CHANNEL, QuantDType.FP32, ScaleStrategy.PER_TOKEN, True))
_reg(K_FP8_STATIC_CHANNEL_SYM, QuantKey(QuantDType.FP8_E4M3, ScaleStrategy.PER_CHANNEL, QuantDType.FP32, ScaleStrategy.PER_TOKEN, True))
_reg(K_FP8_STATIC_TENSOR_SYM, QuantKey(QuantDType.FP8_E4M3, ScaleStrategy.PER_TENSOR, QuantDType.FP32, ScaleStrategy.PER_TOKEN, True))
_reg(K_INT8_STATIC_CHANNEL_SYM, QuantKey(QuantDType.INT8, ScaleStrategy.PER_CHANNEL, QuantDType.FP32, ScaleStrategy.PER_TOKEN, True))
_reg(K_INT8_DYNAMIC_TOKEN_SYM, QuantKey(QuantDType.INT8, ScaleStrategy.PER_CHANNEL, QuantDType.FP32, ScaleStrategy.PER_TOKEN, True))
_reg(K_INT8_W4A16_SYM, QuantKey(QuantDType.INT4, ScaleStrategy.PER_CHANNEL, QuantDType.FP32, ScaleStrategy.NO_SCALE, True))
_reg(K_NF4_W4A16, QuantKey(QuantDType.NF4, ScaleStrategy.PER_CHANNEL, QuantDType.FP16, ScaleStrategy.NO_SCALE, True))
_reg(K_MXFP8_DYNAMIC, QuantKey(QuantDType.MXFP8, ScaleStrategy.PER_GROUP, QuantDType.BF16, ScaleStrategy.PER_TOKEN, False, GroupShape(rows=32)))
_reg(K_MXFP4_DYNAMIC, QuantKey(QuantDType.MXFP4, ScaleStrategy.PER_GROUP, QuantDType.BF16, ScaleStrategy.PER_TOKEN, False, GroupShape(rows=32)))
_reg(K_NVFP4_DYNAMIC, QuantKey(QuantDType.NVFP4, ScaleStrategy.PER_GROUP, QuantDType.BF16, ScaleStrategy.PER_TOKEN, False, GroupShape(rows=16)))


@dataclass(frozen=True)
class ScaleDesc:
    """scale 张量的布局描述（checkpoint 解析产物）。"""

    name: str  # safetensors 张量名
    dtype: str  # F32 / BF16 / F8_E4M3（MX 二次 scale）/ U8（E8M0 指数）
    shape: tuple[int, ...]
    offset: int = 0  # 数据区相对偏移
    length: int = 0

    def is_e8m0(self) -> bool:
        """MX 格式的 E8M0 指数 scale（8-bit 无符号，低 7 位指数）。"""
        return self.dtype.upper() in ("U8", "F8_E8M0")


@dataclass(frozen=True)
class LayerQuantSpec:
    """一层（线性/MoE）的完整量化规格。

    ``weight_key``/``activation_key`` 分别描述权重与激活；``scales`` 是 scale
    张量清单（含 MX 的二次 scale）；``skipped`` 表示命中 ignore 正则（BF16 直通）。
    """

    layer_name: str
    weight_key: QuantKey
    activation_key: QuantKey | None = None
    scales: tuple[ScaleDesc, ...] = ()
    skipped: bool = False
    quant_method: str = ""  # checkpoint quant_method（compressed-tensors 等）

    def effective_key(self) -> QuantKey:
        """激活规格覆盖权重 key 的 act 侧（compressed-tensors 双 key 合成）。"""
        if self.skipped or self.activation_key is None:
            return self.weight_key
        return QuantKey(
            weight_dtype=self.weight_key.weight_dtype,
            weight_scale=self.weight_key.weight_scale,
            act_dtype=self.activation_key.weight_dtype,
            act_scale=self.activation_key.weight_scale,
            symmetric=self.weight_key.symmetric,
            group_shape=self.weight_key.group_shape,
            name=self.weight_key.name,
        )
