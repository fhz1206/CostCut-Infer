"""ccut.quant.registry — QUANTIZATION_METHODS 注册表 + checkpoint 解析。

对齐 vLLM ``config/quantization.py``：
- ``QUANTIZATION_METHODS``：quant_method 名 → ``QuantizationConfig`` 类；
- ``_ONLINE_SHORTHANDS``：在线量化简写（``fp8_per_token`` 等）→ 在线 spec；
- :func:`resolve_checkpoint_quant`：读 ``config.json.quantization_config`` →
  实例化 Config（未注册格式**显式报错**，列出已支持清单 + 建议 L1 后端）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # 仅类型引用（运行时延迟 import）
    from ccut.quant.compressed_tensors import CompressedTensorsConfig

__all__ = [
    "QUANTIZATION_METHODS",
    "ONLINE_SHORTHANDS",
    "QuantizationConfig",
    "resolve_checkpoint_quant",
    "list_supported_quant",
]


class QuantizationConfig:
    """量化配置基类（对齐 vLLM QuantizationConfig）。

    子类实现：
    - ``from_config(cls, checkpoint_cfg)``：解析 checkpoint 的 quantization_config；
    - ``get_layer_spec(layer_name) -> LayerQuantSpec``：按层名分发量化规格。
    """

    name: str = "base"

    @classmethod
    def from_config(cls, quant_cfg: dict, model_dir: str | Path | None = None) -> "QuantizationConfig":
        raise NotImplementedError

    def get_layer_spec(self, layer_name: str):  # type: ignore[no-untyped-def]
        raise NotImplementedError

    def validate(self) -> None:
        """加载完成校验（缺 scale 张量等）。"""
        return None


#: quant_method 名 → Config 类（惰性绑定，避免 import 环）
QUANTIZATION_METHODS: dict[str, type[QuantizationConfig]] = {}


def _register(name: str):
    def wrap(cls: type[QuantizationConfig]) -> type[QuantizationConfig]:
        cls.name = name
        QUANTIZATION_METHODS[name] = cls
        return cls

    return wrap


def _lazy_register_all() -> None:
    """惰性注册全部格式（首次 resolve 时触发）。"""
    global QUANTIZATION_METHODS
    if QUANTIZATION_METHODS:
        return
    from ccut.quant import compressed_tensors as _ct

    for cls in _ct.registered_classes():
        QUANTIZATION_METHODS[cls.name] = cls


#: 在线量化简写（对齐 vLLM _ONLINE_SHORTHANDS 子集；§3.6-3，P8）
ONLINE_SHORTHANDS: dict[str, dict] = {
    "fp8_per_token": {"weight": "fp8_static_channel_sym", "act": "fp8_dynamic_token_sym", "compute": "w8a16"},
    "int8_per_token": {"weight": "int8_static_channel_sym", "act": "int8_dynamic_token_sym", "compute": "w8a16"},
    "int8_per_channel_weight_only": {"weight": "int8_static_channel_sym", "act": None, "compute": "w8a16"},
    "mxfp8": {"weight": "mxfp8_dynamic", "act": None, "compute": "w8a16"},
    "nvfp4_per_token": {"weight": "nvfp4_dynamic", "act": None, "compute": "w4a16"},
}


def list_supported_quant() -> list[str]:
    _lazy_register_all()
    return sorted(QUANTIZATION_METHODS)


class UnquantizedConfig(QuantizationConfig):
    """无量化 checkpoint（BF16/FP16 直通）。"""

    name = "none"

    def __init__(self, weight_dtype: str = "bf16"):
        self.weight_dtype = weight_dtype

    @classmethod
    def from_config(cls, quant_cfg: dict, model_dir: str | Path | None = None) -> "UnquantizedConfig":
        return cls()

    def get_layer_spec(self, layer_name: str):  # type: ignore[no-untyped-def]
        from ccut.quant.spec import LayerQuantSpec, QuantDType, QuantKey

        key = QuantKey(QuantDType.BF16 if self.weight_dtype == "bf16" else QuantDType.FP16, name="unquant")
        return LayerQuantSpec(layer_name, key, skipped=True, quant_method="none")


def resolve_checkpoint_quant(
    model_dir: str | Path,
    online_quantization: str | None = None,
    quant_ignore: list[str] | None = None,
) -> QuantizationConfig:
    """checkpoint 量化解析入口（§3.6-1 主路径）。

    - 无 ``quantization_config`` 且无在线简写 → :class:`UnquantizedConfig`；
    - 有 ``quantization_config`` → 查 QUANTIZATION_METHODS（未注册显式报错）；
    - ``online_quantization`` 为简写名 → 在线量化 Config（与 checkpoint 量化**互斥**，同设报错）。
    """
    model_dir = Path(model_dir)
    cfg_path = model_dir / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"{cfg_path}: 无 config.json")
    with open(cfg_path, "rb") as fh:
        cfg = json.load(fh)
    text_cfg = cfg.get("text_config", cfg)
    quant_cfg = text_cfg.get("quantization_config") or cfg.get("quantization_config")

    online = (online_quantization or "auto").casefold()
    if online in ("auto", "none", ""):
        online = None

    if online is not None:
        if online not in ONLINE_SHORTHANDS:
            raise ValueError(f"未知在线量化简写 {online!r}，可选: {sorted(ONLINE_SHORTHANDS)}")
        if quant_cfg:
            raise ValueError(
                f"在线量化（{online}）与 checkpoint 自带量化（{quant_cfg.get('quant_method')}）互斥，只能设一个"
            )
        from ccut.quant.online import OnlineQuantConfig

        return OnlineQuantConfig.from_config(ONLINE_SHORTHANDS[online], model_dir, ignore_patterns=quant_ignore)

    if not quant_cfg:
        return UnquantizedConfig()

    _lazy_register_all()
    method_name = quant_cfg.get("quant_method", "")
    cls = QUANTIZATION_METHODS.get(method_name)
    if cls is None:
        raise ValueError(
            f"checkpoint 量化格式 {method_name!r} 未注册。已支持: {list_supported_quant()}。"
            "提示：可改用 L1 transformers 兜底后端（--arch-tier auto 会自动降级）。"
        )
    return cls.from_config(quant_cfg, model_dir)
