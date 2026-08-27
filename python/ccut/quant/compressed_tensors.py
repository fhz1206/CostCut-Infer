"""ccut.quant.compressed_tensors — Ornith checkpoint 格式（§0.1 事实 + §3.6-1）。

Ornith 量化事实（已从 config.json 核查）：
- ``quant_method = "compressed-tensors"``，``quantization_status = "compressed"``；
- ``config_groups[0]``（float-quantized）：weights **per-channel 静态** scale
  （``weight_per_tensor_scale=False``、strategy=channel，memoryless_minmax），
  input_activations **per-token 动态** 量化（symmetric）；
- **7 条 ignore 正则**（vLLM ``is_layer_skipped`` 语义，``$`` 锚定）：
  lm_head / embed_tokens / ``mlp.gate``（router）/ shared_expert_gate /
  linear_attn 全部 / visual 全部 → 命中层 BF16 直通。

归一：``config_groups`` → 每层 :class:`LayerQuantSpec`（weight=QuantKey,
activation=QuantKey, scales 布局解析），``test_quant_registry.py`` 做 40 层
逐层黄金断言。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import ClassVar

import numpy as np

from ccut.quant.registry import QuantizationConfig, _register
from ccut.quant.spec import (
    GroupShape,
    K_BF16,
    K_FP8_DYNAMIC_TOKEN_SYM,
    K_FP8_STATIC_CHANNEL_SYM,
    K_FP8_STATIC_TENSOR_SYM,
    K_INT8_STATIC_CHANNEL_SYM,
    LayerQuantSpec,
    QuantDType,
    QuantKey,
    ScaleDesc,
    ScaleStrategy,
    get_quant_key,
)

__all__ = ["CompressedTensorsConfig", "registered_classes"]


def _strategy_to_weight_key(strategy: str, group_size: int | None, dtype: str) -> QuantKey:
    """weights.strategy → 权重 QuantKey。"""
    strategy = strategy.casefold()
    if strategy == "channel":
        base = get_quant_key(K_FP8_STATIC_CHANNEL_SYM if "fp8" in dtype else K_INT8_STATIC_CHANNEL_SYM)
        return QuantKey(
            weight_dtype=base.weight_dtype, weight_scale=ScaleStrategy.PER_CHANNEL,
            act_dtype=base.act_dtype, act_scale=base.act_scale,
            symmetric=base.symmetric, name=base.name,
        )
    if strategy == "tensor":
        base = get_quant_key(K_FP8_STATIC_TENSOR_SYM if "fp8" in dtype else K_INT8_STATIC_CHANNEL_SYM)
        return QuantKey(
            weight_dtype=base.weight_dtype, weight_scale=ScaleStrategy.PER_TENSOR,
            act_dtype=base.act_dtype, act_scale=base.act_scale,
            symmetric=base.symmetric, name=base.name,
        )
    if strategy == "group":
        gs = group_size or 128
        if "fp8" in dtype:
            base = get_quant_key(K_FP8_DYNAMIC_TOKEN_SYM)
            return QuantKey(
                weight_dtype=base.weight_dtype, weight_scale=ScaleStrategy.PER_GROUP,
                act_dtype=base.act_dtype, act_scale=base.act_scale,
                symmetric=base.symmetric, group_shape=GroupShape(rows=gs), name=base.name,
            )
        raise ValueError(f"compressed-tensors int8 group 策略暂不支持（group_size={gs}）")
    raise ValueError(f"未知 weights.strategy: {strategy!r}")


def _strategy_to_act_key(strategy: str, group_size: int | None, dtype: str) -> QuantKey:
    """input_activations.strategy → 激活 QuantKey。"""
    strategy = strategy.casefold()
    if strategy == "token":
        base = get_quant_key(K_FP8_DYNAMIC_TOKEN_SYM if "fp8" in dtype else K_INT8_STATIC_CHANNEL_SYM)
        return QuantKey(
            weight_dtype=base.act_dtype, weight_scale=ScaleStrategy.PER_TOKEN,
            symmetric=True, name=f"act_{base.name}",
        )
    if strategy == "tensor":
        base = get_quant_key(K_FP8_STATIC_TENSOR_SYM if "fp8" in dtype else K_INT8_STATIC_CHANNEL_SYM)
        return QuantKey(
            weight_dtype=base.act_dtype, weight_scale=ScaleStrategy.PER_TENSOR,
            symmetric=True, name=f"act_{base.name}",
        )
    if strategy == "channel":
        base = get_quant_key(K_INT8_STATIC_CHANNEL_SYM)
        return QuantKey(
            weight_dtype=base.act_dtype, weight_scale=ScaleStrategy.PER_CHANNEL,
            symmetric=True, name=f"act_{base.name}",
        )
    if strategy in ("dynamic", "row"):
        return _strategy_to_act_key("token", group_size, dtype)
    raise ValueError(f"未知 input_activations.strategy: {strategy!r}")


def _match_group(group: dict, layer_name: str) -> bool:
    """compressed-tensors group 的 ``targets`` 匹配。

    targets 为**列表**（Ornith 实测：``["Linear"]``），支持：
    - ``Linear``：任何线性投影层（``*_proj`` / fc / gate / lm_head 外的投影）；
    - ``re:<regex>``：正则；
    - ``*`` 通配 / 字面后缀。
    无 targets 的 group 兜底匹配所有层。
    """
    targets = group.get("targets")
    if not targets:
        return True
    if isinstance(targets, str):
        targets = [t for t in targets.split(",") if t.strip()]
    for t in targets:
        t = t.strip()
        if not t:
            continue
        if t == "Linear" or t.startswith("Linear"):
            if (
                "_proj" in layer_name
                or ".fc" in layer_name
                or layer_name.endswith(".fc")
                or layer_name.endswith(".gate")
                or "shared_expert" in layer_name
            ):
                return True
            continue
        if t.startswith("re:"):
            if re.fullmatch(t[3:], layer_name) or re.search(t[3:], layer_name):
                return True
            continue
        if "*" in t:
            if re.fullmatch(t.replace("*", ".*"), layer_name):
                return True
        elif layer_name == t or layer_name.endswith("." + t) or layer_name.endswith(t):
            return True
    return False


class CompressedTensorsConfig(QuantizationConfig):
    """compressed-tensors checkpoint 解析（Ornith 主测格式）。"""

    name = "compressed-tensors"
    _instances: ClassVar[list["CompressedTensorsConfig"]] = []

    def __init__(
        self,
        quant_cfg: dict,
        ignore_patterns: list[str],
        groups: list[dict],
        model_dir: Path | None = None,
    ):
        self.quant_cfg = quant_cfg
        # ignore 模式带 ``re:`` 前缀（compressed-tensors 惯例）→ 剥离后编译
        self.ignore_patterns = [re.compile(p[3:] if p.startswith("re:") else p) for p in ignore_patterns]
        self.groups = groups
        self.model_dir = model_dir
        self._specs: dict[str, LayerQuantSpec] = {}
        self._scale_names: dict[str, tuple[str, ...]] = {}
        self._seen: dict[str, int] = {}

    # -- 解析 ---------------------------------------------------------------
    @classmethod
    def from_config(cls, quant_cfg: dict, model_dir: str | Path | None = None) -> "CompressedTensorsConfig":
        raw_groups = quant_cfg.get("config_groups", [])
        # config_groups 实测为 dict（{"config_group_0": {...}}），兼容 list
        if isinstance(raw_groups, dict):
            groups = list(raw_groups.values())
        elif isinstance(raw_groups, list):
            groups = list(raw_groups)
        else:
            raise ValueError(f"config_groups 结构异常: {type(raw_groups)}")
        if not groups:
            raise ValueError("compressed-tensors quantization_config 缺 config_groups")
        ignore = list(quant_cfg.get("ignore", []))
        return cls(quant_cfg, ignore, groups, Path(model_dir) if model_dir else None)

    # -- 层分发 -------------------------------------------------------------
    def is_layer_skipped(self, layer_name: str) -> bool:
        """7 条 ignore 正则命中 → BF16 直通（vLLM is_layer_skipped 语义）。"""
        for rx in self.ignore_patterns:
            if rx.search(layer_name):
                return True
        return False

    def get_layer_spec(self, layer_name: str) -> LayerQuantSpec:
        """layer_name → LayerQuantSpec（含 ignore 判定与 scale 张量名推断）。

        scale 张量名按 safetensors 惯例推断（``<layer>.weight_scale`` /
        ``<layer>.weight_scale_2``）；真实 offset/length 由 ExpertIndex/
        WeightManager 在启动期从清单填充（ScaleDesc.offset/length 默认 0）。
        """
        if layer_name in self._specs:
            return self._specs[layer_name]
        skipped = self.is_layer_skipped(layer_name)
        for group in self.groups:
            if not _match_group(group, layer_name):
                continue
            w_cfg = group["weights"]
            a_cfg = group.get("input_activations") or {}
            w_key = _strategy_to_weight_key(
                w_cfg.get("strategy", "tensor"), w_cfg.get("group_size"), w_cfg.get("dtype", "fp8")
            )
            a_key = (
                _strategy_to_act_key(a_cfg.get("strategy", "dynamic"), a_cfg.get("group_size"), a_cfg.get("dtype", "fp8"))
                if a_cfg
                else None
            )
            scale_names = (f"{layer_name}.weight_scale",)
            if w_cfg.get("strategy") == "group":
                scale_names = (f"{layer_name}.scales", f"{layer_name}.zero_points")
            scales = tuple(
                ScaleDesc(name=n, dtype="F32", shape=()) for n in scale_names
            )
            spec = LayerQuantSpec(
                layer_name=layer_name,
                weight_key=w_key,
                activation_key=a_key,
                scales=scales,
                skipped=skipped,
                quant_method=self.name,
            )
            self._specs[layer_name] = spec
            return spec
        # 未匹配任何 group：无量化直通（记录，便于排查）
        spec = LayerQuantSpec(
            layer_name=layer_name,
            weight_key=get_quant_key(K_BF16),
            skipped=True,
            quant_method=self.name,
        )
        self._specs[layer_name] = spec
        return spec

    def resolve_scale_names(self, layer_name: str, actual_weights: set[str]) -> tuple[str, ...]:
        """用实际存在的权重名修正 scale 名推断（weight vs weight_scale 后缀事实）。"""
        base = layer_name
        candidates = []
        for suffix in (".weight_scale", ".scales", ".weight_scale_2"):
            if base + suffix in actual_weights:
                candidates.append(base + suffix)
        return tuple(candidates) if candidates else (base + ".weight_scale",)

    def validate(self) -> None:
        """加载期校验：量化层必须有 scale 张量（由清单填充后调用）。"""
        for name, spec in self._specs.items():
            if spec.skipped:
                continue
            if not spec.scales:
                raise ValueError(f"量化层 {name} 无 scale 张量")


def registered_classes() -> list[type[QuantizationConfig]]:
    return [CompressedTensorsConfig]
