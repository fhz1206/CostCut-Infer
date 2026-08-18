"""插件注册表：注意力类型 / MoE 格式 / 量化方法 / 架构归一化的可扩展注册点。

设计目标（可扩展性）：
- 新增组件（新的注意力类型、专家格式、量化方法、模型架构）通过装饰器注册，
  核心分发点（layer / moe / quant / model_config）按名称查找，无需修改核心代码。
- 内置组件在各自模块中注册（import 即注册）；外部组件可随时 ``register_*`` 接入。

用法：:

    from liteengine.registry import register_attention, get_attention

    @register_attention("my_attn")
    def build_my_attn(store, prefix, cfg):
        return MyAttention(store, f"{prefix}.self_attn", cfg)
"""
from __future__ import annotations

_ATTENTIONS: dict[str, callable] = {}
_MOE_FORMATS: dict[str, callable] = {}
_QUANT_METHODS: dict[str, callable] = {}
_ARCH_NORMALIZERS: dict[str, callable] = {}
_ARCH_PATTERNS: list[tuple[str, str]] = []          # [(模式, 归一化器名)]，有序匹配


def register_attention(name: str):
    """注册注意力构建器：``builder(store, prefix, cfg) -> Attention``。"""
    def deco(fn):
        _ATTENTIONS[name] = fn
        return fn
    return deco


def get_attention(name: str) -> callable | None:
    return _ATTENTIONS.get(name)


def list_attentions() -> list[str]:
    return sorted(_ATTENTIONS)


def register_moe_format(name: str):
    """注册 MoE/MLP 构建器：``builder(store, prefix, moe, num_experts,
    expert_cache, layer_idx, quant_cfg) -> 块``（SparseMoeBlock 或 DenseBlock）。"""
    def deco(fn):
        _MOE_FORMATS[name] = fn
        return fn
    return deco


def get_moe_format(name: str) -> callable | None:
    return _MOE_FORMATS.get(name)


def list_moe_formats() -> list[str]:
    return sorted(_MOE_FORMATS)


def register_quant_method(name: str):
    """注册量化反量化处理器：``handler(qweight, qzeros, scales, cfg, dtype) -> ndarray``。"""
    def deco(fn):
        _QUANT_METHODS[name] = fn
        return fn
    return deco


def get_quant_method(name: str) -> callable | None:
    return _QUANT_METHODS.get(name)


def list_quant_methods() -> list[str]:
    return sorted(_QUANT_METHODS)


def register_arch_normalizer(name: str, patterns: tuple[str, ...]):
    """注册架构归一化器：``normalizer(config_dict) -> 引擎内部 dict``。

    ``patterns``：architectures/model_type 的子串匹配（任一命中即选用）。
    匹配顺序按注册顺序；未命中任何注册项的架构走通用回退。
    """
    def deco(fn):
        _ARCH_NORMALIZERS[name] = fn
        for p in patterns:
            _ARCH_PATTERNS.append((p, name))
        return fn
    return deco


def get_arch_normalizer(name: str) -> callable | None:
    return _ARCH_NORMALIZERS.get(name)


def list_arch_normalizers() -> list[str]:
    return sorted(_ARCH_NORMALIZERS)


# ---- 多模态注册点（视觉编码器）----
# 占位：注册机制就绪（可扩展）；真实视觉推理需依赖与模型，方案见 docs/多模态适配方案.md

_VISION_ENCODERS: dict[str, callable] = {}


def register_vision_encoder(name: str):
    """注册视觉编码器（图像 → 视觉 token 嵌入）。"""
    def deco(fn):
        _VISION_ENCODERS[name] = fn
        return fn
    return deco


def get_vision_encoder(name: str) -> callable | None:
    """按名称取视觉编码器（未注册返回 None）。"""
    return _VISION_ENCODERS.get(name)


def list_vision_encoders() -> list[str]:
    """已注册的视觉编码器列表。"""
    return list(_VISION_ENCODERS)


def resolve_arch(archs: str, mtype: str) -> callable | None:
    """按 architectures/model_type 文本匹配已注册的归一化器（None = 走通用回退）。"""
    for pat, name in _ARCH_PATTERNS:
        if pat in archs or pat in mtype:
            return _ARCH_NORMALIZERS[name]
    return None
