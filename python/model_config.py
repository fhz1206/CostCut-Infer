"""通用模型配置归一化：HF config.json → 引擎内部统一 dict（多 MoE 架构支持）。

自动探测架构（``architectures`` / ``model_type``），兼容同系列字段差异：

- Mixtral：``num_local_experts`` / ``num_experts_per_tok`` / ``intermediate_size``（无共享专家）
- Qwen3-MoE：``num_experts`` / ``moe_intermediate_size`` / ``shared_expert_*``（共享专家）
- Qwen3.5-MoE：``text_config`` + ``layer_types`` + ``linear_*``（delta rule + gated full attention）
- DeepSeek-V3：``n_routed_experts`` / ``n_shared_experts`` / ``n_group`` / ``topk_group`` /
  ``first_k_dense_replace``（MLA 字段解析；MLA 实现待扩展）
"""
from __future__ import annotations

from json import load

__all__ = ["load_model_config"]


def _rope_theta(c: dict) -> float:
    rp = c.get("rope_parameters") or {}
    return float(rp.get("rope_theta", c.get("rope_theta", 1e6)))


def _head_dim(c: dict, hidden: int, heads: int) -> int:
    hd = int(c.get("head_dim", 0) or 0)
    return hd if hd > 0 else hidden // heads


def _norm_standard_moe(c: dict, arch: str) -> dict:
    """标准注意力 MoE（Mixtral / Qwen3-MoE 共用字段，Qwen3-MoE 追加共享专家）。"""
    hidden = int(c["hidden_size"])
    heads = int(c["num_attention_heads"])
    kvh = int(c.get("num_key_value_heads", heads))
    n_layers = int(c["num_hidden_layers"])
    moe_inter = int(c.get("moe_intermediate_size", c.get("intermediate_size", 0)))
    return {
        "arch": arch,
        "hidden_size": hidden,
        "num_hidden_layers": n_layers,
        "num_attention_heads": heads,
        "num_key_value_heads": kvh,
        "head_dim": _head_dim(c, hidden, heads),
        "rms_norm_eps": float(c.get("rms_norm_eps", 1e-5)),
        "vocab_size": int(c["vocab_size"]),
        "rope_theta": _rope_theta(c),
        "rope_partial": float((c.get("rope_parameters") or {}).get("partial_rotary_factor", 1.0)),
        "layer_attention_types": ["standard"] * n_layers,
        "moe": {
            "num_experts": int(c.get("num_local_experts", c.get("num_experts"))),
            "top_k": int(c.get("num_experts_per_tok", c.get("num_experts_per_token", 1))),
            "intermediate": moe_inter,
            "shared": bool(c.get("shared_expert_intermediate_size", False)),
            "shared_intermediate": int(c.get("shared_expert_intermediate_size", 0)),
            "shared_expert_prefix": "shared_expert",
            "experts_format": "merged_plain",
        },
        "weight_prefix": "model",
    }


def _norm_qwen35(c: dict) -> dict:
    """Qwen3.5-MoE：现有引擎架构（text_config + layer_types + delta rule）。"""
    tc = c.get("text_config", c)
    layers = tc["layer_types"]
    attn_map = {"linear_attention": "linear_delta", "full_attention": "full_gated"}
    return {
        "arch": "qwen3_5_moe",
        "hidden_size": int(tc["hidden_size"]),
        "num_hidden_layers": int(tc["num_hidden_layers"]),
        "num_attention_heads": int(tc["num_attention_heads"]),
        "num_key_value_heads": int(tc.get("num_key_value_heads", tc["num_attention_heads"])),
        "head_dim": int(tc.get("head_dim", tc["hidden_size"] // tc["num_attention_heads"])),
        "rms_norm_eps": float(tc["rms_norm_eps"]),
        "vocab_size": int(tc["vocab_size"]),
        "rope_theta": float((tc.get("rope_parameters") or {}).get("rope_theta", 1e7)),
        "rope_partial": float((tc.get("rope_parameters") or {}).get("partial_rotary_factor", 0.25)),
        "layer_attention_types": [attn_map.get(t, "full_gated") for t in layers],
        "moe": {
            "num_experts": int(tc["num_experts"]),
            "top_k": int(tc["num_experts_per_tok"]),
            "intermediate": int(tc["moe_intermediate_size"]),
            "shared": True,
            "shared_intermediate": int(tc.get("shared_expert_intermediate_size", 0)),
            "shared_expert_prefix": "shared_expert",
            "experts_format": "quantized_separate",
        },
        "weight_prefix": "model.language_model",
    }


def _norm_deepseek(c: dict) -> dict:
    """DeepSeek-V3/V4 与 Kimi K2（text_config 别名 deepseek_v3）：MLA + 组限制路由 + 前 K 层 dense。

    - MLA：``kv_lora_rank`` / ``q_lora_rank`` / ``qk_rope_head_dim``（kv 头数=1 的低秩压缩注意力）
    - MoE：``n_routed_experts`` / ``n_shared_experts`` / ``n_group`` / ``topk_group``
    - ``first_k_dense_replace``：前 K 层用 dense MLP（非 MoE）
    """
    tc = c.get("text_config", c)                     # Kimi K2 的文本配置为 deepseek_v3 风格
    hidden = int(tc["hidden_size"])
    heads = int(tc["num_attention_heads"])
    n_layers = int(tc["num_hidden_layers"])
    return {
        "arch": "deepseek_moe",
        # MTP（Multi-Token Prediction）模块数——投机解码草稿（DeepSeek-V3.2 约定）
        "mtp_layers": int(c.get("num_nextn_predict_layers", 0)),
        "hidden_size": hidden,
        "num_hidden_layers": n_layers,
        "num_attention_heads": heads,
        "num_key_value_heads": int(tc.get("num_key_value_heads", 1)),   # MLA：1 个 kv 头
        "head_dim": int(tc.get("qk_rope_head_dim", _head_dim(tc, hidden, heads))),
        "rms_norm_eps": float(tc.get("rms_norm_eps", 1e-6)),
        "vocab_size": int(tc["vocab_size"]),
        "rope_theta": _rope_theta(tc),
        "rope_partial": 1.0,
        "attention": "mla",                          # MLA 实现（见 liteengine/attention.py）
        "kv_lora_rank": int(tc.get("kv_lora_rank", 0)),
        "q_lora_rank": int(tc.get("q_lora_rank", 0)),
        "moe": {
            "num_experts": int(tc.get("n_routed_experts", tc.get("num_experts"))),
            "top_k": int(tc.get("num_topk", tc.get("num_experts_per_tok", 8))),
            "intermediate": int(tc.get("moe_intermediate_size", 0)),
            "shared": int(tc.get("n_shared_experts", 1) or 0) > 0,
            "shared_count": int(tc.get("n_shared_experts", 1) or 0),
            "shared_expert_prefix": "shared_experts",
            "experts_format": "merged_plain",
            "group_limit": (int(tc["n_group"]), int(tc["topk_group"]))
            if tc.get("n_group") and tc.get("topk_group") else None,
            "dense_layers": int(tc.get("first_k_dense_replace", 0) or 0),
        },
        "weight_prefix": "model",
    }


def _norm_glm(c: dict) -> dict:
    """GLM4-MoE / GLM-MoE：标准注意力 + 共享/路由专家（``n_shared_experts`` 为计数）。"""
    hidden = int(c["hidden_size"])
    heads = int(c["num_attention_heads"])
    kvh = int(c.get("num_key_value_heads", heads))
    n_layers = int(c["num_hidden_layers"])
    return {
        "arch": "glm_moe",
        "hidden_size": hidden,
        "num_hidden_layers": n_layers,
        "num_attention_heads": heads,
        "num_key_value_heads": kvh,
        "head_dim": _head_dim(c, hidden, heads),
        "rms_norm_eps": float(c.get("rms_norm_eps", 1e-5)),
        "vocab_size": int(c["vocab_size"]),
        "rope_theta": _rope_theta(c),
        "rope_partial": 1.0,
        "layer_attention_types": ["standard"] * n_layers,
        "moe": {
            "num_experts": int(c.get("num_local_experts",
                                     c.get("num_experts", c.get("n_experts", 0)))),
            "top_k": int(c.get("num_experts_per_tok", 8)),
            "intermediate": int(c.get("moe_intermediate_size", 0)),
            "shared": int(c.get("n_shared_experts", 0) or 0) > 0,
            "shared_count": int(c.get("n_shared_experts", 0) or 0),
            "shared_expert_prefix": "shared_experts",
            "experts_format": "merged_plain",
        },
        "weight_prefix": "model",
    }


def _norm_fallback(c: dict, archs: str, mtype: str) -> dict:
    """通用架构回退：未知但**非专属**架构（标准解码器结构）自动归一化。

    判定：hidden_size / num_hidden_layers / num_attention_heads 齐全且**无线性注意力专属字段**
    （delta rule / linear_attention 层）→ 标准 GQA 注意力；
    MoE 由专家字段自动探测（num_local_experts / num_experts / n_routed_experts），
    否则为稠密 MLP（gate_proj/up_proj/down_proj）。
    """
    layer_types = c.get("layer_types") or []
    if any("linear" in str(t).lower() for t in layer_types):
        raise ValueError(f"专属/不支持架构（含线性注意力层）: {archs or mtype}")
    missing = [k for k in ("hidden_size", "num_hidden_layers", "num_attention_heads")
               if k not in c]
    if missing:
        raise ValueError(f"不支持的架构（缺字段 {missing}）: {archs or mtype or '未知'}")
    hidden = int(c["hidden_size"])
    heads = int(c["num_attention_heads"])
    n_layers = int(c["num_hidden_layers"])
    n_exp = c.get("num_local_experts", c.get("num_experts", c.get("n_routed_experts", 0)))
    is_moe = int(n_exp or 0) > 0
    base = {
        "arch": "generic_moe" if is_moe else "generic_dense",
        "hidden_size": hidden,
        "num_hidden_layers": n_layers,
        "num_attention_heads": heads,
        "num_key_value_heads": int(c.get("num_key_value_heads", heads)),
        "head_dim": _head_dim(c, hidden, heads),
        "rms_norm_eps": float(c.get("rms_norm_eps", 1e-5)),
        "vocab_size": int(c.get("vocab_size", 0)),
        "rope_theta": _rope_theta(c),
        "rope_partial": 1.0,
        "layer_attention_types": ["standard"] * n_layers,
        "weight_prefix": "model",
    }
    if is_moe:
        shared_n = int(c.get("n_shared_experts", 0) or 0)
        base["moe"] = {
            "num_experts": int(n_exp),
            "top_k": int(c.get("num_experts_per_tok", c.get("num_experts_per_token", 2))),
            "intermediate": int(c.get("moe_intermediate_size", c.get("intermediate_size", 0))),
            "shared": bool(shared_n) or bool(c.get("shared_expert_intermediate_size", 0)),
            "shared_count": shared_n,
            "shared_expert_prefix": ("shared_experts"
                                     if (shared_n or c.get("shared_expert_intermediate_size"))
                                     else "shared_expert"),
            "experts_format": "merged_plain",
        }
    else:
        base["moe"] = {
            "experts_format": "dense_mlp",
            "intermediate": int(c.get("intermediate_size", 4 * hidden)),
        }
    return base


# ---- 注册表：内置架构归一化器（load_model_config 经 registry 探测；外部可新增注册）----

from engine.registry import register_arch_normalizer, resolve_arch


@register_arch_normalizer("qwen3_5_moe", patterns=("Qwen3_5", "qwen3_5"))
def _reg_qwen35(c: dict) -> dict:
    return _norm_qwen35(c)


@register_arch_normalizer("mixtral", patterns=("Mixtral", "mixtral"))
def _reg_mixtral(c: dict) -> dict:
    return _norm_standard_moe(c, "mixtral")


@register_arch_normalizer("qwen3_moe", patterns=("Qwen3Moe", "qwen3_moe"))
def _reg_qwen3moe(c: dict) -> dict:
    return _norm_standard_moe(c, "qwen3_moe")


@register_arch_normalizer("glm5", patterns=("Glm5", "glm5", "GlmMoeDsa", "glm_moe_dsa"))
def _reg_glm5(c: dict) -> dict:
    """GLM-5（GlmMoeDsa）：MLA（kv_lora/q_lora）+ 256 路由专家/top-8 + 1 共享 + DSA 索引器。

    DSA（DeepSeek Sparse Attention——index_topk/index_head_dim）字段随归一化保留，
    索引器实现记为后续（与 DeepSeek-V4 的 V4-Flash 压缩注意力同类）。
    """
    d = _norm_deepseek(c)
    d["arch"] = "glm5"
    return d



@register_arch_normalizer("glm_moe", patterns=("Glm4Moe", "glm4_moe", "GlmMoe"))
def _reg_glm(c: dict) -> dict:
    return _norm_glm(c)


@register_arch_normalizer("deepseek_moe", patterns=("Deepseek", "deepseek", "kimi"))
def _reg_deepseek(c: dict) -> dict:
    return _norm_deepseek(c)


@register_arch_normalizer("dbrx", patterns=("Dbrx", "dbrx"))
def _reg_dbrx(c: dict) -> dict:
    return _norm_standard_moe(c, "dbrx")


@register_arch_normalizer("phi3_moe", patterns=("Phi3MoE", "phi3_moe"))
def _reg_phi3(c: dict) -> dict:
    return _norm_standard_moe(c, "phi3_moe")


def load_model_config(model_dir: str) -> dict:
    """读取并归一化模型配置（架构经注册表探测；未命中走通用回退）。"""
    with open(f"{model_dir}/config.json", "r", encoding="utf-8") as f:
        c = load(f)
    archs = " ".join(c.get("architectures") or [])
    mtype = str(c.get("model_type", "")).lower()
    normalizer = resolve_arch(archs, mtype)
    if normalizer is not None:
        d = normalizer(c)
    else:
        # 通用回退：未知但非专属架构（标准解码器结构）自动适配
        d = _norm_fallback(c, archs, mtype)
    # vLLM 精度适配约定：模型 torch_dtype（fp16/bf16/fp32/fp8）注入归一化配置
    d.setdefault("dtype", str(c.get("torch_dtype", "float32")))
    return d
