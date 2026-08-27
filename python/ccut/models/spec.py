"""ccut.models.spec — ModelSpec：config.json → 声明式架构规格。

设计（§3.4-3 声明式架构数据）：ModelSpec 是**纯数据**——
- 架构标量（hidden / 层数 / 头数 / 专家数…）；
- **层模板序列**：``[{"type": "linear_attn"|"full_attn", "index": N}, ...]``
  （由 ``layer_pattern`` 计算：Ornith = 每 ``full_attention_interval`` 层一个 full，
  其余 linear；Kimi = 全 linear + 周期 full；Llama = 全 full）；
- MoE / MTP / 量化配置引用。

ModelSpec **不含任何权重与代码**——层前向由 generic 组装器按模板查
blocks + quant method 执行；族模板 JSON（families/）提供「config 键名 →
ModelSpec 字段」的映射规则，避免每架构手写解析。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["LayerTemplate", "MoeSpec", "MtpSpec", "ModelSpec", "parse_model_spec"]


@dataclass(frozen=True)
class LayerTemplate:
    """单层模板：类型 + 索引（层内具体参数由家族模板 + config 填充）。"""

    index: int
    type: str  # linear_attn | full_attn | mlp_only（无注意力层，个别架构）


@dataclass(frozen=True)
class MoeSpec:
    """MoE 参数（零驻留专家流 R2 的输入）。"""

    num_experts: int
    top_k: int
    intermediate_size: int  # 每专家 intermediate
    norm_topk_prob: bool
    has_shared_expert: bool
    shared_intermediate: int  # 0 = 无共享专家

    @property
    def expert_bytes_factor(self) -> int:
        """每专家 3 个投影的「行×列」总量（字节 = ×dtype 字节数）。"""
        # gate/up: [hidden, inter]×2 + down: [inter, hidden] → 3×hidden×inter
        return 3  # 系数，乘 hidden×inter×dtype_bytes


@dataclass(frozen=True)
class MtpSpec:
    """MTP（多 token 预测）参数（P5）。"""

    num_layers: int
    loss_scaling_factor: float
    has_fused_experts: bool = True  # MTP 专家融合 3D 布局（Ornith 实测）


@dataclass
class ModelSpec:
    """架构规格（纯数据）。"""

    # -- 基础 --
    architectures: list[str]
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    vocab_size: int
    tie_word_embeddings: bool
    rms_norm_eps: float
    max_position_embeddings: int
    rope_theta: float
    rope_scaling: dict | None
    # -- 注意力 --
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    layer_templates: list[LayerTemplate] = field(default_factory=list)
    full_attention_interval: int = 0
    attn_logit_softcapping: float | None = None
    use_qk_norm: bool = False
    # -- GDN 线性注意力（Ornith）--
    gdn_num_key_heads: int = 0
    gdn_num_value_heads: int = 0
    gdn_key_head_dim: int = 0
    gdn_value_head_dim: int = 0
    gdn_conv_kernel_dim: int = 0
    attn_output_gate: bool = False
    # -- MoE --
    moe: MoeSpec | None = None
    # -- MTP --
    mtp: MtpSpec | None = None
    # -- 前缀/量化（引用）--
    weight_prefix: str = "model."  # 语言模型张量前缀
    quant_method: str | None = None
    extra: dict = field(default_factory=dict)

    # -- 派生 ---------------------------------------------------------------
    @property
    def num_full_attn_layers(self) -> int:
        return sum(1 for t in self.layer_templates if t.type == "full_attn")

    @property
    def num_linear_attn_layers(self) -> int:
        return sum(1 for t in self.layer_templates if t.type == "linear_attn")

    @property
    def kv_bytes_per_token_full(self) -> int:
        """每 full_attn 层每 token 的 KV 字节（bf16；fp8 由 quant/kv 系数折算）。"""
        return 2 * self.num_key_value_heads * self.head_dim * 2

    def full_attn_layers(self) -> list[int]:
        return [t.index for t in self.layer_templates if t.type == "full_attn"]

    def linear_attn_layers(self) -> list[int]:
        return [t.index for t in self.layer_templates if t.type == "linear_attn"]

    def to_dict(self) -> dict:
        d = json.loads(json.dumps(self.__dict__, default=str))
        d["moe"] = self.moe.__dict__ if self.moe else None
        d["mtp"] = self.mtp.__dict__ if self.mtp else None
        return d


def _layer_templates_from_pattern(num_layers: int, interval: int, full_offset: int = 3) -> list[LayerTemplate]:
    """按 ``full_attention_interval`` 生成层模板序列。

    Ornith 实测：layer 3 是 full_attn（0 基），即 ``(index % interval) == (interval-1)``
    为 full，其余 linear。``full_offset`` 参数化不同家族的 full 层位置
    （Kimi K2 式 hybrid 家族由族模板覆盖）。
    """
    if interval <= 0:
        return [LayerTemplate(i, "full_attn") for i in range(num_layers)]
    templates = []
    for i in range(num_layers):
        templates.append(LayerTemplate(i, "full_attn" if (i % interval) == full_offset else "linear_attn"))
    return templates


def parse_model_spec(config: dict, model_dir: str | Path | None = None) -> ModelSpec:
    """config.json（可含 text_config 包裹）→ ModelSpec。

    键名差异由家族模板处理；本函数取**通用键** + GDN 扩展键
    （Ornith/Qwen3-Next 同名）。未知键进 ``extra``（不丢弃，便于排查）。
    """
    tc = config.get("text_config", config)
    archs = config.get("architectures") or ([tc.get("model_type")] if tc.get("model_type") else ["unknown"])
    num_layers = int(tc["num_hidden_layers"])

    # 注意力
    n_heads = int(tc.get("num_attention_heads", 1))
    n_kv = int(tc.get("num_key_value_heads", n_heads))
    head_dim = int(tc.get("head_dim", tc.get("hidden_size", 0) // n_heads))

    # 层模板：full_attention_interval（Ornith=4，full 在 offset 3）
    interval = int(tc.get("full_attention_interval", 0))
    templates = _layer_templates_from_pattern(num_layers, interval, full_offset=int(tc.get("full_attention_offset", 3)))

    # MoE
    moe: MoeSpec | None = None
    if tc.get("num_experts"):
        shared_inter = int(tc.get("moe_shared_expert_intermediate_size", 0))
        moe = MoeSpec(
            num_experts=int(tc["num_experts"]),
            top_k=int(tc.get("num_experts_per_tok", 1)),
            intermediate_size=int(tc.get("moe_intermediate_size", 0)),
            norm_topk_prob=bool(tc.get("norm_topk_prob", False)),
            has_shared_expert=shared_inter > 0 or bool(tc.get("n_shared_experts", 0)),
            shared_intermediate=shared_inter,
        )

    # MTP
    mtp: MtpSpec | None = None
    if tc.get("mtp_num_hidden_layers"):
        mtp = MtpSpec(
            num_layers=int(tc["mtp_num_hidden_layers"]),
            loss_scaling_factor=float(tc.get("mtp_loss_scaling_factor", 0.1)),
        )

    # GDN 线性注意力（Ornith/Qwen3-Next 键名）
    gdn_k_heads = int(tc.get("linear_num_key_heads", 0))
    gdn_v_heads = int(tc.get("linear_num_value_heads", 0))

    extra = {
        k: v
        for k, v in tc.items()
        if k not in (
            "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "head_dim",
            "hidden_size", "vocab_size", "tie_word_embeddings", "rms_norm_eps",
            "max_position_embeddings", "rope_theta", "rope_scaling", "num_experts",
            "num_experts_per_tok", "moe_intermediate_size", "norm_topk_prob",
            "mtp_num_hidden_layers", "mtp_loss_scaling_factor",
            "linear_num_key_heads", "linear_num_value_heads",
            "linear_key_head_dim", "linear_value_head_dim", "full_attention_interval",
            "full_attention_offset", "attn_output_gate", "linear_conv_kernel_dim",
            "model_type", "architectures", "quantization_config", "kv_cache_scheme",
            "moe_shared_expert_intermediate_size", "n_shared_experts", "attn_logit_softcapping",
            "use_qk_norm",
        )
    }

    quant_cfg = tc.get("quantization_config") or config.get("quantization_config")
    return ModelSpec(
        architectures=[str(a) for a in archs],
        model_type=str(tc.get("model_type", archs[0])),
        hidden_size=int(tc.get("hidden_size", 0)),
        num_hidden_layers=num_layers,
        vocab_size=int(tc.get("vocab_size", 0)),
        tie_word_embeddings=bool(tc.get("tie_word_embeddings", False)),
        rms_norm_eps=float(tc.get("rms_norm_eps", 1e-6)),
        max_position_embeddings=int(tc.get("max_position_embeddings", 32768)),
        rope_theta=float(tc.get("rope_theta", 10000.0)),
        rope_scaling=tc.get("rope_scaling"),
        num_attention_heads=n_heads,
        num_key_value_heads=n_kv,
        head_dim=head_dim,
        layer_templates=templates,
        full_attention_interval=interval,
        attn_logit_softcapping=tc.get("attn_logit_softcapping"),
        use_qk_norm=bool(tc.get("use_qk_norm", False)),
        gdn_num_key_heads=gdn_k_heads,
        gdn_num_value_heads=gdn_v_heads,
        gdn_key_head_dim=int(tc.get("linear_key_head_dim", 0)),
        gdn_value_head_dim=int(tc.get("linear_value_head_dim", 0)),
        gdn_conv_kernel_dim=int(tc.get("linear_conv_kernel_dim", 0)),
        attn_output_gate=bool(tc.get("attn_output_gate", False)),
        moe=moe,
        mtp=mtp,
        weight_prefix="model." if "language_model" in json.dumps(archs or tc) or "language_model" in str(tc) else "model.",
        quant_method=(quant_cfg or {}).get("quant_method"),
        extra=extra,
    )


def load_model_spec(model_dir: str | Path) -> ModelSpec:
    """从目录读 config.json → ModelSpec。"""
    model_dir = Path(model_dir)
    cfg_path = model_dir / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"{cfg_path}")
    with open(cfg_path, "rb") as fh:
        config = json.load(fh)
    return parse_model_spec(config, model_dir)
