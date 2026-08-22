"""DecoderLayer 组装：混合注意力（linear / full）+ MoE（对照 transformers Qwen3_5MoeDecoderLayer）。

结构：residual → input_layernorm(RMSNorm) → token mixer → +residual
     → post_attention_layernorm → MoE(router+专家+shared_expert) → +residual
"""
from __future__ import annotations

import torch
from torch import Tensor

import liteengine.attention  # noqa: F401  触发注意力构建器注册（layer 经 registry 分发）
from liteengine.moe import DenseBlock, MLP, SparseMoeBlock, TopKRouter, torch_weight
from liteengine.core.norm import rms_norm, rms_norm_add
from liteengine.registry import get_attention, get_moe_format, list_attentions, list_moe_formats

__all__ = ["DecoderLayer"]


class DecoderLayer:
    """单层前向。cos/sin/mask 仅 full_attention 层使用（linear_attn 忽略）。"""

    def __init__(self, store, layer_idx: int, cfg: dict, expert_cache=None,
                 quant_cfg=None, compute_dtype: str = "float32",
                 expert_parallel: bool = False):
        self.layer_idx = layer_idx
        self.block_type = str(cfg["layer_types"][layer_idx]) if "layer_types" in cfg else "standard"
        self.eps = float(cfg["rms_norm_eps"])
        prefix = f"{cfg.get('weight_prefix', 'model.language_model')}.layers.{layer_idx}"
        self.input_norm_w = torch_weight(store, f"{prefix}.input_layernorm.weight")
        self.post_norm_w = torch_weight(store, f"{prefix}.post_attention_layernorm.weight")
        # 注意力类型分发：归一化配置提供逐层类型；旧配置按 layer_types 兼容映射。
        # 构建器经注册表查找（liteengine.registry），外部可注册新注意力类型。
        layer_attns = cfg.get("layer_attention_types")
        if layer_attns is not None:
            attn_type = str(layer_attns[layer_idx])
        else:
            attn_type = ("full_gated" if self.block_type == "full_attention" else "linear_delta")
        builder = get_attention(attn_type)
        if builder is None:
            raise ValueError(f"未知注意力类型 {attn_type!r}（可用: {list_attentions()}）")
        self.attn = builder(store, prefix, cfg)
        self.mlp = self._build_moe(store, f"{prefix}.mlp", cfg, expert_cache, quant_cfg,
                                   compute_dtype, expert_parallel)

    def offload(self) -> None:
        """AirLLM 风格层级卸载：释放本层专家反量化缓存（下次前向惰性重建——降低内存峰值）。"""
        if hasattr(self, "mlp") and hasattr(self.mlp, "clear_cache"):
            self.mlp.clear_cache()

    def _build_moe(self, store, prefix: str, cfg: dict, expert_cache=None,
                   quant_cfg=None, compute_dtype: str = "float32",
                   expert_parallel: bool = False) -> SparseMoeBlock:
        moe = cfg.get("moe", {})
        fmt = moe.get("experts_format", "quantized_separate")
        num_experts = int(cfg.get("num_experts", moe.get("num_experts", 0)))
        # 构建器经注册表查找（liteengine.registry），外部可注册新专家格式
        builder = get_moe_format(fmt)
        if builder is None:
            raise ValueError(f"未知专家格式 {fmt!r}（可用: {list_moe_formats()}）")
        block = builder(store, prefix, moe, num_experts, expert_cache,
                        self.layer_idx, quant_cfg, compute_dtype, expert_parallel)
        if isinstance(block, DenseBlock):
            return block                             # 稠密 MLP：无路由无共享
        router = TopKRouter(
            torch_weight(store, f"{prefix}.gate.weight"),
            int(cfg.get("num_experts_per_tok", moe.get("top_k", 1))),
        )
        if moe.get("shared", True):
            shared_pre = moe.get("shared_expert_prefix", "shared_expert")
            shared = MLP(
                torch_weight(store, f"{prefix}.{shared_pre}.gate_proj.weight"),
                torch_weight(store, f"{prefix}.{shared_pre}.up_proj.weight"),
                torch_weight(store, f"{prefix}.{shared_pre}.down_proj.weight"),
            )
            shared_gate_w = torch_weight(store, f"{prefix}.{shared_pre}_gate.weight")
            return SparseMoeBlock(router, block, shared, shared_gate_w)
        return SparseMoeBlock(router, block)         # 无共享专家（Mixtral 等）

    def forward(self, x: Tensor, cos: Tensor | None = None, sin: Tensor | None = None,
                mask: Tensor | None = None, cache=None) -> Tensor:
        """prefill：cache 提供时填充该层状态（KV / conv / recurrent）。"""
        residual = x
        h = rms_norm(x, self.input_norm_w, self.eps)
        if self.block_type == "linear_attention":
            h = self.attn(h, cache, self.layer_idx)
        else:
            h = self.attn(h, cos, sin, mask, cache, self.layer_idx)
        h = residual + h

        # 残差 + post_norm 融合（差异报告 #8 算子融合）：省一次中间张量往返
        h, h_pre = rms_norm_add(h, residual, self.post_norm_w, self.eps)
        h = self.mlp(h)
        return h_pre + h

    def forward_step(self, x: Tensor, cos: Tensor | None, sin: Tensor | None, cache) -> Tensor:
        """decode 单 token：就地更新 cache 中的该层状态（KV 追加 / conv+recurrent 递推）。"""
        residual = x
        h = rms_norm(x, self.input_norm_w, self.eps)
        if self.block_type == "linear_attention":
            h, cache.conv_state[self.layer_idx], cache.rec_state[self.layer_idx] = \
                self.attn.forward_step(h, cache.conv_state[self.layer_idx], cache.rec_state[self.layer_idx])
        else:
            h, cache.attn_kv[self.layer_idx] = self.attn.forward_step(
                h, cos, sin, cache.attn_kv[self.layer_idx])
        h = residual + h

        # 残差 + post_norm 融合（差异报告 #8 算子融合）：省一次中间张量往返
        h, h_pre = rms_norm_add(h, residual, self.post_norm_w, self.eps)
        h = self.mlp(h)
        return h_pre + h

    __call__ = forward
