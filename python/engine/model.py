"""模型外壳（无 cache 路径）：embed + 40 层循环 + lm_head（对照 transformers Qwen3_5MoeModel 文本路径）。

内存友好设计：
- 层按需构建（``layer(i)`` 惰性缓存），M4 将改为按层构建/释放的流式
- embed / lm_head 常驻 fp16（合计 ~2GB），计算片段才转 fp32
"""
from __future__ import annotations

from json import load

import torch
from torch import Tensor
from torch.nn.functional import linear

from engine.cache import Cache, ExpertCache
from engine.layer import DecoderLayer
from engine.moe import torch_weight
from core.norm import rms_norm
from quant import QuantConfig, load_quant_config
from core.rope import compute_inv_freq, rotary_embeddings
from engine.sampling import sample_token

__all__ = ["Qwen3_5MoeModel", "load_text_config", "causal_mask"]


def load_text_config(model_dir: str) -> dict:
    """读取 config.json 的 text_config 作为层配置。"""
    with open(f"{model_dir}/config.json", "r", encoding="utf-8") as f:
        data = load(f)
    return dict(data["text_config"])


def causal_mask(length: int) -> Tensor:
    """因果注意力掩码 (L, L)：右上三角（不含对角线）为 -inf，可广播到 (B, H, L, L)。"""
    m = torch.full((length, length), float("-inf"))
    return m.triu(1)


class Qwen3_5MoeModel:
    """纯 Python 推理外壳（prefill / 无 cache）。

    Args:
        store: WeightStore（惰性读权）。
        cfg: ``load_text_config`` 的 text_config。
    """

    def __init__(self, store, cfg: dict, expert_cache_max: int = 128, layer_offload: bool = False):
        self.store = store
        self.cfg = cfg
        self.expert_cache_max = expert_cache_max
        self.layer_offload = layer_offload   # AirLLM 风格层级卸载（层前向后释放专家缓存——降内存峰值）
        self._expert_cache = ExpertCache(max_entries=expert_cache_max)
        self.num_layers = int(cfg["num_hidden_layers"])
        self.hidden_size = int(cfg["hidden_size"])
        self.eps = float(cfg["rms_norm_eps"])
        self.weight_prefix = str(cfg.get("weight_prefix", "model.language_model"))
        rope_cfg = cfg.get("rope_parameters", {})
        self.inv_freq = compute_inv_freq(
            int(cfg["head_dim"]),
            float(cfg.get("rope_theta", rope_cfg.get("rope_theta", 1e7))),
            float(cfg.get("rope_partial", rope_cfg.get("partial_rotary_factor", 0.25))),
            str(cfg.get("rope_type", rope_cfg.get("rope_type", "default"))),
            dict(cfg.get("rope_scaling", rope_cfg.get("rope_scaling", {}))),
        )
        self._layers: dict[int, DecoderLayer] = {}
        self._embed: Tensor | None = None
        self._lm_head: Tensor | None = None
        self._final_norm_w: Tensor | None = None   # 最终 RMSNorm（lm_head 前，transfomers 必做）
        # 量化配置（config.json quantization_config → 专家反量化分发，支持多规格/算法）
        self.quant_cfg = load_quant_config(str(self.store.model_dir))

    # ---- 惰性构建 ----

    def layer(self, idx: int) -> DecoderLayer:
        if idx not in self._layers:
            self._layers[idx] = DecoderLayer(self.store, idx, self.cfg, self._expert_cache,
                                             self.quant_cfg)
        return self._layers[idx]

    def clear_expert_cache(self) -> None:
        """清空专家反量化缓存（会话切换 / 内存回收）。"""
        self._expert_cache.clear()

    def expert_cache_bytes(self) -> int:
        """当前专家缓存占用字节数（内存审计）。"""
        return self._expert_cache.bytes()

    def embed(self) -> Tensor:
        """词嵌入（fp16 常驻）。"""
        if self._embed is None:
            self._embed = torch.from_numpy(self.store.get(f"{self.weight_prefix}.embed_tokens.weight"))
        return self._embed

    def lm_head(self) -> Tensor:
        """输出投影（fp16 常驻；M3 生成时使用）。"""
        if self._lm_head is None:
            self._lm_head = torch.from_numpy(self.store.get("lm_head.weight"))
        return self._lm_head

    def final_norm(self) -> Tensor:
        """最终层 RMSNorm 权重（``{weight_prefix}.norm.weight``，lm_head 前必须应用）。"""
        if self._final_norm_w is None:
            self._final_norm_w = torch_weight(self.store, f"{self.weight_prefix}.norm.weight")
        return self._final_norm_w

    # ---- 前向 ----

    def forward(self, input_ids: Tensor, num_layers: int | None = None) -> Tensor:
        """input_ids: (L,) int64。返回第 num_layers 层后的 hidden (L, hidden) fp32。

        只跑前 num_layers 层（默认全部）——M2 单层验证传 num_layers=1。
        """
        seq_len = input_ids.shape[0]
        h = self.embed()[input_ids].float()          # (L, hidden)
        pos = torch.arange(seq_len, dtype=torch.int64)
        cos, sin = rotary_embeddings(pos, self.inv_freq)
        n = self.num_layers if num_layers is None else num_layers
        layer_attns = self.cfg.get("layer_attention_types")
        if layer_attns is not None:
            use_mask = any(t != "linear_delta" for t in layer_attns[:n])
        else:
            use_mask = any(t == "full_attention"
                           for t in self.cfg.get("layer_types", ["full_attention"])[:n])
        mask = causal_mask(seq_len) if use_mask else None
        h = h.unsqueeze(0)                            # (1, L, hidden)
        for i in range(n):
            h = self.layer(i)(h, cos, sin, mask)
            if self.layer_offload:
                self.layer(i).offload()
        return h.squeeze(0)                           # (L, hidden)

    # ---- 生成（M3）----

    def logits(self, h: Tensor) -> Tensor:
        """hidden → 词表 logits。按 lm_head 的 dtype 对齐：
        真实模型 lm_head 常驻 fp16（省内存）；GGUF 等 f32 权重则用 f32 计算。"""
        return linear(h.to(self.lm_head().dtype), self.lm_head())

    def generate_speculative(self, input_ids: Tensor, speculator,
                             max_new_tokens: int = 32, temperature: float = 0.7,
                             top_p: float = 0.9, top_k: int = 0,
                             repetition_penalty: float = 1.0,
                             eos_token_id: int | None = None,
                             num_layers: int | None = None) -> list[int]:
        """dspark 投机解码生成：草稿 K token → 主模型并行验证 → 投机采样接受。

        草稿质量只影响提速幅度；接受判定保证输出分布与自回归一致。
        部分接受时回滚 cache 并重新推进接受前缀（delta rule 状态无法按位置截断）。
        """
        from engine.speculator import speculative_accept
        cache = Cache(self.num_layers,
                      max_len=int(input_ids.shape[0]) + max_new_tokens + 16)
        h = self.prefill(input_ids, cache, num_layers)
        pos = int(input_ids.shape[0])
        out_ids: list[int] = []
        K = speculator.num_draft
        embed = self.embed()
        while len(out_ids) < max_new_tokens:
            draft_ids, draft_probs = speculator.draft(h[-1:], embed, start_pos=pos)
            snap = cache.snapshot()
            verify_h = self.prefill(torch.tensor(draft_ids), cache, num_layers,
                                    start_pos=pos)                      # (K, hidden)
            # 草稿位置分布：logits(上下文最后位置) + logits(hidden[0..K-2])
            # （verify_h[i] 是处理草稿 i 之后的隐藏 → logits 预测草稿 i+1）
            ctx_last = self.logits(h[-1:]).float()                               # (1, vocab)
            mid = self.logits(verify_h[:-1]).float() if verify_h.shape[0] > 1 \
                else torch.empty(0, ctx_last.shape[1])
            verify_logits = torch.cat([ctx_last, mid], dim=0)                    # (K, vocab)
            extra_logits = self.logits(verify_h[-1:]).float()                    # 全接受后新 token
            acc, n_acc, new_tok = speculative_accept(
                draft_ids, draft_probs, verify_logits, extra_logits, speculator.t2d,
                temperature, top_p, top_k, repetition_penalty, out_ids, eos_token_id)
            if n_acc < K:
                cache.restore(snap)                    # 部分接受：回滚 + 重新推进接受前缀
                for i, t in enumerate(acc):
                    h = self.decode_step(torch.tensor(t), cache, pos + i,
                                         num_layers).reshape(1, -1)
                pos += n_acc
            else:
                pos += K
            out_ids.extend(acc)
            if new_tok is None:
                break
            out_ids.append(new_tok)
            if eos_token_id is not None and new_tok == eos_token_id:
                break
            if len(out_ids) >= max_new_tokens:
                break
            h = self.decode_step(torch.tensor(new_tok), cache, pos, num_layers).reshape(1, -1)
            pos += 1
        return out_ids[:max_new_tokens]

    def prefill(self, input_ids: Tensor, cache: Cache, num_layers: int | None = None,
                start_pos: int = 0) -> Tensor:
        """对 input_ids 全层前向并填 cache，返回最后一层输出 (L, hidden)。

        ``start_pos``：续接前向的起始位置（投机解码验证草稿时传入——RoPE 位置须与上下文连续）。
        """
        seq_len = input_ids.shape[0]
        h = self.embed()[input_ids].float().unsqueeze(0)                 # (1, L, hidden)
        pos = torch.arange(start_pos, start_pos + seq_len, dtype=torch.int64)
        cos, sin = rotary_embeddings(pos, self.inv_freq)
        n = self.num_layers if num_layers is None else num_layers
        layer_attns = self.cfg.get("layer_attention_types")
        if layer_attns is not None:
            use_mask = any(t != "linear_delta" for t in layer_attns[:n])
        else:
            use_mask = any(t == "full_attention"
                           for t in self.cfg.get("layer_types", ["full_attention"])[:n])
        mask = causal_mask(seq_len) if use_mask else None
        for i in range(n):
            h = self.layer(i)(h, cos, sin, mask, cache)
            if self.layer_offload:
                self.layer(i).offload()
        h = rms_norm(h, self.final_norm(), self.eps)             # 最终归一化（lm_head 前）
        return h.squeeze(0)                                              # (L, hidden)

    def decode_step(self, token: Tensor, cache: Cache, pos: int,
                    num_layers: int | None = None) -> Tensor:
        """单 token 全层 decode（就地更新 cache），返回 (1, 1, hidden)。"""
        idx = token.reshape(())                                        # 标量化（容忍 0 维或 (1,)）
        h = self.embed()[idx].float().unsqueeze(0).unsqueeze(0)        # (1, 1, hidden)
        cos, sin = rotary_embeddings(torch.tensor([pos], dtype=torch.int64), self.inv_freq)
        n = self.num_layers if num_layers is None else num_layers
        for i in range(n):
            h = self.layer(i).forward_step(h, cos, sin, cache)
            if self.layer_offload:
                self.layer(i).offload()
        h = rms_norm(h, self.final_norm(), self.eps)             # 最终归一化（lm_head 前）
        return h                                                        # (1, 1, hidden)

    def generate(self, input_ids: Tensor, max_new_tokens: int = 32,
                 temperature: float = 0.7, top_p: float = 0.9, top_k: int = 0,
                 repetition_penalty: float = 1.0, eos_token_id: int | None = None,
                 num_layers: int | None = None) -> list[int]:
        """生成 max_new_tokens 个新 token id（prefill → 逐 token decode）。

        eos_token_id 提供时命中即提前终止（聊天场景必须，避免空转）。
        """
        cache = Cache(self.num_layers,
                      max_len=int(input_ids.shape[0]) + max_new_tokens + 16)
        h = self.prefill(input_ids, cache, num_layers)
        last = h[-1:]                                                   # (1, hidden)
        pos = input_ids.shape[0]
        out_ids: list[int] = []
        for _ in range(max_new_tokens):
            token = sample_token(self.logits(last), temperature, top_p, top_k,
                                 repetition_penalty, out_ids)
            tid = int(token)
            out_ids.append(tid)
            if eos_token_id is not None and tid == eos_token_id:
                break
            last = self.decode_step(token, cache, pos, num_layers).reshape(1, -1)
            pos += 1
        return out_ids

    def generate_stream(self, input_ids: Tensor, max_new_tokens: int = 32,
                        temperature: float = 0.7, top_p: float = 0.9, top_k: int = 0,
                        repetition_penalty: float = 1.0, eos_token_id: int | None = None,
                        num_layers: int | None = None) -> Iterator[int]:
        """流式生成器：逐 token yield（供 CLI 实时输出）。"""
        cache = Cache(self.num_layers,
                      max_len=int(input_ids.shape[0]) + max_new_tokens + 16)
        h = self.prefill(input_ids, cache, num_layers)
        last = h[-1:]
        pos = input_ids.shape[0]
        out_ids: list[int] = []
        for _ in range(max_new_tokens):
            token = sample_token(self.logits(last), temperature, top_p, top_k,
                                 repetition_penalty, out_ids)
            tid = int(token)
            out_ids.append(tid)
            yield tid
            if eos_token_id is not None and tid == eos_token_id:
                break
            last = self.decode_step(token, cache, pos, num_layers).reshape(1, -1)
            pos += 1
