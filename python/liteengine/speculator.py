"""Qwen3DSpark 投机解码草稿器（dspark speculator）。

参考 ``models/Qwen3.6-35B-A3B-speculator.dspark`` 的 config：
- 5 层 Qwen3 风格 sliding-attention 草稿 transformer（hidden 2048 / 16 头 / 2 KV 头 / head_dim 256 / intermediate 6144）
- 无独立 embed：输入为主模型 aux 层隐藏状态（``aux_hidden_state_layer_ids=[2,10,20,30,37]``）
- ``lm_head`` 输出精简词表（``draft_vocab_size=32000``）；``t2d`` 映射主词表（248320 → 32000）
- ``markov_head``（248320 → 256 → 32000）：主模型 logits → 草稿词表置信度（投机采样接受）

投机解码：草稿模型贪心产出 K 个 token → 主模型并行验证 → 投机采样接受
（接受判定保证输出分布与主模型自回归一致，草稿质量只影响提速幅度）。
"""
from __future__ import annotations

import json

import torch
from torch import Tensor
from torch.nn.functional import linear, silu, softmax

from liteengine.loader import WeightStore
from liteengine.moe import torch_weight
from liteengine.core.norm import rms_norm
from liteengine.core.rope import apply_rotary_pos_emb, compute_inv_freq, rotary_embeddings

__all__ = ["DSparkSpeculator"]


class _DraftLayer:
    """草稿 transformer 层（sliding attention + SwiGLU MLP，fp32）。

    window=2048 在草稿短序列下退化为标准因果注意力；GQA 16 头 / 2 KV 头。
    """

    def __init__(self, store, prefix: str, cfg: dict):
        self.eps = float(cfg["rms_norm_eps"])
        self.head_dim = int(cfg["head_dim"])
        self.num_heads = int(cfg["num_attention_heads"])
        self.num_kv_heads = int(cfg["num_key_value_heads"])
        self.scaling = self.head_dim ** -0.5
        self.input_norm_w = torch_weight(store, f"{prefix}.input_layernorm.weight")
        self.post_norm_w = torch_weight(store, f"{prefix}.post_attention_layernorm.weight")
        self.q_w = torch_weight(store, f"{prefix}.self_attn.q_proj.weight")
        self.k_w = torch_weight(store, f"{prefix}.self_attn.k_proj.weight")
        self.v_w = torch_weight(store, f"{prefix}.self_attn.v_proj.weight")
        self.o_w = torch_weight(store, f"{prefix}.self_attn.o_proj.weight")
        self.q_norm_w = torch_weight(store, f"{prefix}.self_attn.q_norm.weight")
        self.k_norm_w = torch_weight(store, f"{prefix}.self_attn.k_norm.weight")
        self.gate_w = torch_weight(store, f"{prefix}.mlp.gate_proj.weight")
        self.up_w = torch_weight(store, f"{prefix}.mlp.up_proj.weight")
        self.down_w = torch_weight(store, f"{prefix}.mlp.down_proj.weight")

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        residual = x
        h = rms_norm(x, self.input_norm_w, self.eps)
        B, L, _ = h.shape
        q = linear(h, self.q_w).view(B, L, self.num_heads, self.head_dim)
        k = linear(h, self.k_w).view(B, L, self.num_kv_heads, self.head_dim)
        v = linear(h, self.v_w).view(B, L, self.num_kv_heads, self.head_dim)
        q = rms_norm(q, self.q_norm_w, self.eps)
        k = rms_norm(k, self.k_norm_w, self.eps)
        q, k = apply_rotary_pos_emb(q.transpose(1, 2), k.transpose(1, 2), cos, sin)   # (B, H, L, D)
        k = k.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
        v = v.transpose(1, 2).repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scaling
        mask = torch.triu(torch.full((L, L), float("-inf"), dtype=x.dtype), diagonal=1)
        out = softmax(scores + mask, dim=-1, dtype=torch.float32).to(x.dtype) @ v
        out = out.transpose(1, 2).reshape(B, L, -1)
        h = residual + linear(out, self.o_w)
        residual = h
        h = rms_norm(h, self.post_norm_w, self.eps)
        h = silu(linear(h, self.gate_w)) * linear(h, self.up_w)
        return residual + linear(h, self.down_w)

    __call__ = forward


class DSparkSpeculator:
    """dspark 草稿器：加载草稿模型，提供贪心草稿 token 序列与草稿概率。"""

    def __init__(self, model_dir: str, num_draft: int = 8):
        self.store = WeightStore(model_dir)
        with open(f"{model_dir}/config.json", "r", encoding="utf-8") as f:
            self.cfg = json.load(f)
        tcfg = self.cfg["transformer_layer_config"]
        self.hidden = int(tcfg["hidden_size"])
        self.eps = float(tcfg["rms_norm_eps"])
        self.num_layers = int(tcfg["num_hidden_layers"])
        self.head_dim = int(tcfg["head_dim"])
        self.rope_theta = float(tcfg.get("rope_parameters", {}).get("rope_theta", 1e7))
        self.draft_vocab = int(self.cfg.get("draft_vocab_size", 32000))
        self.num_draft = num_draft                              # speculators_config.speculative_tokens
        self.inv_freq = compute_inv_freq(self.head_dim, self.rope_theta, 1.0)
        self.norm_w = torch_weight(self.store, "norm.weight")     # (2048,)
        self.lm_head_w = torch_weight(self.store, "lm_head.weight")   # (32000, 2048)
        # markov_head（vanilla，vLLM DSparkMarkovHead 语义）：上一草稿 token → 低秩转移偏置。
        # bias[v] = markov_w1[prev_t] @ markov_w2[v]；加到基础 logits 后左到右顺序采样。
        self.markov_w1 = torch_weight(self.store, "markov_head.markov_w1.weight")   # (248320, 256)
        self.markov_w2 = torch_weight(self.store, "markov_head.markov_w2.weight")   # (32000, 256)
        t2d = torch.from_numpy(self.store.get("t2d")).long()          # (248320,) 主→草稿
        self.t2d = t2d
        # 草稿 → 主词表：每个草稿 id 取 t2d 映射的首个主 token（未映射的用自身 id）
        inv = torch.full((self.draft_vocab,), -1, dtype=torch.long)
        seen = torch.zeros(self.draft_vocab, dtype=torch.bool)
        for i in range(int(t2d.shape[0])):
            d = int(t2d[i])
            if 0 <= d < self.draft_vocab and not bool(seen[d]):
                inv[d] = i
                seen[d] = True
        self.d2t = torch.where(seen, inv, torch.arange(self.draft_vocab))
        self._layers: dict[int, _DraftLayer] = {}

    def _layer(self, i: int) -> _DraftLayer:
        if i not in self._layers:
            self._layers[i] = _DraftLayer(self.store, f"layers.{i}",
                                          self.cfg["transformer_layer_config"])
        return self._layers[i]

    def draft(self, h_target: Tensor, embed_main: Tensor, start_pos: int = 0,
              num_tokens: int | None = None) -> tuple[list[int], Tensor]:
        """贪心草稿 num_tokens 个 token。

        Args:
            h_target: (1, hidden) 主模型 aux 层目标位置的隐藏状态。
            embed_main: (vocab, hidden) 主模型词嵌入（草稿 token 的下一步输入）。
            start_pos: 草稿起始位置（RoPE 位置推进）。
        Returns:
            (主词表 token 列表, 草稿概率 (num_tokens, draft_vocab) fp32)。
        """
        n = self.num_draft if num_tokens is None else num_tokens
        h = h_target.unsqueeze(0).float()                        # (1, 1, hidden)
        probs: list[Tensor] = []
        tokens: list[int] = []
        prev_t: int | None = None
        with torch.no_grad():
            for step in range(n):
                pos = torch.tensor([start_pos + step])
                cos, sin = rotary_embeddings(pos, self.inv_freq)
                for i in range(self.num_layers):
                    h = self._layer(i)(h, cos, sin)
                h = rms_norm(h, self.norm_w, self.eps)
                logits = linear(h.squeeze(0), self.lm_head_w)    # (1, draft_vocab)
                if prev_t is not None:
                    # 转移偏置（vLLM DSparkMarkovHead）：上一草稿 token 的低秩过渡依赖
                    logits = logits + (self.markov_w1[prev_t] @ self.markov_w2.t()).unsqueeze(0)
                probs.append(softmax(logits, dim=-1, dtype=torch.float32)[0])
                d = int(logits[0].argmax())
                t = int(self.d2t[d])
                tokens.append(t)
                prev_t = t
                # 下一步输入：草稿 token 的主模型嵌入（草稿模型无独立 embed）
                h = embed_main[t].reshape(1, 1, -1).float()
        return tokens, torch.stack(probs)


def speculative_accept(draft_ids: list[int], draft_probs: Tensor, verify_logits: Tensor,
                       extra_logits: Tensor, t2d: Tensor, temperature: float = 0.7,
                       top_p: float = 0.9, top_k: int = 0,
                       repetition_penalty: float = 1.0,
                       prev_ids: list[int] | None = None,
                       eos_token_id: int | None = None) -> tuple[list[int], int, int | None]:
    """投机采样接受：接受最长前缀，拒绝时从主模型分布重采样。

    - ``verify_logits[i]``：主模型在草稿位置 i（处理该草稿**之前**）的预测分布，
      用于判定草稿 i 的概率（``logits(上下文最后位置) + logits(hidden[0..K-2])``）。
    - ``extra_logits``：全接受后（K 个草稿之后）的新 token 分布。
    - 贪婪（``temperature <= 0``）：确定性接受（草稿 token == 主模型 argmax）。

    返回 (接受的前缀 token 列表, 接受数, 新采样 token（拒绝时）或 None（全接受）)。
    """
    from liteengine.sampling import sample_token
    accepted: list[int] = []
    greedy = temperature <= 0.0
    r = torch.rand(len(draft_ids))
    for i, t in enumerate(draft_ids):
        if greedy:
            ok = t == int(verify_logits[i].argmax())
        else:
            p_main = float(softmax(verify_logits[i].float() / temperature, dim=-1)[t])
            d = int(t2d[t]) if 0 <= t < t2d.shape[0] else -1
            p_draft = float(draft_probs[i][d]) if 0 <= d < draft_probs.shape[1] else 0.0
            ok = r[i].item() < min(1.0, p_main / (p_draft + 1e-12))
        if ok:
            accepted.append(t)
        else:
            # 拒绝：从主模型位置 i 的分布重采样
            token = int(sample_token(
                verify_logits[i].unsqueeze(0), temperature, top_p, top_k,
                repetition_penalty, prev_ids + accepted))
            return accepted, len(accepted), token
    # 全部接受：额外采样一个 token（K 个草稿之后的分布）
    token = int(sample_token(
        extra_logits.unsqueeze(0), temperature, top_p, top_k,
        repetition_penalty, prev_ids + accepted))
    return accepted, len(accepted), token
