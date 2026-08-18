"""混合注意力：FullAttention（GQA + RoPE）与 GatedDeltaNet（线性注意力，FLA chunked delta rule）。

逐行对照 transformers 5.15 ``modeling_qwen3_5_moe.py``：
- ``eager_attention_forward`` / ``Qwen3_5MoeAttention``（full_attention 层）
- ``torch_chunk_gated_delta_rule`` / ``causal_conv1d_fn`` / ``Qwen3_5MoeGatedDeltaNet``（linear_attention 层）

本模块为无 cache（prefill）路径；decode 的 recurrent 路径在 M3 补。
"""
from __future__ import annotations

import torch
from torch import Tensor
from torch.nn.functional import conv1d, linear, pad, silu, softplus

from liteengine.cache import kv_append
from liteengine.moe import torch_weight
from liteengine.norm import rms_norm, rms_norm_gated
from liteengine.rope import apply_rotary_pos_emb

__all__ = ["FullAttention", "GatedDeltaNet", "StandardAttention", "MlaAttention",
           "chunk_gated_delta_rule", "recurrent_gated_delta_rule"]


def _causal_conv1d(x: Tensor, weight: Tensor, activation=silu) -> Tensor:
    """因果深度卷积：x (B, C, L)，weight (C, kernel)。padding=kernel-1 后截取前 L。

    内部按 fp32 计算（CPU conv1d 对 fp16 支持不稳），输出还原输入 dtype。
    """
    pad_k = weight.shape[-1] - 1
    out = conv1d(x.float(), weight.float().unsqueeze(1), padding=pad_k,
                 groups=x.shape[1])[:, :, : x.shape[-1]]
    if activation is not None:
        out = activation(out)
    return out.to(x.dtype)


def l2norm(x: Tensor, dim: int = -1, eps: float = 1e-6) -> Tensor:
    """与 FLA 库对齐的 l2 归一化。"""
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


def _repeat_kv(h: Tensor, n_rep: int) -> Tensor:
    """(B, kv_heads, L, D) → (B, kv_heads*n_rep, L, D)。"""
    if n_rep == 1:
        return h
    B, kv, L, D = h.shape
    return h[:, :, None, :, :].expand(B, kv, n_rep, L, D).reshape(B, kv * n_rep, L, D)


def eager_attention(query: Tensor, key: Tensor, value: Tensor, mask: Tensor | None,
                    scaling: float, n_rep: int) -> Tensor:
    """标准 eager attention（对照 transformers ``eager_attention_forward``）：
    repeat_kv → matmul*scaling → +mask → softmax(fp32) → matmul。
    返回 (B, L, H, D)（与 transformers 一致，由调用方 reshape）。"""
    k = _repeat_kv(key, n_rep)
    v = _repeat_kv(value, n_rep)
    attn = torch.matmul(query, k.transpose(-2, -1)) * scaling
    if mask is not None:
        attn = attn + mask
    attn = torch.softmax(attn, dim=-1, dtype=torch.float32).to(query.dtype)
    return torch.matmul(attn, v).transpose(1, 2).contiguous()


def chunk_gated_delta_rule(query: Tensor, key: Tensor, value: Tensor, g: Tensor, beta: Tensor,
                           chunk_size: int = 64, initial_state: Tensor | None = None) -> Tensor:
    """FLA chunked gated delta rule（对照 transformers ``torch_chunk_gated_delta_rule``）。

    输入 q/k/v: (B, L, heads, dim)，g/beta: (B, L, heads)（fp32）。
    返回 (B, L, heads, v_dim)，L2 归一化 q/k 在内部完成。
    ``initial_state``：续接前向的初始 recurrent 状态（缓存 rec_state，投机解码验证用）。
    """
    initial_dtype = query.dtype
    query = l2norm(query, dim=-1)
    key = l2norm(key, dim=-1)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]
    B, H, L, kd = key.shape
    vd = value.shape[-1]
    pad_len = (chunk_size - L % chunk_size) % chunk_size
    query = pad(query, (0, 0, 0, pad_len))
    key = pad(key, (0, 0, 0, pad_len))
    value = pad(value, (0, 0, 0, pad_len))
    beta = pad(beta, (0, pad_len))
    g = pad(g, (0, pad_len))
    total = L + pad_len
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
        for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool), diagonal=0)

    # chunk 内衰减
    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last = (
        torch.zeros(B, H, kd, vd, dtype=value.dtype, device=value.device)
        if initial_state is None
        else initial_state.to(value)
    )
    core = torch.zeros_like(value)
    mask2 = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool), diagonal=1)

    # 逐 chunk 递推
    for i in range(total // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]
        v_prime = k_cumdecay[:, :, i] @ last
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last
        core[:, :, i] = attn_inter + attn @ v_new
        last = (
            last * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    core = core.reshape(core.shape[0], core.shape[1], -1, core.shape[-1])
    core = core[:, :, :L].transpose(1, 2).contiguous()
    return core.to(initial_dtype), last


def recurrent_gated_delta_rule(query: Tensor, key: Tensor, value: Tensor, g: Tensor, beta: Tensor,
                               initial_state: Tensor | None = None,
                               output_final_state: bool = False) -> tuple[Tensor, Tensor | None]:
    """递归版 delta rule（对照 transformers ``torch_recurrent_gated_delta_rule``）。

    逐 token 更新状态；用于 decode（initial_state=缓存状态）与测试中与 chunk 版互证。
    输入 q/k/v: (B, L, heads, dim)，g/beta: (B, L, heads)。
    返回 (core (B, L, heads, v_dim), last_state (B, heads, kd, vd) 或 None)。
    """
    initial_dtype = query.dtype
    query = l2norm(query, dim=-1)
    key = l2norm(key, dim=-1)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]
    B, H, L, kd = key.shape
    vd = value.shape[-1]
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale
    core = torch.zeros(B, H, L, vd, dtype=value.dtype, device=value.device)
    last = (
        torch.zeros(B, H, kd, vd, dtype=value.dtype, device=value.device)
        if initial_state is None
        else initial_state.to(value)
    )
    for i in range(L):
        q_t, k_t, v_t = query[:, :, i], key[:, :, i], value[:, :, i]
        g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, i].unsqueeze(-1)
        last = last * g_t
        kv_mem = (last * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        last = last + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        core[:, :, i] = (last * q_t.unsqueeze(-1)).sum(dim=-2)
    if not output_final_state:
        last = None
    return core.transpose(1, 2).contiguous().to(initial_dtype), last


class FullAttention:
    """Qwen3_5MoeAttention：q_proj 输出含 gate（attn_output_gate），q/k 带 head_dim RMSNorm，
    partial RoPE（64 维），GQA（2 kv heads），输出乘 sigmoid(gate)。"""

    def __init__(self, store, prefix: str, cfg: dict):
        self.head_dim = int(cfg["head_dim"])
        self.num_heads = int(cfg["num_attention_heads"])
        self.num_kv_heads = int(cfg["num_key_value_heads"])
        self.scaling = self.head_dim ** -0.5
        self.eps = float(cfg["rms_norm_eps"])
        self.q_w = torch_weight(store, prefix + ".q_proj.weight")
        self.k_w = torch_weight(store, prefix + ".k_proj.weight")
        self.v_w = torch_weight(store, prefix + ".v_proj.weight")
        self.o_w = torch_weight(store, prefix + ".o_proj.weight")
        self.q_norm_w = torch_weight(store, prefix + ".q_norm.weight")
        self.k_norm_w = torch_weight(store, prefix + ".k_norm.weight")

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor, mask: Tensor | None,
                cache=None, layer_idx: int = 0) -> Tensor:
        """prefill：cache 提供时记录 (key, value) 供 decode 续接。"""
        B, L, _ = x.shape
        qg = linear(x, self.q_w).view(B, L, -1, self.head_dim * 2)
        query, gate = torch.chunk(qg, 2, dim=-1)          # (B, L, H, head_dim), gate 同形
        gate = gate.reshape(B, L, -1)
        query = rms_norm(query, self.q_norm_w, self.eps).transpose(1, 2)   # (B, H, L, D)
        key = rms_norm(linear(x, self.k_w).view(B, L, -1, self.head_dim), self.k_norm_w, self.eps).transpose(1, 2)
        value = linear(x, self.v_w).view(B, L, -1, self.head_dim).transpose(1, 2)
        query, key = apply_rotary_pos_emb(query, key, cos, sin)
        if cache is not None:
            kv_prev = cache.attn_kv[layer_idx]
            if kv_prev is None:
                cache.attn_kv[layer_idx] = (None, None, 0, cache.max_len)
                kv_prev = cache.attn_kv[layer_idx]
            ctx = kv_prev[2]
            new_kv, (key, value) = kv_append(kv_prev, key, value)
            cache.attn_kv[layer_idx] = new_kv
            if ctx > 0:
                cm = torch.zeros(L, ctx + L, dtype=x.dtype, device=x.device)
                cm = cm.masked_fill(
                    torch.triu(torch.ones(L, ctx + L, dtype=torch.bool), diagonal=ctx + 1),
                    float("-inf"))
                mask = cm
        out = eager_attention(query, key, value, mask, self.scaling,
                              self.num_heads // self.num_kv_heads)          # (B, L, H*D)
        out = out.reshape(B, L, -1)
        out = out * torch.sigmoid(gate)
        return linear(out, self.o_w)

    def forward_step(self, x: Tensor, cos: Tensor, sin: Tensor, kv
                     ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        """decode 单 token：x (B, 1, hidden)。kv=(k_prev, v_prev)，返回 (out, new_kv)。

        单 query 对全量缓存自回归（当前 token 只能看到过去与自身，无需掩码）。
        """
        B = x.shape[0]
        qg = linear(x, self.q_w).view(B, 1, -1, self.head_dim * 2)
        query, gate = torch.chunk(qg, 2, dim=-1)
        gate = gate.reshape(B, 1, -1)
        query = rms_norm(query, self.q_norm_w, self.eps).transpose(1, 2)
        key = rms_norm(linear(x, self.k_w).view(B, 1, -1, self.head_dim), self.k_norm_w, self.eps).transpose(1, 2)
        value = linear(x, self.v_w).view(B, 1, -1, self.head_dim).transpose(1, 2)
        query, key = apply_rotary_pos_emb(query, key, cos, sin)
        new_kv, (k, v) = kv_append(kv, key, value)
        out = eager_attention(query, k, v, None, self.scaling,
                              self.num_heads // self.num_kv_heads)          # (B, 1, H*D)
        out = out.reshape(B, 1, -1)
        out = out * torch.sigmoid(gate)
        return linear(out, self.o_w), new_kv

    __call__ = forward


class StandardAttention:
    """标准 GQA 注意力（Mixtral / Qwen3-MoE / GLM 等，无 gate）。

    与 Qwen3.5 的 FullAttention 结构一致但无 gate（q_proj 不翻倍）：
    q/k/v/o 投影 → RoPE → 因果注意力（+ KV 缓存续接）。
    """

    def __init__(self, store, prefix: str, cfg: dict):
        self.num_heads = int(cfg["num_attention_heads"])
        self.num_kv_heads = int(cfg["num_key_value_heads"])
        head_dim = int(cfg.get("head_dim", 0))
        self.head_dim = head_dim if head_dim > 0 else int(cfg["hidden_size"]) // self.num_heads
        self.eps = float(cfg["rms_norm_eps"])
        self.scaling = self.head_dim ** -0.5
        self.q_w = torch_weight(store, f"{prefix}.q_proj.weight")
        self.k_w = torch_weight(store, f"{prefix}.k_proj.weight")
        self.v_w = torch_weight(store, f"{prefix}.v_proj.weight")
        self.o_w = torch_weight(store, f"{prefix}.o_proj.weight")

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor, mask: Tensor | None = None,
                cache=None, layer_idx: int = 0) -> Tensor:
        B, L, _ = x.shape
        q = linear(x, self.q_w).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = linear(x, self.k_w).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = linear(x, self.v_w).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        if cache is not None:
            kv_prev = cache.attn_kv[layer_idx]
            if kv_prev is None:
                cache.attn_kv[layer_idx] = (None, None, 0, cache.max_len)
                kv_prev = cache.attn_kv[layer_idx]
            ctx = kv_prev[2]
            new_kv, (k, v) = kv_append(kv_prev, k, v)
            cache.attn_kv[layer_idx] = new_kv
            if ctx > 0:
                cm = torch.zeros(L, ctx + L, dtype=x.dtype, device=x.device)
                cm = cm.masked_fill(
                    torch.triu(torch.ones(L, ctx + L, dtype=torch.bool), diagonal=ctx + 1),
                    float("-inf"))
                mask = cm
        out = eager_attention(q, k, v, mask, self.scaling,
                              self.num_heads // self.num_kv_heads)          # (B, L, H*D)
        return linear(out.reshape(B, L, -1), self.o_w)

    def forward_step(self, x: Tensor, cos: Tensor, sin: Tensor, kv
                     ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        """decode 单 token（无 gate 版，与 FullAttention.forward_step 同构）。"""
        B = x.shape[0]
        q = linear(x, self.q_w).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k = linear(x, self.k_w).view(B, 1, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = linear(x, self.v_w).view(B, 1, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        new_kv, (k, v) = kv_append(kv, k, v)
        out = eager_attention(q, k, v, None, self.scaling,
                              self.num_heads // self.num_kv_heads)
        return linear(out.reshape(B, 1, -1), self.o_w), new_kv

    __call__ = forward


class MlaAttention:
    """MLA（Multi-head Latent Attention，DeepSeek-V2/V3 与 Kimi K2 风格）：

    q/k/v 经低秩投影压缩（``q_a_proj``→``q_b_proj`` / ``kv_a_proj``→``kv_b_proj``），
    RoPE 仅应用于解耦的 rope 部分；注意力分数 = ``q_nope@k_nope + q_rope@k_rope``（解耦打分）。

    - ``q_head_dim = qk_nope_head_dim + qk_rope_head_dim``，``qk_nope_head_dim = v_head_dim``
    - KV 缓存：k_nope / k_rope / v（本实现为全量缓存，保持与 StandardAttention 一致的续接语义）
    """

    def __init__(self, store, prefix: str, cfg: dict):
        self.num_heads = int(cfg["num_attention_heads"])
        self.kv_lora_rank = int(cfg.get("kv_lora_rank", 512))
        self.q_lora_rank = int(cfg.get("q_lora_rank", 0))
        self.qk_rope_head_dim = int(cfg.get("qk_rope_head_dim", 64))
        self.v_head_dim = int(cfg.get("v_head_dim", 128))
        self.qk_nope_head_dim = self.v_head_dim                # DeepSeek 约定
        self.q_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.eps = float(cfg["rms_norm_eps"])
        self.scaling = self.qk_nope_head_dim ** -0.5
        self.q_a_w = torch_weight(store, f"{prefix}.q_a_proj.weight")
        self.q_b_w = torch_weight(store, f"{prefix}.q_b_proj.weight")
        self.kv_a_w = torch_weight(store, f"{prefix}.kv_a_proj.weight")
        self.kv_b_w = torch_weight(store, f"{prefix}.kv_b_proj.weight")
        self.o_w = torch_weight(store, f"{prefix}.o_proj.weight")
        self.q_norm_w = torch_weight(store, f"{prefix}.q_a_layernorm.weight")
        self.k_norm_w = torch_weight(store, f"{prefix}.k_pe_layernorm.weight")

    def _project(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """返回 (q_nope, q_rope, k_nope, k_rope, v)，均为 (B, H, L, D)。"""
        B, L, _ = x.shape
        q_latent = rms_norm(linear(x, self.q_a_w), self.q_norm_w, self.eps)   # (B, L, q_lora)
        q = linear(q_latent, self.q_b_w).view(B, L, self.num_heads,
                                              self.q_head_dim).transpose(1, 2)
        q_nope, q_rope = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        kv = linear(x, self.kv_a_w)                                            # (B, L, kv_lora+rope)
        kv_latent, k_rope = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_rope = rms_norm(k_rope, self.k_norm_w, self.eps)
        kv_out = linear(kv_latent, self.kv_b_w).view(
            B, L, self.num_heads, self.qk_nope_head_dim + self.v_head_dim).transpose(1, 2)
        k_nope, v = torch.split(kv_out, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        return q_nope, q_rope, k_nope, k_rope, v

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor, mask: Tensor | None = None,
                cache=None, layer_idx: int = 0) -> Tensor:
        B, L, _ = x.shape
        q_nope, q_rope, k_nope, k_rope, v = self._project(x)
        q_rope, k_rope = apply_rotary_pos_emb(q_rope, k_rope, cos, sin)
        if cache is not None:
            kv_prev = cache.attn_kv[layer_idx]
            if kv_prev is None:
                cache.attn_kv[layer_idx] = (k_nope, k_rope, v)
            else:
                k_n_prev, k_r_prev, v_prev = kv_prev
                k_nope = torch.cat([k_n_prev, k_nope], dim=2)
                k_rope = torch.cat([k_r_prev, k_rope], dim=1)    # k_rope 为跨头共享 (B, L, rope)
                v = torch.cat([v_prev, v], dim=2)
                cache.attn_kv[layer_idx] = (k_nope, k_rope, v)
                ctx = k_n_prev.shape[2]
                cm = torch.zeros(L, ctx + L, dtype=x.dtype, device=x.device)
                cm = cm.masked_fill(
                    torch.triu(torch.ones(L, ctx + L, dtype=torch.bool), diagonal=ctx + 1),
                    float("-inf"))
                mask = cm
        attn = (torch.matmul(q_nope, k_nope.transpose(-1, -2))
                + torch.matmul(q_rope, k_rope.transpose(-1, -2))) * self.scaling
        if mask is not None:
            attn = attn + mask
        out = torch.softmax(attn, dim=-1, dtype=torch.float32).to(x.dtype) @ v
        return linear(out.transpose(1, 2).reshape(B, L, -1), self.o_w)

    def forward_step(self, x: Tensor, cos: Tensor, sin: Tensor, kv
                     ) -> tuple[Tensor, tuple]:
        """decode 单 token：kv=(k_nope, k_rope, v) 缓存续接。"""
        B = x.shape[0]
        q_nope, q_rope, k_nope, k_rope, v = self._project(x)
        q_rope, k_rope = apply_rotary_pos_emb(q_rope, k_rope, cos, sin)
        k_n_prev, k_r_prev, v_prev = kv
        k_nope = torch.cat([k_n_prev, k_nope], dim=2)
        k_rope = torch.cat([k_r_prev, k_rope], dim=1)            # k_rope 为跨头共享 (B, L, rope)
        v = torch.cat([v_prev, v], dim=2)
        attn = (torch.matmul(q_nope, k_nope.transpose(-1, -2))
                + torch.matmul(q_rope, k_rope.transpose(-1, -2))) * self.scaling
        out = torch.softmax(attn, dim=-1, dtype=torch.float32).to(x.dtype) @ v
        return linear(out.transpose(1, 2).reshape(B, 1, -1), self.o_w), (k_nope, k_rope, v)

    __call__ = forward


class GatedDeltaNet:
    """Qwen3_5MoeGatedDeltaNet：conv1d + in_proj_qkv/a/b/z + chunked delta rule + gated norm。"""

    def __init__(self, store, prefix: str, cfg: dict):
        self.key_dim = int(cfg["linear_key_head_dim"]) * int(cfg["linear_num_key_heads"])
        self.value_dim = int(cfg["linear_value_head_dim"]) * int(cfg["linear_num_value_heads"])
        self.num_k_heads = int(cfg["linear_num_key_heads"])
        self.num_v_heads = int(cfg["linear_num_value_heads"])
        self.head_k_dim = int(cfg["linear_key_head_dim"])
        self.head_v_dim = int(cfg["linear_value_head_dim"])
        self.eps = float(cfg["rms_norm_eps"])
        self.in_proj_qkv = torch_weight(store, prefix + ".in_proj_qkv.weight")
        self.in_proj_z = torch_weight(store, prefix + ".in_proj_z.weight")
        self.in_proj_b = torch_weight(store, prefix + ".in_proj_b.weight")
        self.in_proj_a = torch_weight(store, prefix + ".in_proj_a.weight")
        self.conv_w = torch_weight(store, prefix + ".conv1d.weight").squeeze(1)  # (C, kernel)
        self.A_log = torch_weight(store, prefix + ".A_log")
        self.dt_bias = torch_weight(store, prefix + ".dt_bias")
        self.norm_w = torch_weight(store, prefix + ".norm.weight")
        self.out_w = torch_weight(store, prefix + ".out_proj.weight")

    def forward(self, x: Tensor, cache=None, layer_idx: int = 0) -> Tensor:
        """prefill：x (B, L, hidden)。cache 提供时记录 conv/recurrent 状态供 decode 续接。"""
        B, L, _ = x.shape
        qkv = linear(x, self.in_proj_qkv).transpose(1, 2)          # (B, C, L) 卷积前输入
        kernel = self.conv_w.shape[-1]
        qkv_pre = qkv
        conv_state = cache.conv_state[layer_idx] if cache is not None else None
        if conv_state is None:
            qkv = _causal_conv1d(qkv_pre, self.conv_w, silu).transpose(1, 2)   # (B, L, C)
        else:
            # 续接：从缓存 conv_state 起步（vLLM causal_conv1d_update 语义，参考已核对）
            inp = torch.cat([conv_state, qkv_pre], dim=-1)         # (B, C, kernel-1+L)
            qkv = _causal_conv1d(inp, self.conv_w, silu)[:, :, kernel - 1:].transpose(1, 2)
        if cache is not None:
            # conv 状态 = 卷积输入（pre-conv）序列末尾 kernel-1 个值（含左侧零填充）。
            # 注意：必须是卷积前的输入值（decode 续接用它做 causal_conv1d_update）。
            state = qkv_pre[:, :, -(kernel - 1):]
            need = kernel - 1
            if state.shape[-1] < need:
                state = torch.cat(
                    [torch.zeros(B, state.shape[1], need - state.shape[-1],
                                 dtype=qkv.dtype, device=qkv.device), state], dim=-1)
            cache.conv_state[layer_idx] = state.contiguous()
        query, key, value = torch.split(qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        query = query.reshape(B, L, -1, self.head_k_dim)
        key = key.reshape(B, L, -1, self.head_k_dim)
        value = value.reshape(B, L, -1, self.head_v_dim)
        z = linear(x, self.in_proj_z).reshape(B, L, -1, self.head_v_dim)
        b = linear(x, self.in_proj_b)
        a = linear(x, self.in_proj_a)
        beta = b.sigmoid()
        g = -self.A_log.float().exp() * softplus(a.float() + self.dt_bias)
        if self.num_v_heads // self.num_k_heads > 1:
            rep = self.num_v_heads // self.num_k_heads
            query = query.repeat_interleave(rep, dim=2)
            key = key.repeat_interleave(rep, dim=2)
        out, last = chunk_gated_delta_rule(query, key, value, g, beta,   # (B, L, v_heads, v_dim)
                                           initial_state=cache.rec_state[layer_idx]
                                           if cache is not None else None)
        if cache is not None:
            cache.rec_state[layer_idx] = last
        out = out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        out = rms_norm_gated(out, self.norm_w, z, self.eps)
        out = out.reshape(B, L, -1)
        return linear(out, self.out_w)

    def forward_step(self, x: Tensor, conv_state: Tensor | None, rec_state: Tensor | None
                     ) -> tuple[Tensor, Tensor, Tensor | None]:
        """decode 单 token：x (B, 1, hidden)。返回 (out, new_conv_state, new_rec_state)。

        对照 transformers ``causal_conv1d_update`` + ``torch_recurrent_gated_delta_rule``。
        """
        B = x.shape[0]
        qkv = linear(x, self.in_proj_qkv).transpose(1, 2)          # (B, C, 1)
        if conv_state is None:
            conv_state = torch.zeros(B, self.conv_w.shape[0], self.conv_w.shape[-1] - 1,
                                     dtype=x.dtype, device=x.device)
        inp = torch.cat([conv_state, qkv], dim=-1)                 # (B, C, kernel)
        out = silu(conv1d(inp.float(), self.conv_w.float().unsqueeze(1),
                          groups=self.conv_w.shape[0])).to(inp.dtype)
        new_conv = inp[:, :, -(self.conv_w.shape[-1] - 1):].contiguous()
        qkv = out.transpose(1, 2)                                  # (B, 1, C)
        query, key, value = torch.split(qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        query = query.reshape(B, 1, -1, self.head_k_dim)
        key = key.reshape(B, 1, -1, self.head_k_dim)
        value = value.reshape(B, 1, -1, self.head_v_dim)
        z = linear(x, self.in_proj_z).reshape(B, 1, -1, self.head_v_dim)
        b = linear(x, self.in_proj_b)
        a = linear(x, self.in_proj_a)
        beta = b.sigmoid()
        g = -self.A_log.float().exp() * softplus(a.float() + self.dt_bias)
        if self.num_v_heads // self.num_k_heads > 1:
            rep = self.num_v_heads // self.num_k_heads
            query = query.repeat_interleave(rep, dim=2)
            key = key.repeat_interleave(rep, dim=2)
        out, new_rec = recurrent_gated_delta_rule(query, key, value, g, beta,
                                                  initial_state=rec_state, output_final_state=True)
        out = out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        out = rms_norm_gated(out, self.norm_w, z, self.eps)
        out = out.reshape(B, 1, -1)
        return linear(out, self.out_w), new_conv, new_rec

    __call__ = forward


# ---- 注册表：内置注意力构建器（layer 按名称查找；外部组件可新增注册）----

from liteengine.registry import register_attention


@register_attention("standard")
def _build_standard(store, prefix, cfg):
    return StandardAttention(store, f"{prefix}.self_attn", cfg)


@register_attention("full_gated")
def _build_full_gated(store, prefix, cfg):
    return FullAttention(store, f"{prefix}.self_attn", cfg)


@register_attention("linear_delta")
def _build_linear_delta(store, prefix, cfg):
    return GatedDeltaNet(store, f"{prefix}.linear_attn", cfg)


@register_attention("mla")
def _build_mla(store, prefix, cfg):
    return MlaAttention(store, f"{prefix}.self_attn", cfg)
