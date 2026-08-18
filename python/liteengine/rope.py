"""RoPE 旋转位置编码（对照 transformers Qwen3_5MoeTextRotaryEmbedding 文本路径）。

文本路径下 mrope 的三个维度完全相同，``apply_interleaved_mrope`` 是恒等变换，
因此直接按 1D 位置生成 cos/sin。
``partial_rotary_factor = 0.25``：只旋转 head_dim 的前 25%（256 → 64 维），
其余维度原样透传（``apply_rotary_pos_emb`` 的 q_pass/k_pass）。
"""
from __future__ import annotations

import torch
from torch import Tensor


def compute_inv_freq(head_dim: int, theta: float = 1e7, partial_rotary_factor: float = 0.25) -> Tensor:
    """``inv_freq = 1 / (theta ** (arange(0, rotary_dim, 2) / rotary_dim))``。"""
    rotary_dim = int(head_dim * partial_rotary_factor)
    arange = torch.arange(0, rotary_dim, 2, dtype=torch.float32)
    return 1.0 / (theta ** (arange / rotary_dim))


def rotary_embeddings(position_ids: Tensor, inv_freq: Tensor) -> tuple[Tensor, Tensor]:
    """文本路径 cos/sin：shape (seq, rotary_dim)。

    ``freqs = position * inv_freq``，``emb = cat((freqs, freqs))``（与 transformers 一致）。
    """
    freqs = position_ids.float().unsqueeze(-1) * inv_freq   # (seq, rotary_dim/2)
    emb = torch.cat((freqs, freqs), dim=-1)                 # (seq, rotary_dim)
    return emb.cos(), emb.sin()


def rotate_half(x: Tensor) -> Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
    """只旋转前 rotary_dim 维，其余透传（partial rotary）。

    q/k: (bs, heads, seq, head_dim)；cos/sin: (seq, rotary_dim)，尾部广播。
    """
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)
    return torch.cat([q_embed, q_pass], dim=-1), torch.cat([k_embed, k_pass], dim=-1)
