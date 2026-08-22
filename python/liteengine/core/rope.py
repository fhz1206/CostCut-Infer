"""RoPE 旋转位置编码（对照 transformers Qwen3_5MoeTextRotaryEmbedding 文本路径）。

文本路径下 mrope 的三个维度完全相同，``apply_interleaved_mrope`` 是恒等变换，
因此直接按 1D 位置生成 cos/sin。
``partial_rotary_factor = 0.25``：只旋转 head_dim 的前 25%（256 → 64 维），
其余维度原样透传（``apply_rotary_pos_emb`` 的 q_pass/k_pass）。
"""
from __future__ import annotations

import math

import torch
from torch import Tensor


def compute_inv_freq(head_dim: int, theta: float = 1e7, partial_rotary_factor: float = 0.25,
                     rope_type: str = "default", rope_scaling: dict | None = None) -> Tensor:
    """计算 RoPE 逆频率。``inv_freq = 1 / (theta ** (arange(0, rotary_dim, 2) / rotary_dim))``。

    rope_type="yarn" 时启用完整 YaRN（NTK-by-parts 波长斜坡插值 + 注意力温度缩放）：
    - NTK-by-parts：``h(θ) = (1-γ)·θ/s + γ·θ``——γ(r) 为波长斜坡（r = 波长/原始最大长度）：
      r < β_slow（高频）插值 θ/s；r > β_fast（低频）保持 θ；中间平滑过渡。
    - 温度缩放：``√(1/t) = 0.1·ln(s) + 1``——缩放频率（等价于注意力 logits 温度）。
    """
    rotary_dim = int(head_dim * partial_rotary_factor)
    arange = torch.arange(0, rotary_dim, 2, dtype=torch.float32)
    inv_freq = 1.0 / (theta ** (arange / rotary_dim))
    if rope_type == "yarn":
        scaling = rope_scaling or {}
        factor = float(scaling.get("factor", 2.0))
        original_max = float(scaling.get("original_max_position_embeddings", 4096))
        beta_fast = float(scaling.get("beta_fast", 32.0))   # 低频保持边界（γ=1）
        beta_slow = float(scaling.get("beta_slow", 1.0))    # 高频插值边界（γ=0）
        freqs = theta ** (arange / rotary_dim)              # 原始频率 θ（角频率 ω=θ^(2i/d)——波长=2π·θ^(2i/d)）
        new_freqs = []
        for f in freqs:
            wavelen = 2 * math.pi * float(f)                # 波长（周期——2π/ω=2π·θ^(2i/d)——i 大波长长）
            r = wavelen / original_max                      # 归一化波长
            if r < beta_slow:
                gamma = 0.0                                  # 高频：插值 θ/s
            elif r > beta_fast:
                gamma = 1.0                                  # 低频：保持 θ
            else:
                gamma = (r - beta_slow) / (beta_fast - beta_slow)
            new_freqs.append((1 - gamma) * float(f) / factor + gamma * float(f))
        inv_freq = 1.0 / torch.tensor(new_freqs, dtype=torch.float32)
        # 注意力温度缩放：√(1/t) = 0.1·ln(s) + 1（等价于缩放 cos/sin 频率）
        temp = 0.1 * math.log(factor) + 1.0
        inv_freq = inv_freq * temp
    return inv_freq


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
