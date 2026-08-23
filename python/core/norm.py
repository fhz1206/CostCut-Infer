"""RMSNorm 与 Gated RMSNorm（对照 transformers Qwen3_5MoeRMSNorm / Qwen3_5MoeRMSNormGated）。

注意 Qwen3_5Moe 的 RMSNorm 与 Llama 不同：输出是 ``norm(x) * (1 + weight)``（weight 初始为 0），
而不是 ``norm(x) * weight``；Gated 变体再乘 ``silu(gate)``。
"""
from __future__ import annotations

import torch
from torch import Tensor
from torch.nn.functional import silu


def rms_norm(x: Tensor, weight: Tensor, eps: float = 1e-6) -> Tensor:
    """Qwen3_5MoeRMSNorm：``norm(x.float()) * (1 + weight.float())``，结果还原为 x 的 dtype。"""
    x_f = x.float()
    variance = x_f.pow(2).mean(-1, keepdim=True)
    out = x_f * torch.rsqrt(variance + eps)
    out = out * (1.0 + weight.float())
    return out.type_as(x)


def rms_norm_add(x: Tensor, residual: Tensor, weight: Tensor,
                 eps: float = 1e-6) -> tuple[Tensor, Tensor]:
    """融合：残差相加与 RMSNorm 合并（vLLM fused add-rmsnorm 的 torch 层面对应）。

    返回 ``(norm(x + residual), x + residual)``——归一化结果 + 残差和（层最终输出用），
    消除 Python 层的一步中间张量往返（差异报告 #8 算子融合）。
    """
    h = x + residual
    return rms_norm(h, weight, eps), h


def rms_norm_gated(x: Tensor, weight: Tensor, gate: Tensor, eps: float = 1e-6) -> Tensor:
    """Qwen3_5MoeRMSNormGated：norm → 乘 weight → 乘 silu(gate)。

    用于 linear_attn 的 norm（head_v_dim 维），gate 来自 in_proj_z 输出 z。
    """
    x_f = x.float()
    variance = x_f.pow(2).mean(-1, keepdim=True)
    out = x_f * torch.rsqrt(variance + eps)
    out = weight * out.type_as(x)
    out = out * silu(gate.float())
    return out.type_as(x)
