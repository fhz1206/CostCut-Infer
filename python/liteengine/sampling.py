"""采样策略：greedy / top-k / top-p（nucleus，HF 风格实现）。"""
from __future__ import annotations

import torch
from torch import Tensor

__all__ = ["sample_token"]


def sample_token(logits: Tensor, temperature: float = 0.7, top_p: float = 0.9,
                 top_k: int = 0, repetition_penalty: float = 1.0,
                 prev_ids: list[int] | None = None) -> Tensor:
    """从 (…, vocab) logits 采样一个 token id（int64 张量，标量或批）。

    - temperature <= 0 或 top_k == 1 且 top_p >= 1：greedy（argmax，可复现）
    - top_k > 0：先截断到 top-k
    - top_p < 1：nucleus 截断后重新归一化
    - repetition_penalty != 1.0 且 prev_ids 非空：对已生成的 token 施加
      HF 风格重复惩罚（logit>0 除以 penalty，否则乘以 penalty）
    """
    if repetition_penalty != 1.0 and prev_ids:
        logits = logits.clone()
        for t in set(prev_ids):
            v = logits[..., t]
            logits[..., t] = v / repetition_penalty if v > 0 else v * repetition_penalty

    if temperature <= 0.0 or (top_k == 1 and top_p >= 1.0):
        return torch.argmax(logits, dim=-1)

    scaled = logits / temperature
    if top_k > 0:
        k = min(int(top_k), scaled.shape[-1])
        topk_vals, _ = torch.topk(scaled, k, dim=-1)
        scaled = torch.where(scaled < topk_vals[..., -1:],
                             torch.full_like(scaled, float("-inf")), scaled)

    probs = torch.softmax(scaled, dim=-1, dtype=torch.float32)
    if top_p < 1.0:
        sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
        cum = torch.cumsum(sorted_probs, dim=-1)
        # HF 风格 nucleus：保留累计概率 <= top_p 的前缀（含跨越 token）
        remove = cum > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_probs = torch.where(remove, torch.zeros_like(sorted_probs), sorted_probs)
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
        chosen = torch.multinomial(sorted_probs, 1)
        return sorted_idx.gather(-1, chosen)[..., 0]

    return torch.multinomial(probs, 1)[..., 0]
