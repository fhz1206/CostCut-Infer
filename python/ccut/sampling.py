"""ccut.sampling — 采样（独立模块，引擎与 L1 后端共用）。

把 ``blocks/heads.temperature_topk`` 的**前向概率分布**与策略解耦成
「应用 → 归一化 → 采样」流水线（对齐 vLLM SamplingParams 语义）::

    logits → repetition/presence/frequency/length 惩罚
           → temperature → top_k → min_p → top_p → typical
           → 概率 → 采样（greedy / 随机 / beam 入口）
           → logprobs 输出（可选）

参数约束（§3.4-4）：
- ``top_k=0`` 禁用 top-k；
- ``top_p=1.0`` 禁用 nucleus；
- ``min_p=0`` 禁用 min-p；
- ``typical_p=1`` 禁用 typical；
- 全部约束可叠加（vLLM 语义：top-k → min-p → top-p → typical）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ccut.blocks.attn_gqa import apply_presence_frequency_penalties, repetition_penalty_logits
from ccut.blocks.heads import logprobs_from_logits, temperature_topk

__all__ = ["SamplingParams", "apply_sampling", "greedy_argmax", "logits_to_logprobs"]


@dataclass
class SamplingParams:
    """采样参数（与 config.sampling schema 字段对应）。"""

    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    min_p: float = 0.0
    typical_p: float = 1.0
    repetition_penalty: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    length_penalty: float = 1.0
    seed: int | None = None
    n: int = 1
    best_of: int | None = None  # 默认 n
    greedy: bool = False
    max_tokens: int | None = None
    stop: list[str] = field(default_factory=list)
    stop_token_ids: list[int] = field(default_factory=list)
    ignore_eos: bool = False
    logprobs: int | None = None
    prompt_logprobs: int | None = None

    def effective_top_k(self) -> int:
        return 0 if self.top_k < 0 else self.top_k


def apply_sampling(
    logits: np.ndarray,
    params: SamplingParams,
    history_counts: np.ndarray | None = None,
    generated: np.ndarray | None = None,
    length: int = 1,
) -> np.ndarray:
    """应用全部约束 → 概率分布（[batch, vocab]）。

    输入 logits [batch, vocab] float32；返回概率 [batch, vocab] float32（已归一化）。
    采样在 :func:`greedy_argmax` / 外部 rng 中完成（与分布解耦）。
    """
    x = logits.astype(np.float32, copy=False)
    if history_counts is None:
        history_counts = np.zeros((x.shape[0], x.shape[1]), dtype=np.float32)
    # 1) repetition penalty
    if params.repetition_penalty != 1.0 and generated is not None and generated.size:
        x = repetition_penalty_logits(x, generated, params.repetition_penalty)
    # 2) presence/frequency/length
    if params.presence_penalty or params.frequency_penalty or params.length_penalty != 1.0:
        x = apply_presence_frequency_penalties(
            x,
            history_counts,
            params.presence_penalty,
            params.frequency_penalty,
            params.length_penalty,
            length,
        )
    # 3) temperature + 约束 → softmax
    return temperature_topk(
        x,
        temperature=params.temperature if not params.greedy else 0.0,
        top_k=params.effective_top_k(),
        top_p=params.top_p,
        min_p=params.min_p,
        typical_p=params.typical_p,
    )


def greedy_argmax(probs: np.ndarray) -> np.ndarray:
    """greedy 采样（argmax of probs，temperature 任意）。"""
    return probs.argmax(axis=-1)


def logits_to_logprobs(
    logits: np.ndarray,
    top_k: int | None = None,
) -> np.ndarray:
    """logits → logprobs（[rows, vocab]，未归一化）。"""
    return logprobs_from_logits(logits.astype(np.float32, copy=False), top_k=top_k)
