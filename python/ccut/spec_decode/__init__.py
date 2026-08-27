"""ccut.spec_decode — P5 投机解码（MTP + Ngram 备选 + 接受率指标）。

设计（§7 P5）：
- **propose**：MTP 模型生成 γ 个 draft（Ornith γ=1）→ draft 序列；
- **verify**：主模型一次性前向 [x, d1..dγ] → γ+1 位置 logits → rejection sampling
  （vLLM speculative 语义）；
- **Ngram 备选**（`--enable-ngram`）：无 MTP 时按 prompt N-gram 匹配提议；
- **接受率指标**（`accept_rate = accepted / total_drafts`）→ metrics.spec。

与 `blocks/mtp.py` 的关系：``speculative_verify`` 是数值核心；本模块
负责：
- 决策入口（``propose`` 选 MTP/Ngram）；
- 请求级状态（draft_history、accepted_so_far）；
- 指标汇总（每个 step 写 metrics.spec）。
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

from ccut.blocks.mtp import MTPLayer, mtp_draft, speculative_verify

__all__ = [
    "SpecDecodeStats",
    "NgramProposer",
    "MTPProposer",
    "SpecDecoder",
    "resolve_proposer",
]


@dataclass
class SpecDecodeStats:
    """投机解码指标。"""

    drafts_proposed: int = 0
    drafts_accepted: int = 0
    bonus_samples: int = 0
    accepted_tokens: int = 0  # 总接受 token 数（= 接受 draft + bonus）
    last_run_ns: int = 0
    avg_run_ms: float = 0.0

    def accept_rate(self) -> float:
        return self.drafts_accepted / self.drafts_proposed if self.drafts_proposed > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "drafts_proposed": self.drafts_proposed,
            "drafts_accepted": self.drafts_accepted,
            "accept_rate": round(self.accept_rate(), 4),
            "bonus_samples": self.bonus_samples,
            "accepted_tokens": self.accepted_tokens,
            "avg_run_ms": round(self.avg_run_ms, 3),
        }


class NgramProposer:
    """N-gram 备选 proposer（无 MTP 时的兜底）。

    在 prompt N-gram 索引里查「最近 N-1 个 token 的下一 token」——找到则
    作为 draft（多轮不深，最多 γ 个）。
    """

    def __init__(self, n: int = 8, max_draft: int = 1):
        self.n = max(2, n)
        self.max_draft = max(0, max_draft)
        # index: (n-1-tuple) -> list[token_id]
        self._index: dict[tuple, list[int]] = {}

    def build(self, prompt: Sequence[int]) -> None:
        if len(prompt) < self.n:
            return
        for i in range(len(prompt) - self.n + 1):
            key = tuple(prompt[i : i + self.n - 1])
            nxt = int(prompt[i + self.n - 1])
            self._index.setdefault(key, []).append(nxt)

    def propose(self, recent: Sequence[int]) -> list[int]:
        if len(recent) < self.n - 1 or self.max_draft == 0:
            return []
        key = tuple(recent[-(self.n - 1) :])
        cands = self._index.get(key)
        if not cands:
            return []
        return [int(cands[0])][: self.max_draft]


class MTPProposer:
    """MTP 包装（blocks/mtp.mtp_draft 的薄包装，统计指标）。"""

    def __init__(self, model_layers, mtp_layer: MTPLayer, embed_fn, lm_head, max_draft: int = 1, greedy: bool = True):
        self.model_layers = model_layers
        self.mtp_layer = mtp_layer
        self.embed_fn = embed_fn
        self.lm_head = lm_head
        self.max_draft = max(0, max_draft)
        self.greedy = greedy

    def propose(self, hidden_prev: np.ndarray, token_ids: np.ndarray) -> list[int]:
        if self.max_draft == 0:
            return []
        return mtp_draft(
            self.model_layers,
            self.mtp_layer,
            self.embed_fn,
            self.lm_head,
            hidden_prev,
            token_ids,
            n_draft=self.max_draft,
            greedy=self.greedy,
        )


def resolve_proposer(spec_cfg: dict, mtp_layer=None, model_layers=None, embed_fn=None, lm_head=None, prompt: Sequence[int] | None = None):
    """``Config.spec_decode`` → proposer 实例。

    - ``enable_mtp=true`` 且 mtp_layer 已构建 → MTPProposer；
    - ``enable_ngram=true`` → NgramProposer（兜底）；
    - 都关 → 返回 None（不投机）。
    """
    if spec_cfg.get("enable_mtp", True) and mtp_layer is not None:
        return MTPProposer(
            model_layers, mtp_layer, embed_fn, lm_head,
            max_draft=int(spec_cfg.get("mtp_draft_tokens", 1)),
        )
    if spec_cfg.get("enable_ngram", False) and prompt is not None:
        ng = NgramProposer(n=int(spec_cfg.get("ngram_window", 8)), max_draft=1)
        ng.build(prompt)
        return ng
    return None


class SpecDecoder:
    """投机解码调度器（每步：propose → verify → 写指标）。"""

    def __init__(self, proposer=None, seed: int | None = None):
        self.proposer = proposer
        self.stats = SpecDecodeStats()
        self._seed = seed

    def step(
        self,
        recent_tokens: list[int],
        main_logits: np.ndarray,
        last_hidden: np.ndarray | None = None,
        ngram_recent: list[int] | None = None,
    ) -> tuple[list[int], int | None]:
        """单步投机：propose → verify → 返回 (accepted_token_ids, bonus_or_None)。

        - ``main_logits``: 主模型对 [x, d1..dγ] 的逐位置 logits（[γ+1, vocab]），
          未传时仅用 proposer 而无验证（接受率按 1.0 估算——**错误**但保留接口）；
        - ``last_hidden``：MTP 的 hidden 前一个 step（仅 MTP 用）；
        - ``ngram_recent``：N-gram 用的近 token 序列（MTP 可不传）。
        """
        t0 = time.monotonic_ns()
        # propose
        if isinstance(self.proposer, MTPProposer) and last_hidden is not None:
            drafts = self.proposer.propose(
                last_hidden.astype(np.float32, copy=False),
                np.asarray(recent_tokens, dtype=np.int64),
            )
        elif isinstance(self.proposer, NgramProposer):
            drafts = self.proposer.propose(ngram_recent or recent_tokens)
        else:
            drafts = []
        # verify
        bonus: int | None = None
        accepted = []
        if drafts:
            try:
                n_acc, bonus = speculative_verify(
                    drafts, main_logits, draft_probs=None, temperature=1.0, seed=self._seed
                )
                accepted = drafts[:n_acc]
            except Exception:
                accepted = []
                bonus = None
        # 写指标
        dt_ms = (time.monotonic_ns() - t0) / 1e6
        n = len(drafts)
        self.stats.drafts_proposed += n
        self.stats.drafts_accepted += len(accepted)
        self.stats.accepted_tokens += len(accepted) + (1 if bonus is not None else 0)
        self.stats.bonus_samples += 1 if bonus is not None else 0
        self.stats.last_run_ns = time.monotonic_ns()
        # 移动平均
        a, n_s = self.stats.avg_run_ms, 0
        self.stats.avg_run_ms = a * (n_s / (n_s + 1)) + dt_ms * (1 / (n_s + 1))
        return accepted, bonus

    def metrics_dict(self) -> dict:
        return self.stats.to_dict()
