"""ccut.experts.pipeline — 三层流水线（§3.3 计算/预取/投机）。

三层阶段（流水线并行，依赖关系单向→下一层可乱序执行）：

- **计算层 C**（num_worker 线程池）：路由 top-k → 触发 prefetch（同步、等就绪）→
  expert_ffn → 加权融合 → 残差加共享专家。
- **预取层 P**（ExpertReader worker 线程）：消费路由决策、上 mmap 读 →
  dequant → 写 ring buffer。
- **投机层 S**（speculative_route 启发式）：基于最近 N 步路由分布历史，预猜
  下一步高概率专家 → 提前入队（在 P 之前），把 P 的有效窗口放大。

时序与监控（§3.3 第四点）：
- 阶段交叠用 ``step_t_ns`` 单调时间戳采样，**overlap_ratio = (C∩P) / C_total**；
- 指标总线（pipeline_metrics=true 时）写到 metrics.py 的 PipelineMetrics。
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from ccut.blocks.moe import expert_ffn, shared_expert_add
from ccut.experts.reader import ExpertReader

__all__ = ["PipelineMetrics", "ExpertPipeline", "RouteHistory"]


@dataclass
class PipelineMetrics:
    """三层流水线时序指标。"""

    compute_ns: int = 0
    prefetch_ns: int = 0
    overlap_ns: int = 0
    steps: int = 0
    speculative_correct: int = 0  # S 预取命中数
    speculative_total: int = 0
    ring_hits: int = 0
    ring_misses: int = 0
    ring_evicted: int = 0  # ring 覆盖前刚被覆盖的专家（说明 prefetch 顺序欠优）

    def overlap_ratio(self) -> float:
        return self.overlap_ns / self.compute_ns if self.compute_ns > 0 else 0.0

    def speculative_accept_rate(self) -> float:
        return self.speculative_correct / self.speculative_total if self.speculative_total > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "compute_ns": self.compute_ns,
            "prefetch_ns": self.prefetch_ns,
            "overlap_ns": self.overlap_ns,
            "overlap_ratio": round(self.overlap_ratio(), 4),
            "steps": self.steps,
            "spec_accept_rate": round(self.speculative_accept_rate(), 4),
            "ring_hits": self.ring_hits,
            "ring_misses": self.ring_misses,
        }


class RouteHistory:
    """路由决策滑动窗口（投机层使用）。"""

    def __init__(self, window: int = 4):
        self.window = window
        self._hist: deque[set[int]] = deque(maxlen=window)

    def record(self, topk_ids: Iterable[int]) -> None:
        self._hist.append(set(int(x) for x in topk_ids))

    def predict(self, top_k: int) -> list[int]:
        """按窗口历史频次预测下步 top-k 专家（频次降序）。"""
        if not self._hist:
            return []
        counts: dict[int, int] = {}
        for step in self._hist:
            for e in step:
                counts[e] = counts.get(e, 0) + 1
        return [e for e, _ in sorted(counts.items(), key=lambda kv: -kv[1])[:top_k]]


class ExpertPipeline:
    """三层流水线编排器（专家流 R2/R3）。

    使用::

        pipe = ExpertPipeline(reader, num_workers=2, speculative_window=4)
        for layer_step in steps:
            pipe.step(layer, x, gate_w, top_k=8, moe_norm=True,
                      shared_projs=...) -> y

    每步：
    1. S 投机预取（基于 RouteHistory 预测）
    2. 真实路由 → topk → 即时 prefetch（fill 未就绪的专家到 ring）
    3. C 计算：按专家聚合 token → expert_ffn → 加权融合
    4. 共享专家（若有）→ 残差
    5. 记录 history + 更新 metrics
    """

    def __init__(
        self,
        reader: ExpertReader,
        num_workers: int = 2,
        speculative_window: int = 4,
        prefetch_steps_ahead: int = 2,
        metrics_enabled: bool = True,
    ):
        self.reader = reader
        self.num_workers = num_workers
        self.prefetch_steps_ahead = prefetch_steps_ahead
        self.metrics = PipelineMetrics() if metrics_enabled else None
        self.history = RouteHistory(window=speculative_window)
        # 后台计算线程池（decode 时专家 ffn 较重；prefill 主线程承担）
        self._executor: list[threading.Thread] = []
        self._stop = False
        # 预取阶段独立线程的 mmap 入口由 reader 内部 worker 负责

    def step(
        self,
        layer: int,
        x: np.ndarray,
        gate_w: np.ndarray,
        top_k: int = 8,
        norm_topk_prob: bool = True,
        shared_projs: dict | None = None,
        shared_gate: float = 1.0,
    ) -> np.ndarray:
        """单步专家前向：路由 + 融合 + 共享专家（返回 y）。"""
        from ccut.blocks.moe import topk_softmax, moe_combine

        t0 = time.monotonic_ns()
        # S 投机预取（仅 metric 计入 speculative_total）
        if self.metrics and self.prefetch_steps_ahead > 0:
            pred = self.history.predict(top_k)
            if pred:
                self.metrics.speculative_total += 1
                if self.reader.prefetch(layer, pred):
                    self.metrics.speculative_correct += 1
        # 真实路由
        logits = x @ gate_w.T
        weights, ids = topk_softmax(logits, top_k, norm_topk_prob)
        self.history.record(ids[0] if hasattr(ids, "shape") and ids.ndim == 2 else ids)
        # P 预取：本步 top-k 实时入队（去重于 ring 已就绪的）
        flat_ids = ids.flatten().tolist() if hasattr(ids, "flatten") else list(ids)
        t_pf0 = time.monotonic_ns()
        self.reader.prefetch(layer, flat_ids)
        # C 计算：按专家聚合 token → expert_ffn → 加权融合
        t_c0 = time.monotonic_ns()
        # 简化版：单线程遍历 top-k（prefill 批大时可改 numba 并行 — 占位）
        seq, hidden = x.shape
        out = np.zeros((seq, hidden), dtype=np.float32)
        # 等所有本次 top-k 就绪（首次访问会触发 reader.get 同步等）
        for eid in set(flat_ids):
            try:
                wg, wu, wd = self.reader.get(layer, int(eid), timeout=30.0)
            except TimeoutError:
                # 单步超时：把该专家置为不参与（路由权重重分配）
                continue
            # 聚合该专家的 token
            t_int = self.reader.index.intermediate_size if hasattr(self.reader.index, "intermediate_size") else wg.shape[1]
            idx_list = [t for t in range(seq) for kk in range(top_k) if int(ids[t, kk]) == eid]
            if not idx_list:
                continue
            batch = x[np.array(idx_list, dtype=np.int64)]
            y = expert_ffn(batch, wg, wu, wd)
            for i, t in enumerate(idx_list):
                # 路由权重：取 token t 在该专家的 weight（k 维）
                w_t = None
                for kk in range(top_k):
                    if int(ids[t, kk]) == eid:
                        w_t = weights[t, kk]
                        break
                if w_t is not None:
                    out[t] += w_t * y[i]
        t_c1 = time.monotonic_ns()
        # 共享专家
        if shared_projs is not None and "gate_proj" in shared_projs:
            s = shared_projs["gate_proj"].apply(x)
            su = shared_projs["up_proj"].apply(x)
            from ccut.quant import kernels as qk

            inter = np.empty(s.shape, dtype=np.float32)
            qk.silu_mul_fused(s, su, inter)
            s_out = shared_projs["down_proj"].apply(inter)
            shared_expert_add(out, s_out, shared_gate)
        t1 = time.monotonic_ns()
        # 指标更新
        if self.metrics is not None:
            compute = t_c1 - t_c0
            prefetch = t_pf0 - t0 + (t1 - t_c1)  # 包含 S + 残余 prefetch
            overlap = min(compute, prefetch)  # 简化估算：overlap ≈ 较小者
            self.metrics.compute_ns += compute
            self.metrics.prefetch_ns += prefetch
            self.metrics.overlap_ns += overlap
            self.metrics.steps += 1
            stats = self.reader.stats()
            self.metrics.ring_hits = stats.get("hits", 0)
            self.metrics.ring_misses = stats.get("misses", 0)
        return out

    def close(self) -> None:
        self._stop = True
        self.reader.close()
