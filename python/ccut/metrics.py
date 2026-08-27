"""ccut.metrics — 指标总线（引擎 + 三机制聚合，§3.4-2）。

设计：
- **三段采集**：engine metrics（请求/调度）、kv coordinator metrics（L1/L2
  命中）、experts pipeline metrics（三层 overlap_ratio / 投机接受率）、
  weight manager metrics（bytes_read / evictions）；
- **聚合** :func:`snapshot` 把全部 metrics 拍平到单 dict，HTTP /metrics 端点用；
- **文件 dump**（可选）：``--metrics-dump-dir`` 写 ``step_NNNN.json``。

不做：
- 不做 in-memory 时间序列（计划显式「不引入时序 DB」）；
- 不做 metrics push（pull 模型——上层服务按需抓）。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["MetricsBus", "snapshot"]


@dataclass
class MetricsBus:
    """指标聚合点（持有四个子系统的 stats 句柄）。"""

    engine_metrics = None  # engine.EngineMetrics
    kv_stats: dict = field(default_factory=dict)
    pipeline_stats: dict = field(default_factory=dict)
    weight_stats: dict = field(default_factory=dict)
    resources_stats: dict = field(default_factory=dict)
    spec_stats: dict = field(default_factory=dict)  # P5 投机
    started_ns: int = 0
    dump_dir: Path | None = None
    dump_every: int = 0  # 0=不写
    _step: int = 0

    def attach_engine(self, m) -> None:
        self.engine_metrics = m
        if not self.started_ns:
            self.started_ns = time.monotonic_ns()

    def tick(self) -> None:
        self._step += 1
        if self.dump_dir and self.dump_every and self._step % self.dump_every == 0:
            self._dump()

    def _dump(self) -> None:
        if self.dump_dir is None:
            return
        try:
            self.dump_dir.mkdir(parents=True, exist_ok=True)
            (self.dump_dir / f"step_{self._step:08d}.json").write_text(
                json.dumps(snapshot(self), default=str, ensure_ascii=False, indent=1),
                encoding="utf-8",
            )
        except OSError:
            pass

    def set_kv(self, fn) -> None:
        """延迟挂载：``fn() -> dict``（coordinator.stats()）。"""
        self._kv_fn = fn

    def set_pipeline(self, fn) -> None:
        self._pipe_fn = fn

    def set_weight(self, fn) -> None:
        self._w_fn = fn

    def set_resources(self, fn) -> None:
        self._r_fn = fn

    def set_spec(self, fn) -> None:
        self._s_fn = fn

    def collect(self) -> dict:
        if hasattr(self, "_kv_fn"):
            self.kv_stats = self._kv_fn() or self.kv_stats
        if hasattr(self, "_pipe_fn"):
            self.pipeline_stats = self._pipe_fn() or self.pipeline_stats
        if hasattr(self, "_w_fn"):
            self.weight_stats = self._w_fn() or self.weight_stats
        if hasattr(self, "_r_fn"):
            self.resources_stats = self._r_fn() or self.resources_stats
        if hasattr(self, "_s_fn"):
            self.spec_stats = self._s_fn() or self.spec_stats
        return snapshot(self)


def snapshot(bus: MetricsBus) -> dict:
    """聚合快照（统一 dict，可直接 JSON 输出）。"""
    out = {
        "ts": time.time(),
        "step": bus._step,
        "uptime_s": (time.monotonic_ns() - bus.started_ns) / 1e9 if bus.started_ns else 0,
    }
    if bus.engine_metrics is not None:
        out["engine"] = bus.engine_metrics.to_dict() if hasattr(bus.engine_metrics, "to_dict") else bus.engine_metrics
    if bus.kv_stats:
        out["kv"] = bus.kv_stats
    if bus.pipeline_stats:
        out["pipeline"] = bus.pipeline_stats
    if bus.weight_stats:
        out["weights"] = bus.weight_stats
    if bus.resources_stats:
        out["resources"] = bus.resources_stats
    if bus.spec_stats:
        out["spec"] = bus.spec_stats
    return out
