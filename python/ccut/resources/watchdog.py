"""ccut.resources.watchdog — 后台周期采样 + 内存压力信号（§3.7）。

- 周期 = ``monitor_interval``（默认 1s）；
- 采样项：CPU% / RSS（psutil）、IO 累计字节（IOLimiter.stats）；
- 内存压力：触发 :class:`MemoryPressureSignal` 回调（调度器反压）；
- 指标写 ``MetricsBus.resources_stats``（pull 模型，HTTP /metrics 端点抓）；
- IO 限速模式 ``throttle``：
  - ``auto``：超限自动反压 + 限制新请求入队；
  - ``warn``：仅记录告警（``logging.warning``），不动调度；
  - ``off``：只采样不打分（预算表已打印足够）。
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

import psutil

from ccut.resources.budget import ResourceBudget
from ccut.resources.limiter import IOLimiter, MemoryPressureSignal, get_io_limiter

__all__ = ["ResourceWatchdog", "WatchdogStats"]

_LOG = logging.getLogger("ccut.resources.watchdog")


@dataclass
class WatchdogStats:
    """最近一次采样值。"""

    cpu_pct: float = 0.0
    rss_gb: float = 0.0
    rss_budget_gb: float = 0.0
    pressure: bool = False
    io_consumed_bytes: int = 0
    io_budget_mbps: float = 0.0
    last_sample_ns: int = 0
    samples: int = 0
    pressure_engaged_count: int = 0

    def to_dict(self) -> dict:
        return {
            "cpu_pct": round(self.cpu_pct, 2),
            "rss_gb": round(self.rss_gb, 3),
            "rss_budget_gb": round(self.rss_budget_gb, 2),
            "pressure": self.pressure,
            "io_consumed_bytes": self.io_consumed_bytes,
            "io_budget_mbps": round(self.io_budget_mbps, 1),
            "last_sample_ns": self.last_sample_ns,
            "samples": self.samples,
            "pressure_engaged_count": self.pressure_engaged_count,
        }


class ResourceWatchdog:
    """后台周期采样线程（daemon，引擎关闭时自动退出）。"""

    def __init__(
        self,
        budget: ResourceBudget,
        metrics_bus=None,
        throttle_callback: Callable[[WatchdogStats], None] | None = None,
    ):
        self.budget = budget
        self.metrics_bus = metrics_bus
        self.pressure = MemoryPressureSignal(budget.mem_budget_gb)
        self.throttle_cb = throttle_callback
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.stats = WatchdogStats(
            rss_budget_gb=budget.mem_budget_gb,
            io_budget_mbps=budget.io_budget_mbps,
        )
        self._proc = psutil.Process(os.getpid())
        self._last_io = 0

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, name="resource-watchdog", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def snapshot(self) -> WatchdogStats:
        return self.stats

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._sample()
            except Exception:
                _LOG.exception("watchdog sample 失败")
            self._stop.wait(self.budget.monitor_interval)

    def _sample(self) -> None:
        cpu = self._proc.cpu_percent(interval=None)
        mem = self._proc.memory_info().rss / (1024**3)
        io = get_io_limiter()
        io_consumed = io.stats()["consumed_bytes"] if io is not None else 0
        pressure_changed = self.pressure.update(mem)
        s = self.stats
        s.cpu_pct = float(cpu)
        s.rss_gb = float(mem)
        s.pressure = self.pressure.engaged
        s.io_consumed_bytes = io_consumed
        s.last_sample_ns = time.monotonic_ns()
        s.samples += 1
        if pressure_changed and self.pressure.engaged:
            s.pressure_engaged_count += 1
        # 写 metrics
        if self.metrics_bus is not None:
            self.metrics_bus.resources_stats = s.to_dict()
        # throttle 模式
        if self.budget.throttle != "off" and (s.pressure or s.cpu_pct > self.budget.cpu_pct * 1.2):
            if self.budget.throttle == "warn":
                _LOG.warning(
                    "R11 超限: cpu=%.1f%% rss=%.2fGB 预算=%.2fGB pressure=%s",
                    s.cpu_pct,
                    s.rss_gb,
                    s.rss_budget_gb,
                    s.pressure,
                )
            elif self.throttle_cb is not None:
                self.throttle_cb(s)
        # 始终打印（auto 时也告警，配置已警告）
        if s.pressure:
            _LOG.info(
                "R11 内存压力触发: rss=%.2fGB / 预算 %.2fGB（已触发 %d 次）",
                s.rss_gb,
                s.rss_budget_gb,
                s.pressure_engaged_count,
            )


# os import 延迟（watchdog 单独 import 时避免顶层强依赖）
import os  # noqa: E402
