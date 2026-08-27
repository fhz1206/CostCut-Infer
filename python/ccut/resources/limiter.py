"""ccut.resources.limiter — 三类门控（CPU/内存/IO）。

设计（§3.7）：
- **CPU 门控**（启动期一次性）：写环境变量 ``OMP_NUM_THREADS`` / ``MKL_NUM_THREADS``
  → 调 ``torch.set_num_threads`` / ``numba.set_num_threads`` → ``windows_io.set_thread_priority_below_normal``。
  注意：必须在创建 numba 编译产物**之前**设置线程数（否则 numba 缓存的是
  启动时线程数）。**先调本函数再 import numba**。
- **内存软上限**：psutil 周期检测（watchdog 调 ``read_rss_gb``）+ 触发信号
  ``MemoryPressureSignal``（不杀进程——让调度器反压、evict decoding）。
- **IO 令牌桶**：``IOLimiter`` 全局单例；``acquire(n_bytes)`` 阻塞等令牌释放。
  限额 = ``io_budget_mbps * 1e6`` bytes/s；启动期由 :func:`apply_resource_limits` 注入。
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from typing import Callable

__all__ = [
    "apply_cpu_limits",
    "apply_memory_limits",
    "IOLimiter",
    "MemoryPressureSignal",
    "apply_resource_limits",
]


def apply_cpu_limits(cpu_threads: int) -> dict:
    """写环境变量 + 调线程 API（启动期一次性，需在 numba import 前调）。"""
    os.environ["OMP_NUM_THREADS"] = str(cpu_threads)
    os.environ["MKL_NUM_THREADS"] = str(cpu_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(cpu_threads)
    applied: dict = {"env": {k: os.environ[k] for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}}
    # torch
    try:
        import torch

        torch.set_num_threads(cpu_threads)
        applied["torch_num_threads"] = cpu_threads
    except Exception:
        pass
    # numba（模块已 import 也可调 set_num_threads）
    try:
        from numba import set_num_threads

        set_num_threads(cpu_threads)
        applied["numba_num_threads"] = cpu_threads
    except Exception:
        pass
    return applied


def apply_memory_limits(mem_budget_gb: float) -> dict:
    """设 RSS 软上限（Windows 通过 ctypes JobObject 限制工作集——本机无编译环境，
    退化为「soft hint」+ 监控期 watchdog 周期检查 + 触发反压信号）。

    POSIX（Linux/macOS）下 ``resource.RLIMIT_AS`` = 虚拟地址空间上限——
    设为 ``mem_budget_gb * 1e9``，超限 ``MemoryError``。本机是 Windows，
    显式 no-op（保留 API 形状，监控 + 反压替代硬限）。
    """
    applied: dict = {"mem_budget_gb": mem_budget_gb}
    try:
        import resource

        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        new = int(mem_budget_gb * (1024**3))
        if new < soft:
            resource.setrlimit(resource.RLIMIT_AS, (new, hard))
            applied["posix_rlimit_as"] = (new, hard)
    except (ImportError, ValueError, OSError):
        applied["posix_rlimit_as"] = None  # Windows 退化
    return applied


class MemoryPressureSignal:
    """内存压力信号（watchdog 检测到 RSS 超限 → 触发 → 调度器反压）。

    用法::

        sig = MemoryPressureSignal()
        if sig.engaged:
            ...  # 调度器减少并发 / 优先 evict
        sig.on_pressure(lambda: ...)

    阈值（启动期）：``mem_budget_gb`` 的 90%（触发）+ 80%（恢复）——迟滞防抖。
    """

    def __init__(self, mem_budget_gb: float, enter_ratio: float = 0.9, exit_ratio: float = 0.8):
        self.budget = mem_budget_gb
        self.enter_ratio = enter_ratio
        self.exit_ratio = exit_ratio
        self._engaged = False
        self._cbs: list[Callable[[], None]] = []
        self._lock = threading.Lock()

    @property
    def engaged(self) -> bool:
        with self._lock:
            return self._engaged

    def on_pressure(self, cb: Callable[[], None]) -> None:
        with self._lock:
            self._cbs.append(cb)

    def update(self, rss_gb: float) -> bool:
        """采样 RSS；返回是否发生状态变化。"""
        changed = False
        with self._lock:
            if not self._engaged and rss_gb >= self.budget * self.enter_ratio:
                self._engaged = True
                changed = True
            elif self._engaged and rss_gb <= self.budget * self.exit_ratio:
                self._engaged = False
                changed = True
        if changed and self._engaged:
            for cb in self._cbs:
                try:
                    cb()
                except Exception:
                    pass
        return changed


class IOLimiter:
    """IO 令牌桶（字节/秒）——全局单例，``acquire`` 阻塞等。

    算法（漏桶）：``burst_bytes`` 启动时灌满；``consume()`` 持续按
    ``bytes_per_sec`` 补充。``acquire(n)`` 阻塞直到桶有 ≥ n 字节。
    """

    def __init__(self, mbps: float, burst_bytes: int | None = None):
        self.bytes_per_sec = mbps * 1e6
        self.capacity = float(burst_bytes if burst_bytes is not None else self.bytes_per_sec)
        self._tokens = self.capacity
        self._last_ns = time.monotonic_ns()
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._consumed_bytes = 0
        self._acquired_calls = 0

    def _refill_locked(self) -> None:
        now = time.monotonic_ns()
        dt = (now - self._last_ns) / 1e9
        if dt > 0:
            self._tokens = min(self.capacity, self._tokens + dt * self.bytes_per_sec)
            self._last_ns = now

    def acquire(self, n: int) -> None:
        """阻塞等令牌（令牌不足时按「恰好够 1 次」唤醒）。"""
        if n <= 0:
            return
        if self.bytes_per_sec <= 0:
            return  # 0 速率 = 不限速
        with self._cv:
            self._refill_locked()
            while self._tokens < n:
                # 算还需等多久
                deficit = n - self._tokens
                wait_s = deficit / self.bytes_per_sec
                self._cv.wait(timeout=min(wait_s, 0.1))
                self._refill_locked()
            self._tokens -= n
            self._consumed_bytes += n
            self._acquired_calls += 1

    def stats(self) -> dict:
        with self._lock:
            return {
                "io_budget_mbps": self.bytes_per_sec / 1e6,
                "capacity_bytes": self.capacity,
                "consumed_bytes": self._consumed_bytes,
                "acquired_calls": self._acquired_calls,
            }


# 全局 IO 限速器（启动期注入；测试用 :func:`get_io_limiter`）
_IO_LIMITER: IOLimiter | None = None


def get_io_limiter() -> IOLimiter | None:
    return _IO_LIMITER


def apply_resource_limits(budget) -> dict:
    """``ResourceBudget`` → 全部三类门控初始化。返回 applied 摘要（metrics.resources）。"""
    out = {
        "cpu": apply_cpu_limits(budget.cpu_threads),
        "memory": apply_memory_limits(budget.mem_budget_gb),
        "io_budget_mbps": budget.io_budget_mbps,
    }
    global _IO_LIMITER
    _IO_LIMITER = IOLimiter(mbps=budget.io_budget_mbps)
    return out
