"""tests.test_resources — R11 资源限制（预算表 + IOLimiter + 内存压力信号）。"""

from __future__ import annotations

import time

import psutil
import pytest

from ccut.platforms import detect_cpu, detect_memory, benchmark_disk
from ccut.resources.budget import (
    ResourcePctResolver,
    build_resource_budget,
    render_budget_table,
)
from ccut.resources.limiter import (
    IOLimiter,
    MemoryPressureSignal,
    apply_cpu_limits,
    apply_resource_limits,
)
from ccut.resources.watchdog import ResourceWatchdog


def test_resource_pct_resolver_auto():
    assert ResourcePctResolver.resolve(50, "auto") == 50
    assert ResourcePctResolver.resolve(50, None) == 50
    assert ResourcePctResolver.resolve(50, "30%") == 30
    assert ResourcePctResolver.resolve(50, 30) == 30
    with pytest.raises(ValueError):
        ResourcePctResolver.resolve(50, "bogus")


@pytest.mark.slow
def test_build_resource_budget():
    cpu = detect_cpu()
    mem = detect_memory()
    disk = benchmark_disk(sample_mb=2)
    b = build_resource_budget(
        {"resource_pct": 50, "resource_throttle": "auto", "resource_monitor_interval": 0.5},
        disk.sequential1m_mbps,
        cpu.logical_cores,
        mem.total_gb,
    )
    assert b.cpu_threads == max(1, int(round(cpu.logical_cores * 0.5)))
    assert b.mem_budget_gb == pytest.approx(mem.total_gb * 0.5, rel=0.1)
    assert b.io_budget_mbps == pytest.approx(disk.sequential1m_mbps * 0.5, rel=0.2)
    lines = render_budget_table(b, 50)
    assert any("R11" in ln for ln in lines)


def test_cpu_limits_apply():
    applied = apply_cpu_limits(2)
    assert applied["env"]["OMP_NUM_THREADS"] == "2"
    assert applied["torch_num_threads"] == 2


@pytest.mark.slow
def test_iolimiter_throttles_bytes():
    """IOLimiter 在低速率下确实限速：2MB @ 10MB/s ≈ 0.2s。"""
    lim = IOLimiter(mbps=10.0, burst_bytes=0)  # burst=0 → 必等令牌
    t0 = time.time()
    lim.acquire(2_000_000)  # 2 MB
    dt = time.time() - t0
    # 容差放宽：Windows 计时精度+首次 0 令牌时起点即等
    assert dt >= 0.15, f"限速应等 ≥0.15s，实测 {dt:.3f}s"
    assert lim.stats()["consumed_bytes"] >= 2_000_000


def test_iolimiter_zero_unlimited():
    lim = IOLimiter(mbps=0)
    t0 = time.time()
    lim.acquire(10_000_000)
    assert time.time() - t0 < 0.05


def test_memory_pressure_hysteresis():
    sig = MemoryPressureSignal(mem_budget_gb=1.0, enter_ratio=0.9, exit_ratio=0.8)
    assert not sig.engaged
    sig.update(0.95)  # 触发
    assert sig.engaged
    sig.update(0.85)  # 仍触发（< exit_ratio 才退）
    assert sig.engaged
    sig.update(0.7)  # 退出
    assert not sig.engaged


@pytest.mark.slow
def test_watchdog_runs_and_collects_samples():
    cpu = detect_cpu()
    mem = detect_memory()
    disk = benchmark_disk(sample_mb=2)
    b = build_resource_budget(
        {"resource_pct": 50, "resource_throttle": "off", "resource_monitor_interval": 0.1},
        disk.sequential1m_mbps,
        cpu.logical_cores,
        mem.total_gb,
    )
    wd = ResourceWatchdog(b)
    wd.start()
    time.sleep(0.35)
    wd.stop()
    s = wd.snapshot()
    assert s.samples >= 1
    assert s.rss_gb > 0
