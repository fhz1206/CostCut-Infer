"""ccut.resources.budget — 资源预算表（启动期一次性解析，输出预算明细）。

「全局 vs 分资源」覆盖优先级（§3.7）：
- ``resource_pct=50`` 为默认（如未设 → 50%）
- ``resource_cpu_pct=auto``（默认跟随全局）| 数字（如 30）覆盖
- ``resource_mem_pct=auto`` 同上
- ``resource_io_pct=auto`` 同上

输出预算表（启动报告 + metrics）::

    R11 资源预算（按全局=50%, 覆盖=无）:
      CPU:  4/8  线程（50%, 实际 threads 数）
      内存: 5.85 GB（50%, 系统总 11.7GB）
      IO:   450 MB/s（50%, 盘速 900MB/s）
      监控: 1.0s/次
      模式: auto（超限自限）

异常：``resource_pct`` 越界（<1/>100）由 schema 校验；分资源覆盖在 schema 后这里
只做"auto=继承"→ 实际值解析。
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

import psutil

__all__ = [
    "ResourceBudget",
    "ResourcePctResolver",
    "build_resource_budget",
    "render_budget_table",
]


_PCT_PAT = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*%?\s*$", re.IGNORECASE)


class ResourcePctResolver:
    """资源百分比解析（大小写不敏感 + auto 继承）。"""

    @staticmethod
    def resolve(global_pct: int, override: str | int | None) -> float:
        if override is None or (isinstance(override, str) and override.strip().casefold() == "auto"):
            return max(1.0, min(100.0, float(global_pct)))
        if isinstance(override, (int, float)):
            return max(1.0, min(100.0, float(override)))
        s = str(override).strip()
        m = _PCT_PAT.match(s)
        if not m:
            raise ValueError(f"无法解析资源百分比: {override!r}（auto / 整数 / '50%'）")
        v = float(m.group(1))
        return max(1.0, min(100.0, v))


@dataclass
class ResourceBudget:
    """资源预算明细（启动期 + 监控期共用）。"""

    cpu_pct: float
    mem_pct: float
    io_pct: float
    cpu_threads: int
    cpu_logical: int
    mem_total_gb: float
    mem_budget_gb: float
    io_disk_mbps: float
    io_budget_mbps: float
    monitor_interval: float
    throttle: str

    def to_dict(self) -> dict:
        return {
            "cpu_pct": round(self.cpu_pct, 1),
            "mem_pct": round(self.mem_pct, 1),
            "io_pct": round(self.io_pct, 1),
            "cpu_threads": self.cpu_threads,
            "cpu_logical": self.cpu_logical,
            "mem_total_gb": round(self.mem_total_gb, 2),
            "mem_budget_gb": round(self.mem_budget_gb, 2),
            "io_disk_mbps": round(self.io_disk_mbps, 1),
            "io_budget_mbps": round(self.io_budget_mbps, 1),
            "monitor_interval": self.monitor_interval,
            "throttle": self.throttle,
        }


def build_resource_budget(
    resources_cfg: dict[str, Any],
    disk_mbps: float,
    cpu_logical: int,
    mem_total_gb: float,
) -> ResourceBudget:
    """``Config.resources`` + 平台探测值 → 预算表。"""
    global_pct = int(resources_cfg.get("resource_pct", 50))
    cpu_pct = ResourcePctResolver.resolve(global_pct, resources_cfg.get("resource_cpu_pct", "auto"))
    mem_pct = ResourcePctResolver.resolve(global_pct, resources_cfg.get("resource_mem_pct", "auto"))
    io_pct = ResourcePctResolver.resolve(global_pct, resources_cfg.get("resource_io_pct", "auto"))
    cpu_threads = max(1, int(round(cpu_logical * cpu_pct / 100.0)))
    mem_budget = mem_total_gb * mem_pct / 100.0
    io_budget = disk_mbps * io_pct / 100.0
    return ResourceBudget(
        cpu_pct=cpu_pct,
        mem_pct=mem_pct,
        io_pct=io_pct,
        cpu_threads=cpu_threads,
        cpu_logical=cpu_logical,
        mem_total_gb=mem_total_gb,
        mem_budget_gb=mem_budget,
        io_disk_mbps=disk_mbps,
        io_budget_mbps=io_budget,
        monitor_interval=float(resources_cfg.get("resource_monitor_interval", 1.0)),
        throttle=str(resources_cfg.get("resource_throttle", "auto")),
    )


def render_budget_table(b: ResourceBudget, global_pct: int) -> list[str]:
    """预算表（启动报告 + metrics 文本行）。"""
    return [
        f"R11 资源预算（按全局={global_pct}%, 实际: CPU={b.cpu_pct:.0f}% 内存={b.mem_pct:.0f}% IO={b.io_pct:.0f}%）",
        f"  CPU:  {b.cpu_threads}/{b.cpu_logical}  线程",
        f"  内存: {b.mem_budget_gb:.2f} GB（系统 {b.mem_total_gb:.1f}GB）",
        f"  IO:   {b.io_budget_mbps:.0f} MB/s（盘速 {b.io_disk_mbps:.0f}MB/s）",
        f"  监控: {b.monitor_interval:.1f}s/次 | 模式: {b.throttle}",
    ]
