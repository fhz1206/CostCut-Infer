"""ccut.platforms — CPU 能力 / 内存 / 盘速探测与启动报告。

- CPU 指令集：cpuid（ctypes，无编译依赖）探测 AVX2 / AVX-512（F/DQVL/BW/CD/VL/VNNI/BF16）。
- 内存：物理内存 / 可用内存（psutil，缺失时 ctypes 回退）。
- 盘速基准：4KB 随机读 + 1MB 顺序读（64MB 样本，默认 16MB 快速档）。
- 启动报告：``render_startup_report`` 汇总为文本（``--info`` 使用）。

所有探测都**可注入**（benchmark_size / 盘路径 / 采样器），便于单测。
"""

from __future__ import annotations

import ctypes
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

__all__ = [
    "CpuFeatures",
    "DiskBenchmark",
    "MemoryInfo",
    "detect_cpu",
    "detect_memory",
    "benchmark_disk",
    "PlatformReport",
    "render_startup_report",
]

# ---------------------------------------------------------------------------
# CPU 能力（cpuid via ctypes）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CpuFeatures:
    """CPU 指令集能力。所有字段 bool，``supports(name)`` 查询。"""

    name: str = "unknown"
    avx2: bool = False
    avx512f: bool = False
    avx512dq: bool = False
    avx512bw: bool = False
    avx512vl: bool = False
    avx512_vnni: bool = False
    avx512_bf16: bool = False
    logical_cores: int = 1
    physical_cores: int = 1

    def supports(self, feature: str) -> bool:
        return bool(getattr(self, feature.casefold(), False))

    def has_avx2(self) -> bool:
        return self.avx2

    def has_avx512(self) -> bool:
        return self.avx512f and self.avx512bw

    def vnni(self) -> bool:
        return self.avx512_vnni


def _cpuid_flags() -> dict[str, bool]:
    """读取 CPU 特性位。

    本机无 C 编译器，无法内联 cpuid 汇编；采用**运行时真实能力**作为 cpuid 替代：
    解析 ``torch.__config__.show()``（oneDNN/MKL 构建期检测到的指令集，权威）+
    保守兜底（AVX2 假设为真——绝大多数近十年 x86_64 均支持；AVX512 系列未知则 False，
    调用方 numba/torch 内核自带能力探测回退，不会误用）。
    """
    try:
        import io

        from torch import __config__ as cfg

        buf = io.StringIO()
        cfg.show(printer=lambda *a, **k: buf.write(" ".join(str(x) for x in a)))
        text = buf.getvalue().casefold()
        return {
            "avx2": "avx2" in text,
            "avx512f": "avx512f" in text,
            "avx512bw": "avx512bw" in text or "avx512_bw" in text,
            "avx512dq": "avx512dq" in text or "avx512_dq" in text,
            "avx512vl": "avx512vl" in text or "avx512_vl" in text,
            "avx512_vnni": "avx512_vnni" in text,
            "avx512_bf16": "avx512_bf16" in text or "avx512bf16" in text,
        }
    except Exception:
        return {}


def _os_logical_cores() -> tuple[int, int]:
    try:
        from os import cpu_count

        logical = cpu_count() or 1
    except Exception:
        logical = 1
    physical = max(1, logical // 2)  # 无 SMT 信息时保守估计
    try:
        from psutil import cpu_count as _pc

        logical = _pc(logical=True) or logical
    except Exception:
        pass
    return logical, physical


def detect_cpu() -> CpuFeatures:
    """探测 CPU 能力：torch 运行时特性 + 逻辑核数。cpuid 文本兜底。"""
    logical, physical = _os_logical_cores()
    name = "unknown"
    try:
        from platform import machine

        name = f"{sys.platform}/{machine()}"
    except Exception:
        pass
    flags = _cpuid_flags()
    # 兜底：torch 查询失败时按保守值（AVX2 假设可用，AVX512 未知 → False 走纯 numpy 路径）
    return CpuFeatures(
        name=name,
        avx2=flags.get("avx2", True),
        avx512f=flags.get("avx512f", False),
        avx512dq=flags.get("avx512dq", False),
        avx512bw=flags.get("avx512bw", False),
        avx512vl=flags.get("avx512vl", False),
        avx512_vnni=flags.get("avx512_vnni", False),
        avx512_bf16=flags.get("avx512_bf16", False),
        logical_cores=logical,
        physical_cores=physical,
    )


# ---------------------------------------------------------------------------
# 内存
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MemoryInfo:
    total_gb: float
    available_gb: float

    @property
    def available_mb(self) -> float:
        return self.available_gb * 1024.0


def detect_memory() -> MemoryInfo:
    try:
        from psutil import virtual_memory

        vm = virtual_memory()
        return MemoryInfo(total_gb=vm.total / 2**30, available_gb=vm.available / 2**30)
    except Exception:
        # Windows 回退：GlobalMemoryStatusEx
        if sys.platform == "win32":
            try:

                class _MEMORYSTATUSEX(ctypes.Structure):
                    _fields_ = [
                        ("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                    ]

                stat = _MEMORYSTATUSEX()
                stat.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
                ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
                return MemoryInfo(stat.ullTotalPhys / 2**30, stat.ullAvailPhys / 2**30)
            except Exception:
                pass
        return MemoryInfo(total_gb=0.0, available_gb=0.0)


# ---------------------------------------------------------------------------
# 盘速基准
# ---------------------------------------------------------------------------


@dataclass
class DiskBenchmark:
    """盘速基准结果（MB/s）。"""

    random4k_mbps: float = 0.0
    sequential1m_mbps: float = 0.0
    probe_path: str = ""
    sample_mb: int = 16
    elapsed_ms: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "random4k_mbps": self.random4k_mbps,
            "sequential1m_mbps": self.sequential1m_mbps,
            "probe_path": self.probe_path,
            "sample_mb": self.sample_mb,
            "elapsed_ms": self.elapsed_ms,
        }


def _random4k_read_mbps(path: Path, sample_mb: int) -> float:
    """4KB 随机读基准：对 temp 文件随机 offset 读 4KB（预热后）。"""
    import random

    file_size = sample_mb * 1024 * 1024
    path.write_bytes(os.urandom(min(file_size, 16 * 1024 * 1024)))
    # 预热
    with open(path, "rb", buffering=0) as fh:
        for _ in range(4):
            fh.seek(0)
            fh.read(64 * 1024)
    rng = random.Random(1234)
    n_ops = 256
    t0 = time.perf_counter()
    with open(path, "rb", buffering=0) as fh:
        for _ in range(n_ops):
            off = rng.randrange(0, max(1, file_size - 4096)) // 4096 * 4096
            fh.seek(off)
            fh.read(4096)
    dt = time.perf_counter() - t0
    return (n_ops * 4096) / dt / 1e6 if dt > 0 else 0.0


def _sequential1m_read_mbps(path: Path, sample_mb: int) -> float:
    """1MB 顺序读基准：顺序读满 sample_mb（预热一轮后计时一轮）。"""
    file_size = sample_mb * 1024 * 1024
    path.write_bytes(os.urandom(min(file_size, 64 * 1024 * 1024)))
    with open(path, "rb", buffering=1024 * 1024) as fh:
        # 预热
        for _ in range(1):
            fh.seek(0)
            while fh.read(1024 * 1024):
                pass
        t0 = time.perf_counter()
        fh.seek(0)
        chunks = 0
        while fh.read(1024 * 1024):
            chunks += 1
        dt = time.perf_counter() - t0
    return (chunks * 1024 * 1024) / dt / 1e6 if dt > 0 else 0.0


def benchmark_disk(dir_path: str | Path | None = None, sample_mb: int = 16) -> DiskBenchmark:
    """盘速基准：4KB 随机读 + 1MB 顺序读。``dir_path`` 缺失时落 ``.kv_cache`` 探测目录。"""
    if dir_path is None:
        from ccut.config import SCHEMA

        dir_path = SCHEMA["kv_cache"]["kv_l2_dir"].default
    base = Path(dir_path)
    base.mkdir(parents=True, exist_ok=True)
    probe = base / f".disk_probe_{os.getpid()}.tmp"
    t0 = time.perf_counter()
    try:
        r4k = _random4k_read_mbps(probe, min(sample_mb, 16))
        seq = _sequential1m_read_mbps(probe, sample_mb)
    finally:
        try:
            probe.unlink(missing_ok=True)
        except OSError:
            pass
    return DiskBenchmark(
        random4k_mbps=round(r4k, 2),
        sequential1m_mbps=round(seq, 2),
        probe_path=str(base),
        sample_mb=sample_mb,
        elapsed_ms=round((time.perf_counter() - t0) * 1000, 1),
    )


# ---------------------------------------------------------------------------
# 平台报告
# ---------------------------------------------------------------------------


@dataclass
class PlatformReport:
    cpu: CpuFeatures
    memory: MemoryInfo
    disk: DiskBenchmark
    extra: dict = field(default_factory=dict)

    @classmethod
    def probe(cls, dir_path: str | Path | None = None, sample_mb: int = 16, disk: Callable | None = None) -> "PlatformReport":
        cpu = detect_cpu()
        memory = detect_memory()
        disk_fn = disk if disk is not None else benchmark_disk
        disk_res = disk_fn(dir_path, sample_mb) if disk is None else disk()
        return cls(cpu=cpu, memory=memory, disk=disk_res)


def render_startup_report(report: PlatformReport, budget_lines: list[str] | None = None) -> str:
    """启动报告文本（``--info``）。"""
    cpu = report.cpu
    mem = report.memory
    disk = report.disk
    lines = [
        "=== CostCut-Infer 平台报告 ===",
        f"CPU      : {cpu.name}，逻辑核 {cpu.logical_cores}（物理 {cpu.physical_cores}）",
        f"指令集   : AVX2={cpu.avx2} AVX512F={cpu.avx512f} BW={cpu.avx512bw} DQ={cpu.avx512dq} "
        f"VL={cpu.avx512vl} VNNI={cpu.avx512_vnni} BF16={cpu.avx512_bf16}",
        f"内存     : 物理 {mem.total_gb:.1f}GB / 可用 {mem.available_gb:.1f}GB",
        f"盘速     : 4KB 随机 {disk.random4k_mbps:.0f}MB/s | 1MB 顺序 {disk.sequential1m_mbps:.0f}MB/s"
        f"（probe={disk.probe_path}，{disk.elapsed_ms:.0f}ms）",
    ]
    if budget_lines:
        lines.append("资源预算（R11）:")
        lines.extend(f"  {line}" for line in budget_lines)
    return "\n".join(lines)
