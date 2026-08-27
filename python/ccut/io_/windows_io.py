"""ccut.io_.windows_io — ctypes Windows 系统 API（无编译依赖）。

- 顺序读 readahead 提示：Windows 无 POSIX ``posix_fadvise``；NTFS 对顺序 IO 有自身
  readahead，本模块通过**保证请求顺序性** + ``SetFilePointerEx`` 大偏移跳读实现语义
  等价（§3.3 第 2 点）。``advise_sequential`` 在 POSIX 上可用 ``posix_fadvise``。
- 线程优先级：``set_thread_priority_below_normal``（ctypes ``SetThreadPriority``），
  R11 CPU 降档用（§3.7）；非 Windows 平台静默 no-op。
- 预取指令：numba 内联 asm 的 prefetcht0 序列放在 experts/reader.py（能力探测回退），
  本模块只提供纯 ctypes 能力探测辅助。
"""

from __future__ import annotations

import ctypes
import os
import sys
from typing import IO

__all__ = [
    "advise_sequential",
    "advise_random",
    "set_thread_priority_below_normal",
    "set_thread_priority_normal",
    "is_windows",
]

is_windows = sys.platform == "win32"

# Win32 线程优先级常量（SetThreadPriority）
_THREAD_PRIORITY_BELOW_NORMAL = -1
_THREAD_PRIORITY_NORMAL = 0

_kernel32 = None


def _get_kernel32():
    global _kernel32
    if not is_windows:
        return None
    if _kernel32 is None:
        _kernel32 = ctypes.windll.kernel32
    return _kernel32


def advise_sequential(fh: IO[bytes], hint_bytes: int | None = None) -> bool:
    """顺序读提示（§3.3）。

    POSIX：``posix_fadvise(POSIX_FADV_SEQUENTIAL)``；
    Windows：NTFS 内置顺序 readahead，无显式 API——返回 True 表示「按 NTFS 语义
    保证顺序请求即可」（调用方应保持 offset 单调递增）。
    """
    try:
        if hasattr(os, "posix_fadvise"):
            fd = fh.fileno()
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_SEQUENTIAL)
            return True
        if is_windows:
            return True  # NTFS 顺序 readahead，请求顺序性由调用方保证
    except (OSError, ValueError):
        pass
    return False


def advise_random(fh: IO[bytes]) -> bool:
    """随机读提示（放弃 readahead，避免浪费预读）。POSIX 专用；Windows no-op。"""
    try:
        if hasattr(os, "posix_fadvise"):
            os.posix_fadvise(fh.fileno(), 0, 0, os.POSIX_FADV_RANDOM)
            return True
    except (OSError, ValueError):
        pass
    return False


def _set_thread_priority(priority: int) -> bool:
    """对**当前线程**设置优先级。失败不抛错（权限不足时静默降级）。"""
    if not is_windows:
        return False
    k32 = _get_kernel32()
    if k32 is None:
        return False
    try:
        handle = k32.GetCurrentThread()
        return bool(k32.SetThreadPriority(handle, priority))
    except Exception:
        return False


def set_thread_priority_below_normal() -> bool:
    """当前线程降档（R11 CPU 合作式自治，§3.7）。"""
    return _set_thread_priority(_THREAD_PRIORITY_BELOW_NORMAL)


def set_thread_priority_normal() -> bool:
    """当前线程恢复正常优先级。"""
    return _set_thread_priority(_THREAD_PRIORITY_NORMAL)
