"""ccut.io_ — IO 底座（包名 io 与 stdlib 冲突，沿用 io_ 约定）。

- :mod:`ccut.io_.safetensors_io`：mmap 读 safetensors（只解析头、按段读、零拷贝视图）。
- :mod:`ccut.io_.windows_io`：ctypes Windows 系统 API（顺序读 readahead 提示、线程优先级）。
"""
