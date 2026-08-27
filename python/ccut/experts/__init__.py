"""ccut.experts — 零驻留流式专家加载（需求 2）。

- :mod:`ccut.experts.index`：专家清单扫描（shard 头解析）+ 落盘缓存 + 校验。
- :mod:`ccut.experts.reader`：mmap ExpertReader + ring buffer + AVX2 预取核。
- :mod:`ccut.experts.pipeline`：三层流水线状态机（C/P/S 阶段、队列、时序指标）。
"""
