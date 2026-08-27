"""ccut.resources — R11 资源限制（CPU/内存/IO 比例门控，默认 50%）。

设计（§3.7 R11）：
- **budget.py** 读 ``Config.resources`` 解析「全局 / 分资源」覆盖，输出**预算表**
  （启动时打印一次 + 写到 metrics.resources）；
- **limiter.py** 三类门控实现：
  - **CPU**：torch 线程数 + numba 线程数（写环境变量 + 内部 set_num_threads）+
    进程优先级（windows_io.set_thread_priority_below_normal）；
  - **内存**：私有 RSS 软上限（``resource.RLIMIT_AS``/``psutil`` 周期检查，超限
    触发「反压」— 让调度器少接新请求 + 优先 evict decoding，不杀进程）；
  - **IO**：令牌桶（``--resource-io-pct`` 换算为 MB/s 字节/秒），所有 read 路径
    走 :func:`acquire` 等待令牌（无全局 → 0=不限速）。
- **watchdog.py** 后台线程周期采样（默认 1s）：CPU%、RSS、IO 累计字节
  → 写入 metrics.resources；超限按 ``resource_throttle`` 模式（auto / warn / off）动作。
"""

from __future__ import annotations

from ccut.resources.budget import (
    ResourceBudget,
    ResourcePctResolver,
    build_resource_budget,
    render_budget_table,
)

__all__ = [
    "ResourceBudget",
    "ResourcePctResolver",
    "build_resource_budget",
    "render_budget_table",
]
