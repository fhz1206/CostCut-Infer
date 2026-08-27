"""ccut.weights.stream — 层流式预取流（§3.5 R10 上下游协同）。

设计：
- **流协议**（manager 之上）：prefetch 队列 + 预取窗口 = ring_layers + prefetch_ahead；
- **顺序约束**：prefill 阶段顺序流（layer N → N+1 → N+2...），decode 阶段
  投机预取（本步算 N 时已预填 N+1/N+2）；
- **sublayer 切分提示**：当 manager 切分时，stream 负责把「前半先到 / 后半异步补」
  的状态对外透出——上层（generic 层前向）首次访问后半段触发阻塞补读。
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Iterable

from ccut.weights.manager import LayerSlice, WeightManager

__all__ = ["WeightStream", "PrefetchWorker"]


class WeightStream:
    """权重流式预取（顺序 + 投机）。

    使用::

        stream = WeightStream(manager, prefetch_ahead=2)
        for layer_idx in layers:
            stream.refill_ahead(current=layer_idx)         # 推进流
            slc = stream.current(layer_idx)                # 获取当前层
            ...                                            # 计算
            stream.unref(layer_idx)
    """

    def __init__(self, manager: WeightManager, prefetch_ahead: int = 2):
        self.manager = manager
        self.prefetch_ahead = max(0, prefetch_ahead)
        self._max_layer = 0
        for l in manager.layer_tensor_names.keys():
            self._max_layer = max(self._max_layer, l)
        self._worker = PrefetchWorker(manager, prefetch_ahead=prefetch_ahead)
        self._worker.start()
        # 启动期预填第 0 层（保 prefill 第 1 步就绪）
        manager.prefetch_layer(0)

    def refill_ahead(self, current: int) -> None:
        """推进流：当前层 ref + 推 N+1~N+ahead 预填。"""
        self.manager.ring.ref(current)
        for k in range(1, self.prefetch_ahead + 1):
            nxt = current + k
            if nxt > self._max_layer:
                break
            self._worker.enqueue(nxt)
        self._worker.wake()

    def current(self, layer: int) -> LayerSlice:
        """获取当前层 slice（命中即返回；未命中同步等补读）。"""
        return self.manager.slot(layer)

    def unref(self, layer: int) -> None:
        self.manager.unref(layer)

    def stop(self) -> None:
        self._worker.stop()
        self._worker.join(timeout=2.0)
        self.manager.close()


@dataclass
class _Job:
    layer: int
    enq_ns: int = 0


class PrefetchWorker(threading.Thread):
    """预取 worker：单线程消费队列（mmap 是顺序 IO，单线程足矣）。"""

    def __init__(self, manager: WeightManager, prefetch_ahead: int = 2):
        super().__init__(name="weight-prefetch", daemon=True)
        self.manager = manager
        self._queue: deque[_Job] = deque()
        self._cv = threading.Condition()
        self._stop = False
        self._prefetch_ahead = prefetch_ahead

    def enqueue(self, layer: int) -> None:
        import time

        with self._cv:
            # 去重：同层未消费则不入
            if any(j.layer == layer for j in self._queue):
                return
            self._queue.append(_Job(layer=layer, enq_ns=time.monotonic_ns()))
            self._cv.notify()

    def wake(self) -> None:
        with self._cv:
            self._cv.notify()

    def stop(self) -> None:
        with self._cv:
            self._stop = True
            self._cv.notify_all()

    def run(self) -> None:
        while True:
            with self._cv:
                while not self._queue and not self._stop:
                    self._cv.wait()
                if self._stop and not self._queue:
                    return
                job = self._queue.popleft()
            if job.layer in self.manager.ring:
                continue
            try:
                self.manager.prefetch_layer(job.layer)
            except Exception:
                # 单层预取失败：跳过（层前向会同步触发，错误在那里暴露）
                pass
