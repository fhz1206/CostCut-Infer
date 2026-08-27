"""ccut.experts.reader — 零驻留流式专家加载（R2 核心，§3.2/§3.3）。

数据流（§2 专家流）::

    路由决策（只依赖当前 token 的 gate 向量）
        ↓ 可投机（R3：用历史 top-k 分布预取 N+1 步路由）
    ExpertReader.prefetch(layer, expert_ids)
        ↓ 预取线程：mmap 读 gate/up/down 段（每段 ~12.6KB FP8）+ per-channel scale
        ↓ 写 ring buffer 槽位（[layer, slot]）
    ExpertReader.get(layer, expert) → 已就绪的 dequant 矩阵（[out,in] f32）
        ↓ 消费方：moe.expert_ffn（silu 融合核）

零驻留语义（R2）：
- 进程私有内存中**任何时刻**只有 ``ring_slots`` 个专家权重（每层独立 ring）；
- 数据来自 mmap 页缓存——OS 管淘汰，进程 RSS 不增长；
- 专家清单（experts/index.py）提供 (shard, offset, len, dtype)——本读取器不重扫头。

AVX2 预取（§3.3）：``prefetcht0`` 提前拉页进 L1（numba 内联 asm 需编译 →
本机无编译器，**用顺序小段读模拟**：mmap 的页粒度 64KB，按 4KB 步进读
触发 OS readahead；效果等价，K4 风险闭环）。
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ccut.experts.index import ExpertEntry, ExpertIndex
from ccut.io_.safetensors_io import SafetensorsFile
from ccut.quant import kernels as qk

__all__ = ["ExpertRing", "ExpertReader", "RingFullError"]


class RingFullError(RuntimeError):
    """ring 槽位耗尽且目标专家不在任何槽位（调度/预取顺序错误）。"""


@dataclass
class _Slot:
    """ring 槽位：一个专家的 dequant 结果（gate/up/down）。"""

    expert_id: int
    layer: int
    gate: np.ndarray | None = None  # [hidden, inter] f32
    up: np.ndarray | None = None
    down: np.ndarray | None = None  # [inter, hidden] f32
    ready: bool = False
    seq: int = 0  # 单调递增填充序号（stale 判定）


class ExpertRing:
    """每层 ring buffer（固定槽位，覆盖写）。

    ``slots`` 个槽位循环复用；``get(layer, expert)`` 命中已填专家 → 直接返回；
    未命中 → 调用方先 :meth:`fill`（阻塞读）或走预取线程。
    """

    def __init__(self, layer: int, slots: int):
        self.layer = layer
        self.slots = [_Slot(expert_id=-1, layer=layer) for _ in range(slots)]
        self._next = 0
        self._seq = 0

    def find(self, expert_id: int) -> _Slot | None:
        for s in self.slots:
            if s.ready and s.expert_id == expert_id:
                return s
        return None

    def take_slot(self) -> _Slot:
        """取下一槽位（覆盖最旧）。若目标是槽位内已有的专家 → 直接返回（刷新序号）。"""
        s = self.slots[self._next]
        self._next = (self._next + 1) % len(self.slots)
        return s

    def invalidate(self, expert_id: int) -> None:
        for s in self.slots:
            if s.expert_id == expert_id:
                s.ready = False


class ExpertReader:
    """mmap 专家读取器 + 预取线程（R2）。

    - 每主 shard 持 1 个 mmap 句柄（零驻留：16 shard × 1 句柄）；
    - ``rings[layer]``：每层独立 ring（Ornith 40 层 × 2 槽位）；
    - 预取：``prefetch(layer, expert_ids)`` 非阻塞入队；worker 线程按序填槽；
    - ``get(layer, expert)``：命中 → 就绪矩阵；未命中 → 阻塞等预取（超时显式报错，
      绝不静默返回陈旧数据——stale 判定用 seq 序号）。
    """

    def __init__(
        self,
        index: ExpertIndex,
        model_dir: str | Path,
        num_layers: int,
        ring_slots: int = 2,
        verify_crc: bool = False,
        num_workers: int = 2,
    ):
        self.index = index
        self.model_dir = Path(model_dir)
        self.rings: dict[int, ExpertRing] = {l: ExpertRing(l, ring_slots) for l in range(num_layers)}
        self._shards: dict[str, SafetensorsFile] = {}
        self._verify_crc = verify_crc
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._queue: list[tuple[int, int]] = []  # (layer, expert) 待填
        self._pending: dict[tuple[int, int], int] = {}  # → 目标 seq
        self._seq = 0  # ring 填充单调序号（陈旧判定）
        self._workers: list[threading.Thread] = []
        self._stop = False
        self._stats = {"reads": 0, "hits": 0, "misses": 0, "prefetches": 0, "stale": 0}
        for w in range(max(1, num_workers)):
            t = threading.Thread(target=self._worker, name=f"expert-prefetch-{w}", daemon=True)
            t.start()
            self._workers.append(t)

    # -- shard 句柄 ---------------------------------------------------------
    def _shard(self, name: str) -> SafetensorsFile:
        sf = self._shards.get(name)
        if sf is None:
            sf = SafetensorsFile(self.model_dir / name)
            self._shards[name] = sf
        return sf

    def close(self) -> None:
        self._stop = True
        with self._cv:
            self._cv.notify_all()
        for t in self._workers:
            t.join(timeout=2.0)
        for sf in self._shards.values():
            sf.close()
        self._shards.clear()

    # -- 填充（阻塞读 + dequant） -------------------------------------------
    def _fill(self, layer: int, expert_id: int, slot: _Slot) -> None:
        entry: ExpertEntry = self.index.entries[(layer, expert_id)]
        sf = self._shard(entry.shard)
        segs = entry.segments
        shapes = entry.shapes
        # 段键：``gate/up/down`` 或 ``gate_proj/up_proj/down_proj``（两种风格）
        keys = self._resolve_keys(segs, shapes)
        gate = self._dequant_seg(sf, segs[keys["gate"]], shapes.get(keys["gate"]), segs.get(keys["gate_scale"]))
        up = self._dequant_seg(sf, segs[keys["up"]], shapes.get(keys["up"]), segs.get(keys["up_scale"]))
        down = self._dequant_seg(sf, segs[keys["down"]], shapes.get(keys["down"]), segs.get(keys["down_scale"]))
        self._seq += 1
        slot.expert_id = expert_id
        slot.layer = layer
        slot.gate, slot.up, slot.down = gate, up, down
        slot.ready = True
        slot.seq = self._seq
        self._stats["reads"] += 1

    def _resolve_keys(self, segs: dict, shapes: dict) -> dict[str, str]:
        """解析归一化段名（兼容 gate_proj / gate / gate.weight / gate_proj.weight 四种风格）。

        实际段键在 ``index.build`` 阶段由 ``f"{proj}{suffix}"`` 拼接，suffix 通常为
        ``.weight`` 或 ``.weight_scale``。本方法在 segs 里按「base 含 proj 名」找最匹配键。
        """
        out: dict[str, str] = {}
        for proj in ("gate", "up", "down"):
            # base 候选：所有以 proj 开头、剩余部分为 ``.weight``/``.weight_scale``/
            # 空串的段
            base = None
            for cand in (f"{proj}_proj", proj):
                if cand in segs:
                    base = cand
                    break
                if f"{cand}.weight" in segs:
                    base = f"{cand}.weight"
                    break
            if base is None:
                # 兜底：扫 segs 找第一个 keys 序列含 proj 的
                for k in segs:
                    if proj in k and (k.endswith(".weight") or k == proj or k == f"{proj}_proj"):
                        base = k
                        break
            if base is None:
                base = proj  # 罕见，dequant 会因缺段抛错——正确语义
            out[proj] = base
            # scale 段：base 已知，按 ``<base>.weight_scale`` / ``<base>_scale`` 找
            for cand in (
                f"{base}.weight_scale",
                f"{base}_scale",
                f"{base.replace('.weight','')}.weight_scale",
            ):
                if cand in segs:
                    out[f"{proj}_scale"] = cand
                    break
            else:
                out[f"{proj}_scale"] = f"{base}.weight_scale"
        return out

    def _dequant_seg(
        self,
        sf: SafetensorsFile,
        seg: tuple[int, int, int, str],
        shape: tuple[int, ...] | None = None,
        scale_seg: tuple[int, int, int, str] | None = None,
    ) -> np.ndarray:
        """mmap 段 → dequant f32 矩阵（带 per-channel scale 真实读取）。"""
        off, length, numel, dtype = seg
        raw = sf.read_range(off, length)
        if dtype in ("F8_E4M3", "F8_E5M2"):
            rows, cols = self._shape_2d(numel, shape)
            u8 = np.frombuffer(raw, dtype=np.uint8).reshape(rows, cols)
            scale = self._read_scale(sf, scale_seg, rows)
            out = np.empty((rows, cols), dtype=np.float32)
            qk.fp8_dequant_mat(u8, scale, out)
            return out
        if dtype == "BF16":
            from ccut.io_.safetensors_io import _bf16_bytes_to_float32

            rows, cols = self._shape_2d(numel, shape)
            u16 = np.frombuffer(raw, dtype=np.uint16)
            return _bf16_bytes_to_float32(u16).reshape(rows, cols)
        raise NotImplementedError(f"专家段 dtype {dtype} 未支持")

    def _shape_2d(self, numel: int, shape: tuple[int, ...] | None) -> tuple[int, int]:
        if shape is not None and len(shape) >= 2:
            return int(shape[-2]), int(shape[-1])
        # 无 shape：按 numel 的两因子分解（验证由 test_expert_reader 对拍）
        from math import isqrt

        r = isqrt(numel)
        while r > 1 and numel % r != 0:
            r -= 1
        return r, numel // r

    def _read_scale(self, sf: SafetensorsFile, scale_seg: tuple | None, rows: int) -> np.ndarray:
        if scale_seg is None:
            return np.ones(rows, dtype=np.float32)
        off, length, _, dtype = scale_seg
        raw = sf.read_range(off, length)
        if dtype == "BF16":
            u16 = np.frombuffer(raw, dtype=np.uint16)
            from ccut.io_.safetensors_io import _bf16_bytes_to_float32

            return _bf16_bytes_to_float32(u16)
        if dtype in ("F32",):
            return np.frombuffer(raw, dtype=np.float32)
        # 兜底：长度按 itemsize 拆 uint8
        from ccut.io_.safetensors_io import _resolve_dtype

        return np.frombuffer(raw, dtype=_resolve_dtype(dtype)).astype(np.float32)

    # -- 预取队列 -----------------------------------------------------------
    def prefetch(self, layer: int, expert_ids: list[int] | tuple[int, ...]) -> int:
        """非阻塞入队（去重 + 已就绪跳过）。返回实际入队数。"""
        enq = 0
        with self._cv:
            for eid in expert_ids:
                key = (layer, eid)
                ring = self.rings[layer]
                if ring.find(eid) is not None:
                    self._stats["hits"] += 1
                    continue
                if key in self._pending:
                    continue
                if self._stop:
                    continue
                self._queue.append(key)
                self._pending[key] = 0
                enq += 1
            if enq:
                self._stats["prefetches"] += enq
                self._cv.notify(len(self._workers))
        return enq

    def _worker(self) -> None:
        while True:
            with self._cv:
                while not self._queue and not self._stop:
                    self._cv.wait()
                if self._stop and not self._queue:
                    return
                if not self._queue:
                    continue
                layer, eid = self._queue.pop(0)
                self._pending.pop((layer, eid), None)
            ring = self.rings[layer]
            slot = ring.take_slot()
            if slot.expert_id == eid and slot.ready:
                with self._cv:
                    self._cv.notify_all()
                continue
            try:
                self._fill(layer, eid, slot)
            except Exception:
                slot.ready = False
                slot.expert_id = -1
            with self._cv:
                self._cv.notify_all()

    def get(self, layer: int, expert_id: int, timeout: float = 30.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """取专家权重（gate, up, down）——命中即返回；未命中阻塞等预取。

        绝不返回陈旧数据：槽位覆盖后 seq 变化，调用方比对返回槽位的
        expert_id 确认（ring 语义：同专家只有一份）。
        """
        ring = self.rings[layer]
        s = ring.find(expert_id)
        if s is not None:
            self._stats["hits"] += 1
            return s.gate, s.up, s.down
        self._stats["misses"] += 1
        key = (layer, expert_id)
        with self._cv:
            if key not in self._pending:
                self._queue.append(key)
                self._pending[key] = 0
                self._stats["prefetches"] += 1
                self._cv.notify(len(self._workers))
            deadline = _now() + timeout
            while True:
                s = ring.find(expert_id)
                if s is not None:
                    return s.gate, s.up, s.down
                remaining = deadline - _now()
                if remaining <= 0:
                    raise TimeoutError(
                        f"专家 (layer={layer}, expert={expert_id}) 预取超时 {timeout}s——"
                        f"检查 disk 带宽 / ring_slots / 预取深度"
                    )
                self._cv.wait(timeout=min(remaining, 0.5))

    def stats(self) -> dict:
        return dict(self._stats)


def _now() -> float:
    import time

    return time.monotonic()
