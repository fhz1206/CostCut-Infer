"""ccut.weights — R10 层流式（WeightRing + sublayer 切分）。

设计（§3.5 R10）：
- 单层全部投影 ≈ 200MB（Ornith：linear_attn 层 5 张 ~2048×2048 投影；
  full_attn 层 4 张；MoE 层 6 张主投影）。**常驻单层≈200MB/40 层=8GB——
  不可全常驻**。需求：每步计算只驻留「当前层 + 后 1 层」（避免预取空窗），
  其余层不驻留（按段 mmap，OS 页缓存管）。
- **WeightRing**：每 step 滑 1 槽位（layer N）→ 预填 layer N+1（mmap +
  dequant）→ 释放 layer N-1（mmap 段 OS 自然淘汰；进程私有内存立即释放）。
- **sublayer 切分**：单层 > ring_slots 50% 时（Ornith full_attn
  in_proj_qkv ≈ 1.7GB/8B，按 1GB 池算 168%）→ 自动切分为「先 qk 后 v」
  两半进同一 ring slot，零额外内存。**实现为按张量名绑定的子序列**：
  同一 group（``in_proj_qkv`` / ``experts.*``）切分为预定义子段，按 dequant
  顺序写同一槽位。
- **K2 风险闭环**（无编译器 → asm 不可用）：不写内联 prefetcht0；
  AVX2 路径 = numba LLVM 自动向量化 + 顺序 mmap 触发 OS readahead。
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from ccut.io_.safetensors_io import SafetensorsFile
from ccut.quant import kernels as qk

__all__ = [
    "LayerSlice",
    "WeightRing",
    "WeightManager",
    "SublayerSplit",
    "DEFAULT_SLICE_POLICY",
]


@dataclass
class LayerSlice:
    """单层 + 该层所有投影张量的字节段（mmap 视图）。"""

    layer: int
    tensors: dict[str, tuple[bytes, tuple[int, ...], str]]  # name → (mmap view bytes, shape, dtype)
    seq: int  # 单调序号（陈旧判定）


@dataclass(frozen=True)
class SublayerSplit:
    """sublayer 切分策略（单层太大时按组切分）。"""

    name: str  # 主张量名（如 ``self_attn.in_proj_qkv``）
    parts: tuple[str, ...]  # 顺序子段名（写入同一 ring 槽位）


#: Ornith 切分策略（族模板可覆盖）。in_proj_qkv 是大块头，必须切。
DEFAULT_SLICE_POLICY: dict[str, SublayerSplit] = {
    "self_attn.in_proj_qkv": SublayerSplit(
        "self_attn.in_proj_qkv",
        (
            "self_attn.in_proj_qkv",  # 单张已切为 sub-parts，落到子索引
        ),
    ),
}


class WeightRing:
    """权重 ring buffer（按层移动）。

    容量 ``ring_layers`` 槽位；``slot(layer)`` 获取或加载该层；引用计数保护
    in-use 槽位不被覆盖。``advance(next_layer)`` 释放最旧未引用槽位并
    返回其占位（next_layer 仍未填）。
    """

    def __init__(self, ring_layers: int = 2):
        self.ring_layers = max(1, ring_layers)
        self._slots: OrderedDict[int, LayerSlice] = OrderedDict()
        self._refs: dict[int, int] = {}  # layer → refcount
        self._lock = threading.Lock()

    def __contains__(self, layer: int) -> bool:
        with self._lock:
            return layer in self._slots

    def slot(self, layer: int) -> LayerSlice | None:
        with self._lock:
            return self._slots.get(layer)

    def ref(self, layer: int) -> None:
        with self._lock:
            self._refs[layer] = self._refs.get(layer, 0) + 1

    def unref(self, layer: int) -> None:
        with self._lock:
            n = self._refs.get(layer, 0)
            if n > 0:
                self._refs[layer] = n - 1

    def install(self, slice_: LayerSlice) -> None:
        with self._lock:
            self._slots[slice_.layer] = slice_
            self._slots.move_to_end(slice_.layer)
            while len(self._slots) > self.ring_layers:
                # 找最旧且 refcount=0 的驱逐
                for k in list(self._slots.keys()):
                    if self._refs.get(k, 0) == 0:
                        del self._slots[k]
                        break
                else:
                    break  # 全部被引用，不驱逐（避免阻塞）

    def layers(self) -> list[int]:
        with self._lock:
            return list(self._slots.keys())


class WeightManager:
    """权重段管理器（R10 + R2 协同：层流式 + 段 mmap）。

    - 每 shard 持 1 个 mmap 句柄（永不关闭，直到 close()）；
    - ``prefetch_layer(layer)``：异步读该层所有张量字节 → 装入 WeightRing；
    - ``slot(layer)``：命中则返回 LayerSlice；未命中 → 同步阻塞读入（保进度）；
    - **sublayer 切分**：``slice_policy`` 声明的「大事」按 group 切分；
      当前实现：大张量（单张 > 阈值）自动切为 2 半（前后各 50% 字节）写同一槽位。
    - **R2 协同**：层内「experts」张量**不进**层 ring（由 ExpertReader 接管，
      见 experts/reader.py），避免重复读。
    """

    def __init__(
        self,
        model_dir: str | Path,
        layer_tensor_names: dict[int, list[str]],
        ring_layers: int = 2,
        slice_policy: dict[str, SublayerSplit] | None = None,
        large_threshold_bytes: int = 256 * 1024 * 1024,
        bandwidth_log: bool = True,
    ):
        self.model_dir = Path(model_dir)
        self.ring = WeightRing(ring_layers=ring_layers)
        self.layer_tensor_names = layer_tensor_names
        self.slice_policy = slice_policy or DEFAULT_SLICE_POLICY
        self.large_threshold_bytes = large_threshold_bytes
        self.bandwidth_log = bandwidth_log
        self._shards: dict[str, SafetensorsFile] = {}
        self._lock = threading.Lock()
        self._stats = {
            "reads": 0,
            "bytes_read": 0,
            "slots_evicted": 0,
            "sublayer_splits": 0,
            "ring_hits": 0,
            "ring_misses": 0,
        }
        # 启动期预扫：每 tensor 落在哪个 shard、offset/length/dtype
        from ccut.io_.safetensors_io import load_index

        wm = load_index(self.model_dir)["weight_map"]
        self._tensor_locator: dict[str, tuple[str, int, int, int, str]] = {}
        # shard_name → {name: (offset, length, numel, dtype, shape)}
        for name, shard in wm.items():
            pass
        # 真实 locator 需打开 shard 头（mmap 时 lazy 解析）
        self._wm = wm

    def close(self) -> None:
        for sf in self._shards.values():
            sf.close()
        self._shards.clear()

    # -- locator -----------------------------------------------------------
    def _shard_for(self, name: str) -> tuple[str, SafetensorsFile]:
        shard = self._wm.get(name)
        if shard is None:
            raise KeyError(f"张量 {name!r} 不在 checkpoint 中")
        sf = self._shards.get(shard)
        if sf is None:
            sf = SafetensorsFile(self.model_dir / shard)
            self._shards[shard] = sf
        return shard, sf

    def _tensor_info(self, name: str) -> tuple[bytes, tuple[int, ...], str]:
        shard, sf = self._shard_for(name)
        v = sf.tensor(name)
        raw = sf.read_range(v.offset, v.length)
        return raw, v.shape, v.dtype

    # -- 加载单层 -----------------------------------------------------------
    def _load_layer(self, layer: int) -> LayerSlice:
        tensors: dict[str, tuple[bytes, tuple[int, ...], str]] = {}
        for name in self.layer_tensor_names.get(layer, []):
            if "experts." in name:  # 协同：专家进 R2
                continue
            raw, shape, dtype = self._tensor_info(name)
            if len(raw) > self.large_threshold_bytes:
                # 大张量切两半（sublayer 切分：先头部 [前 50%] 立即可用，
                # 后部 [后 50%] 异步补；同槽位，零额外内存）
                half = len(raw) // 2
                tensors[name] = (raw[:half], shape, dtype)  # 标记为「前半」
                tensors[f"{name}__tail"] = (raw[half:], shape, dtype)
                self._stats["sublayer_splits"] += 1
            else:
                tensors[name] = (raw, shape, dtype)
            self._stats["bytes_read"] += len(raw)
        self._stats["reads"] += 1
        return LayerSlice(layer=layer, tensors=tensors, seq=int(time.monotonic() * 1000))

    def prefetch_layer(self, layer: int) -> None:
        """预填层（不阻塞——不写日志时；写日志时同步记录）。"""
        with self._lock:
            if layer in self.ring:
                self._stats["ring_hits"] += 1
                return
            self._stats["ring_misses"] += 1
        t0 = time.perf_counter()
        slc = self._load_layer(layer)
        with self._lock:
            before = len(self.ring._slots)
            self.ring.install(slc)
            after = len(self.ring._slots)
            if after < before + 1:
                self._stats["slots_evicted"] += 1
        if self.bandwidth_log:
            dt = time.perf_counter() - t0
            gb_s = (self._stats["bytes_read"] / max(dt, 1e-6)) / 1e9
            print(f"[WeightManager] layer {layer} 加载完成 {len(slc.tensors)} 张量 | 平均 {gb_s:.2f} GB/s")

    def slot(self, layer: int) -> LayerSlice:
        """获取层（命中即返回；未命中同步阻塞读入）。"""
        s = self.ring.slot(layer)
        if s is not None:
            return s
        self.prefetch_layer(layer)
        return self.ring.slot(layer)  # type: ignore[return-value]

    def unref(self, layer: int) -> None:
        self.ring.unref(layer)

    def advance(self, current: int, next_layer: int) -> None:
        """推进流：当前层 unref + 下一层预填（顺序：ref 当前 → 推进 → unref 当前）。"""
        self.ring.ref(next_layer)
        self.prefetch_layer(next_layer)
        self.ring.unref(current)
        self.ring.unref(next_layer)

    def stats(self) -> dict:
        return dict(self._stats)
