"""ccut.kv.coordinator — KV 双层统一调度（R1 门面）。

三级 lookup（§3.1 R1）::

    lookup(prefix_hash)
      ├─ L1 命中 → 直接引用（refcount+1）
      ├─ L2 命中 → alloc L1 槽位 → read_payload → load_payload（装回）→ 引用
      └─ miss    → 调用方继续写新块

写入路径：
- decode 每步：当前批 token 攒满 ``block_size`` → ``commit_block``
  （L1 写 + L2 异步下沉判定——水位驱逐由 :meth:`maintain` 周期执行）；
- 非满尾块（请求未结束）留在「活动块」，请求结束时若长度 > 阈值则提交，
  否则丢弃（尾块 < block_size 不做前缀复用——与 vLLM 一致）。

驱逐策略（水位，§3.1）：
- ``utilization ≥ evict_high_water`` → 取 L1 可驱逐块（refcount=0 且超出
  hot window）按 LRU 下沉 L2（disk.store）→ mark_on_disk → release_block；
- ``utilization ≤ evict_low_water`` → 停止；
- L2 满 → disk._evict 轮转最旧文件（自动）。

块池按 full_attn 层各自独立（前缀哈希是 token 序列级，跨层同哈希——
coordinator 持有 ``{layer: BlockPool}``，lookup 需带层号或遍历全部层）。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ccut.kv.blocks import PREFIX_HASH_NULL, BlockPool, block_hash
from ccut.kv.disk import DiskLayer

__all__ = ["KVCoordinator", "KVHit"]


@dataclass
class KVHit:
    """前缀命中结果：命中的满块链（token 序列前缀）。"""

    n_tokens: int
    block_hashes: list[int]  # 根→叶 序
    token_ids: list[int] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return self.n_tokens == 0


class KVCoordinator:
    """KV 双层协调器（引擎持有单例）。

    参数来自 ``Config``（kv_cache 节 + quant.kv_cache_dtype 折算 per_token_bytes）。
    """

    def __init__(
        self,
        pools: dict[int, BlockPool],
        disk: DiskLayer,
        block_size: int = 16,
        policy: str = "hybrid",
        evict_high_water: float = 0.8,
        evict_low_water: float = 0.6,
        hot_window_steps: int = 16,
        kv_dtype: str = "bf16",
    ):
        self.pools = pools  # layer_idx → BlockPool
        self.disk = disk
        self.block_size = block_size
        self.policy = policy.casefold()
        self.evict_high_water = evict_high_water
        self.evict_low_water = evict_low_water
        self.hot_window_steps = hot_window_steps
        self.kv_dtype = kv_dtype
        self._step = 0
        # 活动块：(layer, block_id) → 已攒 token（未提交）
        self._active: dict[int, tuple[int, list[int], int]] = {}
        self._stats = {"l1_hits": 0, "l2_hits": 0, "misses": 0, "evictions": 0, "l2_writes": 0}

    # -- 步推进 -------------------------------------------------------------
    def step(self) -> None:
        """每推理步调用（冷却窗口 + 水位维护）。"""
        self._step += 1
        for p in self.pools.values():
            p.step()
        if self.policy != "memory_first":
            self.maintain()

    def maintain(self) -> None:
        """水位驱逐：高水位 → 下沉 L2 直到低水位。"""
        if self.policy == "memory_first":
            return
        for layer, pool in self.pools.items():
            u = pool.utilization()
            if u < self.evict_high_water:
                continue
            for bid in pool.evictable(self.hot_window_steps):
                if pool.utilization() <= self.evict_low_water:
                    break
                b = pool.block(bid)
                payload = pool.payload_bytes(bid)
                if self.policy == "disk_first":
                    self.disk.store(layer, b.block_hash, b.prev_hash, b.token_ids, payload)
                    self._stats["l2_writes"] += 1
                else:  # hybrid：L2 有副本才直接释放，否则下沉
                    if self.disk.lookup(b.block_hash, layer) is None:
                        self.disk.store(layer, b.block_hash, b.prev_hash, b.token_ids, payload)
                        self._stats["l2_writes"] += 1
                pool.mark_on_disk(bid)
                pool.release_block(bid)
                self._stats["evictions"] += 1

    # -- 前缀查询 -----------------------------------------------------------
    def lookup_prefix(
        self,
        layers: list[int],
        token_ids: list[int],
    ) -> KVHit:
        """沿块边界向前匹配前缀（从根哈希逐块走）。

        ``layers``：参与匹配的 full_attn 层（全部层必须同时命中才算命中——
        单池实现下同哈希各池独立存，逐层 find）。
        返回最长前缀命中（token 数必为 block_size 倍数）。
        """
        hit = KVHit(n_tokens=0, block_hashes=[])
        h = PREFIX_HASH_NULL
        consumed: list[int] = []
        i = 0
        n = len(token_ids)
        while i + self.block_size <= n:
            chunk = token_ids[i : i + self.block_size]
            h = block_hash(h, chunk)
            all_found = True
            for layer in layers:
                if self.pools[layer].find(h) is None:
                    if self.policy in ("hybrid", "disk_first"):
                        rec = self.disk.lookup(h, layer)
                        if rec is None:
                            all_found = False
                            break
                        # L2 命中：装回（各层分别装）
                        pool = self.pools[layer]
                        try:
                            bid = pool.alloc()
                        except MemoryError:
                            self.maintain()
                            bid = pool.alloc()
                        payload = self.disk.read_payload(rec)
                        pool.load_payload(bid, list(rec.token_ids), payload, rec.prev_hash)
                        pool.ref(bid)
                        self._stats["l2_hits"] += 1
                    else:
                        all_found = False
                        break
                else:
                    self._stats["l1_hits"] += 1
            if not all_found:
                break
            hit.block_hashes.append(h)
            hit.n_tokens += self.block_size
            consumed.extend(chunk)
            i += self.block_size
        hit.token_ids = consumed
        if hit.n_tokens == 0:
            self._stats["misses"] += 1
        return hit

    # -- 写入 ---------------------------------------------------------------
    def begin_request(self, layers: list[int], prompt_len: int) -> None:
        """请求开始：各层分配活动块（prompt 攒块用）。"""
        for layer in layers:
            pool = self.pools[layer]
            bid = pool.alloc()
            self._active[layer] = (bid, [], PREFIX_HASH_NULL)

    def feed_tokens(self, layers: list[int], token_ids: list[int]) -> int:
        """喂入 token（prefill 分块 / decode 单 token）→ 返回本次提交的块数。

        攒满 ``block_size`` 即 :meth:`commit_block`（L1 写；L2 由 maintain 水位下沉）。
        """
        committed = 0
        for layer in layers:
            bid, acc, prev = self._active.get(layer, (-1, [], PREFIX_HASH_NULL))
            if bid < 0:
                pool = self.pools[layer]
                bid = pool.alloc()
                acc, prev = [], PREFIX_HASH_NULL
            acc.extend(token_ids)
            while len(acc) >= self.block_size:
                chunk = acc[: self.block_size]
                acc = acc[self.block_size:]
                self.commit_block(layer, bid, chunk, prev)
                prev = block_hash(prev, chunk)
                # 写后换槽位（满块已注册哈希；活动块用新槽）
                pool = self.pools[layer]
                new_bid = pool.alloc()
                bid = new_bid
                committed += 1
            self._active[layer] = (bid, acc, prev)
        return committed

    def commit_block(self, layer: int, block_id: int, token_ids: list[int], prev_hash: int) -> int:
        """注册已写入 L1 槽位的满块（K/V 字节由引擎算完注意力后写入，本方法只挂哈希）。"""
        pool = self.pools[layer]
        return pool.register_meta(block_id, token_ids, prev_hash)

    def end_request(self, layers: list[int], tail_threshold: int = 0) -> None:
        """请求结束：释放活动块引用；尾块达到阈值则提交，否则丢弃。"""
        for layer in list(self._active):
            if layer not in self.pools:
                continue
            pool = self.pools[layer]
            bid, acc, prev = self._active.pop(layer)
            if acc and len(acc) >= tail_threshold and len(acc) >= self.block_size:
                self.commit_block(layer, bid, acc[: self.block_size], prev)
            if bid >= 0:
                b = pool.block(bid)
                if b.token_ids and b.refcount > 0:
                    pool.unref(bid)
                    pool.release_block(bid)
                elif b.token_ids and b.block_hash:
                    # 已提交满块：仅解引用（保持 L1 可复用）
                    pass

    def ref_hit_blocks(self, layers: list[int], hit: KVHit) -> None:
        """命中块加引用（防驱逐，请求生命周期内）。"""
        for h in hit.block_hashes:
            for layer in layers:
                bid = self.pools[layer].find(h)
                if bid is not None:
                    self.pools[layer].ref(bid)

    def unref_hit_blocks(self, layers: list[int], hit: KVHit) -> None:
        for h in hit.block_hashes:
            for layer in layers:
                bid = self.pools[layer].find(h)
                if bid is not None:
                    self.pools[layer].unref(bid)

    # -- 指标 ---------------------------------------------------------------
    def stats(self) -> dict:
        u = [p.utilization() for p in self.pools.values()]
        return {
            **self._stats,
            "l1_util_avg": round(sum(u) / len(u), 4) if u else 0.0,
            "l1_blocks_total": sum(p.num_blocks for p in self.pools.values()),
            "l2": self.disk.stats(),
            "step": self._step,
        }


_EMPTY_KV = np.zeros(0, dtype=np.float32)


# BlockPool 需要 register_meta（只注册哈希/元数据，K/V 已写好）
def _register_meta(self, block_id: int, token_ids: list[int], prev_hash: int) -> int:
    b = self.blocks[block_id]
    b.token_ids = list(token_ids)
    b.prev_hash = prev_hash
    b.block_hash = block_hash(prev_hash, token_ids)
    b.last_access_step = self._step
    b.on_disk = False
    self._hash2id[b.block_hash] = block_id
    return b.block_hash


if not hasattr(BlockPool, "register_meta"):
    BlockPool.register_meta = _register_meta
