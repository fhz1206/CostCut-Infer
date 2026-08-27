"""tests.test_kv — KV 双层存储（哈希 + 写读 + L2 装回 + 崩溃安全）。"""

from __future__ import annotations

import shutil

import numpy as np
import pytest

from ccut.kv.blocks import PREFIX_HASH_NULL, BlockPool, block_hash
from ccut.kv.coordinator import KVCoordinator
from ccut.kv.disk import DiskLayer


def test_prefix_hash_deterministic_and_prefix_sensitive():
    h1 = block_hash(0, [1, 2, 3, 4])
    h2 = block_hash(0, [1, 2, 3, 4])
    h3 = block_hash(0, [1, 2, 3, 5])
    assert h1 == h2 and h1 != h3 and h1 != 0
    # 注：滚动哈希是逐块级——「前 4 token 的块哈希」不等于「全 8 token 的块哈希」
    # （实现按整块一次消化 prev_hash || 全块，prev=0 || [1..4] ≠ prev=0 || [1..8]）。
    # 测试只覆盖「整块级确定性 + 敏感性 + 与 0 区分」。
    h_other = block_hash(0, [1, 2, 3, 4, 5, 6, 7, 9])  # 单点差异
    assert h_other != h1


def test_l1_round_trip():
    pool = BlockPool(0, 4, 2, 8, 1024 * 1024)
    k = np.random.randn(4, 2, 8).astype(np.float32)
    v = np.random.randn(4, 2, 8).astype(np.float32)
    bid = pool.alloc()
    bh = pool.write_block(bid, [1, 2, 3, 4], k, v, PREFIX_HASH_NULL)
    k2, v2 = pool.read_kv(bid)
    assert np.allclose(k2, k) and np.allclose(v2, v)
    assert pool.find(bh) == bid


def test_l2_store_rescan(tmp_dir):
    pool = BlockPool(0, 4, 2, 8, 1024 * 1024)
    k = np.random.randn(4, 2, 8).astype(np.float32)
    v = np.random.randn(4, 2, 8).astype(np.float32)
    bid = pool.alloc()
    bh = pool.write_block(bid, [1, 2, 3, 4], k, v, PREFIX_HASH_NULL)
    disk = DiskLayer(tmp_dir, max_bytes=64 * 1024 * 1024)
    assert disk.store(0, bh, PREFIX_HASH_NULL, [1, 2, 3, 4], pool.payload_bytes(bid))
    rec = disk.lookup(bh, 0)
    assert rec is not None
    # 重启重建
    disk2 = DiskLayer(tmp_dir, max_bytes=64 * 1024 * 1024)
    rec2 = disk2.lookup(bh, 0)
    assert rec2 is not None and disk2.read_payload(rec2) == pool.payload_bytes(bid)


def test_l2_crash_safe_torn_tail(tmp_dir):
    pool = BlockPool(0, 4, 2, 8, 1024 * 1024)
    bid = pool.alloc()
    bh = pool.write_block(bid, [1, 2, 3, 4], np.zeros((4, 2, 8), np.float32), np.zeros((4, 2, 8), np.float32), PREFIX_HASH_NULL)
    disk = DiskLayer(tmp_dir, max_bytes=64 * 1024 * 1024)
    disk.store(0, bh, PREFIX_HASH_NULL, [1, 2, 3, 4], pool.payload_bytes(bid))
    # 撕裂尾部记录
    f = disk._files[0]
    with open(f, "r+b") as fh:
        fh.seek(0, 2)
        fh.write(b"KVB2\x00")
    disk2 = DiskLayer(tmp_dir, max_bytes=64 * 1024 * 1024)
    assert disk2.lookup(bh, 0) is not None  # 已有记录不应丢


def test_coordinator_prefix_hit_and_l2_reload(tmp_dir):
    """KV 双层：L1 命中 → 驱逐下沉 L2 → 装回（disk_first 强制每驱逐都写）。"""
    pools = {0: BlockPool(0, 4, 2, 8, l1_bytes=4 * 1024)}
    disk = DiskLayer(tmp_dir, max_bytes=32 * 1024 * 1024)
    co = KVCoordinator(pools, disk, block_size=4, policy="disk_first", evict_high_water=0.3, evict_low_water=0.1, hot_window_steps=0)
    co.begin_request([0], 8)
    p = pools[0]
    toks = list(range(1, 9))
    prev = PREFIX_HASH_NULL
    for ci in range(2):
        chunk = toks[ci * 4 : (ci + 1) * 4]
        k = np.random.randn(4, 2, 8).astype(np.float32)
        v = np.random.randn(4, 2, 8).astype(np.float32)
        prev = p.write_block(p.alloc(), chunk, k, v, prev)
    co.ref_hit_blocks([0], co.lookup_prefix([0], toks))
    co.maintain()
    co.unref_hit_blocks([0], co.lookup_prefix([0], toks))
    co.maintain()
    assert co.stats()["l2_writes"] >= 1
    hit2 = co.lookup_prefix([0], toks)
    assert hit2.n_tokens == 8
    assert co.stats()["l2_hits"] >= 1
