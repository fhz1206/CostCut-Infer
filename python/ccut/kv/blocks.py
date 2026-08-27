"""ccut.kv.blocks — L1 内存块池 + 前缀滚动哈希（R1 核心）。

设计（§3.1 R1）：
- **固定块**：``block_size`` token/块（默认 16），块字节 = ``block_size × per_token_bytes``；
- **前缀滚动哈希**：``h_i = BLAKE2B(h_{i-1} || token_ids[块])``（跨进程稳定，
  替代 vLLM 的不稳定 ``hash()``；取前 8 字节为大端 uint64）；
- **引用计数**：块被活跃请求引用时不可驱逐；请求结束 → refcount 归零 →
  进入「冷块」队列（hot window 内仍不驱逐）；
- **水位驱逐**：L1 占用 ≥ ``evict_high_water`` → 冷块按 LRU 下沉 L2
  （coordinator 调 disk.store）；占用 ≤ ``evict_low_water`` 停止。
- **零驻留语义**：块池是定长 bytearray 预分配（启动一次），块复用不扩容——
  RSS 恒等于 ``l1_bytes``（R9 预算表的可预测部分）。
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

import numpy as np

__all__ = ["KVBlock", "BlockPool", "PrefixIndex", "block_hash", "PREFIX_HASH_NULL"]

PREFIX_HASH_NULL = 0  # 根前缀哈希（空前缀）


def block_hash(prev_hash: int, token_ids: list[int] | tuple[int, ...]) -> int:
    """前缀滚动哈希：``H(prev_hash_be64 || u32 tokens...)`` → uint64。

    BLAKE2B（digest_size=8，跨进程稳定）；prev=0 表示从根开始。
    """
    import hashlib

    buf = struct.pack(">Q", prev_hash & 0xFFFFFFFFFFFFFFFF)
    for t in token_ids:
        buf += struct.pack(">I", int(t) & 0xFFFFFFFF)
    return struct.unpack(">Q", hashlib.blake2b(buf, digest_size=8).digest())[0]


@dataclass
class KVBlock:
    """一个 KV 块（定长槽位 + 元数据）。"""

    block_id: int
    prev_hash: int
    token_ids: list[int] = field(default_factory=list)
    block_hash: int = 0
    refcount: int = 0
    last_access_step: int = 0
    on_disk: bool = False  # 已下沉 L2（L1 槽位仍可能被复用——数据已失效）

    @property
    def full(self) -> bool:
        # 满块判定由调用方按 block_size 给 token_ids 长度
        return True


class BlockPool:
    """单层 KV 块池（定长 buffer，启动预分配）。

    每 full_attn 层一个池；``per_token_bytes`` 由
    ``quant.kv.kv_bytes_per_token(num_kv_heads, head_dim, mode)`` 给出。

    块内存布局（每块）：``K[block_tokens, kv_heads, head_dim] || V[...]``
    （K/V 连续两段，读回时切分）。
    """

    def __init__(
        self,
        layer_idx: int,
        block_size: int,
        kv_heads: int,
        head_dim: int,
        l1_bytes: int,
        dtype: np.dtype = np.dtype(np.float32),
    ):
        self.layer_idx = layer_idx
        self.block_size = block_size
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        # 每 token 字节（K+V 两侧）：2 × kv_heads × head_dim × itemsize
        self.per_token_bytes = 2 * kv_heads * head_dim * dtype.itemsize
        # 整块字节（对齐到 64B，避免撕裂）
        raw = block_size * self.per_token_bytes
        self.block_bytes = (raw + 63) & ~63
        self.num_blocks = max(1, l1_bytes // self.block_bytes)
        self._buf = np.zeros(self.num_blocks * self.block_bytes, dtype=np.uint8)
        self.blocks: dict[int, KVBlock] = {i: KVBlock(block_id=i, prev_hash=PREFIX_HASH_NULL) for i in range(self.num_blocks)}
        self._free: list[int] = list(range(self.num_blocks))
        self._step = 0
        # 哈希 → block_id（每池独立；同哈希跨池由 coordinator 统一管）
        self._hash2id: dict[int, int] = {}

    # -- 生命周期 -----------------------------------------------------------
    def step(self) -> None:
        """推进逻辑步（冷却窗口计时用）。"""
        self._step += 1

    def used_bytes(self) -> int:
        """已分配块占用（未含 L2 下沉的——槽位复用后按新数据计）。"""
        return (self.num_blocks - len(self._free)) * self.block_bytes

    def utilization(self) -> float:
        return self.used_bytes() / (self.num_blocks * self.block_bytes)

    def alloc(self) -> int:
        if not self._free:
            raise MemoryError(f"layer {self.layer_idx} 块池耗尽（{self.num_blocks} 块）——coordinator 应先下沉 L2")
        return self._free.pop()

    def free(self, block_id: int) -> None:
        b = self.blocks[block_id]
        if block_id not in self._free:
            self._free.append(block_id)
        b.token_ids = []
        b.refcount = 0
        b.on_disk = False

    # -- 读写 ---------------------------------------------------------------
    def _f32(self) -> np.ndarray:
        """整池 buffer 的 float32 视图（block_bytes 为 64 倍数 → 4 字节对齐成立）。"""
        return self._buf.view(np.float32)

    def write_block(
        self,
        block_id: int,
        token_ids: list[int],
        k: np.ndarray,
        v: np.ndarray,
        prev_hash: int,
    ) -> int:
        """写入满块并注册哈希。``k``/``v``: [block_tokens, kv_heads, head_dim]。

        返回块哈希（coordinator 用它挂 L2 索引）。
        """
        b = self.blocks[block_id]
        off = block_id * self.block_bytes
        k32 = k.astype(np.float32, copy=False).reshape(-1)
        v32 = v.astype(np.float32, copy=False).reshape(-1)
        view = self._f32()
        base = off // 4
        view[base : base + k32.size] = k32
        view[base + k32.size : base + k32.size + v32.size] = v32
        b.token_ids = list(token_ids)
        b.prev_hash = prev_hash
        b.block_hash = block_hash(prev_hash, token_ids)
        b.last_access_step = self._step
        b.on_disk = False
        self._hash2id[b.block_hash] = block_id
        return b.block_hash

    def read_kv(self, block_id: int) -> tuple[np.ndarray, np.ndarray]:
        b = self.blocks[block_id]
        n = len(b.token_ids)
        elems = n * self.kv_heads * self.head_dim
        base = block_id * self.block_bytes // 4
        view = self._f32()
        k = view[base : base + elems].reshape(n, self.kv_heads, self.head_dim).copy()
        v = view[base + elems : base + 2 * elems].reshape(n, self.kv_heads, self.head_dim).copy()
        b.last_access_step = self._step
        return k, v

    def payload_bytes(self, block_id: int) -> bytes:
        """整块序列化（L2 下沉用）：``K 段 || V 段``（仅 token_ids 长度部分）。"""
        b = self.blocks[block_id]
        n = len(b.token_ids)
        seg_len = n * self.kv_heads * self.head_dim * self.dtype.itemsize
        off = block_id * self.block_bytes
        return bytes(self._buf[off : off + 2 * seg_len])

    def register_meta(self, block_id: int, token_ids: list[int], prev_hash: int) -> int:
        """只注册哈希/元数据（K/V 字节已由调用方写入槽位 buffer）。"""
        b = self.blocks[block_id]
        b.token_ids = list(token_ids)
        b.prev_hash = prev_hash
        b.block_hash = block_hash(prev_hash, token_ids)
        b.last_access_step = self._step
        b.on_disk = False
        self._hash2id[b.block_hash] = block_id
        return b.block_hash

    def load_payload(self, block_id: int, token_ids: list[int], payload: bytes, prev_hash: int) -> int:
        """L2 装回：payload 写入槽位 + 注册哈希。"""
        b = self.blocks[block_id]
        b.token_ids = list(token_ids)
        b.prev_hash = prev_hash
        b.block_hash = block_hash(prev_hash, token_ids)
        b.last_access_step = self._step
        b.on_disk = False
        off = block_id * self.block_bytes
        self._buf[off : off + len(payload)] = np.frombuffer(payload, dtype=np.uint8)
        self._hash2id[b.block_hash] = block_id
        return b.block_hash

    # -- 哈希查询 -----------------------------------------------------------
    def find(self, h: int) -> int | None:
        bid = self._hash2id.get(h)
        if bid is None:
            return None
        if self.blocks[bid].on_disk:
            return None
        return bid

    def mark_on_disk(self, block_id: int) -> None:
        b = self.blocks[block_id]
        if b.block_hash and self._hash2id.get(b.block_hash) == block_id:
            del self._hash2id[b.block_hash]
        b.on_disk = True

    def evictable(self, hot_window: int) -> list[int]:
        """可驱逐块（refcount=0 且超出 hot window），LRU 序（最旧在前）。"""
        out = [
            i
            for i, b in self.blocks.items()
            if b.refcount == 0 and b.token_ids and (self._step - b.last_access_step) >= hot_window
        ]
        out.sort(key=lambda i: self.blocks[i].last_access_step)
        return out

    def ref(self, block_id: int) -> None:
        self.blocks[block_id].refcount += 1

    def unref(self, block_id: int) -> None:
        b = self.blocks[block_id]
        if b.refcount > 0:
            b.refcount -= 1

    def block(self, block_id: int) -> KVBlock:
        return self.blocks[block_id]

    def release_block(self, block_id: int) -> None:
        """释放槽位（unref 后）：清哈希 + 回空闲。调用方保证 refcount 已 0。"""
        b = self.blocks[block_id]
        if b.block_hash and self._hash2id.get(b.block_hash) == block_id:
            del self._hash2id[b.block_hash]
        self.free(block_id)
