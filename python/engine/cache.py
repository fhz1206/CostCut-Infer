"""解码缓存与专家反量化缓存（M3 decode / M4 专家流式）。"""
from __future__ import annotations

from collections import OrderedDict

import torch
from torch import Tensor

__all__ = ["Cache", "ExpertCache", "kv_append"]


def kv_append(kv, key: Tensor, value: Tensor):
    """预分配 KV 追加（差异报告 #1 KV 缓存预分配）：kv=(k, v, length, max_len)。

    首次调用按 max_len 预分配（0 时按需扩展），后续位置索引写入——
    消除 decode 每步 ``torch.cat`` 的 O(ctx) 复制。
    返回 (新 kv, 活动切片 (k[..., :length+1], v[..., :length+1]))。
    """
    # 兼容旧 (k, v) 二元组调用（直接调用方）；新格式 (k, v, length, max_len)
    if len(kv) == 2:
        kv = (kv[0], kv[1], 0 if kv[0] is None else kv[0].shape[2], 0)
    k, v, length, max_len = kv
    seq = key.shape[2]
    if k is None:
        m = max_len if max_len > 0 else max(length + seq + 1024, 2048)
        k = torch.zeros(key.shape[0], key.shape[1], m, key.shape[3],
                        dtype=key.dtype, device=key.device)
        v = torch.zeros(value.shape[0], value.shape[1], m, value.shape[3],
                        dtype=value.dtype, device=value.device)
    elif k.shape[2] < length + seq:
        # 旧 (k, v) 二元组调用（无预分配）：空间不足时扩容（等价旧 concat 语义）
        new_m = max(k.shape[2] * 2, length + seq)
        pad = new_m - k.shape[2]
        k = torch.cat([k, torch.zeros(k.shape[0], k.shape[1], pad, k.shape[3],
                                      dtype=k.dtype, device=k.device)], dim=2)
        v = torch.cat([v, torch.zeros(v.shape[0], v.shape[1], pad, v.shape[3],
                                      dtype=v.dtype, device=v.device)], dim=2)
    k[:, :, length:length + seq, :] = key
    v[:, :, length:length + seq, :] = value
    new_len = length + seq
    return (k, v, new_len, max_len), (k[:, :, :new_len], v[:, :, :new_len])


class Cache:
    """逐层解码缓存（M3 decode 路径）。

    - full_attention 层：KV 缓存 (B, kv_heads, L, head_dim)，续接时追加
    - linear_attention 层：conv 状态 (B, conv_dim, kernel-1) + recurrent 状态
      (B, v_heads, k_head_dim, v_head_dim)
    """

    def __init__(self, num_layers: int, max_len: int = 0):
        self.num_layers = num_layers
        self.max_len = max_len
        # attn_kv: (k, v, length, max_len)——预分配 + 位置索引写入
        # （差异报告 #1：消除 decode 每步 torch.cat 的 O(ctx) 复制）
        self.attn_kv: list[tuple | None] = [None] * num_layers
        self.conv_state: list[Tensor | None] = [None] * num_layers
        self.rec_state: list[Tensor | None] = [None] * num_layers

    def reset(self) -> None:
        self.attn_kv = [None] * self.num_layers
        self.conv_state = [None] * self.num_layers
        self.rec_state = [None] * self.num_layers

    def snapshot(self) -> "Cache":
        """深拷贝状态（投机解码验证前快照，部分接受时回滚）。

        预分配 KV 完整克隆（含 length），回滚后位置索引从快照长度继续。
        """
        snap = Cache(self.num_layers, max_len=self.max_len)
        snap.attn_kv = [None if v is None else
                        (v[0].clone(), v[1].clone(), v[2], v[3]) for v in self.attn_kv]
        snap.conv_state = [None if v is None else v.clone() for v in self.conv_state]
        snap.rec_state = [None if v is None else v.clone() for v in self.rec_state]
        return snap

    def restore(self, snap: "Cache") -> None:
        """恢复快照状态。"""
        self.attn_kv = snap.attn_kv
        self.conv_state = snap.conv_state
        self.rec_state = snap.rec_state


class ExpertCache:
    """跨层共享的专家反量化缓存（全局条目上限，LRU 淘汰）。

    键：(layer_idx, expert_idx)；值：{"gate_proj": w, "up_proj": w, "down_proj": w}。
    ``max_entries`` 为总驻留专家数上限（与层数无关），保证缓存内存有硬上限
    （每专家约 12MB fp32，128 条 ≈ 1.5GB 封顶）。
    """

    def __init__(self, max_entries: int = 128):
        self.max_entries = max(1, max_entries)
        self._data: OrderedDict[tuple[int, int], dict[str, Tensor]] = OrderedDict()

    def get(self, key: tuple[int, int]) -> dict[str, Tensor] | None:
        entry = self._data.get(key)
        if entry is not None:
            self._data.move_to_end(key)
        return entry

    def put(self, key: tuple[int, int], entry: dict[str, Tensor]) -> None:
        self._data[key] = entry
        while len(self._data) > self.max_entries:
            self._data.popitem(last=False)          # LRU 淘汰最久未用

    def clear(self) -> None:
        self._data.clear()

    def __len__(self) -> int:
        return len(self._data)

    def bytes(self) -> int:
        """当前缓存占用字节数（内存审计）。"""
        return sum(t.numel() * t.element_size() for e in self._data.values() for t in e.values())


class PagedCache:
    """PagedAttention 分页 KV 缓存（参考 vLLM——物理块 + 块表映射 + prefix caching）。

    - 物理块：num_blocks × BLOCK_SIZE 的预分配 KV（按需分配——避免连续预留 O(ctx²)）
    - 块表：序列 token → 物理块映射（decode 时按块索引读写）
    - prefix caching：相同前缀 token 命中相同物理块（引用计数——多请求复用）
    """

    BLOCK_SIZE = 32  # vLLM 默认 16——用 32 减少块表开销

    def __init__(self, num_layers: int, num_kv_heads: int, head_dim: int,
                 num_blocks: int = 256):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_blocks = num_blocks
        self.block_size = self.BLOCK_SIZE
        # 物理块存储：list of (key_blocks, value_blocks) per layer——惰性分配
        self.blocks: list[dict[int, tuple]] = [{} for _ in range(num_layers)]
        self.free: set[int] = set(range(num_blocks))
        self.ref_count: dict[int, int] = {}     # 块引用计数（prefix caching 复用）
        self.prefix_cache: dict[tuple, int] = {}  # token 前缀 → 块（prefix caching）

    def alloc_block(self) -> int | None:
        """分配一个物理块（无空闲返回 None）。"""
        if not self.free:
            return None
        b = self.free.pop()
        self.ref_count[b] = 1
        return b

    def free_block(self, block: int) -> None:
        """释放物理块（引用计数归零后回收）。"""
        self.ref_count[block] = self.ref_count.get(block, 1) - 1
        if self.ref_count[block] <= 0:
            self.free.add(block)
            for layer in self.blocks:
                layer.pop(block, None)

    def prefix_match(self, token_ids: list[int]) -> int:
        """prefix caching：匹配已缓存前缀，返回复用长度（公共前缀长度）。"""
        common = 0
        # 从长到短匹配（先块级——再渐进前缀——支持短缓存前缀复用）
        for n in range(len(token_ids), 0, -1):
            if tuple(token_ids[:n]) in self.prefix_cache:
                common = n
                break
        return common

    def append(self, layer: int, block: int, pos: int, key, value) -> None:
        """写入物理块（block 内 pos 位置）。"""
        k_blk, v_blk = self.blocks[layer].get(block, (None, None))
        if k_blk is None:
            k_blk = key.new_empty((self.block_size, *key.shape))
            v_blk = value.new_empty((self.block_size, *value.shape))
            self.blocks[layer][block] = (k_blk, v_blk)
        k_blk[pos] = key
        v_blk[pos] = value

    def read(self, layer: int, block: int, start: int, end: int):
        """读取物理块 [start, end) 区间。"""
        k_blk, v_blk = self.blocks[layer][block]
        return k_blk[start:end], v_blk[start:end]

    def total_blocks(self) -> int:
        """已用块数。"""
        return self.num_blocks - len(self.free)
