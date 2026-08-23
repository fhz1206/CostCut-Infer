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
