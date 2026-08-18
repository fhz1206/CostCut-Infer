"""safetensors 惰性权重读取：WeightStore（内存友好）。

设计要点：
- 只解析 index.json 的 weight_map（几十 KB），不加载任何权重
- 每个 safetensors 分片用 ``safe_open`` 打开（mmap 惰性），仅当请求某张量时才
  真正读入数据；未请求的张量不占内存
- 提供 ``get_slice`` 局部读取能力，为后续专家流式（按需取行/组）预留
- ``close()`` 释放全部分片句柄
"""
from __future__ import annotations

from json import load
from pathlib import Path
from typing import Any, Optional

from safetensors import safe_open
from numpy import ndarray

__all__ = ["WeightStore"]


class WeightStore:
    """按 index.json 惰性读取 safetensors 权重的存储。

    Args:
        model_dir: 模型目录（含 model.safetensors.index.json 与分片文件）。
        index_name: 索引文件名（默认 model.safetensors.index.json）。
    """

    def __init__(self, model_dir: str, index_name: str = "model.safetensors.index.json"):
        self.model_dir = Path(model_dir)
        self.index_name = index_name
        self.metadata: dict[str, Any] = {}
        self._weight_map: dict[str, str] = {}
        self._shards: dict[str, Any] = {}
        self._load_index()

    # ---- 索引 ----

    def _load_index(self) -> None:
        index_path = self.model_dir / self.index_name
        if index_path.exists():
            with open(index_path, "r", encoding="utf-8") as f:
                data = load(f)
            self._weight_map = dict(data.get("weight_map", {}))
            self.metadata = dict(data.get("metadata", {}))
        else:
            # 单文件模型（无 index.json，如 speculator.dspark）：全部张量映射到 model.safetensors
            single = self.model_dir / "model.safetensors"
            if not single.exists():
                raise FileNotFoundError(f"找不到权重索引 {self.index_name} 或单文件权重")
            with safe_open(str(single), framework="numpy") as f:
                self._weight_map = {name: "model.safetensors" for name in f.keys()}
            self.metadata = {}

    def keys(self) -> list[str]:
        """全部张量名（不读取任何数据）。"""
        return list(self._weight_map)

    def has(self, name: str) -> bool:
        return name in self._weight_map

    def shard_of(self, name: str) -> str:
        return self._weight_map[name]

    # ---- 张量读取 ----

    def _open_shard(self, shard: str) -> Any:
        handle = self._shards.get(shard)
        if handle is None:
            handle = safe_open(str(self.model_dir / shard), framework="numpy")
            self._shards[shard] = handle
        return handle

    def _require(self, name: str) -> str:
        shard = self._weight_map.get(name)
        if shard is None:
            raise KeyError(f"张量 {name!r} 不在索引 {self.index_name} 中")
        return shard

    def get(self, name: str) -> ndarray:
        """读取单个张量（仅此张量实际加载入内存）。"""
        shard = self._require(name)
        try:
            return self._open_shard(shard).get_tensor(name)
        except TypeError:
            # bf16 张量（numpy 无原生 bf16，如 speculator.dspark）：经 torch 框架读取转 float32
            import torch
            with safe_open(str(self.model_dir / shard), framework="torch") as f:
                return f.get_tensor(name).float().numpy()

    def get_slice(self, name: str) -> Any:
        """返回 safetensors.Slice：可按区间局部读取（专家流式用）。"""
        shard = self._require(name)
        return self._open_shard(shard).get_slice(name)

    def tensor_info(self, name: str) -> tuple[list[int], str]:
        """张量形状与 dtype（不读取数据）。"""
        sl = self.get_slice(name)
        return list(sl.get_shape()), sl.get_dtype()

    # ---- 生命周期 ----

    def close(self) -> None:
        for handle in self._shards.values():
            try:
                handle.close()
            except Exception:
                pass
        self._shards.clear()

    def __enter__(self) -> "WeightStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()
