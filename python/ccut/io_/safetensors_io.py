"""ccut.io_.safetensors_io — 轻量 safetensors 读取器（mmap、按段、零拷贝）。

设计（§3.2 权重布局利用）：
- safetensors = 8B 小端长度 + JSON 头 + 连续 tensor 段。本读取器**只解析头**，
  每个 tensor 是文件内一段 ``(offset, length)``。
- ``mmap`` 后读 tensor 不产生「驻留」——OS 页缓存按页管理，进程私有内存里只有
  正在算的拷贝。``view()`` 返回零拷贝 numpy 视图（只读）；``read()`` 返回私有拷贝。
- 与 ``safetensors`` 库读数一致性由 ``tests/test_safetensors_io.py`` 对拍。
"""

from __future__ import annotations

import json
import mmap
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = ["TensorView", "SafetensorsFile", "load_index"]

# safetensors dtype → numpy dtype
_DTYPE_MAP = {
    "F64": np.float64,
    "F32": np.float32,
    "F16": np.float16,
    # BF16：numpy 无原生 bfloat16（2.x 亦然）→ 按 16-bit 无符号读，
    # 需要 float 语义时走 read_bf16_as_float32（位操作）或 torch.bfloat16。
    "BF16": np.uint16,
    "I64": np.int64,
    "U64": np.uint64,
    "I32": np.int32,
    "U32": np.uint32,
    "I16": np.int16,
    "U16": np.uint16,
    "I8": np.int8,
    "U8": np.uint8,
    "BOOL": np.bool_,
    "F8_E4M3": np.uint8,  # FP8 以字节读，dequant 时转（CPU 无原生 FP8 dtype）
    "F8_E5M2": np.uint8,
}


def _resolve_dtype(safetensors_dtype: str) -> np.dtype:
    if safetensors_dtype == "BF16":
        # numpy 2.x 无原生 bfloat16；BF16 段按 16-bit 无符号读，
        # 需要 float 语义时由调用方走 torch（torch.bfloat16）或 _bf16_to_float32。
        return np.dtype(np.uint16)
    return np.dtype(_DTYPE_MAP[safetensors_dtype])


def _bf16_bytes_to_float32(raw_u16: np.ndarray) -> np.ndarray:
    """BF16 字节 → float32（纯 numpy：高 16 位左移拼接）。"""
    u16 = raw_u16.astype(np.uint32)
    f32_bits = (u16 << 16)
    return f32_bits.view(np.float32)


@dataclass(frozen=True)
class TensorView:
    """文件内一段 tensor 的元信息（不含数据）。"""

    name: str
    dtype: str  # safetensors 原始 dtype 名
    shape: tuple[int, ...]
    offset: int  # 相对数据区起点
    length: int  # 字节数
    ndim: int = 0

    @property
    def np_dtype(self) -> np.dtype:
        return _resolve_dtype(self.dtype)

    @property
    def numel(self) -> int:
        n = 1
        for d in self.shape:
            n *= int(d)
        return n

    def verify(self) -> None:
        expected = self.numel * self.np_dtype.itemsize
        if expected != self.length:
            raise ValueError(f"{self.name}: 声明 {self.length}B 与 shape/dtype 推算 {expected}B 不符")


class SafetensorsFile:
    """mmap 只读一个 safetensors 分片。

    句柄 ≠ 驻留数据：16 个 shard 各持 1 个 mmap 句柄常驻，页缓存按页淘汰。
    """

    def __init__(self, path: str | Path, use_mmap: bool = True):
        self.path = Path(path)
        self._use_mmap = use_mmap
        self._fh = open(self.path, "rb")
        self._mm: mmap.mmap | None = None
        self._header: dict = {}
        self._tensors: dict[str, TensorView] = {}
        self._data_start = 0
        self._parse_header()

    # -- 头解析 -------------------------------------------------------------
    def _parse_header(self) -> None:
        header_len_bytes = self._fh.read(8)
        if len(header_len_bytes) < 8:
            raise ValueError(f"{self.path}: 文件过短，非 safetensors")
        (header_len,) = struct.unpack("<Q", header_len_bytes)
        if header_len > 64 * 1024 * 1024:
            raise ValueError(f"{self.path}: 头过大（{header_len}B），疑似损坏")
        raw_header = self._fh.read(header_len)
        meta = json.loads(raw_header.decode("utf-8"))
        self._data_start = 8 + header_len
        self._header = meta.pop("__metadata__", {})
        for name, info in meta.items():
            shape = tuple(int(d) for d in info["shape"])
            offset = int(info["data_offsets"][0])
            length = int(info["data_offsets"][1]) - offset
            view = TensorView(
                name=name,
                dtype=info["dtype"],
                shape=shape,
                offset=offset,
                length=length,
                ndim=len(shape),
            )
            view.verify()
            self._tensors[name] = view

    # -- 访问 ---------------------------------------------------------------
    @property
    def metadata(self) -> dict:
        return dict(self._header)

    @property
    def names(self) -> list[str]:
        return list(self._tensors)

    def __contains__(self, name: str) -> bool:
        return name in self._tensors

    def __len__(self) -> int:
        return len(self._tensors)

    def tensor(self, name: str) -> TensorView:
        try:
            return self._tensors[name]
        except KeyError:
            raise KeyError(f"{self.path}: 无 tensor {name!r}") from None

    def _buffer(self, view: TensorView) -> memoryview | bytes:
        abs_offset = self._data_start + view.offset
        if self._use_mmap and self._mm is None:
            size = self.path.stat().st_size
            if size == 0:
                raise ValueError(f"{self.path}: 空文件")
            try:
                self._mm = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)
            except OSError:
                # 某些 FS 不支持 mmap → 回退普通读
                self._use_mmap = False
        if self._mm is not None:
            return self._mm[abs_offset : abs_offset + view.length]
        self._fh.seek(abs_offset)
        return self._fh.read(view.length)

    def view(self, name: str) -> np.ndarray:
        """零拷贝 numpy 视图（只读；仅 mmap 模式安全复用，非 mmap 模式每次新读）。"""
        v = self.tensor(name)
        buf = self._buffer(v)
        if self._use_mmap and self._mm is not None:
            arr = np.frombuffer(buf, dtype=v.np_dtype, count=v.numel)
            return arr.reshape(v.shape)
        # 非 mmap：内存副本（仍为 ndarray）
        return np.frombuffer(bytes(buf), dtype=v.np_dtype, count=v.numel).reshape(v.shape)

    def read(self, name: str, dtype: str | None = None) -> np.ndarray:
        """私有拷贝（可写）。``dtype`` 可选重解释（如 BF16 段 → float32）。"""
        v = self.tensor(name)
        arr = self.view(name).copy()
        if dtype is not None:
            arr = arr.astype(_resolve_dtype(dtype))
        return arr

    def read_bf16_as_float32(self, name: str) -> np.ndarray:
        """BF16 段 → float32（纯 numpy 位操作）。"""
        v = self.tensor(name)
        if v.dtype != "BF16":
            raise ValueError(f"{name}: 非 BF16 段（{v.dtype}）")
        raw = self.view(name).astype(np.uint16)
        return _bf16_bytes_to_float32(raw).reshape(v.shape)

    def read_range(self, offset: int, length: int) -> bytes:
        """按绝对数据区偏移读任意字节段（专家权重段用，§3.2）。"""
        abs_offset = self._data_start + offset
        if self._mm is not None:
            return bytes(self._mm[abs_offset : abs_offset + length])
        self._fh.seek(abs_offset)
        return self._fh.read(length)

    def close(self) -> None:
        if self._mm is not None:
            try:
                self._mm.close()
            except ValueError:
                pass
            self._mm = None
        try:
            self._fh.close()
        except OSError:
            pass

    def __enter__(self) -> "SafetensorsFile":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# ---------------------------------------------------------------------------
# index.json（model.safetensors.index.json）解析
# ---------------------------------------------------------------------------


def load_index(model_dir: str | Path) -> dict:
    """解析 ``model.safetensors.index.json`` → ``{tensor_name: shard_filename}``。

    返回 ``{"weight_map": {...}, "metadata": {...}}``；无 index 时按存在的 shard 兜底
    （单分片 ``model.safetensors`` 优先；否则扫 ``*.safetensors`` 第一张）。
    """
    model_dir = Path(model_dir)
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path, "rb") as fh:
            data = json.load(fh)
        return {
            "weight_map": data.get("weight_map", {}),
            "metadata": {k: v for k, v in data.items() if k != "weight_map"},
        }
    # 无 index：单文件约定优先（HF 习惯），否则扫 *.safetensors
    single = model_dir / "model.safetensors"
    if single.exists():
        return {"weight_map": {}, "metadata": {"single_file": single.name}}
    shards = sorted(model_dir.glob("*.safetensors"))
    if shards:
        return {"weight_map": {}, "metadata": {"single_file": shards[0].name}}
    raise FileNotFoundError(
        f"{model_dir}: 无 model.safetensors.index.json / model.safetensors / 任何 .safetensors"
    )


def iter_shards(model_dir: str | Path) -> list[Path]:
    """按 index 列出全部 shard 文件路径（去重保序）。"""
    info = load_index(model_dir)
    model_dir = Path(model_dir)
    seen: list[Path] = []
    for shard in info["weight_map"].values():
        p = model_dir / shard
        if p.exists() and p not in seen:
            seen.append(p)
    if not seen and info["metadata"].get("single_file"):
        p = model_dir / info["metadata"]["single_file"]
        if p.exists() and p not in seen:
            seen.append(p)
    return seen
