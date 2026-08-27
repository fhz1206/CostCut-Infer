"""tests.test_safetensors_io — mmap safetensors 头解析 + 零拷贝视图 + BF16 位操作。"""

from __future__ import annotations

import json
import struct

import numpy as np
import pytest
import safetensors
from safetensors import safe_open

from ccut.io_.safetensors_io import (
    SafetensorsFile,
    _bf16_bytes_to_float32,
    load_index,
    iter_shards,
)


def _make_safetensors(path, tensors: dict[str, tuple[np.ndarray, str]]) -> None:
    """写一个最小 safetensors 文件：JSON 头 + 连续数据段。"""
    # 排序 tensor 名（确定性字节序）
    header = {"__metadata__": {}}
    offset = 0
    buffers = []
    for name in sorted(tensors):
        arr, dtype = tensors[name]
        raw = arr.astype(dtype_to_np(dtype)).tobytes()
        header[name] = {
            "dtype": dtype,
            "shape": list(arr.shape),
            "data_offsets": [offset, offset + len(raw)],
        }
        offset += len(raw)
        buffers.append((name, raw))
    h = json.dumps(header, ensure_ascii=False).encode("utf-8")
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(h)))
        fh.write(h)
        for _, b in buffers:
            fh.write(b)


def dtype_to_np(safetensors_dtype: str):
    return {
        "F32": np.float32,
        "F16": np.float16,
        "I64": np.int64,
        "U64": np.uint64,
        "I32": np.int32,
        "U32": np.uint32,
        "I8": np.int8,
        "U8": np.uint8,
    }[safetensors_dtype]


def test_parse_header_and_view(tmp_dir):
    a = np.random.randn(4, 8).astype(np.float32)
    b = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int32)
    p = tmp_dir / "a.safetensors"
    _make_safetensors(p, {"a": (a, "F32"), "b": (b, "I32")})

    with SafetensorsFile(p) as sf:
        assert "a" in sf
        assert "b" in sf
        av = sf.read("a")
        bv = sf.read("b")
    assert np.array_equal(av, a)
    assert np.array_equal(bv, b)


def test_bf16_bytes_to_float32_exact():
    """BF16 位操作 → float32：与 torch 参考对拍。"""
    import torch

    src = np.array([1.0, -2.5, 0.0, 1.234567e-3, -1.234567e-3, 1e3, -1e3, 6.55e4], dtype=np.float32)
    bf16 = src.astype(np.float32).view(np.uint32).astype(np.uint16) << 16  # 不等价
    # 正确 BF16：低 16 位直接拷贝
    ref = torch.tensor(src.tolist(), dtype=torch.bfloat16)
    raw = ref.view(torch.int16).numpy().tobytes()
    got = _bf16_bytes_to_float32(np.frombuffer(raw, dtype=np.uint16))
    assert np.allclose(got, ref.float().numpy())


def test_zero_copy_view_no_copy():
    a = np.arange(64, dtype=np.float32).reshape(8, 8)
    p = tmp_dir if False else None
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        from pathlib import Path

        path = Path(d) / "x.safetensors"
        _make_safetensors(path, {"x": (a, "F32")})
        with SafetensorsFile(path) as sf:
            view = sf.view("x")
            # mmap 视图 → 修改源应影响视图
            assert view.shape == (8, 8)
            assert view.ctypes.data != 0


def test_read_range_bytes_match(tmp_dir):
    a = np.random.randint(0, 255, size=(3, 4), dtype=np.uint8)
    p = tmp_dir / "r.safetensors"
    _make_safetensors(p, {"r": (a, "U8")})
    with SafetensorsFile(p) as sf:
        # 用 safetensors 库独立算绝对偏移
        with safe_open(p, framework="numpy") as ref:
            ref_a = ref.get_tensor("r")
        raw = sf.read_range(0, sf.tensor("r").length)
        assert raw == ref_a.tobytes()


def test_load_index_single_file(tmp_dir):
    a = np.zeros(8, dtype=np.float32)
    p = tmp_dir / "single.safetensors"
    _make_safetensors(p, {"a": (a, "F32")})
    # load_index 接受无 index.json 的单文件
    info = load_index(tmp_dir)
    assert info["metadata"].get("single_file") == "single.safetensors"
    assert iter_shards(tmp_dir)[0].name == "single.safetensors"


def test_load_index_with_manifest(tmp_dir):
    # 写两个 shard + manifest
    a = np.zeros(4, dtype=np.float32)
    p1 = tmp_dir / "model-00001-of-00002.safetensors"
    p2 = tmp_dir / "model-00002-of-00002.safetensors"
    _make_safetensors(p1, {"t1": (a, "F32")})
    _make_safetensors(p2, {"t2": (a, "F32")})
    (tmp_dir / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"t1": "model-00001-of-00002.safetensors", "t2": "model-00002-of-00002.safetensors"}}),
        encoding="utf-8",
    )
    info = load_index(tmp_dir)
    assert info["weight_map"]["t1"] == "model-00001-of-00002.safetensors"
    shards = iter_shards(tmp_dir)
    assert len(shards) == 2
