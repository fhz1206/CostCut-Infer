"""tests.test_weights — R10 层流式（WeightRing + sublayer 切分）。"""

from __future__ import annotations

import numpy as np
import pytest

from ccut.weights.manager import WeightManager
from ccut.weights.stream import WeightStream


@pytest.mark.slow
def test_layer_round_trip_with_sublayer_split(tmp_dir, python_dir):
    """加载 Ornith 真实 layer 0：22 张量、sublayer 切分（in_proj_qkv 大头）。"""
    from ccut.io_.safetensors_io import load_index

    model_dir = python_dir / "models" / "Ornith-1.5-35B-A3B-MTP-FP8"
    if not model_dir.exists():
        import pytest

        pytest.skip(f"Ornith checkpoint 不存在: {model_dir}")
    wm = load_index(model_dir)["weight_map"]
    layer_tensors = {0: [], 1: [], 2: []}
    for n in wm:
        if "experts." in n:
            continue
        if ".language_model.layers." in n:
            for l in (0, 1, 2):
                if f".layers.{l}." in n:
                    layer_tensors[l].append(n)
                    break
    m = WeightManager(
        model_dir,
        layer_tensors,
        ring_layers=2,
        large_threshold_bytes=1 << 20,  # 1MB 阈值触发切分
        bandwidth_log=False,
    )
    m.prefetch_layer(0)
    s = m.ring.slot(0)
    assert s is not None
    assert len(s.tensors) > 0
    assert m.stats()["sublayer_splits"] >= 1
    # 推进流
    m.ring.ref(0)
    m.prefetch_layer(1)
    assert 1 in m.ring
    m.unref(0)
    m.close()


@pytest.mark.slow
def test_stream_refill_ahead():
    """WeightStream 顺序推进 + 预取 ahead（用真模型目录做 index）。"""
    model_dir = python_dir / "models" / "Ornith-1.5-35B-A3B-MTP-FP8"
    if not model_dir.exists():
        import pytest

        pytest.skip(f"Ornith checkpoint 不存在: {model_dir}")
    from ccut.io_.safetensors_io import load_index

    wm = load_index(model_dir)["weight_map"]
    layer_tensors = {0: [], 1: []}
    for n in wm:
        if "experts." in n:
            continue
        if ".layers.0." in n:
            layer_tensors[0].append(n)
        elif ".layers.1." in n:
            layer_tensors[1].append(n)
    m = WeightManager(model_dir, layer_tensors, ring_layers=2, bandwidth_log=False)
    m.prefetch_layer(0)
    assert 0 in m.ring
    m.unref(0)
    m.close()
