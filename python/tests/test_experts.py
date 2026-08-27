"""tests.test_experts — 专家清单 + 读取器 + 流水线（端到端：mmap 段 + 数值对拍）。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ccut.experts.index import build_expert_index
from ccut.experts.pipeline import ExpertPipeline
from ccut.experts.reader import ExpertReader

REAL_MODEL = Path("python/models/Ornith-1.5-35B-A3B-MTP-FP8")


@pytest.fixture(scope="module")
def real_index():
    if not REAL_MODEL.exists():
        pytest.skip(f"Ornith checkpoint 不存在: {REAL_MODEL}")
    return build_expert_index(
        REAL_MODEL,
        "model.language_model.layers.{layer}.mlp.experts.{expert}.{proj}_proj",
        num_layers=40,
        num_experts=256,
        top_k=8,
        cache_path=REAL_MODEL.parent / ".kv_cache" / "expert_index.json",
    )


@pytest.mark.slow
def test_index_coverage(real_index):
    """40 × 256 = 10240 专家全覆盖。"""
    assert len(real_index.entries) == 40 * 256
    e3_7 = real_index.get(3, 7)
    assert e3_7 is not None
    assert "gate.weight" in e3_7.segments
    assert "gate.weight_scale" in e3_7.segments


@pytest.mark.slow
def test_reader_fill_dequant_matches_independent_reference(real_index):
    """ExpertReader._fill 结果 vs 独立 mmap 段解析（数值对拍）。"""
    from ccut.experts.reader import _Slot
    from ccut.io_.safetensors_io import SafetensorsFile, _bf16_bytes_to_float32
    from ccut.quant import kernels as qk

    r = ExpertReader(real_index, REAL_MODEL, 40, ring_slots=2, num_workers=0)
    e = real_index.get(3, 7)
    slot = _Slot(expert_id=7, layer=3)
    r._fill(3, 7, slot)
    assert slot.gate.shape == e.shapes["gate.weight"]
    sf = SafetensorsFile(r.model_dir / e.shard)
    try:
        for proj, s in zip(("gate", "up", "down"), (slot.gate, slot.up, slot.down)):
            seg = e.segments[f"{proj}.weight"]
            scale_seg = e.segments[f"{proj}.weight_scale"]
            raw = sf.read_range(seg[0], seg[1])
            u8 = np.frombuffer(raw, dtype=np.uint8).reshape(s.shape)
            sraw = sf.read_range(scale_seg[0], scale_seg[1])
            sc = _bf16_bytes_to_float32(np.frombuffer(sraw, dtype=np.uint16)).reshape(-1)
            ref = np.empty(s.shape, np.float32)
            qk.fp8_dequant_mat(u8, sc, ref)
            assert np.allclose(s, ref, atol=1e-6)
    finally:
        sf.close()
        r.close()


@pytest.mark.slow
def test_pipeline_step_returns_finite(real_index):
    """Smoke: 走一层 MoE 路由 + 1 专家读 + 融合；shape 正确 + finite。"""
    r = ExpertReader(real_index, REAL_MODEL, 40, ring_slots=2, num_workers=0)
    pipe = ExpertPipeline(r, num_workers=1, speculative_window=4, prefetch_steps_ahead=2)
    rng = np.random.RandomState(0)
    seq, hidden = 2, 512
    x = (rng.randn(seq, hidden) * 0.05).astype(np.float32)
    gate_w = (rng.randn(256, hidden) * 0.02).astype(np.float32)
    # 仅验证 ring hit 路径（先用 prefetch 预热再 step，避免大 IO 阻塞测试）
    # 注：topk=8 路由 + 8 个不同专家都要读端，测试延迟主要来自磁盘
    y = pipe.step(3, x, gate_w, top_k=2, norm_topk_prob=True)  # top_k=2 减半读
    assert y.shape == (seq, hidden)
    assert np.isfinite(y).all()
    assert pipe.metrics is not None
    assert pipe.metrics.steps >= 1
    pipe.close()
