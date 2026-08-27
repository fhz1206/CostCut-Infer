"""tests.test_quant — 量化子系统冒烟（K5 端到端：compressed-tensors 解析 + 各 method W8A16 数值）。"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from ccut.quant import kernels
from ccut.quant.kv import (
    kv_bytes_per_token,
    quantize_kv_token,
    dequantize_kv_token,
    resolve_kv_dtype,
)
from ccut.quant.method import make_method_for_spec
from ccut.quant.online import quantize_buffer_inplace
from ccut.quant.registry import (
    ONLINE_SHORTHANDS,
    list_supported_quant,
    resolve_checkpoint_quant,
)
from ccut.quant.spec import K_BF16, LayerQuantSpec, ScaleDesc, get_quant_key


def test_fp8_e4m3_round_trip():
    """FP8 e4m3 与 torch 参考全值域对拍。"""
    torch.manual_seed(0)
    x = (torch.randn(64, 32) * 3).numpy()
    scale = np.abs(x).max(axis=0) / 448.0
    q = np.clip(x / scale[None, :], -448, 448)
    codes = kernels.float32_to_fp8_e4m3(q.ravel())
    dec = kernels.fp8_e4m3_to_float32(codes).reshape(x.shape) * scale[None, :]
    tr = torch.from_numpy(q).to(torch.float8_e4m3fn).float().numpy() * scale[None, :]
    mask = np.abs(tr) > 1e-3
    rel = np.abs(dec[mask] - tr[mask]) / np.abs(tr[mask])
    assert float(rel.max()) < 0.05  # FP8 量化后 5% 容差


def test_int8_w8a8_matches_torch():
    """INT8 W8A8 整数累加 vs torch 参考。"""
    torch.manual_seed(1)
    xa = torch.randn(4, 16)
    wb = torch.randint(-127, 127, (16, 8)).float()
    scale_t = torch.rand(8) + 0.1
    sc = xa.abs().max(dim=1).values / 127
    xq = torch.clamp(torch.round(xa * (127 / sc)[:, None]), -127, 127).to(torch.int8)
    acc = xq.long() @ wb.to(torch.int8).long()
    y_ref = acc.float() * sc[:, None] * scale_t[None, :]
    from ccut.quant.int8 import _quantize_per_token, _int8_matmul

    xq2, xs2 = _quantize_per_token(xa.numpy())
    acc2 = _int8_matmul(xq2, wb.numpy())
    y = acc2.astype(np.float32) * xs2[:, None] * scale_t.numpy()[None, :]
    assert np.allclose(y, y_ref.numpy(), atol=1e-3)


def test_compressed_tensors_ignore_rules():
    """Ornith 7 条 ignore 正则：lm_head / embed_tokens / router / shared_expert_gate /
    linear_attn 全部 / visual 全部 → BF16 直通。"""
    cfg = resolve_checkpoint_quant("python/models/Ornith-1.5-35B-A3B-MTP-FP8")
    assert cfg.is_layer_skipped("lm_head")
    assert cfg.is_layer_skipped("model.language_model.embed_tokens")
    assert cfg.is_layer_skipped("model.language_model.layers.0.mlp.gate")
    assert cfg.is_layer_skipped("model.language_model.layers.0.linear_attn.in_proj_qkv")
    assert not cfg.is_layer_skipped("model.language_model.layers.3.self_attn.q_proj")
    assert not cfg.is_layer_skipped("model.language_model.layers.3.mlp.experts.0.gate_proj")


def test_online_quant_mutually_exclusive():
    """checkpoint 量化 + 在线量化简写互斥。"""
    with pytest.raises(ValueError):
        resolve_checkpoint_quant("python/models/Ornith-1.5-35B-A3B-MTP-FP8", online_quantization="fp8_per_token")


def test_online_quant_unknown_rejected():
    """未知简写显式报错（不静默）。"""
    with tempfile.TemporaryDirectory() as d:
        Path(d, "config.json").write_text(json.dumps({"architectures": ["LlamaForCausalLM"]}))
        with pytest.raises(ValueError):
            resolve_checkpoint_quant(d, online_quantization="NOPE")


def test_supported_quant_registered():
    names = list_supported_quant()
    assert "compressed-tensors" in names


def test_kv_bf16_vs_fp8_factor_2x():
    bf16 = kv_bytes_per_token(2, 256, "bf16")
    fp8 = kv_bytes_per_token(2, 256, "fp8")
    assert bf16 == 2 * 2 * 256 * 2
    # fp8 应约为 bf16 的一半（±scale 开销）
    assert 0.4 <= fp8 / bf16 <= 0.6


def test_kv_dtype_auto_for_ornith():
    """Ornith kv_cache_scheme=null → BF16（无 fp8 显式声明）。"""
    mode = resolve_kv_dtype("python/models/Ornith-1.5-35B-A3B-MTP-FP8", "auto")
    assert mode == "bf16"


def test_method_dispatch_for_compressed_tensors_layer():
    """method.make_method_for_spec 路由：FP8 → Fp8LinearMethod。"""
    cfg = resolve_checkpoint_quant("python/models/Ornith-1.5-35B-A3B-MTP-FP8")
    spec = cfg.get_layer_spec("model.language_model.layers.0.mlp.experts.0.gate_proj")
    m = make_method_for_spec(spec)
    assert m.compute_path == "w8a16"


def test_w4a16_dequant_correctness():
    """int4 weight-only dequant vs 参考。"""
    torch.manual_seed(13)
    w4 = torch.randint(0, 255, (10, 16), dtype=torch.uint8).numpy()
    s4 = (torch.rand(32) + 0.1).numpy()
    from ccut.quant.weight_only import WeightOnlyMethod

    m = WeightOnlyMethod(
        LayerQuantSpec(
            "t.int4",
            get_quant_key("int4_w4a16_sym"),
            scales=(ScaleDesc(name="t.int4.weight_scale", dtype="F32", shape=()),),
        )
    )
    x = torch.randn(4, 32).numpy()
    y = m.apply(w4.tobytes(), {"t.int4.weight_scale": s4}, x)
    # 参考（位级正确版：用 np.int8 中间结果避免 uint8 减法回绕）
    packed = w4.copy()
    lo = np.empty(packed.shape, dtype=np.int8)
    hi = np.empty(packed.shape, dtype=np.int8)
    for i in range(packed.shape[0]):
        for j in range(packed.shape[1]):
            b = int(packed[i, j])
            l = b & 0x0F
            h = (b >> 4) & 0x0F
            if l >= 8:
                l -= 16
            if h >= 8:
                h -= 16
            lo[i, j] = l
            hi[i, j] = h
    W = np.empty((10, 32), dtype=np.float32)
    W[:, 0::2] = lo.astype(np.float32) * s4[0::2]
    W[:, 1::2] = hi.astype(np.float32) * s4[1::2]
    ref = x @ W.T
    assert np.allclose(y, ref, atol=1e-5)


def test_online_quantize_bf16_buffer():
    """BF16 字节段 → FP8 码 + per-channel scale 往返。"""
    torch.manual_seed(7)
    wb = torch.randn(8, 16) * 0.1
    bf16_bytes = wb.bfloat16().view(torch.int16).numpy().tobytes()
    sc = (wb.abs().max(dim=1).values / 448.0).numpy()
    out = np.empty((8, 16), dtype=np.uint8)
    quantize_buffer_inplace(bf16_bytes, sc, out)
    back = kernels.fp8_e4m3_to_float32(out.ravel()).reshape(8, 16) * sc[:, None]
    rel = np.abs(back - wb.numpy()) / np.maximum(np.abs(wb.numpy()), 1e-2)
    assert float(rel.max()) < 0.5  # FP8 量化 50% 容差
