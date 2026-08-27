"""tests.test_blocks — 共享积木（norm/rope/attn/gdn/mla/moe/heads）数值对拍 torch。"""

from __future__ import annotations

import math

import numpy as np
import torch

from ccut.blocks.attn_gdn import GDNState, gdn_step, short_conv1d
from ccut.blocks.attn_gqa import build_causal_mask, gqa_attention_fast
from ccut.blocks.attn_mla import MLAState, mla_decode, mla_prefill
from ccut.blocks.heads import temperature_topk
from ccut.blocks.moe import topk_softmax
from ccut.blocks.norm import rms_norm
from ccut.blocks.rope import apply_rope, build_rope


def test_rms_norm_matches_torch():
    np.random.seed(0)
    x = np.random.randn(4, 64).astype(np.float32)
    w = 1 + np.random.randn(64).astype(np.float32)
    y = rms_norm(x, w, 1e-6)
    ref = (x / np.sqrt((x * x).mean(-1, keepdims=True) + 1e-6)) * w
    assert np.allclose(y, ref, atol=1e-5)


def test_rope_none_scaling_pos0_identity():
    """位置 0 RoPE 应为恒等变换。"""
    inv, _ = build_rope(64, 10000.0, 1024)
    q = np.random.randn(1, 8, 1, 64).astype(np.float32)
    k = np.random.randn(1, 2, 1, 64).astype(np.float32)
    pos = np.array([[0]], dtype=np.int64)
    qr, kr = apply_rope(q, k, inv, pos)
    assert np.allclose(qr[0, :, 0], q[0, :, 0], atol=1e-5)
    assert np.allclose(kr[0, :, 0], k[0, :, 0], atol=1e-5)


def test_rope_matches_torch_reference():
    inv, _ = build_rope(64, 10000.0, 1024)
    q = np.random.randn(1, 8, 4, 64).astype(np.float32)
    k = np.random.randn(1, 2, 4, 64).astype(np.float32)
    pos = np.arange(4)[None, :]
    qr, kr = apply_rope(q, k, inv, pos)
    hf = 32
    freqs = torch.arange(0, 64, 2).float() / 64
    invf = 10000.0 ** (-freqs)
    angs = torch.arange(4)[:, None].float() * invf[None, :]
    cos, sin = angs.cos(), angs.sin()

    def rot(t):
        t1, t2 = t[..., :hf], t[..., hf:]
        c, s_ = cos[None], sin[None]
        return torch.cat([t1 * c - t2 * s_, t2 * c + t1 * s_], -1)

    assert np.allclose(qr[0], rot(torch.tensor(q[0])).numpy(), atol=1e-4)
    assert np.allclose(kr[0], rot(torch.tensor(k[0])).numpy(), atol=1e-4)


def test_rope_linear_scaling_halves_freq():
    inv, _ = build_rope(64, 10000.0, 2048, {"type": "linear", "factor": 2.0})
    inv_none, _ = build_rope(64, 10000.0, 1024)
    assert np.allclose(inv, inv_none / 2.0)


def test_gqa_attention_vs_torch():
    """GQA 注意力 vs torch 参考（causal）。"""
    torch.manual_seed(1)
    b, h, s, d, kv = 1, 8, 8, 16, 2
    qt = torch.randn(b, h, s, d)
    kt = torch.randn(b, kv, s, d)
    vt = torch.randn(b, kv, s, d)
    mask = build_causal_mask(s, 0)
    out = gqa_attention_fast(qt.numpy(), kt.numpy(), vt.numpy(), mask)
    kx = torch.repeat_interleave(kt, h // kv, dim=1)
    vx = torch.repeat_interleave(vt, h // kv, dim=1)
    sc = (qt @ kx.transpose(-1, -2)) / np.sqrt(d) + torch.tensor(mask).float()
    pr = torch.softmax(sc, -1)
    reft = (pr @ vx).numpy()
    assert np.allclose(out, reft, atol=1e-4)


def test_gdn_step_matches_python_reference():
    """GDN 递推核 vs 纯 Python 参考（per-step）。"""
    kd, vd, seq = 8, 6, 6
    np.random.seed(3)
    q_t = np.random.randn(seq, kd).astype(np.float32)
    k_t = np.random.randn(seq, kd).astype(np.float32)
    v_t = np.random.randn(seq, vd).astype(np.float32)
    a_t = np.random.randn(seq).astype(np.float32)
    b_t = np.random.randn(seq).astype(np.float32)
    a_log = -0.5
    st = GDNState(1, kd, vd)
    outs = []
    for t in range(seq):
        o = np.empty(vd, np.float32)
        gdn_step(st.states[0], q_t[t], k_t[t], v_t[t], float(a_t[t]), float(b_t[t]), a_log, o)
        outs.append(o)

    sp = lambda x: x if x > 20 else (0.0 if x < -20 else math.log1p(math.exp(x)))
    ref = np.zeros((kd, vd), np.float32)
    outs2 = []
    for t in range(seq):
        decay = math.exp(-sp(float(a_t[t])) + a_log)
        bb = 1 / (1 + math.exp(-float(b_t[t])))
        for j in range(vd):
            vh = ref[:, j] @ k_t[t]
            delta = (v_t[t, j] - vh) * bb
            ref[:, j] = ref[:, j] * decay + k_t[t] * delta
        outs2.append(ref.T @ q_t[t])
    assert np.allclose(np.array(outs), np.array(outs2), atol=1e-5)


def test_short_conv1d_causal():
    """causal conv1d: out[t, c] = sum_{j=0..K-1} w[c, j] * x[t-j, c]（t-j<0 补 0）。"""
    np.random.seed(4)
    x = np.random.randn(6, 3).astype(np.float32)
    w = np.random.randn(3, 4).astype(np.float32)
    y = short_conv1d(x, w, 4)
    assert y.shape == (6, 3)
    # 参考：t=0 → 仅 j=0 → y[0,c] = w[c,0] * x[0,c]
    assert np.allclose(y[0], w[:, 0] * x[0])
    # t=1 → j=0,1 → y[1,c] = w[c,0]*x[1,c] + w[c,1]*x[0,c]
    assert np.allclose(y[1], w[:, 0] * x[1] + w[:, 1] * x[0])


def test_moe_topk_softmax_normalized():
    np.random.seed(5)
    gl = np.random.randn(6, 32).astype(np.float32)
    w, idx = topk_softmax(gl, 4)
    assert np.allclose(w.sum(axis=1), 1.0, atol=1e-5)
    assert all(len(set(idx[i].tolist())) == 4 for i in range(6))
    # 降序
    for i in range(6):
        assert bool((np.diff(w[i]) <= 1e-6).all())


def test_temperature_topk_constraints():
    np.random.seed(6)
    lg = np.random.randn(3, 100).astype(np.float32)
    p = temperature_topk(lg, 1.0, top_k=5, top_p=0.9, min_p=0.05)
    assert int((p[0] > 0).sum()) <= 5
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-5)


def test_mla_prefill_causal():
    """MLA prefill causal 正确性：位置 0 只见自身，后续位置修改不应影响 out[0]。"""
    rank, d_rope, heads, d_nope, d_v = 8, 4, 4, 6, 5
    st = MLAState(rank, d_rope, max_tokens=16)
    seq = 4
    np.random.seed(7)
    x = np.random.randn(seq, 10).astype(np.float32)
    kv_a_proj = np.random.randn(10, rank + d_rope).astype(np.float32)
    q_pe = np.random.randn(seq, d_rope).astype(np.float32)
    c = x @ kv_a_proj[..., :rank]
    st.append(c, q_pe)
    q_nope = np.random.randn(seq, heads, d_nope).astype(np.float32)
    q_pe_r = np.random.randn(seq, heads, d_rope).astype(np.float32)
    k_up = np.random.randn(rank, heads * d_nope).astype(np.float32)
    v_up = np.random.randn(rank, heads * d_v).astype(np.float32)
    out = mla_prefill(st, q_nope, q_pe_r, k_up, v_up, scale=0.1, start_pos=0)
    assert out.shape == (seq, heads, d_v)
    assert np.isfinite(out).all()
    # causal 正确性：构造 st2 改后续 c，out[0] 应不变
    st2 = MLAState(rank, d_rope, max_tokens=16)
    c2 = x @ kv_a_proj[..., :rank]
    c2[1:] *= 5  # 修改后续 token
    st2.append(c2, q_pe)
    out2 = mla_prefill(st2, q_nope, q_pe_r, k_up, v_up, scale=0.1, start_pos=0)
    assert np.allclose(out[0], out2[0], atol=1e-4)
