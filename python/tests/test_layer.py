"""M2 单层前向单元测试。

覆盖（组件数学对照 + 真实权重 sanity）：
- RMSNorm / RMSNormGated 与参考公式对照（fp32 精确）
- RoPE：rotate_half / apply_rotary_pos_emb（partial rotary 旋转 + 透传）
- GatedDeltaNet delta rule：chunk 版 vs recurrent 版互证（负 g 衰减语义）
- eager_attention vs torch.scaled_dot_product_attention
- MoE：router / 量化专家（[in,out] 转置存储约定）/ shared 与手动参考对照
- 真实权重单层前向：layer 0（linear_attention）、layer 3（full_attention）

运行：python -m unittest discover -s tests -v
"""
import unittest

import torch
from numpy import float16, int32
from numpy.random import default_rng
from torch.nn.functional import scaled_dot_product_attention, silu

from liteengine.attention import (
    chunk_gated_delta_rule,
    eager_attention,
    recurrent_gated_delta_rule,
)
from liteengine.layer import DecoderLayer
from liteengine.loader import WeightStore
from liteengine.model import causal_mask, load_text_config
from liteengine.moe import MLP, QuantizedExperts, SparseMoeBlock, TopKRouter
from liteengine.core.norm import rms_norm, rms_norm_gated
from liteengine.quant import dequantize_awq
from liteengine.core.rope import apply_rotary_pos_emb, compute_inv_freq, rotary_embeddings

MODEL_DIR = "python/models/Qwen3.6-35B-A3B-AWQ-4bit"


class TestNorm(unittest.TestCase):
    def test_rms_norm_fp32(self):
        x = torch.randn(2, 3, 8, dtype=torch.float32)
        w = torch.randn(8, dtype=torch.float32)
        x_f = x.float()
        ref = (x_f * torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + 1e-6)) * (1.0 + w)
        torch.testing.assert_close(rms_norm(x, w), ref, atol=1e-6, rtol=0)

    def test_rms_norm_gated_fp32(self):
        """Gated 变体是 weight*norm（无 1+w！）再乘 silu(gate)。"""
        x = torch.randn(2, 3, 8, dtype=torch.float32)
        w = torch.randn(8, dtype=torch.float32)
        gate = torch.randn(2, 3, 8, dtype=torch.float32)
        normed = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)
        ref = (w * normed) * silu(gate)
        torch.testing.assert_close(rms_norm_gated(x, w, gate), ref, atol=1e-6, rtol=0)


class TestRope(unittest.TestCase):
    def test_inv_freq_and_embeddings(self):
        inv = compute_inv_freq(256, 1e7, 0.25)
        self.assertEqual(tuple(inv.shape), (32,))          # rotary_dim=64 → 32 个频率
        pos = torch.arange(5)
        cos, sin = rotary_embeddings(pos, inv)
        self.assertEqual(tuple(cos.shape), (5, 64))

    def test_apply_rotary_partial(self):
        inv = compute_inv_freq(256, 1e7, 0.25)
        pos = torch.arange(5)
        cos, sin = rotary_embeddings(pos, inv)
        q = torch.randn(1, 4, 5, 256)
        k = torch.randn(1, 2, 5, 256)
        q2, k2 = apply_rotary_pos_emb(q, k, cos, sin)
        rot = 64
        q_rot = q[..., :rot] * cos + (torch.cat((-q[..., rot // 2:rot], q[..., :rot // 2]), dim=-1)) * sin
        torch.testing.assert_close(q2[..., :rot], q_rot, atol=1e-6, rtol=0)
        torch.testing.assert_close(q2[..., rot:], q[..., rot:], atol=0, rtol=0)  # 透传
        torch.testing.assert_close(k2[..., rot:], k[..., rot:], atol=0, rtol=0)


class TestDeltaRule(unittest.TestCase):
    def test_chunk_matches_recurrent(self):
        """L=70 跨 2 个 chunk；g 恒负（真实模型的衰减语义）。"""
        torch.manual_seed(0)
        B, L, H, kd, vd = 1, 70, 4, 8, 8
        q = torch.randn(B, L, H, kd)
        k = torch.randn(B, L, H, kd)
        v = torch.randn(B, L, H, vd)
        g = -torch.rand(B, L, H) * 0.1
        beta = torch.sigmoid(torch.randn(B, L, H))
        out_c, _ = chunk_gated_delta_rule(q, k, v, g, beta)
        out_r, _ = recurrent_gated_delta_rule(q, k, v, g, beta)
        self.assertLess((out_c - out_r).abs().max().item(), 1e-3)


class TestEagerAttention(unittest.TestCase):
    def test_matches_sdpa(self):
        """eager attention（repeat_kv+softmax+mask）与 torch SDPA 对照。"""
        torch.manual_seed(1)
        B, H, kv, L, D, n_rep = 1, 4, 2, 8, 16, 2
        q = torch.randn(B, H, L, D)
        k = torch.randn(B, kv, L, D)
        v = torch.randn(B, kv, L, D)
        mask = causal_mask(L)
        out = eager_attention(q, k, v, mask, scaling=D ** -0.5, n_rep=n_rep)   # (B, L, H, D)
        # SDPA 要求 q/k/v 头数一致：先把 kv 头展开到 q 的头数
        ref = scaled_dot_product_attention(q, k.repeat_interleave(n_rep, dim=1),
                                           v.repeat_interleave(n_rep, dim=1),
                                           attn_mask=mask, dropout_p=0.0)
        ref = ref.transpose(1, 2)   # (B, L, H, D)，与 eager_attention 布局一致
        self.assertLess((out - ref).abs().max().item(), 1e-3)


class _FakeStore:
    def __init__(self, d):
        self.d = d

    def get(self, name):
        return self.d[name]


class TestMoe(unittest.TestCase):
    def _build(self):
        """合成数据：量化专家按真实文件 [in, out] 转置存储约定生成。"""
        rng = default_rng(0)
        num_experts, hidden, interm = 4, 64, 32
        w = {}
        for e in range(num_experts):
            for proj, (out_, in_) in [('gate_proj', (interm, hidden)), ('up_proj', (interm, hidden)),
                                      ('down_proj', (hidden, interm))]:
                w[f'ex.{e}.{proj}.qweight'] = rng.integers(-(2**31), 2**31, size=(in_, out_ // 8), dtype=int32)
                w[f'ex.{e}.{proj}.qzeros'] = rng.integers(0, 2**31, size=(in_ // 32, out_ // 8), dtype=int32)
                # scale ×0.01 使反量化权重量级 ~0.15（贴近真实 AWQ），避免 fp16 溢出
                w[f'ex.{e}.{proj}.scales'] = rng.standard_normal((in_ // 32, out_)).astype(float16)
        w['router'] = rng.standard_normal((num_experts, hidden)).astype(float16)
        w['shared.gate'] = rng.standard_normal((interm, hidden)).astype(float16)
        w['shared.up'] = rng.standard_normal((interm, hidden)).astype(float16)
        w['shared.down'] = rng.standard_normal((hidden, interm)).astype(float16)
        w['shared_gate'] = rng.standard_normal((1, hidden)).astype(float16)
        return _FakeStore(w), w, num_experts, hidden, interm

    def test_sparse_moe_matches_reference(self):
        store, w, num_experts, hidden, interm = self._build()
        experts = QuantizedExperts(store, 'ex', num_experts)
        router = TopKRouter(torch.from_numpy(w['router']).float(), top_k=2)
        shared = MLP(*[torch.from_numpy(w[k]).float() for k in ('shared.gate', 'shared.up', 'shared.down')])
        block = SparseMoeBlock(router, experts, shared, torch.from_numpy(w['shared_gate']).float())

        torch.manual_seed(0)
        x = torch.randn(5, hidden)
        out = block(x)

        # ---- 手动参考（同一公式，[in,out] 转置约定）----
        logits = x @ torch.from_numpy(w['router'].T).float()
        probs = torch.softmax(logits, dim=-1)
        scores_ref, idx_ref = probs.topk(2)
        scores_ref = scores_ref / scores_ref.sum(-1, keepdim=True)
        ref = torch.zeros_like(x)
        for t in range(5):
            for kk in range(2):
                e = int(idx_ref[t, kk])
                gw = torch.from_numpy(dequantize_awq(w[f'ex.{e}.gate_proj.qweight'], w[f'ex.{e}.gate_proj.qzeros'],
                                                     w[f'ex.{e}.gate_proj.scales']))
                uw = torch.from_numpy(dequantize_awq(w[f'ex.{e}.up_proj.qweight'], w[f'ex.{e}.up_proj.qzeros'],
                                                     w[f'ex.{e}.up_proj.scales']))
                dw = torch.from_numpy(dequantize_awq(w[f'ex.{e}.down_proj.qweight'], w[f'ex.{e}.down_proj.qzeros'],
                                                     w[f'ex.{e}.down_proj.scales']))
                h = silu(x[t] @ gw) * (x[t] @ uw)
                ref[t] += (h @ dw) * scores_ref[t, kk]
        mlp_out = (silu(x @ torch.from_numpy(w['shared.gate'].T).float()) * (x @ torch.from_numpy(w['shared.up'].T).float())) @ torch.from_numpy(w['shared.down'].T).float()
        sg = torch.sigmoid(x @ torch.from_numpy(w['shared_gate'].T).float())
        ref = ref + sg * mlp_out
        # 合成数据（随机 int4 + N(0,1) scale）数值可达 1e4，用相对误差（fp32 舍入 ~1e-6）
        rel = ((out - ref).abs() / (ref.abs() + 1e-6)).max().item()
        self.assertLess(rel, 1e-3)


class TestRealLayerForward(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.store = WeightStore(MODEL_DIR)
        cls.cfg = load_text_config(MODEL_DIR)

    @classmethod
    def tearDownClass(cls):
        cls.store.close()

    def _run_layer(self, idx: int, seq: int = 8):
        torch.manual_seed(0)
        layer = DecoderLayer(self.store, idx, self.cfg)
        x = torch.randn(1, seq, self.cfg["hidden_size"])
        if layer.block_type == "full_attention":
            inv = compute_inv_freq(int(self.cfg["head_dim"]),
                                   float(self.cfg["rope_parameters"]["rope_theta"]),
                                   float(self.cfg["rope_parameters"].get("partial_rotary_factor", 0.25)))
            cos, sin = rotary_embeddings(torch.arange(seq), inv)
            mask = causal_mask(seq)
        else:
            cos = sin = mask = None
        return layer(x, cos, sin, mask)

    def test_layer0_linear_attention(self):
        h = self._run_layer(0)
        self.assertEqual(tuple(h.shape), (1, 8, 2048))
        self.assertTrue(torch.isfinite(h).all())
        self.assertLess(h.std().item(), 10.0)

    def test_layer3_full_attention(self):
        h = self._run_layer(3)
        self.assertEqual(tuple(h.shape), (1, 8, 2048))
        self.assertTrue(torch.isfinite(h).all())
        self.assertLess(h.std().item(), 10.0)


if __name__ == "__main__":
    unittest.main()
