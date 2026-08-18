"""GGUF 支持测试：合成 GGUF 写入 → 加载 → 配置归一化 → 模型前向/生成冒烟。"""
import os
import struct
import tempfile
import unittest

import numpy as np
import torch

from liteengine.gguf import (GGML_TYPE_F32, GGML_TYPE_Q4_0, GGUFReader,
                             GGUFWeightStore, gguf_metadata_to_config,
                             hf_to_gguf_name)
from liteengine.model import Qwen3_5MoeModel
from liteengine.model_config import _norm_fallback

HIDDEN, N_LAYERS, HEADS, KVH, INTER, VOCAB = 32, 1, 4, 2, 64, 64


def _write_metadata(f, md: dict):
    for k, (v, t) in md.items():
        kb = k.encode("utf-8")
        f.write(struct.pack("<Q", len(kb)))
        f.write(kb)
        if t == "u32":
            f.write(struct.pack("<I", 4))
            f.write(struct.pack("<I", int(v)))
        elif t == "f32":
            f.write(struct.pack("<I", 6))
            f.write(struct.pack("<f", float(v)))
        elif t == "str":
            vb = str(v).encode("utf-8")
            f.write(struct.pack("<I", 8))
            f.write(struct.pack("<Q", len(vb)))
            f.write(vb)


def _quant_q4_0(arr: np.ndarray) -> bytes:
    """F32 → Q4_0 块编码（块 32：2 字节 fp16 缩放 + 16 字节 int4），验证反量化的往返。"""
    out = b""
    flat = arr.reshape(-1)
    for b in range(0, len(flat), 32):
        blk = flat[b:b + 32]
        amax = np.abs(blk).max()
        d = amax / -8 if amax != 0 else 0.0
        if d == 0:
            d = 1.0
        q = np.clip(np.round(blk / d) + 8, 0, 15).astype(np.uint8)
        out += struct.pack("<e", np.float16(d))
        for j in range(0, 32, 2):
            out += bytes([q[j] | (q[j + 1] << 4)])
    return out


def _write_synthetic_gguf(path: str, quant_embed: bool = False) -> None:
    rng = np.random.default_rng(0)
    md = {
        "general.architecture": ("llama", "str"),
        "llama.block_count": (N_LAYERS, "u32"),
        "llama.embedding_length": (HIDDEN, "u32"),
        "llama.attention.head_count": (HEADS, "u32"),
        "llama.attention.head_count_kv": (KVH, "u32"),
        "llama.attention.layer_norm_rms_epsilon": (1e-5, "f32"),
        "llama.rope.freq_base": (10000.0, "f32"),
        "llama.feed_forward_length": (INTER, "u32"),
        "llama.vocab_size": (VOCAB, "u32"),
    }
    hd = HIDDEN // HEADS
    tensors = {
        "token_embd.weight": (rng.standard_normal((VOCAB, HIDDEN)) * 0.1).astype(np.float32),
        "output_norm.weight": np.ones(HIDDEN, dtype=np.float32),
        "output.weight": (rng.standard_normal((VOCAB, HIDDEN)) * 0.1).astype(np.float32),
        "blk.0.attn_norm.weight": np.ones(HIDDEN, dtype=np.float32),
        "blk.0.attn_q.weight": (rng.standard_normal((HEADS * hd, HIDDEN)) * 0.1).astype(np.float32),
        "blk.0.attn_k.weight": (rng.standard_normal((KVH * hd, HIDDEN)) * 0.1).astype(np.float32),
        "blk.0.attn_v.weight": (rng.standard_normal((KVH * hd, HIDDEN)) * 0.1).astype(np.float32),
        "blk.0.attn_output.weight": (rng.standard_normal((HIDDEN, HEADS * hd)) * 0.1).astype(np.float32),
        "blk.0.ffn_norm.weight": np.ones(HIDDEN, dtype=np.float32),
        "blk.0.ffn_gate.weight": (rng.standard_normal((INTER, HIDDEN)) * 0.1).astype(np.float32),
        "blk.0.ffn_up.weight": (rng.standard_normal((INTER, HIDDEN)) * 0.1).astype(np.float32),
        "blk.0.ffn_down.weight": (rng.standard_normal((HIDDEN, INTER)) * 0.1).astype(np.float32),
    }
    with open(path, "wb") as f:
        f.write(b"GGUF")
        f.write(struct.pack("<I", 3))
        f.write(struct.pack("<Q", len(tensors)))
        f.write(struct.pack("<Q", len(md)))
        _write_metadata(f, md)
        offsets, pos = {}, 0
        for name, arr in tensors.items():
            f.write(struct.pack("<Q", len(name)))
            f.write(name.encode("utf-8"))
            f.write(struct.pack("<I", len(arr.shape)))   # n_dims（1 维张量如 output_norm 须写 1）
            for d in reversed(arr.shape):
                f.write(struct.pack("<Q", d))
            f.write(struct.pack("<I", GGML_TYPE_F32))
            f.write(struct.pack("<Q", pos))
            offsets[name] = pos
            pos += arr.nbytes
        for name, arr in tensors.items():
            if quant_embed and name == "token_embd.weight":
                # Q4_0 编码（32 字节对齐模拟：reader 用块长推进，无需整体对齐）
                pass
            f.write(arr.tobytes())


class TestGGUF(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = os.path.join(tempfile.gettempdir(), "syn_llama_test.gguf")
        _write_synthetic_gguf(cls.path)

    def test_parse_and_config(self):
        reader = GGUFReader(self.path)
        self.assertEqual(reader.version, 3)
        self.assertIn("token_embd.weight", reader.tensors)
        cfg = gguf_metadata_to_config(reader)
        self.assertEqual(cfg["hidden_size"], HIDDEN)
        self.assertEqual(cfg["num_hidden_layers"], N_LAYERS)
        reader.close()

    def test_q4_0_dequant_roundtrip(self):
        """Q4_0 编码 → 读取反量化，验证与量化前近似（误差 < 0.5）。"""
        arr = (np.random.default_rng(1).standard_normal(64) * 1.0).astype(np.float32)
        path = os.path.join(tempfile.gettempdir(), "syn_q4_0.gguf")
        # 用最小 GGUF：仅一个 Q4_0 张量
        with open(path, "wb") as f:
            f.write(b"GGUF")
            f.write(struct.pack("<I", 3))
            f.write(struct.pack("<Q", 1))
            f.write(struct.pack("<Q", 0))
            name = b"test.weight"
            f.write(struct.pack("<Q", len(name)))
            f.write(name)
            f.write(struct.pack("<I", 1))
            f.write(struct.pack("<Q", 64))
            f.write(struct.pack("<I", GGML_TYPE_Q4_0))
            f.write(struct.pack("<Q", 0))
            f.write(_quant_q4_0(arr))
        reader = GGUFReader(path)
        got = reader.get_f32("test.weight").reshape(-1)
        err = float(np.abs(got - arr).max())
        self.assertLess(err, 0.5, f"Q4_0 往返误差过大: {err}")
        reader.close()

    def test_name_mapping_inverse(self):
        pairs = [
            ("model.embed_tokens.weight", "token_embd.weight"),
            ("model.layers.0.input_layernorm.weight", "blk.0.attn_norm.weight"),
            ("model.layers.1.self_attn.q_proj.weight", "blk.1.attn_q.weight"),
            ("model.layers.2.mlp.experts.down_proj.5.weight", "blk.2.ffn_exps.5.w2.weight"),
            ("model.layers.3.mlp.experts.gate_up_proj.7.weight", "blk.3.ffn_exps.7.w1.weight"),
        ]
        from liteengine.gguf import gguf_name_to_hf
        for hf, g in pairs:
            self.assertEqual(hf_to_gguf_name(hf), g, f"{hf} 映射错误")
            self.assertEqual(gguf_name_to_hf(g), hf, f"{g} 逆映射错误")

    def test_model_forward(self):
        """GGUF 加载 → 通用稠密配置 → 模型前向/生成冒烟。"""
        store = GGUFWeightStore(self.path)
        raw = gguf_metadata_to_config(store.reader)
        cfg = _norm_fallback(raw, "llama", "gguf")
        self.assertEqual(cfg["arch"], "generic_dense")
        model = Qwen3_5MoeModel(store, cfg)
        torch.manual_seed(0)
        ids = torch.randint(0, VOCAB, (4,))
        out = model.generate(ids, max_new_tokens=3, temperature=0.0)
        self.assertTrue(all(0 <= t < VOCAB for t in out))
        self.assertEqual(len(out), 3)


if __name__ == "__main__":
    unittest.main()
