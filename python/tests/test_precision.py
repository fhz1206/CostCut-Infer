"""精度适配测试（vLLM 约定）：torch_dtype 探测 / fp16 权重保留 / FP8 缩放灵活支持。"""
import json
import os
import tempfile
import unittest

import numpy as np
import torch

from io_.loader import WeightStore
from model_config import load_model_config
from engine.moe import torch_weight_native
from quant import dequantize_fp8

MODEL_DIR = "python/models/Qwen3.6-35B-A3B-AWQ-4bit"


class TestTorchDtype(unittest.TestCase):
    @staticmethod
    def _write(cfg_dict, name):
        d = os.path.join(tempfile.gettempdir(), name)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "config.json"), "w", encoding="utf-8") as f:
            json.dump(cfg_dict, f)
        return d

    def test_torch_dtype_injected(self):
        """模型 torch_dtype（fp16/bf16）注入归一化配置；无则默认 float32。"""
        base = {"architectures": ["WeirdForCausalLM"], "model_type": "weird_dense",
                "hidden_size": 512, "num_hidden_layers": 8, "num_attention_heads": 8,
                "vocab_size": 32000, "intermediate_size": 2048}
        c1 = load_model_config(self._write({**base, "torch_dtype": "float16"}, "cfg_td1"))
        self.assertEqual(c1["dtype"], "float16")
        c2 = load_model_config(self._write({**base, "torch_dtype": "bfloat16"}, "cfg_td2"))
        self.assertEqual(c2["dtype"], "bfloat16")
        c3 = load_model_config(self._write(base, "cfg_td3"))
        self.assertEqual(c3["dtype"], "float32")


class TestNativeDtypeLoading(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.store = WeightStore(MODEL_DIR)

    def test_fp16_scales_preserved(self):
        sc = torch_weight_native(
            self.store,
            "model.language_model.layers.0.mlp.experts.0.gate_proj.scales")
        self.assertEqual(sc.dtype, torch.float16)          # fp16 原 dtype 保留

    def test_quant_int_converted(self):
        qw = torch_weight_native(
            self.store,
            "model.language_model.layers.0.mlp.experts.0.gate_proj.qweight")
        self.assertEqual(qw.dtype, torch.float32)          # 量化整数 → float32


class TestFp8Scaling(unittest.TestCase):
    def test_per_tensor(self):
        out = dequantize_fp8(np.array([0x38, 0x38], dtype=np.uint8), np.float32(2.0))
        self.assertAlmostEqual(float(out[0]), 2.0, places=6)

    def test_per_channel_in(self):
        out = dequantize_fp8(np.array([0x38, 0x38], dtype=np.uint8),
                             np.array([2.0, 3.0]))
        self.assertAlmostEqual(float(out[0]), 2.0, places=6)
        self.assertAlmostEqual(float(out[1]), 3.0, places=6)

    def test_per_channel_out(self):
        qw = np.array([[0x38, 0x38], [0x38, 0x38]], dtype=np.uint8)
        out = dequantize_fp8(qw, np.array([2.0, 3.0]))    # [out] → 行广播
        self.assertAlmostEqual(float(out[0, 0]), 2.0, places=6)
        self.assertAlmostEqual(float(out[1, 0]), 3.0, places=6)


if __name__ == "__main__":
    unittest.main()
