"""liteengine 配置解析单元测试（engine.toml 的 [default]/[model]/[inference]/[chat]）。

运行：python -m unittest discover -s tests -v
"""
import json
import os
import tempfile
import unittest

from config import EngineConfig
from model_config import load_model_config

MODEL_DIR = "python/models/Qwen3.6-35B-A3B-AWQ-4bit"   # model_dir 归一化后（models/ 前缀 → python/models）


class TestEngineConfig(unittest.TestCase):
    def test_parse_default(self):
        """默认 engine.toml：模型、缓存上限、采样与聊天参数。"""
        c = EngineConfig(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      "..", "engine.toml"))
        self.assertEqual(c.default_model, "Qwen3.6-35B-A3B-AWQ-4bit")
        # speculator.dspark 经 dspark_model 字段注册，不占 [model] 块
        self.assertEqual(len(c.models), 3)
        m = c.models["Qwen3.6-35B-A3B-AWQ-4bit"]
        self.assertEqual(m.model_dir, MODEL_DIR)
        self.assertEqual(m.expert_cache_max, 128)
        self.assertEqual(m.dspark_model, "Qwen3.6-35B-A3B-speculator.dspark")
        self.assertEqual(m.dspark_model_dir, "python/models/Qwen3.6-35B-A3B-speculator.dspark")
        self.assertEqual(c.models["Router"].model_dir, "python/models/Router")   # path 默认补全 + 归一化
        self.assertEqual(c.inference.top_k, 0)
        self.assertEqual(c.inference.repetition_penalty, 1.0)   # engine.toml 当前为 1（不干预）
        self.assertEqual(c.inference.max_new_tokens, 2048)
        self.assertEqual(c.chat.max_history, 20)

    def _write(self, txt: str) -> str:
        p = os.path.join(tempfile.gettempdir(), "cfg_unit.toml")
        with open(p, "w", encoding="utf-8") as f:
            f.write(txt)
        return p

    def test_missing_model_block(self):
        with self.assertRaises(ValueError):
            EngineConfig(self._write("[inference]\n"))

    def test_duplicate_name(self):
        with self.assertRaises(ValueError):
            EngineConfig(self._write('[model]\nname = "A"\n[model]\nname = "A"\n'))

    def test_default_not_registered(self):
        with self.assertRaises(ValueError):
            EngineConfig(self._write('[default]\nmodel = "X"\n[model]\nname = "A"\n'))

    def test_custom_values(self):
        txt = ('[default]\nmodel = "A"\n'
               '[model]\nname = "A"\nexpert_cache_max = 64\n'
               '[inference]\ntemperature = 1.0\ntop_k = 50\nrepetition_penalty = 1.2\n'
               'max_new_tokens = 100\n')
        c = EngineConfig(self._write(txt))
        self.assertEqual(c.models["A"].expert_cache_max, 64)
        self.assertEqual(c.inference.top_k, 50)
        self.assertEqual(c.inference.repetition_penalty, 1.2)
        self.assertEqual(c.inference.max_new_tokens, 100)


class TestAllPrecision(unittest.TestCase):
    """所有精度模型的配置归一化适配（fp32/fp16/bf16/fp8——torch_dtype 注入 + 归一化）。

    真实环境同一模型可有多种精度版本（如 Qwen3.5 的 AWQ int4 / fp16 / bf16 / fp8），
    归一化配置必须全部适配（torch_dtype 注入 + 架构/量化字段不变）。
    """

    @staticmethod
    def _write_qwen35(name: str, torch_dtype: str) -> str:
        d = os.path.join(tempfile.gettempdir(), name)
        os.makedirs(d, exist_ok=True)
        cfg = {"architectures": ["Qwen3_5MoeForCausalLM"], "model_type": "qwen3_5_moe",
               "hidden_size": 64, "num_hidden_layers": 2, "num_attention_heads": 4,
               "num_key_value_heads": 2, "vocab_size": 32000, "rms_norm_eps": 1e-6,
               "num_experts": 8, "num_experts_per_tok": 2, "moe_intermediate_size": 32,
               "n_shared_experts": 1, "layer_types": ["linear_attention", "full_attention"],
               "torch_dtype": torch_dtype}
        with open(os.path.join(d, "config.json"), "w", encoding="utf-8") as f:
            json.dump(cfg, f)
        return d

    def test_all_precision_normalize(self):
        for name, td in [("fp32", "float32"), ("fp16", "float16"),
                         ("bf16", "bfloat16"), ("fp8", "float8_e4m3fn")]:
            c = load_model_config(self._write_qwen35(f"q35_{name}", td))
            self.assertEqual(c["dtype"], td, f"{name} 的 torch_dtype 注入错误")
            self.assertEqual(c["arch"], "qwen3_5_moe", f"{name} 的架构归一化错误")
            self.assertEqual(c["moe"]["num_experts"], 8, f"{name} 的 MoE 字段错误")
            self.assertTrue(c["moe"]["shared"], f"{name} 的共享专家字段错误")

    def test_quant_method_stable_across_precision(self):
        """同一模型的量化方法在不同精度下保持一致（AWQ 配置不受 torch_dtype 影响）。"""
        base = {"architectures": ["Qwen3_5MoeForCausalLM"], "model_type": "qwen3_5_moe",
                "hidden_size": 64, "num_hidden_layers": 2, "num_attention_heads": 4,
                "num_key_value_heads": 2, "vocab_size": 32000, "rms_norm_eps": 1e-6,
                "num_experts": 8, "num_experts_per_tok": 2, "moe_intermediate_size": 32,
                "n_shared_experts": 1,
                "quantization_config": {"quant_method": "awq", "bits": 4, "group_size": 32}}
        c_fp16 = load_model_config(self._write_qwen35("q35_awq_fp16", "float16"))
        c_fp8 = load_model_config(self._write_qwen35("q35_awq_fp8", "float8_e4m3fn"))
        self.assertEqual(c_fp16["dtype"], "float16")
        self.assertEqual(c_fp8["dtype"], "float8_e4m3fn")


if __name__ == "__main__":
    unittest.main()
