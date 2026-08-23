"""多 MoE 架构适配单元测试：StandardAttention / MergedExperts / 无共享 SparseMoeBlock / 配置归一化 / Mixtral 层组装。"""
import json
import os
import tempfile
import unittest

import numpy as np
import torch

from engine.attention import StandardAttention
from engine.layer import DecoderLayer
from engine.model import causal_mask
from model_config import load_model_config
from engine.moe import MergedExperts, SparseMoeBlock, TopKRouter


class _FakeStore:
    def __init__(self, data):
        self._d = data

    def get(self, name):
        return self._d[name]


def _mixtral_cfg() -> dict:
    return {
        "arch": "mixtral",
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "rms_norm_eps": 1e-5,
        "vocab_size": 32000,
        "rope_theta": 1e6,
        "rope_partial": 1.0,
        "layer_attention_types": ["standard", "standard"],
        "moe": {"num_experts": 4, "top_k": 2, "intermediate": 32,
                "shared": False, "experts_format": "merged_plain"},
        "weight_prefix": "model",
    }


class TestStandardAttention(unittest.TestCase):
    def test_forward_and_step(self):
        h, nh, kvh, hd = 64, 4, 2, 16
        cfg = {"hidden_size": h, "num_attention_heads": nh, "num_key_value_heads": kvh,
               "rms_norm_eps": 1e-5}
        w = {}
        for proj, (out, in_) in [("q_proj", (nh * hd, h)), ("k_proj", (kvh * hd, h)),
                                 ("v_proj", (kvh * hd, h)), ("o_proj", (h, nh * hd))]:
            w[f"attn.{proj}.weight"] = np.random.randn(out, in_).astype(np.float32)
        attn = StandardAttention(_FakeStore(w), "attn", cfg)
        x = torch.randn(1, 8, h)
        out = attn(x, torch.randn(8, hd), torch.randn(8, hd), causal_mask(8))
        self.assertEqual(tuple(out.shape), (1, 8, h))
        self.assertTrue(torch.isfinite(out).all())
        kv = (torch.randn(1, kvh, 4, hd), torch.randn(1, kvh, 4, hd))
        out_s, kv2 = attn.forward_step(torch.randn(1, 1, h), torch.randn(1, hd),
                                       torch.randn(1, hd), kv)
        self.assertEqual(kv2[2], 5)              # 预分配缓存：kv2=(k, v, length, max_len)，活动长度 5


class TestMergedExperts(unittest.TestCase):
    def test_no_shared_block(self):
        h, E, inter = 64, 4, 32
        data = {
            "experts.gate_up_proj.weight": np.random.randn(E, 2 * inter, h).astype(np.float32),
            "experts.down_proj.weight": np.random.randn(E, h, inter).astype(np.float32),
            "gate.weight": np.random.randn(E, h).astype(np.float32),
        }
        exp = MergedExperts(_FakeStore(data), "experts", E)
        block = SparseMoeBlock(TopKRouter(torch.from_numpy(data["gate.weight"]), 2), exp)
        out = block(torch.randn(1, 8, h))
        self.assertEqual(tuple(out.shape), (1, 8, h))
        self.assertTrue(torch.isfinite(out).all())


class TestModelConfig(unittest.TestCase):
    @staticmethod
    def _write(cfg_dict, name):
        d = os.path.join(tempfile.gettempdir(), name)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "config.json"), "w", encoding="utf-8") as f:
            json.dump(cfg_dict, f)
        return d

    def test_mixtral_normalize(self):
        cfg = {"architectures": ["MixtralForCausalLM"], "model_type": "mixtral",
               "hidden_size": 4096, "num_hidden_layers": 32, "num_attention_heads": 32,
               "num_key_value_heads": 8, "vocab_size": 32000, "rms_norm_eps": 1e-5,
               "intermediate_size": 14336, "num_local_experts": 8, "num_experts_per_tok": 2}
        c = load_model_config(self._write(cfg, "cfg_mixtral"))
        self.assertEqual(c["arch"], "mixtral")
        self.assertFalse(c["moe"]["shared"])
        self.assertEqual(c["moe"]["experts_format"], "merged_plain")
        self.assertEqual(c["weight_prefix"], "model")
        self.assertEqual(c["layer_attention_types"], ["standard"] * 32)

    def test_qwen3moe_normalize(self):
        cfg = {"architectures": ["Qwen3MoeForCausalLM"], "model_type": "qwen3_moe",
               "hidden_size": 2048, "num_hidden_layers": 30, "num_attention_heads": 16,
               "num_key_value_heads": 4, "vocab_size": 151936, "rms_norm_eps": 1e-6,
               "moe_intermediate_size": 768, "num_experts": 128, "num_experts_per_tok": 8,
               "shared_expert_intermediate_size": 2048}
        c = load_model_config(self._write(cfg, "cfg_qwen3moe"))
        self.assertEqual(c["arch"], "qwen3_moe")
        self.assertTrue(c["moe"]["shared"])
        self.assertEqual(c["moe"]["experts_format"], "merged_plain")

    def test_kimi_normalize(self):
        """Kimi K2/K2.5：text_config 别名 deepseek_v3 → MLA + MoE。"""
        cfg = {"model_type": "kimi_k25", "architectures": ["KimiK25ForConditionalGeneration"],
               "text_config": {"model_type": "kimi_k2", "hidden_size": 5120, "num_hidden_layers": 61,
                               "num_attention_heads": 64, "vocab_size": 152064, "rms_norm_eps": 1e-6,
                               "kv_lora_rank": 512, "q_lora_rank": 1536, "qk_rope_head_dim": 64,
                               "n_routed_experts": 256, "n_shared_experts": 1,
                               "moe_intermediate_size": 768, "num_topk": 8,
                               "n_group": 8, "topk_group": 4, "first_k_dense_replace": 3}}
        c = load_model_config(self._write(cfg, "cfg_kimi"))
        self.assertEqual(c["arch"], "deepseek_moe")
        self.assertEqual(c["attention"], "mla")
        self.assertEqual(c["moe"]["num_experts"], 256)
        self.assertTrue(c["moe"]["shared"])

    def test_glm_normalize(self):
        """GLM4-MoE：标准注意力 + 路由/共享专家（shared_experts 前缀）。"""
        cfg = {"model_type": "glm4_moe", "architectures": ["Glm4MoeForCausalLM"],
               "hidden_size": 4096, "num_hidden_layers": 28, "num_attention_heads": 32,
               "num_key_value_heads": 8, "vocab_size": 151552, "rms_norm_eps": 1e-5,
               "moe_intermediate_size": 1408, "num_experts": 64, "num_experts_per_tok": 8,
               "n_shared_experts": 1}
        c = load_model_config(self._write(cfg, "cfg_glm"))
        self.assertEqual(c["arch"], "glm_moe")
        self.assertEqual(c["layer_attention_types"][0], "standard")
        self.assertEqual(c["moe"]["shared_expert_prefix"], "shared_experts")
        self.assertEqual(c["moe"]["num_experts"], 64)

    def test_deepseek_v4_normalize(self):
        """DeepSeek-V4：MLA（kv 头=1）+ 路由/共享专家。"""
        cfg = {"model_type": "deepseek_v4", "architectures": ["DeepseekV4ForCausalLM"],
               "hidden_size": 7168, "num_hidden_layers": 61, "num_attention_heads": 128,
               "num_key_value_heads": 1, "vocab_size": 129280, "rms_norm_eps": 1e-6,
               "kv_lora_rank": 512, "q_lora_rank": 1024, "qk_rope_head_dim": 64,
               "n_routed_experts": 256, "n_shared_experts": 1, "moe_intermediate_size": 2048,
               "num_topk": 8, "n_group": 8, "topk_group": 4, "first_k_dense_replace": 3}
        c = load_model_config(self._write(cfg, "cfg_v4"))
        self.assertEqual(c["attention"], "mla")
        self.assertEqual(c["num_key_value_heads"], 1)
        self.assertEqual(c["moe"]["num_experts"], 256)

    def test_glm5_normalize(self):
        """GLM-5（GlmMoeDsa）：MLA + 256 路由/top-8 + 1 共享 + DSA 索引器字段。"""
        cfg = {"model_type": "glm_moe_dsa", "architectures": ["GlmMoeDsaForCausalLM"],
               "hidden_size": 6144, "num_hidden_layers": 78, "num_attention_heads": 64,
               "num_key_value_heads": 64, "vocab_size": 154880, "rms_norm_eps": 1e-5,
               "moe_intermediate_size": 2048, "n_routed_experts": 256,
               "n_shared_experts": 1, "num_experts_per_tok": 8,
               "kv_lora_rank": 512, "q_lora_rank": 2048, "qk_rope_head_dim": 64,
               "v_head_dim": 256, "first_k_dense_replace": 3,
               "index_topk": 2048, "index_head_dim": 128}
        c = load_model_config(self._write(cfg, "cfg_glm5"))
        self.assertEqual(c["arch"], "glm5")
        self.assertEqual(c["attention"], "mla")
        self.assertEqual(c["moe"]["num_experts"], 256)
        self.assertTrue(c["moe"]["shared"])

    def test_kimi_k3_and_r1(self):
        """Kimi K2.6/K3（"kimi" 模式 → deepseek_moe）与 DeepSeek-R1（DeepseekV3ForCausalLM）。"""
        k3 = {"model_type": "kimi_k3", "architectures": ["KimiK3ForConditionalGeneration"],
              "text_config": {"model_type": "kimi_k2", "hidden_size": 5120,
                              "num_hidden_layers": 61, "num_attention_heads": 64,
                              "vocab_size": 152064, "rms_norm_eps": 1e-6,
                              "kv_lora_rank": 512, "q_lora_rank": 1536, "qk_rope_head_dim": 64,
                              "n_routed_experts": 256, "n_shared_experts": 1,
                              "moe_intermediate_size": 768, "num_topk": 8,
                              "n_group": 8, "topk_group": 4, "first_k_dense_replace": 3}}
        c = load_model_config(self._write(k3, "cfg_k3"))
        self.assertEqual(c["arch"], "deepseek_moe")
        self.assertEqual(c["attention"], "mla")
        r1 = {"model_type": "deepseek_v3", "architectures": ["DeepseekV3ForCausalLM"],
              "hidden_size": 7168, "num_hidden_layers": 61, "num_attention_heads": 128,
              "num_key_value_heads": 1, "vocab_size": 129280, "rms_norm_eps": 1e-6,
              "kv_lora_rank": 512, "q_lora_rank": 1536, "qk_rope_head_dim": 64,
              "n_routed_experts": 256, "n_shared_experts": 1, "moe_intermediate_size": 2048,
              "num_topk": 8, "n_group": 8, "topk_group": 4, "first_k_dense_replace": 3}
        c = load_model_config(self._write(r1, "cfg_r1"))
        self.assertEqual(c["arch"], "deepseek_moe")
        self.assertEqual(c["moe"]["num_experts"], 256)

    def test_unknown_moe_fallback(self):
        """未知但非专属架构（MoE 字段齐全）→ 通用回退 generic_moe（merged_plain）。"""
        cfg = {"architectures": ["WeirdMoeForCausalLM"], "model_type": "weird_moe",
               "hidden_size": 1024, "num_hidden_layers": 12, "num_attention_heads": 16,
               "num_key_value_heads": 4, "vocab_size": 32000, "rms_norm_eps": 1e-5,
               "num_local_experts": 16, "num_experts_per_tok": 2,
               "moe_intermediate_size": 512}
        c = load_model_config(self._write(cfg, "cfg_unk_moe"))
        self.assertEqual(c["arch"], "generic_moe")
        self.assertEqual(c["moe"]["experts_format"], "merged_plain")
        self.assertEqual(c["moe"]["num_experts"], 16)

    def test_unknown_dense_fallback(self):
        """未知但非专属架构（无专家字段）→ 通用回退 generic_dense（dense_mlp）。"""
        cfg = {"architectures": ["WeirdForCausalLM"], "model_type": "weird_dense",
               "hidden_size": 512, "num_hidden_layers": 8, "num_attention_heads": 8,
               "vocab_size": 32000, "intermediate_size": 2048}
        c = load_model_config(self._write(cfg, "cfg_unk_dense"))
        self.assertEqual(c["arch"], "generic_dense")
        self.assertEqual(c["moe"]["experts_format"], "dense_mlp")

    def test_dbrx_and_phi3moe(self):
        """DBRX / Phi3-MoE 显式探测。"""
        dbrx = {"architectures": ["DbrxForCausalLM"], "model_type": "dbrx",
                "hidden_size": 4096, "num_hidden_layers": 40, "num_attention_heads": 48,
                "vocab_size": 100352, "num_local_experts": 16, "num_experts_per_tok": 4,
                "intermediate_size": 3072}
        self.assertEqual(load_model_config(self._write(dbrx, "cfg_dbrx"))["arch"], "dbrx")
        phi = {"architectures": ["Phi3MoEForCausalLM"], "model_type": "phi3_moe",
               "hidden_size": 2560, "num_hidden_layers": 32, "num_attention_heads": 32,
               "vocab_size": 32064, "num_local_experts": 16, "num_experts_per_tok": 2,
               "intermediate_size": 2048}
        self.assertEqual(load_model_config(self._write(phi, "cfg_phi"))["arch"], "phi3_moe")

    def test_linear_attention_rejected(self):
        """专属架构（含线性注意力层）不应走通用回退。"""
        cfg = {"architectures": ["WeirdLinearForCausalLM"], "model_type": "weird_linear",
               "hidden_size": 512, "num_hidden_layers": 4, "num_attention_heads": 8,
               "layer_types": ["linear_attention", "full_attention"]}
        with self.assertRaises(ValueError):
            load_model_config(self._write(cfg, "cfg_unk_linear"))


class TestMixtralLayer(unittest.TestCase):
    """Mixtral 风格 DecoderLayer 组装（标准注意力 + 合并专家 + 无共享）。"""

    def test_layer_assembly(self):
        cfg = _mixtral_cfg()
        hidden, hd = cfg["hidden_size"], cfg["head_dim"]
        nh, kvh = cfg["num_attention_heads"], cfg["num_key_value_heads"]
        inter, E = cfg["moe"]["intermediate"], cfg["moe"]["num_experts"]
        L0 = "model.layers.0"
        data = {}
        for name, shape in [
            (f"{L0}.input_layernorm.weight", (hidden,)),
            (f"{L0}.post_attention_layernorm.weight", (hidden,)),
            (f"{L0}.self_attn.q_proj.weight", (nh * hd, hidden)),
            (f"{L0}.self_attn.k_proj.weight", (kvh * hd, hidden)),
            (f"{L0}.self_attn.v_proj.weight", (kvh * hd, hidden)),
            (f"{L0}.self_attn.o_proj.weight", (hidden, nh * hd)),
            (f"{L0}.mlp.gate.weight", (E, hidden)),
            (f"{L0}.mlp.experts.gate_up_proj.weight", (E, 2 * inter, hidden)),
            (f"{L0}.mlp.experts.down_proj.weight", (E, hidden, inter)),
        ]:
            data[name] = np.random.randn(*shape).astype(np.float32)
        layer = DecoderLayer(_FakeStore(data), 0, cfg)
        x = torch.randn(1, 8, hidden)
        out = layer(x, torch.randn(8, hd), torch.randn(8, hd), causal_mask(8))
        self.assertEqual(tuple(out.shape), (1, 8, hidden))
        self.assertTrue(torch.isfinite(out).all())
