"""插件注册表测试：内置注册 / 未知名查找 / 外部组件接入演示 / 架构模式匹配。"""
import unittest

from liteengine.registry import (get_arch_normalizer, get_attention,
                                 get_moe_format, get_quant_method,
                                 list_arch_normalizers, list_attentions,
                                 list_moe_formats, list_quant_methods,
                                 register_arch_normalizer, register_attention,
                                 register_quant_method, resolve_arch)


class TestRegistryBuiltins(unittest.TestCase):
    def test_builtins_registered(self):
        """内置组件（注意力 / MoE 格式 / 量化 / 架构）经 import 自动注册。"""
        import liteengine.attention          # noqa: F401  触发注册
        import liteengine.moe                # noqa: F401
        import liteengine.model_config       # noqa: F401
        from liteengine.quant import dequantize   # noqa: F401
        self.assertEqual(set(list_attentions()),
                         {"standard", "full_gated", "linear_delta", "mla"})
        self.assertEqual(set(list_moe_formats()),
                         {"quantized_separate", "merged_plain", "dense_mlp"})
        self.assertIn("awq", list_quant_methods())
        self.assertIn("gptq", list_quant_methods())
        for a in ("qwen3_5_moe", "mixtral", "qwen3_moe", "glm_moe",
                  "deepseek_moe", "dbrx", "phi3_moe"):
            self.assertIsNotNone(get_arch_normalizer(a), f"缺少归一化器 {a}")

    def test_unknown_lookup_none(self):
        self.assertIsNone(get_attention("no_such_attn"))
        self.assertIsNone(get_moe_format("no_such_fmt"))
        self.assertIsNone(get_quant_method("no_such_quant"))
        self.assertIsNone(get_arch_normalizer("no_such_arch"))
        self.assertIsNone(resolve_arch("NoSuchArchForCausalLM", "no_such"))


class TestRegistryExtensibility(unittest.TestCase):
    def test_register_new_components(self):
        """外部组件接入演示：注册新注意力 / 量化方法 / 架构归一化器后即可解析。"""

        @register_attention("demo_attn")
        def build_demo(store, prefix, cfg):
            return "demo-attn-built"

        @register_quant_method("demo_quant")
        def handle_demo(qweight, qzeros, scales, cfg, dtype="float32"):
            return "demo-quant-handled"

        @register_arch_normalizer("demo_arch", patterns=("DemoForCausalLM", "demo_arch"))
        def norm_demo(c):
            return {"arch": "demo_arch"}

        self.assertEqual(get_attention("demo_attn")(None, "p", {}), "demo-attn-built")
        self.assertEqual(get_quant_method("demo_quant")(None, None, None, None),
                         "demo-quant-handled")
        self.assertEqual(resolve_arch("DemoForCausalLM", ""), norm_demo)
        self.assertIsNone(resolve_arch("AnotherUnknownArch", "x"))

    def test_arch_pattern_matching(self):
        import liteengine.model_config      # noqa: F401  触发注册
        self.assertIsNotNone(resolve_arch("MixtralForCausalLM", ""))
        self.assertIsNotNone(resolve_arch("", "qwen3_moe"))
        self.assertIsNotNone(resolve_arch("KimiK25ForConditionalGeneration", "kimi_k25"))
        self.assertIsNotNone(resolve_arch("DbrxForCausalLM", ""))
        # 稠密 Phi3（无 MoE）不应命中 phi3_moe 归一化器 → 走通用回退
        self.assertIsNone(resolve_arch("Phi3ForCausalLM", ""))


if __name__ == "__main__":
    unittest.main()
