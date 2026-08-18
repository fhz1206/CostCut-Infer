"""AWQ 反量化与惰性权重加载单元测试（M1 验收）。

覆盖：
- 查表解包 vs 朴素移位参考（多形状）
- 反量化 vs 朴素参考（fp32 / fp16 输出）
- 真实模型张量抽样（依赖本地模型文件，验证布局与数值合理性）
- WeightStore 惰性读取、元数据查询、异常路径

运行（仓库根目录）：python -m unittest discover -s tests -v
"""
import unittest

from numpy import allclose, array_equal, asarray, empty, float16, float32, int16, int32
from numpy.random import default_rng

from liteengine.loader import WeightStore
from liteengine.quant import (QuantConfig, _unpack_colwise, _unpack_int4_colwise,
                              dequantize, dequantize_awq)

MODEL_DIR = "python/models/Qwen3.6-35B-A3B-AWQ-4bit"


def naive_unpack(packed):
    """朴素参考：8 次移位解包（int4 低位在前），并按 AWQ 列序重排还原真实列序。"""
    packed = asarray(packed)
    n, c = packed.shape
    out = empty((n, c * 8), dtype=int32)
    for i in range(8):
        out[:, i::8] = (packed >> (4 * i)) & 0xF
    # AWQ 非标准打包列序：线性解包后须按逆序 [0,4,1,5,2,6,3,7] 重排（vLLM _REVERSE_AWQ_PACK_ORDER）
    return out.reshape(n, -1, 8)[:, :, (0, 4, 1, 5, 2, 6, 3, 7)].reshape(n, c * 8)


class TestUnpack(unittest.TestCase):
    def test_unpack_matches_naive(self):
        rng = default_rng(0)
        for shape in [(8, 4), (64, 64), (2048, 64)]:
            q = rng.integers(-(2**31), 2**31, size=shape, dtype=int32)
            self.assertTrue(
                array_equal(_unpack_int4_colwise(q), naive_unpack(q)),
                f"shape={shape} 解包与朴素参考不一致",
            )


class TestDequantizeAwq(unittest.TestCase):
    def test_dequant_matches_naive(self):
        rng = default_rng(1)
        out, in_, gs = 128, 256, 32
        groups = out // gs
        qw = rng.integers(-(2**31), 2**31, size=(out, in_ // 8), dtype=int32)
        qz = rng.integers(0, 2**31, size=(groups, in_ // 8), dtype=int32)
        sc = rng.standard_normal((groups, in_)).astype(float16)

        ref_w = naive_unpack(qw).astype(int16)
        ref_z = naive_unpack(qz).astype(int16).repeat(gs, axis=0)
        ref_s = sc.astype(float32).repeat(gs, axis=0)
        ref = (ref_w - ref_z).astype(float32) * ref_s

        got = dequantize_awq(qw, qz, sc)
        self.assertEqual(got.shape, (out, in_))
        self.assertTrue(
            allclose(got, ref, atol=1e-3),
            f"反量化与朴素参考不一致，max diff={abs(got - ref).max():.5f}",
        )

    def test_dequant_float16_output(self):
        rng = default_rng(2)
        out, in_, gs = 64, 128, 32
        groups = out // gs
        qw = rng.integers(-(2**31), 2**31, size=(out, in_ // 8), dtype=int32)
        qz = rng.integers(0, 2**31, size=(groups, in_ // 8), dtype=int32)
        sc = rng.standard_normal((groups, in_)).astype(float16)
        got = dequantize_awq(qw, qz, sc, dtype="float16")
        self.assertEqual(got.dtype, float16)

    def test_real_model_tensor(self):
        """真实张量：形状正确、数值合理（依赖本地模型文件）。"""
        store = WeightStore(MODEL_DIR)
        try:
            prefix = "model.language_model.layers.0.mlp.experts.0.gate_proj"
            qw = store.get(prefix + ".qweight")
            qz = store.get(prefix + ".qzeros")
            sc = store.get(prefix + ".scales")
            self.assertEqual(list(qw.shape), [2048, 64])
            self.assertEqual(list(qz.shape), [64, 64])
            self.assertEqual(list(sc.shape), [64, 512])

            d = dequantize_awq(qw, qz, sc)
            self.assertEqual(list(d.shape), [2048, 512])
            self.assertLess(abs(float(d.mean())), 0.01)
            self.assertGreater(float(d.std()), 1e-4)
        finally:
            store.close()


class TestQuantCompat(unittest.TestCase):
    """量化兼容层：多规格（对称 / gptq 线性序 / group_size / bits）合成测试。"""

    def _build(self, out=64, in_=128, gs=32, bits=4, sym=False, seed=3):
        rng = default_rng(seed)
        qw = rng.integers(-(2**31), 2**31, size=(out, in_ * bits // 32), dtype=int32)
        qz = None if sym else rng.integers(0, 2**31, size=(out // gs, in_ * bits // 32), dtype=int32)
        sc = rng.standard_normal((out // gs, in_)).astype(float16)
        return qw, qz, sc

    def test_sym_matches_reference(self):
        qw, _, sc = self._build(sym=True)
        w = _unpack_int4_colwise(qw)                     # sym 仍按 AWQ 打包 → AWQ 序
        s = asarray(sc, dtype=float32).repeat(32, axis=0)
        got = dequantize(qw, None, sc,
                         QuantConfig(quant_method="awq", bits=4, group_size=32, sym=True))
        self.assertTrue(allclose(got, w.astype(float32) * s, atol=1e-3))

    def test_gptq_linear_order(self):
        qw, qz, sc = self._build()
        got = dequantize(qw, qz, sc, QuantConfig(quant_method="gptq", bits=4, group_size=32))
        w = _unpack_colwise(qw, 4)
        z = _unpack_colwise(qz, 4).astype(int16).repeat(32, axis=0)
        s = asarray(sc, dtype=float32).repeat(32, axis=0)
        ref = (w.astype(int16) - z).astype(float32) * s
        self.assertTrue(allclose(got, ref, atol=1e-3))

    def test_group_size_64(self):
        qw, qz, sc = self._build(gs=64)
        got = dequantize(qw, qz, sc, QuantConfig(quant_method="gptq", bits=4, group_size=64))
        self.assertEqual(got.shape, (64, 128))

    def test_bits8_unpack(self):
        rng = default_rng(4)
        qw = rng.integers(0, 2**31, size=(32, 16), dtype=int32)
        w = _unpack_colwise(qw, 8)
        self.assertEqual(w.shape, (32, 64))
        self.assertTrue(w.min() >= 0 and w.max() <= 255)

    def test_awq_config_parse(self):
        cfg = QuantConfig.from_dict({"quant_method": "awq", "zero_point": True,
                                     "group_size": 32, "bits": 4})
        self.assertEqual(cfg.quant_method, "awq")
        self.assertFalse(cfg.sym)
        cfg2 = QuantConfig.from_dict({"quant_method": "gptq", "sym": True, "group_size": 128})
        self.assertTrue(cfg2.sym)
        self.assertEqual(cfg2.group_size, 128)


class TestWeightStore(unittest.TestCase):
    def test_lazy_index_and_info(self):
        store = WeightStore(MODEL_DIR)
        try:
            self.assertGreater(len(store.keys()), 90000)
            self.assertTrue(
                store.has("model.language_model.layers.0.mlp.experts.0.gate_proj.qweight")
            )
            shape, dtype = store.tensor_info(
                "model.language_model.layers.0.mlp.experts.0.gate_proj.qweight"
            )
            self.assertEqual(shape, [2048, 64])
            self.assertEqual(dtype, "I32")
        finally:
            store.close()

    def test_get_small_tensor(self):
        store = WeightStore(MODEL_DIR)
        try:
            w = store.get("model.language_model.layers.0.input_layernorm.weight")
            self.assertEqual(list(w.shape), [2048])
        finally:
            store.close()

    def test_missing_tensor_raises(self):
        store = WeightStore(MODEL_DIR)
        try:
            with self.assertRaises(KeyError):
                store.get("not.exist.tensor")
        finally:
            store.close()


if __name__ == "__main__":
    unittest.main()
