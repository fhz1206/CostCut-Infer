"""FP8（E4M3/E5M2）与 NVFP4（E2M1）量化反量化测试。"""
import unittest

import numpy as np

from liteengine.quant import (QuantConfig, dequantize, e2m1_to_f32, e4m3_to_f32,
                              e5m2_to_f32)
from liteengine.registry import get_quant_method


class TestFpBitLayouts(unittest.TestCase):
    def test_e4m3(self):
        self.assertAlmostEqual(e4m3_to_f32(0x38), 1.0, places=6)    # exp 7, m 0
        self.assertAlmostEqual(e4m3_to_f32(0xC3), -2.75, places=6)  # sign, exp 8, m 3

    def test_e5m2(self):
        self.assertAlmostEqual(e5m2_to_f32(0x3C), 1.0, places=6)
        self.assertAlmostEqual(e5m2_to_f32(0xBC), -1.0, places=6)
        self.assertAlmostEqual(e5m2_to_f32(0xC0), -2.0, places=6)   # exp 16

    def test_e2m1(self):
        self.assertAlmostEqual(e2m1_to_f32(5), 3.0, places=6)       # 0b0101
        self.assertEqual(e2m1_to_f32(8), -0.0)                      # 0b1000


class TestFpDispatch(unittest.TestCase):
    def test_registered(self):
        for m in ("fp8", "e4m3", "e5m2", "nvfp4"):
            self.assertIsNotNone(get_quant_method(m), f"缺少 {m} 处理器")

    def test_nvfp4_dequant(self):
        """NVFP4：16 个 4-bit（1 块），x = e2m1(q) * s_block * s_global。"""
        qw = np.array([0x51] * 8, dtype=np.uint8)   # 低 4 位 1→0.5，高 4 位 5→3.0
        out = dequantize(qw, None, np.array([[0.5, 2.0]]),
                         QuantConfig(quant_method="nvfp4"))
        self.assertEqual(out.shape, (16,))
        self.assertAlmostEqual(out[0], 0.5, places=5)
        self.assertAlmostEqual(out[1], 3.0, places=5)

    def test_fp8_dequant(self):
        out = dequantize(np.array([0x3C], dtype=np.uint8), None, np.array([2.0]),
                         QuantConfig(quant_method="fp8", dtype="fp8_e5m2"))
        self.assertAlmostEqual(out[0], 2.0, places=5)
        out2 = dequantize(np.array([0x38], dtype=np.uint8), None, np.array([3.0]),
                          QuantConfig(quant_method="e4m3"))
        self.assertAlmostEqual(out2[0], 3.0, places=5)


if __name__ == "__main__":
    unittest.main()
