"""M3 生成链路单元测试。

覆盖：
- prefill vs decode 逐位置 logits 一致性（KV cache + conv/recurrent 状态续接的正确性）
- sample_token 边界（greedy / top-k / top-p）
- generate / generate_stream 冒烟（确定性、流式前缀一致、token 范围）

运行：python -m unittest discover -s tests -v
"""
import unittest

import torch

from liteengine.cache import Cache
from liteengine.loader import WeightStore
from liteengine.model import Qwen3_5MoeModel, load_text_config
from liteengine.sampling import sample_token

MODEL_DIR = "models/Qwen3.6-35B-A3B-AWQ-4bit"
N_LAYERS = 5        # 覆盖 3×linear_attention + 1×full_attention + 1×linear
VOCAB = 248320


class TestSampleToken(unittest.TestCase):
    def test_boundaries(self):
        logits = torch.tensor([[0.1, 0.9, 0.3, 0.7]])
        self.assertEqual(sample_token(logits, temperature=0.0).item(), 1)              # greedy
        self.assertEqual(sample_token(logits, temperature=0.7, top_k=1, top_p=1.0).item(), 1)
        self.assertEqual(sample_token(logits, temperature=0.7, top_p=0.0).item(), 1)   # nucleus 只留 top1

    def test_repetition_penalty(self):
        """惩罚已生成的 token：10/1.1=9.09 < 9.5 → argmax 从 0 转移到 1。"""
        logits = torch.tensor([[10.0, 9.5, 1.0, 0.5]])
        self.assertEqual(sample_token(logits, temperature=0.0).item(), 0)
        self.assertEqual(sample_token(logits, temperature=0.0, repetition_penalty=1.0,
                                      prev_ids=[0]).item(), 0)                        # 1.0 不生效
        self.assertEqual(sample_token(logits, temperature=0.0, repetition_penalty=1.1,
                                      prev_ids=[0]).item(), 1)


class TestPrefillDecodeConsistency(unittest.TestCase):
    """KV/状态续接正确性的关键验证：prefill 的逐位置 logits == 逐步 decode 的 logits。"""

    @classmethod
    def setUpClass(cls):
        cls.store = WeightStore(MODEL_DIR)
        cls.cfg = load_text_config(MODEL_DIR)
        cls.model = Qwen3_5MoeModel(cls.store, cls.cfg)

    @classmethod
    def tearDownClass(cls):
        cls.store.close()

    def test_logits_consistency(self):
        torch.manual_seed(0)
        L = 6
        ids = torch.randint(0, 1000, (L,))
        # 参考：prefill 全量 → 各位置 logits
        h = self.model.prefill(ids, Cache(N_LAYERS), num_layers=N_LAYERS)
        ref = self.model.logits(h).float()                         # (L, vocab) fp16→fp32
        # decode：先 prefill 位置 0，再逐步 decode 位置 1..L-1（喂真实下一 token）
        cache = Cache(N_LAYERS)
        h0 = self.model.prefill(ids[:1], cache, num_layers=N_LAYERS)
        got = [self.model.logits(h0).float()]                      # 位置 0
        for i in range(1, L):
            h = self.model.decode_step(ids[i], cache, i, num_layers=N_LAYERS)
            got.append(self.model.logits(h.reshape(1, -1)).float())
        got = torch.cat(got, dim=0)                                 # (L, vocab)（stack 会变 (L,1,V) 导致广播错误）
        # 相对误差（logits 为 fp16 计算，量级 ~O(10)，允许 fp16 舍入）
        rel = ((got - ref).abs() / (ref.abs() + 1.0)).max().item()
        self.assertLess(rel, 1e-2)


class TestGenerate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.store = WeightStore(MODEL_DIR)
        cls.cfg = load_text_config(MODEL_DIR)
        cls.model = Qwen3_5MoeModel(cls.store, cls.cfg)

    @classmethod
    def tearDownClass(cls):
        cls.store.close()

    def test_greedy_deterministic(self):
        ids = torch.randint(0, 1000, (8,))
        out1 = self.model.generate(ids, max_new_tokens=4, temperature=0.0, num_layers=N_LAYERS)
        out2 = self.model.generate(ids, max_new_tokens=4, temperature=0.0, num_layers=N_LAYERS)
        self.assertEqual(out1, out2)
        self.assertTrue(all(0 <= t < VOCAB for t in out1))

    def test_stream_prefix(self):
        ids = torch.randint(0, 1000, (8,))
        gen = self.model.generate(ids, max_new_tokens=3, temperature=0.0, num_layers=N_LAYERS)
        stream = list(self.model.generate_stream(ids, max_new_tokens=3, temperature=0.0,
                                                num_layers=N_LAYERS))
        self.assertEqual(stream, gen[:3])

    def test_top_p_sampling(self):
        ids = torch.randint(0, 1000, (8,))
        out = self.model.generate(ids, max_new_tokens=3, temperature=0.7, top_p=0.9,
                                  num_layers=N_LAYERS)
        self.assertTrue(all(0 <= t < VOCAB for t in out))

    def test_eos_passthrough(self):
        """eos_token_id 透传：命中即提前终止（≤ max_new_tokens）。"""
        ids = torch.randint(0, 1000, (4,))
        out = self.model.generate(ids, max_new_tokens=3, temperature=0.0,
                                  eos_token_id=248044, num_layers=N_LAYERS)
        self.assertLessEqual(len(out), 3)
        self.assertTrue(all(0 <= t < VOCAB for t in out))


class TestSpeculativeDecode(unittest.TestCase):
    """dspark 投机解码：投机（低温）== 标准（低温）等价性——接受判定保证分布一致。

    投机路径：草稿 K token → 主模型并行验证 → 投机采样接受；
    低温（0.01）下接受退化为 top-1 相等（贪婪），输出应与标准生成完全一致。
    """

    @classmethod
    def setUpClass(cls):
        cls.store = WeightStore(MODEL_DIR)
        cls.model = Qwen3_5MoeModel(cls.store, load_text_config(MODEL_DIR))
        from liteengine.speculator import DSparkSpeculator
        cls.spec = DSparkSpeculator("models/Qwen3.6-35B-A3B-speculator.dspark")

    def test_greedy_equivalence(self):
        torch.manual_seed(0)
        ids = torch.randint(0, 1000, (5,))
        kw = dict(max_new_tokens=5, temperature=0.01, num_layers=N_LAYERS)
        spec_out = self.model.generate_speculative(ids, self.spec, **kw)
        std_out = self.model.generate(ids, **kw)
        self.assertEqual(spec_out, std_out)

    def test_output_range(self):
        """投机输出 token 范围合法（采样/接受不越界）。"""
        torch.manual_seed(1)
        ids = torch.randint(0, 1000, (5,))
        out = self.model.generate_speculative(
            ids, self.spec, max_new_tokens=5, temperature=0.7,
            top_p=0.9, num_layers=N_LAYERS)
        self.assertTrue(all(0 <= t < VOCAB for t in out))


if __name__ == "__main__":
    unittest.main()
