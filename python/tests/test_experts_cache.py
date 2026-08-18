"""M4 专家反量化缓存单元测试。

覆盖：
- 缓存正确性：清缓存前后输出逐位一致（反量化确定性，缓存只省计算不改数值）
- LRU 全局上限：条目 ≤ max_entries（跨层共享，与层数无关）
- 缓存字节审计与清理接口
- 所有层共享同一 ExpertCache 实例（内存硬上限的前提）
- ExpertCache 单元级 LRU 淘汰行为

运行：python -m unittest discover -s tests -v
"""
import unittest

import torch

from liteengine.cache import Cache, ExpertCache
from liteengine.loader import WeightStore
from liteengine.model import Qwen3_5MoeModel, load_text_config

MODEL_DIR = "python/models/Qwen3.6-35B-A3B-AWQ-4bit"
N_LAYERS = 5
VOCAB = 248320


class TestExpertCache(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.store = WeightStore(MODEL_DIR)
        cls.cfg = load_text_config(MODEL_DIR)

    @classmethod
    def tearDownClass(cls):
        cls.store.close()

    def test_cache_deterministic(self):
        """清缓存前后 prefill 输出逐位一致。"""
        model = Qwen3_5MoeModel(self.store, self.cfg, expert_cache_max=16)
        ids = torch.randint(0, 1000, (8,))
        h1 = model.prefill(ids, Cache(N_LAYERS), num_layers=N_LAYERS)
        model.clear_expert_cache()
        h2 = model.prefill(ids, Cache(N_LAYERS), num_layers=N_LAYERS)
        torch.testing.assert_close(h1, h2, atol=0, rtol=0)

    def test_lru_global_cap(self):
        """全局条目上限（跨层共享），清空后可重建。"""
        model = Qwen3_5MoeModel(self.store, self.cfg, expert_cache_max=8)
        ids = torch.randint(0, 1000, (8,))
        model.generate(ids, max_new_tokens=3, temperature=0.0, num_layers=N_LAYERS)
        self.assertLessEqual(len(model._expert_cache), 8)
        model.clear_expert_cache()
        self.assertEqual(len(model._expert_cache), 0)
        out = model.generate(ids, max_new_tokens=2, temperature=0.0, num_layers=N_LAYERS)
        self.assertTrue(all(0 <= t < VOCAB for t in out))

    def test_cache_bytes_bound(self):
        """缓存字节 ≈ 条目 × 12MB（每专家 fp32），受 max_entries 约束。"""
        model = Qwen3_5MoeModel(self.store, self.cfg, expert_cache_max=16)
        ids = torch.randint(0, 1000, (8,))
        model.generate(ids, max_new_tokens=3, temperature=0.0, num_layers=N_LAYERS)
        n = len(model._expert_cache)
        self.assertLessEqual(n, 16)
        self.assertLessEqual(model.expert_cache_bytes(), n * 12 * 2**20 + 1)

    def test_shared_across_layers(self):
        """所有层共享同一 ExpertCache 实例（内存硬上限的前提）。"""
        model = Qwen3_5MoeModel(self.store, self.cfg)
        ids = {id(model.layer(i).mlp.experts._cache) for i in range(N_LAYERS)}
        self.assertEqual(len(ids), 1)


class TestExpertCacheUnit(unittest.TestCase):
    def test_lru_eviction(self):
        """LRU 淘汰：最久未用的条目先被淘汰。"""
        cache = ExpertCache(max_entries=3)
        for i in range(4):
            cache.put((0, i), {"w": torch.zeros(1)})
        self.assertEqual(len(cache), 3)
        self.assertIsNone(cache.get((0, 0)))           # 最旧，已被淘汰
        cache.get((0, 1))                              # 访问 1 → 刷新 LRU
        cache.put((0, 5), {"w": torch.zeros(1)})       # 触发淘汰：2 应为最旧
        self.assertIsNone(cache.get((0, 2)))
        self.assertIsNotNone(cache.get((0, 1)))
        self.assertIsNotNone(cache.get((0, 3)))

    def test_bytes_report(self):
        cache = ExpertCache(max_entries=2)
        cache.put((0, 0), {"w": torch.zeros(16, dtype=torch.float32)})
        self.assertEqual(cache.bytes(), 64)            # 16 × 4B


if __name__ == "__main__":
    unittest.main()
