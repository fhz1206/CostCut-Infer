"""tests.test_sampling — 采样：温度/top_k/top_p/min_p + 重复/存在/频率惩罚。"""

from __future__ import annotations

import numpy as np

from ccut.sampling import SamplingParams, apply_sampling, greedy_argmax


def test_apply_sampling_sums_to_one():
    np.random.seed(0)
    lg = np.random.randn(4, 50).astype(np.float32)
    p = apply_sampling(lg, SamplingParams(temperature=1.0, top_k=5, top_p=0.9, min_p=0.05))
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-5)
    assert int((p[0] > 0).sum()) <= 5  # top_k 约束生效


def test_greedy_argmax_in_range():
    np.random.seed(1)
    lg = np.random.randn(2, 30).astype(np.float32)
    p = apply_sampling(lg, SamplingParams(temperature=1.0, top_k=0))
    out = greedy_argmax(p)
    assert out.shape == (2,)
    assert (out >= 0).all() and (out < 30).all()


def test_repetition_penalty_dampens_existing():
    """repetition_penalty > 1 时，已生成 token 的 logit 应被抑制。"""
    np.random.seed(2)
    lg = np.random.randn(1, 20).astype(np.float32)
    generated = np.array([[3, 7]])
    p1 = apply_sampling(lg, SamplingParams(temperature=1.0, repetition_penalty=1.0), generated=generated)
    p2 = apply_sampling(lg, SamplingParams(temperature=1.0, repetition_penalty=2.0), generated=generated)
    # 重复抑制 → token 3/7 在 p2 中概率应明显低于 p1
    assert p2[0, 3] < p1[0, 3] * 0.7
    assert p2[0, 7] < p1[0, 7] * 0.7
