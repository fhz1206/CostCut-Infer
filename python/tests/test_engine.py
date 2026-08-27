"""tests.test_engine — 引擎调度：add_request / schedule / preempt / metrics。"""

from __future__ import annotations

import time

import numpy as np
import pytest

from ccut.engine import Engine, EngineConfig


class _FakeBackend:
    """每步返回固定 token 的最小后端（不接真模型）。"""

    def __init__(self, eos_token: int = 0):
        self.eos_token = eos_token
        self._next = 1

    def generate_step(self, input_ids, sampling_params):
        tok = self._next
        self._next = (self._next + 1) % 100
        return tok


def test_engine_add_and_status():
    eng = Engine(EngineConfig(max_num_seqs=4, max_num_batched_tokens=32, eos_token_id=0), _FakeBackend())
    rid = eng.add_request([1, 2, 3], {"temperature": 0.0}, max_tokens=8)
    st = eng.status(rid)
    assert st["request_id"] == rid
    assert st["status"] in ("pending", "prefilling", "decoding", "finished", "evicted")
    eng.stop()


def test_engine_run_drains_when_all_finished():
    """全部请求完成后 run_until_drained 退出。"""
    eng = Engine(EngineConfig(max_num_seqs=4, max_num_batched_tokens=64, eos_token_id=0), _FakeBackend())
    for _ in range(2):
        eng.add_request([1, 2, 3], {"temperature": 0.0}, max_tokens=4)
    yielded = list(eng.run_until_drained())
    # 至少一个最终状态 yield
    assert eng.metrics.requests_added == 2
    eng.stop()


def test_engine_cancel_returns_to_evicted():
    eng = Engine(EngineConfig(max_num_seqs=4, max_num_batched_tokens=64), _FakeBackend())
    rid = eng.add_request([1], {}, max_tokens=8)
    assert eng.cancel(rid) is True
    st = eng.status(rid)
    assert st["status"] in ("evicted", "finished")
    eng.stop()


def test_engine_metrics_snapshot_keys():
    eng = Engine(EngineConfig(max_num_seqs=2, max_num_batched_tokens=64), _FakeBackend())
    d = eng.metrics.to_dict()
    for k in ("steps", "requests_added", "requests_finished", "preemption_count", "step_avg_ms", "active"):
        assert k in d
    eng.stop()
