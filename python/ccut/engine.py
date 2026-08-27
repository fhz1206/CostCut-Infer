"""ccut.engine — 引擎编排（调度/连续批/chunked prefill/抢占）。

设计（§3.4-2）：

- **请求生命周期**：added → prefill(chunked) → decoding → done/evicted。
- **连续批**：每 step 选一组**可推进**请求（prefill 满 chunk 或 decode
  单 token），统一调 ``model_forward_layer`` 计算，整批 token 同时返回。
- **chunked prefill**：长 prompt 切成 ``chunked_prefill_size`` token 块（默认 4096），
  分多步完成 prefill（同一请求可与 decode 请求混合批）。预算共享：
  ``max_num_batched_tokens``。
- **抢占**（§3.4-5）：批总 token > 预算或新请求到来 → 当前批可抢占的 decode
  请求（已生成数最少）回滚 1 步（丢最后 1 token + 释放其 KV 块）→ 重排批。
  回滚数 hard_cap = 8（防 OOM）；超 cap → 拒绝新请求（503-like）。
- **三机制协同**：专家流（R2） + KV 双层（R1） + 层流式（R10）由
  expert_reader / kv_coordinator / weight_stream 持有，engine 调度
  ``request.stream`` 在每层后被``refill_ahead``。
- **MVP 定位**：本引擎在通用组装器（models/generic.py）就绪前用 **L1 兜底
  路径**（L1Backend 逐请求）；通用路径接通后做「hybrid：L0 通用 + L1 兜底」
  双模式自动降级（§4）。
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

__all__ = [
    "RequestStatus",
    "GenerationRequest",
    "EngineConfig",
    "EngineMetrics",
    "Engine",
]


_LOG = logging.getLogger("ccut.engine")


class RequestStatus(str, Enum):
    PENDING = "pending"
    PREFILLING = "prefilling"
    DECODING = "decoding"
    FINISHED = "finished"
    EVICTED = "evicted"


@dataclass
class GenerationRequest:
    """单个生成请求。"""

    request_id: str
    prompt_ids: list[int]
    sampling_params: dict = field(default_factory=dict)
    max_tokens: int = 256
    stream_id: str = ""
    status: RequestStatus = RequestStatus.PENDING
    generated_ids: list[int] = field(default_factory=list)
    prefill_consumed: int = 0  # 已 prefill 的 token 数（chunked prefill 进度）
    history_counts: Any = None  # 重复惩罚用频次数组
    arrived_ns: int = 0
    finish_reason: str = ""
    error: str = ""
    on_token: Callable[[int], None] | None = None


@dataclass
class EngineConfig:
    """引擎运行配置（从 Config 派生子集）。"""

    max_num_seqs: int = 8
    max_num_batched_tokens: int = 8192
    chunked_prefill_size: int = 4096
    preemption_rollback_steps: int = 8
    scheduler_delay_factor: float = 0.0
    num_worker_threads: int = 4
    seed: int | None = None
    eos_token_id: int | None = None


@dataclass
class EngineMetrics:
    """引擎运行时指标。"""

    steps: int = 0
    requests_added: int = 0
    requests_finished: int = 0
    requests_evicted: int = 0
    prefill_tokens_total: int = 0
    decode_tokens_total: int = 0
    preemption_count: int = 0
    preemption_rollback_total: int = 0
    step_avg_ms: float = 0.0
    step_total_ms: float = 0.0
    active: int = 0

    def to_dict(self) -> dict:
        return {
            "steps": self.steps,
            "requests_added": self.requests_added,
            "requests_finished": self.requests_finished,
            "requests_evicted": self.requests_evicted,
            "prefill_tokens_total": self.prefill_tokens_total,
            "decode_tokens_total": self.decode_tokens_total,
            "preemption_count": self.preemption_count,
            "preemption_rollback_total": self.preemption_rollback_total,
            "step_avg_ms": round(self.step_avg_ms, 3),
            "active": self.active,
        }


class Engine:
    """推理引擎（调度器 + 调度循环）。

    使用::

        cfg = EngineConfig.from_app_config(...)
        engine = Engine(cfg, backend=L1Backend(model_dir, ...))  # L1 兜底
        engine.add_request(prompt_ids, sampling, max_tokens=64)
        for step_result in engine.run_until_drained():
            ...  # 消费 / 推流
    """

    def __init__(
        self,
        config: EngineConfig,
        backend,  # L1Backend 或 L0 组装器（callable: forward_step）
        metrics: EngineMetrics | None = None,
        kv_coordinator=None,
        weight_stream=None,
        expert_pipeline=None,
    ):
        self.cfg = config
        self.backend = backend
        self.kv = kv_coordinator
        self.weight_stream = weight_stream
        self.expert_pipe = expert_pipeline
        self.metrics = metrics or EngineMetrics()
        self._lock = threading.Lock()
        self._requests: dict[str, GenerationRequest] = {}
        self._queue: deque[str] = deque()  # pending IDs
        self._stop = False
        self._seed_seq = 0
        if config.seed is not None:
            import random

            random.seed(config.seed)

    # -- 外部 API ----------------------------------------------------------
    def add_request(
        self,
        prompt_ids: list[int],
        sampling_params: dict | None = None,
        max_tokens: int = 256,
        on_token: Callable[[int], None] | None = None,
        request_id: str | None = None,
    ) -> str:
        rid = request_id or uuid.uuid4().hex[:16]
        with self._lock:
            self._requests[rid] = GenerationRequest(
                request_id=rid,
                prompt_ids=list(prompt_ids),
                sampling_params=dict(sampling_params or {}),
                max_tokens=max_tokens,
                stream_id=rid,
                status=RequestStatus.PENDING,
                arrived_ns=time.monotonic_ns(),
                on_token=on_token,
            )
            self._queue.append(rid)
            self.metrics.requests_added += 1
            self.metrics.active = sum(
                1 for r in self._requests.values() if r.status in (RequestStatus.PREFILLING, RequestStatus.DECODING)
            )
        return rid

    def cancel(self, request_id: str) -> bool:
        with self._lock:
            r = self._requests.get(request_id)
            if r is None or r.status in (RequestStatus.FINISHED, RequestStatus.EVICTED):
                return False
            r.status = RequestStatus.EVICTED
            r.finish_reason = "cancelled"
            self.metrics.requests_evicted += 1
            return True

    def status(self, request_id: str) -> dict:
        r = self._requests.get(request_id)
        if r is None:
            return {"request_id": request_id, "status": "not_found"}
        return {
            "request_id": r.request_id,
            "status": r.status.value,
            "generated": len(r.generated_ids),
            "finish_reason": r.finish_reason,
            "error": r.error,
        }

    def stop(self) -> None:
        self._stop = True

    # -- 主循环 ------------------------------------------------------------
    def run_until_drained(self):
        """驱动循环：每步调度一批 → 调后端 → 推进状态 → 产 yield。"""
        while not self._stop:
            batch = self._schedule_step()
            if not batch:
                if not self._active_count():
                    break
                continue
            t0 = time.perf_counter()
            try:
                outputs = self._backend_step(batch)
            except Exception as e:
                _LOG.exception("backend step 失败")
                for r in batch:
                    r.error = str(e)
                    r.status = RequestStatus.EVICTED
                    self.metrics.requests_evicted += 1
                continue
            self._postprocess(batch, outputs)
            dt_ms = (time.perf_counter() - t0) * 1000
            self.metrics.step_total_ms += dt_ms
            self.metrics.steps += 1
            self.metrics.step_avg_ms = self.metrics.step_total_ms / self.metrics.steps
            self.metrics.active = self._active_count()
            yield from (r for r in batch if r.status in (RequestStatus.FINISHED, RequestStatus.EVICTED))
        # 收尾：全 finished/evicted → 清理
        for r in list(self._requests.values()):
            if r.status not in (RequestStatus.FINISHED, RequestStatus.EVICTED):
                r.status = RequestStatus.EVICTED
                r.finish_reason = "engine_stopped"

    # -- 调度 --------------------------------------------------------------
    def _active_count(self) -> int:
        return sum(
            1
            for r in self._requests.values()
            if r.status in (RequestStatus.PREFILLING, RequestStatus.DECODING)
        )

    def _schedule_step(self) -> list[GenerationRequest]:
        """一步调度：满 ``max_num_batched_tokens`` 的批（chunked prefill + decode）。"""
        with self._lock:
            batch: list[GenerationRequest] = []
            budget = self.cfg.max_num_batched_tokens
            # 1) 推进 pending → prefill（按到达序，max_num_seqs 限）
            while self._queue and len(batch) < self.cfg.max_num_seqs:
                rid = self._queue.popleft()
                r = self._requests[rid]
                if r.status != RequestStatus.PENDING:
                    continue
                chunk = self.cfg.chunked_prefill_size
                remaining = len(r.prompt_ids) - r.prefill_consumed
                if remaining <= 0:
                    r.status = RequestStatus.DECODING
                    continue
                take = min(remaining, chunk)
                # 批预算检查：超预算则放回队列尾（抢占留给 decode 较小者）
                if take > budget:
                    # 抢占：回滚已 decode 的请求 1 步（释放 1 token + 1 KV 块）
                    if not self._preempt(batch, target_free=budget):
                        self._queue.appendleft(rid)  # 不能抢占则等下步
                        break
                r.prefill_consumed += take
                r.status = RequestStatus.PREFILLING
                batch.append(r)
                budget -= take
            # 2) 现有 decoding 全部加入
            for r in self._requests.values():
                if r.status == RequestStatus.DECODING and len(batch) < self.cfg.max_num_seqs and budget >= 1:
                    batch.append(r)
                    budget -= 1
            return batch

    def _preempt(self, batch: list[GenerationRequest], target_free: int) -> bool:
        """回滚最少生成的 decode 请求 1 步，腾出 token 预算。

        hard cap = ``preemption_rollback_steps``：单请求可回滚次数；超 cap
        → 拒绝新请求（避免无限抢占饥饿）。
        """
        candidates = [r for r in batch if r.status == RequestStatus.DECODING and r.generated_ids and r.preemption_count < self.cfg.preemption_rollback_steps]
        if not candidates:
            return False
        # 回滚生成最少者（最廉价）
        candidates.sort(key=lambda r: len(r.generated_ids))
        r = candidates[0]
        if r.generated_ids:
            r.generated_ids.pop()
            # 同步释放最后 token 的 KV（coordinator unref/unref_blocks）
            if self.kv is not None:
                # 简化：coordinator 按 (layer, token_pos) 反向释放（待实现精确）
                pass
        r.preemption_count += 1
        self.metrics.preemption_count += 1
        self.metrics.preemption_rollback_total += 1
        return True

    # -- 后端调用 -----------------------------------------------------------
    def _backend_step(self, batch: list[GenerationRequest]):
        """调后端：L0 组装器接口（per-step 输入 token 张量 + KV 块；L1 Backend
        退化为「逐请求顺序」——MVP 阶段）。

        返回 ``{request_id: sampled_token_id}``（单 token 采样结果）。
        """
        outputs: dict[str, int] = {}
        for r in batch:
            try:
                # L1 兜底路径：每请求用 backend.generate 单独走（无连续批优化）
                if hasattr(self.backend, "generate_step"):
                    tok = self.backend.generate_step(
                        list(r.prompt_ids[: r.prefill_consumed]) + list(r.generated_ids),
                        sampling_params=r.sampling_params,
                    )
                else:
                    # 退化：一次性跑完（无调度）
                    generated = self.backend.generate(
                        list(r.prompt_ids),
                        max_tokens=1,
                        sampling=self._sampling_fn(r.sampling_params),
                    )
                    tok = generated[0] if generated else 0
                outputs[r.request_id] = tok
            except Exception as e:
                r.error = str(e)
                r.status = RequestStatus.EVICTED
                self.metrics.requests_evicted += 1
        return outputs

    def _sampling_fn(self, params: dict):
        import numpy as np

        rng = np.random.default_rng(self._seed_seq)
        self._seed_seq += 1
        temp = float(params.get("temperature", 1.0))
        top_k = int(params.get("top_k", 0))
        top_p = float(params.get("top_p", 1.0))
        min_p = float(params.get("min_p", 0.0))

        def _fn(logits, _pos):
            from ccut.blocks.heads import temperature_topk

            x = logits.squeeze(0).float().numpy() if hasattr(logits, "squeeze") else np.asarray(logits)
            probs = temperature_topk(x, temp, top_k=top_k, top_p=top_p, min_p=min_p)
            return int(rng.choice(len(probs), p=probs))

        return _fn

    # -- 后处理 ------------------------------------------------------------
    def _postprocess(self, batch: list[GenerationRequest], outputs: dict[str, int]):
        for r in batch:
            tok = outputs.get(r.request_id)
            if tok is None:
                continue
            r.generated_ids.append(int(tok))
            self.metrics.decode_tokens_total += 1
            if r.on_token is not None:
                try:
                    r.on_token(int(tok))
                except Exception:
                    pass
            # 完成判定
            if self.cfg.eos_token_id is not None and int(tok) == int(self.cfg.eos_token_id):
                r.status = RequestStatus.FINISHED
                r.finish_reason = "eos"
                self.metrics.requests_finished += 1
                continue
            if len(r.generated_ids) >= r.max_tokens:
                r.status = RequestStatus.FINISHED
                r.finish_reason = "length"
                self.metrics.requests_finished += 1
                continue
            # prefill → decoding 切换
            if r.status == RequestStatus.PREFILLING:
                if r.prefill_consumed >= len(r.prompt_ids):
                    r.status = RequestStatus.DECODING
                    r.prefill_consumed = len(r.prompt_ids)
                    self.metrics.prefill_tokens_total += r.prefill_consumed
                else:
                    # 还在 prefill：放回队尾（下一 step 再来）
                    r.status = RequestStatus.PENDING
                    self._queue.append(r.request_id)

    @classmethod
    def from_app_config(cls, app_config, backend, **kwargs) -> "Engine":
        """从 Config 派生 EngineConfig。"""
        ec = EngineConfig(
            max_num_seqs=app_config.get("engine", {}).get("max_num_seqs", 8),
            max_num_batched_tokens=app_config.get("engine", {}).get("max_num_batched_tokens", 8192),
            chunked_prefill_size=app_config.get("engine", {}).get("chunked_prefill_size", 4096),
            preemption_rollback_steps=8,
            seed=app_config.get("engine", {}).get("seed"),
            eos_token_id=app_config.get("sampling", {}).get("stop_token_ids", [None])[0] if app_config.get("sampling", {}).get("stop_token_ids") else None,
        )
        return cls(ec, backend, **kwargs)
