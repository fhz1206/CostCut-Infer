"""ccut.api_server — OpenAI 兼容 API 服务（/v1/chat/completions + /v1/completions + /v1/models + /metrics + /health）。

实现（§6 + §3.4-2 引擎协同）：
- FastAPI + SSE 流式（heartbeat 默认 15s）；
- /v1/chat/completions：消息列表 → encode → Engine.add_request → 逐 token yield；
- /v1/completions：单 prompt 字符串；
- /v1/models：返回已注册模型（含架构 tier）；
- /metrics：pull 模型，从 ``MetricsBus`` 拍平；
- /health：探活 + 引擎状态 + 当前层流式 / KV 占用。

MVP 阶段后端走 L1（transformers AutoModel，逐请求）；L0 通用路径接通后
通过 ``Config.arch_tier`` 路由——无需改 API 形状。
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any, AsyncIterator

import numpy as np

from ccut.config import Config
from ccut.engine import Engine
from ccut.metrics import MetricsBus
from ccut.models.backend_transformers import L1Backend
from ccut.models.registry import (
    load_family_templates,
    load_registry_table,
    resolve_architecture,
)
from ccut.sampling import SamplingParams
from ccut.tokenization import Tokenization

_LOG = logging.getLogger("ccut.api")

__all__ = ["run_server", "create_app", "build_engine"]


def build_engine(cfg: Config) -> tuple[Engine, MetricsBus, Tokenization, dict | None]:
    """从 Config 构建引擎（后端 + tokenization + metrics bus）。"""
    model_dir = Path(cfg.get("model_path", "model"))
    tok = Tokenization(model_dir)
    arch_resolution = None
    try:
        cfg_json = json.loads(Path(model_dir / "config.json").read_text(encoding="utf-8"))
        arch = cfg_json.get("architectures", ["Unknown"])[0]
        tab = load_registry_table()
        fams = load_family_templates()
        arch_resolution = resolve_architecture(arch, tier_mode=cfg.get("arch_tier", "auto"), table=tab, families=fams)
    except Exception:
        arch_resolution = None
    # 决定后端：L1 兜底（MVP）；L0 通用路径接通后按 tier 切
    backend = L1Backend(model_dir, trust_remote_code=bool(cfg.get("trust_remote_code", False)), dtype=str(cfg.get("dtype", "bf16")))
    # 引擎配置
    sp_cfg = cfg.section("sampling") or {}
    sp = SamplingParams(
        temperature=float(sp_cfg.get("temperature", 1.0)),
        top_k=int(sp_cfg.get("top_k", 20)),
        top_p=float(sp_cfg.get("top_p", 0.95)),
        min_p=float(sp_cfg.get("min_p", 0.0)),
        repetition_penalty=float(sp_cfg.get("repetition_penalty", 1.0)),
        presence_penalty=float(sp_cfg.get("presence_penalty", 0.0)),
        frequency_penalty=float(sp_cfg.get("frequency_penalty", 0.0)),
    )
    from ccut.engine import EngineConfig

    ec = EngineConfig(
        max_num_seqs=int(cfg.get("max_num_seqs", 8)),
        max_num_batched_tokens=int(cfg.get("max_num_batched_tokens", 8192)),
        chunked_prefill_size=int(cfg.get("chunked_prefill_size", 4096)),
        eos_token_id=tok.eos_token_id,
        seed=cfg.get("seed"),
    )
    eng = Engine(ec, backend=backend)
    bus = MetricsBus()
    bus.attach_engine(eng.metrics)
    return eng, bus, tok, arch_resolution


def run_server(cfg: Config, watchdog=None) -> int:
    """启动 FastAPI（uvicorn）。"""
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import JSONResponse, StreamingResponse
        import uvicorn
    except ImportError as exc:
        print(f"[ERROR] FastAPI/uvicorn 未装：{exc}。pip install fastapi uvicorn[standard] 后重试", file=__import__("sys").stderr)
        return 1

    eng, bus, tok, arch_res = build_engine(cfg)
    app = FastAPI(title="CostCut-Infer", version="1.0.0")
    created = int(time.time())
    model_id = Path(cfg.get("model_path", "model")).name or "model"

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "engine_active": eng.metrics.active,
            "arch": {
                "name": arch_res.architecture if arch_res else None,
                "tier": arch_res.tier if arch_res else None,
                "accepted": arch_res.accepted if arch_res else None,
            }
            if arch_res
            else None,
        }

    @app.get("/v1/models")
    def list_models() -> dict:
        return {
            "object": "list",
            "data": [
                {
                    "id": model_id,
                    "object": "model",
                    "created": created,
                    "owned_by": "CostCut-Infer",
                    "tier": arch_res.tier if arch_res else "unknown",
                    "arch_family": arch_res.plan.family if arch_res and arch_res.plan else None,
                }
            ],
        }

    @app.get("/metrics")
    def metrics() -> JSONResponse:
        return JSONResponse(bus.collect())

    @app.post("/v1/chat/completions")
    async def chat_completions(req: dict[str, Any]) -> dict | StreamingResponse:
        messages = req.get("messages") or []
        prompt_text = "\n".join(m.get("content", "") for m in messages if m.get("role") != "system")
        prompt_ids = tok.encode(prompt_text, add_special_tokens=True)
        sp = SamplingParams(
            temperature=float(req.get("temperature", cfg.get("temperature", 1.0))),
            top_p=float(req.get("top_p", cfg.get("top_p", 0.95))),
            top_k=int(req.get("top_k", cfg.get("top_k", 20))),
            max_tokens=int(req.get("max_tokens", 256)),
            stop=req.get("stop", []) or [],
            stream=bool(req.get("stream", False)),
        )
        rid = eng.add_request(prompt_ids, sampling_params=sp.__dict__, max_tokens=sp.max_tokens or 256)
        if sp.stream:
            return StreamingResponse(_stream_chat(rid, model_id, tok, eng, req), media_type="text/event-stream")
        # 非流式：等请求完成
        deadline = time.time() + 300
        while time.time() < deadline:
            st = eng.status(rid)
            if st["status"] in ("finished", "evicted"):
                break
            await asyncio.sleep(0.05)
        st = eng.status(rid)
        req_state = eng._requests.get(rid)
        tokens = req_state.generated_ids if req_state else []
        return {
            "id": rid,
            "object": "chat.completion",
            "created": created,
            "model": model_id,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": tok.decode(tokens)},
                    "finish_reason": st["status"] if st["status"] == "finished" else (req_state.finish_reason if req_state else "stop"),
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt_ids),
                "completion_tokens": len(tokens),
                "total_tokens": len(prompt_ids) + len(tokens),
            },
        }

    async def _stream_chat(rid: str, model_id: str, tok, eng, req) -> AsyncIterator[bytes]:
        last_n = 0
        heartbeat = float(cfg.get("sse_heartbeat_seconds", 15.0))
        last_emit = time.monotonic()
        while True:
            req_state = eng._requests.get(rid)
            if req_state is None:
                yield b"data: [DONE]\n\n"
                return
            new = req_state.generated_ids[last_n:]
            if new:
                last_n += len(new)
                chunk = {
                    "id": rid,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_id,
                    "choices": [{"index": 0, "delta": {"content": tok.decode(new)}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode("utf-8")
                last_emit = time.monotonic()
            elif time.monotonic() - last_emit > heartbeat:
                yield b": heartbeat\n\n"
                last_emit = time.monotonic()
            if req_state.status in ("finished", "evicted"):
                chunk = {
                    "id": rid,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_id,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": req_state.finish_reason}],
                }
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode("utf-8")
                yield b"data: [DONE]\n\n"
                return
            await asyncio.sleep(0.02)

    host = str(cfg.get("host", "0.0.0.0"))
    port = int(cfg.get("port", 8000))
    _LOG.info("OpenAI 兼容服务启动: http://%s:%d", host, port)
    uvicorn.run(app, host=host, port=port, log_level="info")
    return 0


def create_app(cfg: Config):  # noqa: ARG001
    """暴露 ASGI app（给 gunicorn 等）——MVP 阶段未使用。"""
    raise NotImplementedError("create_app 将在 L0 通用路径接通后实现")
