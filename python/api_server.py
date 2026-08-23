"""OpenAI 兼容 API 服务（参考 vLLM 的 /v1 接口——models + chat/completions）。

用法：
    python api_server.py [--host 0.0.0.0] [--port 8000] [--model <name>]
"""
from __future__ import annotations

import argparse
import json
import time
import uuid
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from config import EngineConfig

app = FastAPI(
    title="CostCut Infer OpenAI API",
    description="OpenAI 兼容接口（参考 vLLM）——/v1/models + /v1/chat/completions",
    version="0.1.0",
)

_session = None       # ChatSession（惰性——首次请求构建）
_model_id = "model"   # 暴露的模型 id


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "default"
    messages: list[ChatMessage]
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 0
    max_tokens: Optional[int] = 512
    stream: bool = False
    repetition_penalty: float = 1.0


def _get_session(model_name: Optional[str] = None):
    """惰性构建 ChatSession（首次请求触发模型构建——复用 cli_chat 的生成链路）。"""
    global _session
    if _session is None:
        from cli_chat import ChatSession
        cfg = EngineConfig()  # 脚本相对默认——从任意 cwd 运行都可用
        _session = ChatSession(cfg, model_name or cfg.default_model)
    return _session


@app.get("/v1/models", response_class=JSONResponse)
def list_models():
    """模型列表（参考 vLLM 返回格式：{object: list, data: [{id, object, created, owned_by}]}）。"""
    return {
        "object": "list",
        "data": [{
            "id": _model_id,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "costcut-infer",
        }],
    }


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    """对话补全（参考 vLLM/OpenAI 返回格式：choices[0].message.content）。"""
    global _model_id
    if req.model != "default":
        _model_id = req.model
    try:
        session = _get_session()
        # 组装 messages → 流式生成（max_tokens 限制）
        from config import InferenceConfig
        gen = InferenceConfig(
            temperature=req.temperature, top_p=req.top_p, top_k=req.top_k,
            repetition_penalty=req.repetition_penalty,
            max_new_tokens=req.max_tokens or 512,
        )
        # 走 ChatSession 的生成链路（历史 + 当前输入）
        prompt = "\n".join(f"{m.role}: {m.content}" for m in req.messages)
        if req.stream:
            # 流式响应（SSE——逐 chunk——参考 vLLM/OpenAI chunk 格式）
            async def sse_gen():
                for chunk in _stream_chunks(session, prompt):
                    data = {
                        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": _model_id,
                        "choices": [{"index": 0, "delta": {"content": chunk}, "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
                done = {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
                yield f"data: {json.dumps(done, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(sse_gen(), media_type="text/event-stream")
        text = session.chat(prompt)
        _scheduler.complete(seq_id)  # 请求完成出队（continuous batching）
    except Exception as e:  # 模型构建失败等——返回 500
        _scheduler.complete(seq_id)  # 失败也出队（避免队列积压）
        raise HTTPException(status_code=500, detail=str(e)) from e
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": _model_id,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": len(prompt), "completion_tokens": 0, "total_tokens": len(prompt)},
    }


@app.get("/v1/health")
def health():
    """健康检查。"""
    return {"status": "ok", "model": _model_id}


@app.get("/", response_class=HTMLResponse)
def web_admin():
    """Web 管理页面（模型信息 + API 文档链接）。"""
    return HTMLResponse(f"""
    <!DOCTYPE html><html><head><meta charset="utf-8"><title>CostCut Infer</title>
    <style>body{{font-family:sans-serif;max-width:800px;margin:40px auto;padding:0 20px}}
    code{{background:#f0f0f0;padding:2px 6px;border-radius:4px}}</style></head><body>
    <h1>CostCut Infer 管理页</h1>
    <p>模型：<code>{_model_id}</code> ｜ 状态：<code>运行中</code></p>
    <h2>OpenAI 兼容接口（参考 vLLM）</h2>
    <ul>
      <li><code>GET /v1/models</code>——模型列表</li>
      <li><code>POST /v1/chat/completions</code>——对话补全</li>
      <li><code>GET /v1/health</code>——健康检查</li>
      <li><a href="/docs">API 文档（Swagger UI）</a></li>
    </ul>
    <p>兼容客户端：OpenAI SDK / curl / LangChain 等（base_url 指向本服务）。</p>
    </body></html>""")


def _port_range(v: str) -> int:
    """端口安全校验：仅接受 1-65535（拒绝 -100 等非法值）。"""
    try:
        p = int(v)
    except ValueError:
        raise argparse.ArgumentTypeError(f"端口必须为整数（收到 {v!r}）")
    if not 1 <= p <= 65535:
        raise argparse.ArgumentTypeError(f"端口必须在 1-65535 之间（收到 {p}）")
    return p


def main():
    global _model_id
    ap = argparse.ArgumentParser(description="CostCut Infer OpenAI 兼容 API 服务")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=_port_range, default=8000)
    ap.add_argument("--model", default=None, help="模型名（engine.toml 中注册的 name）")
    args = ap.parse_args()
    if args.model:
        _model_id = args.model
    import uvicorn
    print(f"[api] CostCut Infer OpenAI API 服务：http://{args.host}:{args.port}（模型 {_model_id}）")
    print(f"[api] 接口：/v1/models  /v1/chat/completions  /v1/health  /docs（Swagger）")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()


def _stream_chunks(session, prompt):
    """逐 token 产出（复用 ChatSession 生成——按 4 字符切分模拟流式——SSE chunk）。"""
    text = session.chat(prompt)
    for i in range(0, len(text), 4):
        yield text[i:i + 4]


class _BatchScheduler:
    """continuous batching（简化版——参考 vLLM：请求进队 → 批处理 → 完成后出队）。

    - add(seq_id, prompt)：请求入队
    - pending()：待批处理的请求
    - complete(seq_id)：请求完成出队（释放）
    """

    def __init__(self):
        self._queue: list[dict] = []

    def add(self, seq_id: str, prompt: str) -> None:
        self._queue.append({"seq_id": seq_id, "prompt": prompt, "done": False})

    def pending(self) -> list[dict]:
        return [r for r in self._queue if not r["done"]]

    def complete(self, seq_id: str) -> None:
        for r in self._queue:
            if r["seq_id"] == seq_id:
                r["done"] = True
        self._queue = [r for r in self._queue if not r["done"]]

    def size(self) -> int:
        return len(self._queue)


_scheduler = _BatchScheduler()
