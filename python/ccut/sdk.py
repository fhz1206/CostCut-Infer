"""ccut.sdk — Python SDK（OpenAI 风格 + 流式 + 投机解码集成）。

L1 形态：包装 OpenAI 客户端协议（httpx + SSE），与 ``ccut.api_server`` 一一对应；
L0 形态（MVP 后）：本地直接调 ``Engine``，跳过 HTTP。

用法::

    from ccut.sdk import CostCutInferClient
    client = CostCutInferClient(base_url="http://localhost:8000", api_key=None)
    for tok in client.chat_stream("hello"):
        print(tok, end="")
"""

from __future__ import annotations

import json
from typing import Any, Iterator

import httpx

__all__ = ["CostCutInferClient", "ChatMessage"]


class ChatMessage:
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}


class CostCutInferClient:
    """HTTP 客户端（与 OpenAI Python SDK 兼容的子集）。"""

    def __init__(self, base_url: str = "http://localhost:8000", api_key: str | None = None, timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._cli = httpx.Client(base_url=self.base_url, timeout=timeout, headers={"Authorization": f"Bearer {api_key}"} if api_key else {})

    def health(self) -> dict:
        r = self._cli.get("/health")
        r.raise_for_status()
        return r.json()

    def list_models(self) -> dict:
        r = self._cli.get("/v1/models")
        r.raise_for_status()
        return r.json()

    def chat(
        self,
        messages: list[ChatMessage] | list[dict],
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 20,
        max_tokens: int = 256,
        stop: list[str] | None = None,
    ) -> str:
        r = self._cli.post(
            "/v1/chat/completions",
            json={
                "messages": [m.to_dict() if isinstance(m, ChatMessage) else m for m in messages],
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "max_tokens": max_tokens,
                "stop": stop or [],
                "stream": False,
            },
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    def chat_stream(
        self,
        messages: list[ChatMessage] | list[dict],
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 20,
        max_tokens: int = 256,
        stop: list[str] | None = None,
    ) -> Iterator[str]:
        """SSE 流式：逐 token yield content。"""
        with self._cli.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "messages": [m.to_dict() if isinstance(m, ChatMessage) else m for m in messages],
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "max_tokens": max_tokens,
                "stop": stop or [],
                "stream": True,
            },
        ) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                if line.startswith("data: "):
                    payload = line[6:].strip()
                    if payload == "[DONE]":
                        return
                    try:
                        d = json.loads(payload)
                        delta = d["choices"][0].get("delta", {})
                        content = delta.get("content")
                        if content:
                            yield content
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue
