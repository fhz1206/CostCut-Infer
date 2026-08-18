"""liteengine 配置：解析 engine.toml（[default] / [model] / [inference] / [chat]）。

面向 liteengine 纯 Python 引擎，字段精简为 name/path/expert_cache_max、
采样参数与聊天设置；不再包含旧 transformers 加载器的 mmap/shard_lazy 等选项。
"""
from __future__ import annotations

from re import match
from tomllib import loads
from dataclasses import dataclass
from pathlib import Path

__all__ = ["EngineConfig", "ModelConfig", "InferenceConfig", "ChatConfig"]


def _normalize_model_tables(text: str) -> str:
    """把重复的 [model] 区块（一个模型一块）归一化为 TOML 合法的 [[model]] 数组元素。"""
    lines = text.splitlines(keepends=True)
    out: list[str] = []
    for line in lines:
        stripped = line.rstrip("\r\n")
        if match(r"^\[model\][ \t]*(?:#.*)?$", stripped):
            out.append("[[model]]" + stripped[len("[model]"):] + line[len(stripped):])
        else:
            out.append(line)
    return "".join(out)


@dataclass
class ModelConfig:
    name: str                        # 模型短名（/model 切换用，须唯一）
    path: str = ""                   # 模型目录；留空默认 models/<name>
    expert_cache_max: int = 128      # 专家反量化缓存上限（条目，每条约 12MB）
    dspark_model: str = ""           # 投机解码草稿模型目录；留空 = 禁用投机（标准自回归）

    @property
    def model_dir(self) -> str:
        """模型实际目录：path 未指定时默认 models/<name>。"""
        return self.path or f"models/{self.name}"

    @property
    def dspark_model_dir(self) -> str:
        """投机草稿模型目录：直接路径存在则用；否则按 models/ 下名字解析。

        兼容两种写法：``dspark_model = "models/Qwen3.6-35B-A3B-speculator.dspark"``
        或简写 ``dspark_model = "Qwen3.6-35B-A3B-speculator.dspark"``。
        """
        if not self.dspark_model:
            return ""
        v = self.dspark_model.strip()
        if Path(v).is_dir():
            return v
        return f"models/{v}"


@dataclass
class InferenceConfig:
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 0                   # 0 = 关闭 top-k 截断
    repetition_penalty: float = 1.0  # 1.0 = 不惩罚
    max_new_tokens: int = 2048
    system_prompt: str = "你是一个乐于助人的中文助手。请始终用中文回答，保持准确、简洁、条理清晰。"
    compute_dtype: str = "float32"        # 计算精度：fp32（CPU 实测最优）/ fp16 / bf16（可选）
    expert_parallel: bool = False         # 专家多线程并行（本机实测慢 ~5 倍——默认关，他机可试）
    int4_fused_matmul: bool = False       # int4 融合 matmul（本机实测慢——默认关，他机可试）
    speculate: bool = False               # 投机解码（本机实测慢——默认关；/speculate 可运行时切换）


@dataclass
class ChatConfig:
    max_history: int = 20
    auto_save_history: bool = False
    history_file: str = ".chat_history.json"


class EngineConfig:
    """liteengine 的 engine.toml 解析器。"""

    def __init__(self, config_path: str = "engine.toml"):
        self.config_path = Path(config_path)
        self.default_model: str = ""
        self.models: dict[str, ModelConfig] = {}
        self.inference = InferenceConfig()
        self.chat = ChatConfig()
        self._load()

    def _load(self) -> None:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        with open(self.config_path, "r", encoding="utf-8") as f:
            data = loads(_normalize_model_tables(f.read()))

        self.default_model = str(data.get("default", {}).get("model", ""))

        model_data = data.get("model")
        if model_data is None:
            raise ValueError("engine.toml 缺少 [model] 区块（每个模型一个 [model] 块，name 必填）")
        if not isinstance(model_data, list):
            raise ValueError("engine.toml 的 [model] 必须一个模型一块（重复 [model] 头）")
        for m in model_data:
            name = str(m.get("name", "")).strip()
            if not name:
                raise ValueError("[model] 缺少必填字段 name（模型短名，需唯一）")
            if name in self.models:
                raise ValueError(f"[model] 存在重复的 name：{name}")
            self.models[name] = ModelConfig(
                name=name,
                path=str(m.get("path", "")),
                expert_cache_max=int(m.get("expert_cache_max", 128)),
                dspark_model=str(m.get("dspark_model", "")),
            )

        if self.default_model and self.default_model not in self.models:
            raise ValueError(
                f"[default] 的 model={self.default_model!r} 未在 [model] 块中注册，"
                "请检查 engine.toml 的 name 是否一致。"
            )

        inf = data.get("inference", {})
        self.inference = InferenceConfig(
            temperature=float(inf.get("temperature", 0.7)),
            top_p=float(inf.get("top_p", 0.9)),
            top_k=int(inf.get("top_k", 0)),
            repetition_penalty=float(inf.get("repetition_penalty", 1.0)),
            max_new_tokens=int(inf.get("max_new_tokens", 2048)),
            system_prompt=str(inf.get("system_prompt", "You are a helpful assistant.")),
            compute_dtype=str(inf.get("compute_dtype", "float32")),
            expert_parallel=bool(inf.get("expert_parallel", False)),
            int4_fused_matmul=bool(inf.get("int4_fused_matmul", False)),
            speculate=bool(inf.get("speculate", False)),
        )

        chat = data.get("chat", {})
        self.chat = ChatConfig(
            max_history=int(chat.get("max_history", 20)),
            auto_save_history=bool(chat.get("auto_save_history", False)),
            history_file=str(chat.get("history_file", ".chat_history.json")),
        )

    def get_model_config(self, name: str) -> ModelConfig:
        if name in self.models:
            return self.models[name]
        return ModelConfig(name=name)
