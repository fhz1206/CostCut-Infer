from __future__ import annotations

"""
Chat Engine Configuration Module
Python 3.14 Compatible

配置文件 engine.toml 结构：
    [default]    默认模型（填某个 [model] 块的 name）
    [model]      模型列表（每个模型一个 [model] 块，字段均可省略；只支持这种写法）
    [inference]  生成参数
    [chat]       对话历史设置
"""
import re
import tomllib
from pathlib import Path
from dataclasses import dataclass, field

import os
os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"

# shard_lazy 未自定义时默认跳过的权重前缀（视觉编码器 / MTP 预测头）
DEFAULT_SHARD_SKIP_PREFIXES = ["model.visual.", "mtp."]


def _normalize_model_tables(text: str) -> str:
    """把重复的 [model] 区块（一个模型一块）归一化为 TOML 合法的
    [[model]] 数组元素（仅支持 [model] 这一种写法）。"""
    lines = text.splitlines(keepends=True)
    out: list[str] = []
    for line in lines:
        stripped = line.rstrip("\r\n")
        if re.match(r"^\[model\][ \t]*(?:#.*)?$", stripped):
            out.append("[[model]]" + stripped[len("[model]"):] + line[len(stripped):])
        else:
            out.append(line)
    return "".join(out)


@dataclass
class ModelConfig:
    """单个模型的配置。name 必填，其余字段均可省略（使用默认值）。"""

    name: str                        # 模型短名：/model 切换、/models 显示时使用，必须唯一
    path: str = ""                   # 模型目录；留空时默认 models/<name>
    type: str = "transformers"       # 加载方式：transformers / custom
    dtype: str = "float16"           # 权重精度：float16 / bfloat16 / float32 / auto
    device_map: str = "auto"         # 设备分配：auto / cpu / cuda:0 / ...

    # ---- 可选加载优化（engine.toml 的 [model] 块内直接平铺设置）----
    mmap: bool = False               # 内存映射：权重按需分页读入，降低峰值内存
    shard_lazy: bool = False         # 分片按需加载：跳过用不到的 safetensors 分片
    weight_sharing: bool = False     # 权重共享：多个条目指向同一目录时复用已加载模型
    expert_offload: bool = False     # MoE 专家卸载：专家权重换出到磁盘，按需换入
    expert_offload_cache: int = 8    # expert_offload 时内存中驻留的专家数（LRU 上限）
    lazy: bool = True                # 懒加载：默认 true（首次对话才加载，启动零开销）
    shard_skip_prefixes: list = field(default_factory=list)  # 分片跳过前缀（配合 shard_lazy）

    @property
    def model_dir(self) -> str:
        """模型实际目录：path 未指定时默认 models/<name>。"""
        return self.path or f"models/{self.name}"

    @property
    def effective_shard_skip_prefixes(self) -> list[str]:
        """实际生效的分片跳过前缀（未自定义时用默认值）。"""
        prefixes = self.shard_skip_prefixes or DEFAULT_SHARD_SKIP_PREFIXES
        return [str(p) for p in prefixes]


@dataclass
class InferenceConfig:
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 2048
    repetition_penalty: float = 1.1
    system_prompt: str = "You are a helpful assistant."


@dataclass
class ChatConfig:
    max_history: int = 20
    auto_save_history: bool = False
    history_file: str = ".chat_history.json"


class EngineConfig:
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
            text = f.read()

        # 手写的重复 [model] 区块（一个模型一块）：TOML 不允许重复声明同一张表，
        # 解析前把每个 [model] 头归一化为 [[model]] 数组元素。
        data = tomllib.loads(_normalize_model_tables(text))

        self.default_model = str(data.get("default", {}).get("model", ""))

        # 只允许 [model] 写法：残留的 models 相关表（旧格式）一律拒绝
        if "models" in data:
            raise ValueError(
                "engine.toml 的模型区块只允许 [model] 写法（每个模型一个 [model] 块），\n"
                "不再支持 [models] / [[models]] 等旧写法，请迁移到 [model] 区块。"
            )
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
                type=str(m.get("type", "transformers")),
                dtype=str(m.get("dtype", "float16")),
                device_map=str(m.get("device_map", "auto")),
                mmap=bool(m.get("mmap", False)),
                shard_lazy=bool(m.get("shard_lazy", False)),
                weight_sharing=bool(m.get("weight_sharing", False)),
                expert_offload=bool(m.get("expert_offload", False)),
                expert_offload_cache=int(m.get("expert_offload_cache", 8)),
                lazy=bool(m.get("lazy", True)),
                shard_skip_prefixes=[str(p) for p in m.get("shard_skip_prefixes", [])],
            )

        inf = data.get("inference", {})
        self.inference = InferenceConfig(
            temperature=float(inf.get("temperature", 0.7)),
            top_p=float(inf.get("top_p", 0.9)),
            max_tokens=int(inf.get("max_tokens", 2048)),
            repetition_penalty=float(inf.get("repetition_penalty", 1.1)),
            system_prompt=str(inf.get("system_prompt", "You are a helpful assistant.")),
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
        # 未注册的模型按默认配置处理（目录默认 models/<name>）
        return ModelConfig(name=name)
