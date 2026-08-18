"""liteengine：纯 Python 的 Qwen3.6-35B-A3B-AWQ 本地推理引擎（CPU / 低内存）。

设计约定（性能 / 内存友好）：
1. 惰性加载：权重不整包入内存，按需读取（见 liteengine.loader.WeightStore）
2. 避免全量拷贝：张量读取、反量化尽量复用缓冲、按需转换 dtype
3. 导入风格：模块内统一使用 ``from ... import ...``，减少符号表膨胀与查找开销

目录分层：
- 模型/架构：model / model_config / layer / attention / moe / gguf
- 量化：quant/（config / unpack / awq / gptq / fp8 / nvfp4 / dequantize）
- 基础设施：loader / cache / norm / rope / tensor / sampling
- 配置/扩展：config（engine.toml）/ registry（插件注册表）/ speculator（投机）
- 兼容（旧模块）：engine / lazy_loader
"""
from liteengine.config import EngineConfig
from liteengine.loader import WeightStore
from liteengine.model import Qwen3_5MoeModel, load_text_config
from liteengine.quant import dequantize, dequantize_awq

__all__ = ["EngineConfig", "WeightStore", "Qwen3_5MoeModel", "load_text_config",
           "dequantize", "dequantize_awq"]
