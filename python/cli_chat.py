#!/usr/bin/env python3
"""CLI Chat —— liteengine 纯 Python 推理引擎 + transformers tokenizer。

用法：python cli_chat.py（可用环境变量 ENGINE_TOML 指定配置文件）
特性：懒加载（首次对话才构建模型）、流式输出、历史记录、/model 切换。
注意：liteengine 仅支持 Qwen3_5Moe 文本架构，速度受 CPU 限制（约 60s/token）。
"""
# === 双击直跑兼容：让不带 encoding 的文本 open() 默认 UTF-8 ===
# 中文 Windows 下双击运行时不带 PYTHONUTF8，open() 默认按 GBK 解码；
# 此补丁必须在导入任何第三方库之前执行，从根上消除 GBK 解码错误。
import builtins as _builtins

_orig_open = _builtins.open


def _open_utf8_default(file, mode="r", buffering=-1, encoding=None,
                       errors=None, newline=None, closefd=True, opener=None):
    if "b" not in mode and encoding is None:
        encoding = "utf-8"
    return _orig_open(file, mode, buffering, encoding, errors, newline, closefd, opener)


_builtins.open = _open_utf8_default

import os
import sys
from json import load, dump
from pathlib import Path
from typing import Optional

# === 进程内统一 UTF-8 编码（不重启进程）===
if os.name == "nt":
    try:
        import ctypes
        _kernel32 = ctypes.windll.kernel32
        _kernel32.SetConsoleOutputCP(65001)
        _kernel32.SetConsoleCP(65001)
    except Exception:
        pass

for _stream in (sys.stdout, sys.stderr):
    try:
        if not _stream.isatty():
            _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"

from config import EngineConfig
from io_.loader import WeightStore
from engine.engine.model import Qwen3_5MoeModel, load_text_config

# liteengine 仅支持此架构
SUPPORTED_ARCH = "Qwen3_5MoeForConditionalGeneration"


def arch_supported(model_dir: str) -> bool:
    """检查模型目录的 config.json 架构是否被 liteengine 支持。"""
    try:
        with open(Path(model_dir) / "config.json", "r", encoding="utf-8") as f:
            return SUPPORTED_ARCH in load(f).get("architectures", [])
    except Exception:
        return False


class ChatSession:
    """liteengine 对话会话：tokenizer + 模型惰性构建、历史、流式生成。"""

    def __init__(self, engine_config: EngineConfig, model_name: Optional[str] = None):
        self.config = engine_config
        self._cfg = engine_config   # 兼容 _get_model 的 self._cfg.inference.layer_offload 使用
        self.model_name = model_name or engine_config.default_model
        self.model_cfg = self.config.get_model_config(self.model_name)
        self.history: list[dict[str, str]] = []
        self._tokenizer = None
        self._store = None
        self._model = None
        self.speculative_enabled = bool(self.model_cfg.dspark_model)   # 配置 dspark_model 非空即启用
        self._speculator = None
        self._load_history()

    # ---- 惰性构建 ----

    def _get_tokenizer(self):
        """tokenizer（transformers 工具，仅作编码/解码；推理仍为 liteengine）。"""
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_cfg.model_dir, local_files_only=True
            )
            if not getattr(self._tokenizer, "chat_template", None):
                tpl = Path(self.model_cfg.model_dir) / "chat_template.jinja"
                if tpl.exists():
                    self._tokenizer.chat_template = tpl.read_text(encoding="utf-8")
        return self._tokenizer

    def _get_model(self):
        """liteengine 模型（首次调用构建，40 层约需 1-2 分钟）。"""
        if self._model is None:
            self._store = WeightStore(self.model_cfg.model_dir)
            cfg = load_text_config(self.model_cfg.model_dir)
            # 用户可配置：engine.toml [model] 的 rope_type/rope_scaling（YaRN 长文本外推）
            cfg["rope_type"] = self.model_cfg.rope_type
            cfg["rope_scaling"] = self.model_cfg.rope_scaling
            self._model = Qwen3_5MoeModel(
                self._store,
                cfg,
                expert_cache_max=self.model_cfg.expert_cache_max,
                layer_offload=self._cfg.inference.layer_offload,
            )
        return self._model

    def _get_speculator(self):
        """dspark 投机草稿模型（配置 dspark_model 非空时启用）。"""
        if self._speculator is None and self.model_cfg.dspark_model:
            from engine.engine.speculator import DSparkSpeculator
            self._speculator = DSparkSpeculator(self.model_cfg.dspark_model_dir)
        return self._speculator

    def toggle_speculative(self) -> None:
        """切换投机解码（需配置 dspark_model 非空）。"""
        if not self.model_cfg.dspark_model:
            print("[System] 当前模型未配置 dspark_model（engine.toml [model] 中设置），投机解码不可用")
            return
        self.speculative_enabled = not self.speculative_enabled
        print(f"[System] 投机解码：{'已启用（dspark 草稿）' if self.speculative_enabled else '已禁用（标准自回归）'}")

    # ---- 历史 ----

    def _load_history(self) -> None:
        if self.config.chat.auto_save_history and Path(self.config.chat.history_file).exists():
            try:
                with open(self.config.chat.history_file, "r", encoding="utf-8") as f:
                    self.history = load(f)
            except Exception:
                self.history = []

    def _save_history(self) -> None:
        if self.config.chat.auto_save_history:
            try:
                with open(self.config.chat.history_file, "w", encoding="utf-8") as f:
                    dump(self.history[-self.config.chat.max_history:], f,
                         ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"[Warning] Failed to save history: {e}")

    # ---- 生成 ----

    def _format_prompt(self, user_input: str) -> str:
        """组装对话提示词（系统提示词 + 历史 + 当前输入，含 <think> 链路）。"""
        messages = [{"role": "system", "content": self.config.inference.system_prompt}]
        messages.extend(self.history[-self.config.chat.max_history:])
        messages.append({"role": "user", "content": user_input})
        return self._get_tokenizer().apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def generate_stream(self, user_input: str) -> str:
        """流式生成回复（首次调用触发模型构建）。"""
        if self._model is None:
            print("[System] 正在构建模型（首次对话，约需 1-2 分钟，请稍候）...", flush=True)
        model = self._get_model()
        tok = self._get_tokenizer()
        prompt = self._format_prompt(user_input)
        ids = tok(prompt, return_tensors="pt")["input_ids"][0]
        gen = self.config.inference

        print("\nAssistant: ", end="", flush=True)
        out_ids: list[int] = []
        text = ""
        spec = self._get_speculator() if self.speculative_enabled else None
        if spec is not None:
            # dspark 投机解码：草稿 K token → 主模型并行验证 → 投机采样接受
            tokens_iter = iter(model.generate_speculative(
                ids, spec,
                max_new_tokens=gen.max_new_tokens,
                temperature=gen.temperature,
                top_p=gen.top_p,
                top_k=gen.top_k,
                repetition_penalty=gen.repetition_penalty,
                eos_token_id=tok.eos_token_id,
            ))
        else:
            tokens_iter = model.generate_stream(
                ids,
                max_new_tokens=gen.max_new_tokens,
                temperature=gen.temperature,
                top_p=gen.top_p,
                top_k=gen.top_k,
                repetition_penalty=gen.repetition_penalty,
                eos_token_id=tok.eos_token_id,
            )
        for tid in tokens_iter:
            out_ids.append(tid)
            new_text = tok.decode(out_ids, skip_special_tokens=True)
            if len(new_text) > len(text):
                print(new_text[len(text):], end="", flush=True)
                text = new_text
        print()

        response = tok.decode(out_ids, skip_special_tokens=True)
        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": response})
        self._save_history()
        return response

    def clear_history(self) -> None:
        self.history = []
        self._save_history()
        print("[System] Chat history cleared.")

    def switch_model(self, model_name: str) -> None:
        """切换到其它 [model] 块（重建 tokenizer/模型）。"""
        if model_name not in self.config.models:
            print(f"[Error] Model '{model_name}' not found in config.")
            return
        cfg = self.config.get_model_config(model_name)
        if not arch_supported(cfg.model_dir):
            print(f"[Error] 模型 '{model_name}' 的架构不被 liteengine 支持"
                  f"（仅支持 {SUPPORTED_ARCH}）。")
            return
        self.model_name = model_name
        self.model_cfg = cfg
        self._tokenizer = None
        self._store = None
        self._model = None
        self.history = []
        print(f"[System] Switched to model: {model_name}")


def print_help() -> None:
    help_text = """
Available Commands:
  /help           Show this help message
  /model [name]   Switch to a different model (e.g., /model Qwen3.6-35B-A3B-AWQ-4bit)
  /models         List all configured models
  /speculate      Toggle dspark speculative decoding (requires dspark_model configured)
  /clear          Clear chat history
  /exit, /quit    Exit the application
"""
    print(help_text)


def list_models(config: EngineConfig) -> None:
    print("\nConfigured Models:")
    for i, (name, cfg) in enumerate(config.models.items(), 1):
        mark = "" if arch_supported(cfg.model_dir) else "  [liteengine 不支持]"
        print(f"  {i}. {name}{mark}")
    if not config.models:
        print("  No models configured.")
    print()


def main() -> None:
    # 默认配置路径按脚本所在目录解析（python/engine.toml），与运行 CWD 无关
    config_path = os.environ.get("ENGINE_TOML") or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "engine.toml")
    try:
        config = EngineConfig(config_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        sys.exit(1)

    session = ChatSession(config)
    print("=" * 50)
    print("CLI Chat (liteengine 纯 Python 推理引擎)")
    print(f"Default Model: {session.model_name}")
    if session.model_cfg.dspark_model:
        print(f"Speculative (dspark): enabled — {session.model_cfg.dspark_model}")
    print("Type /help for commands, /exit to quit")
    print("=" * 50)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n[System] Goodbye!")
            break

        if not user_input:
            continue

        if user_input.startswith("/"):
            parts = user_input.split(maxsplit=1)
            cmd = parts[0].lower()
            if cmd in ("/exit", "/quit"):
                print("[System] Goodbye!")
                break
            elif cmd == "/help":
                print_help()
            elif cmd == "/clear":
                session.clear_history()
            elif cmd == "/model":
                if len(parts) < 2:
                    print("[Error] Usage: /model <name>")
                else:
                    session.switch_model(parts[1])
            elif cmd == "/models":
                list_models(config)
            elif cmd == "/speculate":
                session.toggle_speculative()
            else:
                print(f"[Error] Unknown command: {cmd}")
            continue

        try:
            session.generate_stream(user_input)
        except Exception as e:
            print(f"\n[Error] {str(e)}")


if __name__ == "__main__":
    main()
