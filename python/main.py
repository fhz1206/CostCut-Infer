"""CostCut Infer（python 版）统一入口。

用法：
    python main.py chat [--model <name>]    # CLI 对话（原 cli_chat.py）
    python main.py tui  [--model <name>]    # TUI 对话（textual——原 tui_chat.py）
    python main.py api  [--host ..] [--port ..] [--model <name>]   # OpenAI 兼容 API（原 api_server.py）
"""
from __future__ import annotations

import argparse
import sys


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python main.py", description="CostCut Infer (python) unified entry — no args = TUI")
    sub = ap.add_subparsers(dest="cmd")

    p_chat = sub.add_parser("chat", help="CLI 对话（原 cli_chat.py）")
    p_chat.add_argument("--model", default=None, help="模型名（engine.toml 中注册）")

    p_tui = sub.add_parser("tui", help="TUI 对话（textual——原 tui_chat.py）")
    p_tui.add_argument("--model", default=None, help="模型名（engine.toml 中注册）")

    p_api = sub.add_parser("api", help="OpenAI 兼容 API 服务（原 api_server.py——参考 vLLM）")
    p_api.add_argument("--host", default="0.0.0.0", help="监听地址（默认 0.0.0.0）")
    p_api.add_argument("--port", type=int, default=8000, help="监听端口（默认 8000）")
    p_api.add_argument("--model", default=None, help="模型名（engine.toml 中注册）")
    return ap


def main() -> None:
    args = _build_parser().parse_args()
    cmd = args.cmd or "tui"  # no argument -> TUI by default

    if cmd == "chat":
        # 转发参数给 cli_chat.main（argparse 兼容）
        argv = ["cli_chat.py"]
        if args.model:
            argv.append(f"--model={args.model}")
        sys.argv = argv
        import cli_chat
        cli_chat.main()
    elif cmd == "tui":
        import tui_chat
        tui_chat.main()
    elif cmd == "api":
        import api_server
        api_server.main()


if __name__ == "__main__":
    main()
