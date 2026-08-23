"""CostCut Infer TUI (textual — mirror of cli_chat chat).

Usage: python tui_chat.py [--model <name>]
"""
from __future__ import annotations

import argparse

from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.widgets import Button, Input, OptionList, Static
from textual.widgets.option_list import Option


class ChatTUI(App):
    """TUI: select ALL params (model/device/sampling/features) then chat (all output in English)."""

    CSS = """
    #title { height: 1; text-style: bold; color: #5af; }
    .group-label { height: 1; color: #fa5; text-style: bold; margin-top: 1; }
    OptionList { height: auto; max-height: 3; margin-bottom: 0; }
    #start { margin-top: 1; width: 40; }
    #hint { height: 1; color: #888; margin-top: 1; }
    #history { height: 1fr; border: solid #555; padding: 1; overflow-y: auto; }
    #input-row { height: 3; padding: 1; }
    #input { width: 1fr; }
    #send { width: 12; }
    .user { color: #5af; }
    .assistant { color: #af5; }
    """

    def __init__(self, model_name: str | None = None):
        super().__init__()
        self._model_name = model_name
        self._session = None
        self._params = {  # TUI-selected params (all selectable — no CLI args needed)
            "device_kind": "cpu", "threads": 0, "temperature": 0.7, "top_p": 0.9,
            "top_k": 0, "max_tokens": 512, "speculate": False, "streaming": True,
            "mode": "chat",  # chat | server (OpenAI API)
        }

    def compose(self) -> ComposeResult:
        yield Static("CostCut Infer TUI — configure, then press Start", id="title")
        yield Static("— Mode —", classes="group-label")
        yield OptionList(Option("Chat mode", id="chat"), Option("Server mode (OpenAI API)", id="server"),
                         id="mode-select")
        yield Static("— Model —", classes="group-label")
        # Step 1: model selection (from engine.toml)
        from config import EngineConfig
        cfg = EngineConfig()  # 脚本相对默认——从任意 cwd 运行都可用
        options = []
        for name, m in cfg.models.items():
            note = "" if m.path else " (no path — display only)"
            options.append(Option(f"{name}{note}", id=name))
        if not options:
            options.append(Option("(no models — check engine.toml [model])", id=""))
        yield OptionList(*options, id="model-select")
        # Step 2: device kind
        yield Button("Start chat (all selected)", id="start", variant="primary")
        yield Static("Select each option above, then press Start. (Ctrl+C to quit)", id="hint")
        with Vertical(id="history"):
            pass
        with Vertical(id="input-row"):
            yield Input(placeholder="Type a message… (Ctrl+C to quit)", id="input")
            yield Button("Send", id="send")

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        """Collect ALL param selections (model/device/sampling/features) into _params."""
        wid, val = event.option_list.id, event.option.id  # widget id + selected value
        if wid == "mode-select":
            self._params["mode"] = val
        elif wid == "model-select":
            self._model_name = val

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "start":
            if self._params["mode"] == "server":
                # Server mode: launch OpenAI-compatible API service
                self._start_server()
            else:
                # All params selected -> enter chat (English)
                for wid in ("mode-select", "model-select", "start"):
                    self.query_one(f"#{wid}", OptionList if wid != "start" else Button).display = False
                self.query_one("#title", Static).update(
                    f"Model: {self._model_name} (parameters from engine.toml)"
                )
                self.query_one("#input", Input).focus()
        elif event.button.id == "send":
            self._send()

    def on_mount(self) -> None:
        self.query_one("#input", Input).focus()

    def _start_server(self) -> None:
        """Server mode: launch the OpenAI-compatible API service (reference vLLM)."""
        self.query_one("#title", Static).update(
            f"Starting server (model: {self._model_name}) on http://0.0.0.0:8000 ... (Ctrl+C to stop)")
        import api_server
        if self._model_name:
            api_server._model_id = self._model_name
        api_server.main()

    def _get_session(self):
        if self._session is None:
            from cli_chat import ChatSession
            from config import EngineConfig
            cfg = EngineConfig()  # 脚本相对默认——从任意 cwd 运行都可用
            # parameters come from engine.toml (device/sampling/features — no TUI overrides)
            self._session = ChatSession(cfg, self._model_name or cfg.default_model)
            self.query_one("#title", Static).update("Model building… (first time ~1-2 min)")
        return self._session

    def _append(self, who: str, text: str) -> None:
        history = self.query_one("#history", Vertical)
        history.mount(Static(f"{who}: {text}", classes="user" if who == "你" else "assistant"))


    def on_input_submitted(self, _: Input.Submitted) -> None:
        self._send()

    def _send(self) -> None:
        inp = self.query_one("#input", Input)
        text = inp.value.strip()
        if not text:
            return
        inp.value = ""
        self._append("你", text)
        try:
            session = self._get_session()
            self._append("助手", session.chat(text))
        except Exception as e:
            self._append("错误", str(e))


def main():
    ap = argparse.ArgumentParser(description="CostCut Infer TUI 对话")
    ap.add_argument("--model", default=None, help="模型名（engine.toml 中注册的 name）")
    args = ap.parse_args()
    ChatTUI(args.model).run()


if __name__ == "__main__":
    main()
