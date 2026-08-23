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
    /* opencode 调研成果：tokyonight 风格——深蓝背景 + 蓝强调 + 绿成功 + 灰层次 */
    Screen { background: #1a1b26; }
    #brand { height: 6; content-align: center middle; background: #16161e; }
    #brand-name { color: #7aa2f7; text-align: center; text-style: bold; }
    #brand-tag { color: #565f89; text-align: center; }
    .section { border: tall #414868; margin: 2 6; padding: 1 3; background: #16161e; }
    .section-title { color: #7aa2f7; text-style: bold; text-align: center; margin-bottom: 1; }
    OptionList { height: auto; max-height: 4; border: none; background: transparent; }
    OptionList > .option-list--option { color: #c0caf5; }
    OptionList > .option-list--option:hover { background: #2f334d; }
    OptionList:focus { border: tall #7aa2f7; }
    #start { margin: 1 6; width: 100%; background: #7aa2f7; color: #1a1b26; }
    #start:hover { background: #9db4f0; }
    #footer { height: 1; color: #414868; text-align: center; }
    #history { height: 1fr; border: solid #414868; padding: 1; overflow-y: auto; }
    #input-row { height: 3; padding: 1; }
    #input { width: 1fr; }
    #send { width: 12; }
    .user { color: #7aa2f7; }
    .assistant { color: #9ece6a; }
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
        with Vertical(id="brand"):
            yield Static("CostCut Infer", id="brand-name")
            yield Static("MoE inference engine — CPU / GPU / NPU / APU", id="brand-tag")
        with Vertical(classes="section"):
            yield Static("MODE", classes="section-title")
            yield OptionList(Option("Chat — interactive conversation", id="chat"),
                             Option("Server — OpenAI-compatible API", id="server"),
                             id="mode-select")
        yield Button("Start", id="start", variant="primary")
        yield Static("Pick a mode, then press Start.  (Ctrl+C to quit)", id="footer")

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        """Collect ALL param selections (model/device/sampling/features) into _params."""
        wid, val = event.option_list.id, event.option.id  # widget id + selected value
        if wid == "mode-select":
            self._params["mode"] = val

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "start":
            if self._params["mode"] == "server":
                # Server mode: launch OpenAI-compatible API service
                self._start_server()
            else:
                # All params selected -> enter chat (English)
                # hide redundant home components (brand/mode/start/footer-hint) — chat page only
                for wid in ("brand", "mode-select", "start"):
                    self.query_one(f"#{wid}", OptionList if wid == "mode-select" else (Button if wid == "start" else Vertical)).display = False
                self.query_one("#footer", Static).update("Chat mode — type a message below. (Ctrl+C to quit)")
                # mount chat UI dynamically (not on home page)
                self.mount(Vertical(id="history"))
                self.mount(Vertical(
                    Input(placeholder="Type a message… (Ctrl+C to quit)", id="input"),
                    Button("Send", id="send"),
                    id="input-row",
                ))
                self.query_one("#input", Input).focus()
        elif event.button.id == "send":
            self._send()

    def on_mount(self) -> None:
        # 首页焦点在模式选择（#input 在 Start 后动态挂载——首页无输入框）
        self.query_one("#mode-select", OptionList).focus()

    def _start_server(self) -> None:
        """Server mode: launch the OpenAI-compatible API service (reference vLLM)."""
        # hide redundant home components (brand/mode/start) — server page only
        for wid in ("brand", "mode-select", "start"):
            self.query_one(f"#{wid}", OptionList if wid == "mode-select" else (Button if wid == "start" else Vertical)).display = False
        self.query_one("#footer", Static).update(
            f"Starting server (model: {self._model_name}) on http://0.0.0.0:8000 ... (Ctrl+C to stop)")
        import api_server
        from config import EngineConfig
        api_server._model_id = EngineConfig().default_model
        api_server.main()

    def _get_session(self):
        if self._session is None:
            from cli_chat import ChatSession
            from config import EngineConfig
            cfg = EngineConfig()  # 脚本相对默认——从任意 cwd 运行都可用
            # parameters come from engine.toml (device/sampling/features — no TUI overrides)
            # model from toml (default_model) — no TUI selection
            self._session = ChatSession(cfg, cfg.default_model)
            self.query_one("#footer", Static).update("Model building… (first time ~1-2 min)")
        return self._session

    def _append(self, who: str, text: str) -> None:
        history = self.query_one("#history", Vertical)
        history.mount(Static(f"{who}: {text}", classes="user" if who == "You" else "assistant"))


    def on_input_submitted(self, _: Input.Submitted) -> None:
        self._send()

    def _send(self) -> None:
        inp = self.query_one("#input", Input)
        text = inp.value.strip()
        if not text:
            return
        inp.value = ""
        self._append("You", text)
        try:
            session = self._get_session()
            self._append("Assistant", session.chat(text))
        except Exception as e:
            self._append("错误", str(e))


def main():
    ap = argparse.ArgumentParser(description="CostCut Infer TUI 对话")
    ap.add_argument("--model", default=None, help="模型名（engine.toml 中注册的 name）")
    args = ap.parse_args()
    ChatTUI(args.model).run()


if __name__ == "__main__":
    main()
