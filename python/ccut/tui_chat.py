"""ccut.tui_chat — 终端单请求对话 + 冷启动基准。

用法（被 ``CostCut-Infer.py chat`` / ``CostCut-Infer.py bench`` 调用）::

    run_tui(cfg)            # 进入交互输入
    run_bench(cfg)          # 冷启动基准
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Iterable

import numpy as np

from ccut.api_server import build_engine
from ccut.config import Config
from ccut.sampling import SamplingParams

_LOG = logging.getLogger("ccut.tui")

__all__ = ["run_tui", "run_bench"]


def run_tui(cfg: Config) -> int:
    """TUI 单请求（rich fallback to print）：输入 → 生成 → 逐 token 打印。"""
    try:
        from rich.console import Console
        from rich.markdown import Markdown
    except ImportError:
        Console = None  # type: ignore
    eng, bus, tok, arch_res = build_engine(cfg)
    if Console is not None:
        console = Console()
        console.print(f"[bold green]CostCut-Infer[/bold green] ready (tier={arch_res.tier if arch_res else '?'})")
        try:
            while True:
                console.print("[bold cyan]>>>[/bold cyan] ", end="")
                prompt = input()
                if prompt.strip().lower() in ("/exit", "/quit", "exit", "quit"):
                    break
                _generate_to_console(eng, tok, console, prompt, cfg)
        except (EOFError, KeyboardInterrupt):
            pass
    else:
        print(f"CostCut-Infer ready (tier={arch_res.tier if arch_res else '?'})")
        while True:
            try:
                prompt = input(">>> ")
            except (EOFError, KeyboardInterrupt):
                break
            if prompt.strip().lower() in ("/exit", "/quit", "exit", "quit"):
                break
            _generate_to_print(eng, tok, prompt, cfg)
    return 0


def _generate_to_console(eng, tok, console, prompt: str, cfg: Config) -> None:
    prompt_ids = tok.encode(prompt, add_special_tokens=True)
    sp = SamplingParams(
        temperature=float(cfg.get("temperature", 1.0)),
        top_p=float(cfg.get("top_p", 0.95)),
        top_k=int(cfg.get("top_k", 20)),
        max_tokens=int(cfg.get("max_tokens", 256)),
    )
    rid = eng.add_request(prompt_ids, sampling_params=sp.__dict__, max_tokens=sp.max_tokens or 256)
    last_n = 0
    start = time.perf_counter()
    ttft = None
    while True:
        rs = eng._requests.get(rid)
        if rs is None:
            break
        new = rs.generated_ids[last_n:]
        if new:
            if ttft is None:
                ttft = time.perf_counter() - start
            console.print(tok.decode(new), end="")
            last_n += len(new)
        if rs.status in ("finished", "evicted"):
            break
        time.sleep(0.02)
    dt = time.perf_counter() - start
    n = max(1, len(eng._requests.get(rid).generated_ids) if eng._requests.get(rid) else 0)
    tps = n / max(dt, 1e-6)
    console.print(f"\n\n[dim]gen {n} tokens in {dt:.2f}s | TTFT {ttft*1000 if ttft else 0:.0f}ms | {tps:.2f} tok/s[/dim]")


def _generate_to_print(eng, tok, prompt: str, cfg: Config) -> None:
    prompt_ids = tok.encode(prompt, add_special_tokens=True)
    sp = SamplingParams(
        temperature=float(cfg.get("temperature", 1.0)),
        top_p=float(cfg.get("top_p", 0.95)),
        top_k=int(cfg.get("top_k", 20)),
        max_tokens=int(cfg.get("max_tokens", 256)),
    )
    rid = eng.add_request(prompt_ids, sampling_params=sp.__dict__, max_tokens=sp.max_tokens or 256)
    last_n = 0
    start = time.perf_counter()
    while True:
        rs = eng._requests.get(rid)
        if rs is None:
            break
        new = rs.generated_ids[last_n:]
        if new:
            print(tok.decode(new), end="", flush=True)
            last_n += len(new)
        if rs.status in ("finished", "evicted"):
            break
        time.sleep(0.02)
    print()


def run_bench(cfg: Config) -> int:
    """冷启动基准：warmup + 1 请求推理 + 统计 TTFT / TPS。"""
    eng, bus, tok, arch_res = build_engine(cfg)
    n = 64
    prompt = "Hello, " * 32
    prompt_ids = tok.encode(prompt, add_special_tokens=True)[:256]
    sp = SamplingParams(
        temperature=0.0,
        max_tokens=n,
        greedy=True,
    )
    # warmup
    print("[bench] warmup...")
    rid = eng.add_request(prompt_ids, sampling_params=sp.__dict__, max_tokens=8)
    while eng._requests[rid].status not in ("finished", "evicted"):
        time.sleep(0.05)
    # main
    print("[bench] main run...")
    t0 = time.perf_counter()
    rid = eng.add_request(prompt_ids, sampling_params=sp.__dict__, max_tokens=n)
    last_n = 0
    ttft = None
    while True:
        rs = eng._requests.get(rid)
        if rs is None:
            break
        if ttft is None and rs.generated_ids:
            ttft = time.perf_counter() - t0
        last_n = len(rs.generated_ids)
        if rs.status in ("finished", "evicted"):
            break
        time.sleep(0.02)
    dt = time.perf_counter() - t0
    tps = last_n / max(dt, 1e-6)
    print(f"[bench] generated {last_n} tokens in {dt:.2f}s | TTFT {ttft*1000 if ttft else 0:.0f}ms | {tps:.2f} tok/s")
    print(f"[bench] engine metrics: {eng.metrics.to_dict()}")
    return 0
