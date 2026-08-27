"""CostCut-Infer 推理引擎 — 启动入口。

用法::

    python CostCut-Infer.py --list-params
    python CostCut-Infer.py --list-architectures
    python CostCut-Infer.py --info
    python CostCut-Infer.py serve --api-port 8000
    python CostCut-Infer.py chat --prompt "..."  # TUI 单请求
    python CostCut-Infer.py bench  # 自检基准

启动流程（§4 启动期）：
1. 解析三源配置（CLI > toml > env[CCUT_…] > 默认）；
2. 平台探测（CPU/内存/盘速）+ 启动报告；
3. 架构账本查找（registry.resolve_architecture）；
4. 模型装配（quant + KV + 权重 + 引擎）；
5. 资源限制（CPU/内存/IO 门控）+ watchdog 启动；
6. 进入目标模式（serve / chat / bench）。
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import ccut
from ccut.config import Config, ConfigError, parse_cli_args, render_params_table
from ccut.platforms import PlatformReport, benchmark_disk, detect_cpu, detect_memory, render_startup_report
from ccut.resources import build_resource_budget, render_budget_table
from ccut.resources.limiter import apply_resource_limits
from ccut.resources.watchdog import ResourceWatchdog
from ccut.models.registry import (
    list_families,
    list_l0_architectures,
    load_family_templates,
    load_registry_table,
    resolve_architecture,
)

_LOG = logging.getLogger("ccut")
_DEFAULT_TOML = Path(__file__).resolve().parent / "engine.toml"


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="CostCut-Infer",
        description=(
            f"CostCut-Infer v{ccut.__version__} — CPU 单机零驻留 MoE 推理引擎。"
            "参数大小写不敏感；详细列表见 --list-params。"
        ),
    )
    # 顶层开关（无前缀）
    p.add_argument("--config", type=Path, default=_DEFAULT_TOML, help=f"toml 配置文件（默认 {_DEFAULT_TOML}）")
    p.add_argument("--list-params", action="store_true", help="打印全部参数表后退出")
    p.add_argument("--list-architectures", action="store_true", help="打印已支持架构清单后退出")
    p.add_argument("--info", action="store_true", help="打印平台/资源/启动报告后退出")
    p.add_argument("--version", action="version", version=f"CostCut-Infer v{ccut.__version__}")

    sub = p.add_subparsers(dest="mode", title="mode", required=False)
    sub.add_parser("serve", help="OpenAI 兼容 API 服务（api_server.py）")
    sub.add_parser("chat", help="TUI 单请求对话（tui_chat.py）")
    sub.add_parser("bench", help="冷启动基准（warmup + 1 请求推理）")
    return p


def _dispatch_special_flags(args: argparse.Namespace, cfg: Config | None) -> bool:
    """处理 --list-params / --list-architectures / --info，返回是否已处理（应退出）。"""
    if args.list_params:
        print(render_params_table())
        return True
    if args.list_architectures:
        try:
            tab = load_registry_table()
            fams = list_families()
        except FileNotFoundError as e:
            print(f"[ERROR] {e}", file=sys.stderr)
            return True
        print("=== CostCut-Infer 架构清单 ===")
        print(f"vLLM registry 统计：{tab['total']} 条 | 层级 {tab['tier_counts']}")
        print(f"已落地家族: {fams}")
        print()
        print(f"L0（原生快速）: {len(list_l0_architectures(tab))} 条")
        for a in list_l0_architectures(tab)[:30]:
            entry = tab["architectures"][a]
            print(f"  {a:60s}  family={entry.get('family')!r}")
        if len(list_l0_architectures(tab)) > 30:
            print(f"  ... +{len(list_l0_architectures(tab)) - 30} more")
        print()
        print("L1（transformers 兜底）: 用 L1Backend 自动降级（详细 reason 见账本）")
        print("L2（显式拒绝）: 见 registry_table.json architectures.<name>.reason")
        return True
    if args.info:
        cfg = cfg or Config.build()
        report = PlatformReport.probe()
        budget = build_resource_budget(
            cfg.section("resources"),
            report.disk.sequential1m_mbps,
            report.cpu.logical_cores,
            report.memory.total_gb,
        )
        print(render_startup_report(report, render_budget_table(budget, int(cfg.get("resource_pct", 50)))))
        return True
    return False


def _try_load_architecture(cfg: Config):
    """config.json 优先（model_path 推断架构名），无 model_path 跳过。"""
    model_path = cfg.get("model_path", "model")
    config_json = Path(model_path) / "config.json"
    if not config_json.exists():
        return None
    try:
        data = json.loads(config_json.read_text(encoding="utf-8"))
    except Exception:
        return None
    archs = data.get("architectures") or [data.get("model_type") or "Unknown"]
    table = load_registry_table()
    families = load_family_templates()
    return resolve_architecture(archs[0], tier_mode=cfg.get("arch_tier", "auto"), table=table, families=families)


def _apply_resources(cfg: Config) -> tuple[ResourceWatchdog | None, object | None]:
    """启动期资源门控：CPU/IO 写限制 + watchdog 后台采样。"""
    probe = PlatformReport.probe()
    budget = build_resource_budget(
        cfg.section("resources"),
        probe.disk.sequential1m_mbps,
        probe.cpu.logical_cores,
        probe.memory.total_gb,
    )
    applied = apply_resource_limits(budget)
    _LOG.info("资源限制已应用: %s", applied)
    metrics_bus = None  # 主循环 attach
    wd = ResourceWatchdog(budget, metrics_bus=metrics_bus)
    wd.start()
    return wd, budget


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = _build_argparser()
    args = parser.parse_args(argv)

    # 三源配置解析：CLI + toml + env
    raw_argv = [a for a in (argv or sys.argv[1:]) if not a.startswith(("serve", "chat", "bench", "--list", "--info", "--version", "--help", "-h"))]
    try:
        cfg = Config.build(argv=raw_argv, toml_path=args.config)
    except ConfigError as e:
        print(f"[CONFIG ERROR] {e}", file=sys.stderr)
        return 2

    if _dispatch_special_flags(args, cfg):
        return 0

    # 启动期报告（先打印一次，让用户看到状态）
    probe = PlatformReport.probe()
    budget = build_resource_budget(
        cfg.section("resources"),
        probe.disk.sequential1m_mbps,
        probe.cpu.logical_cores,
        probe.memory.total_gb,
    )
    print(render_startup_report(probe, render_budget_table(budget, int(cfg.get("resource_pct", 50)))))

    # 架构解析（仅 print；装配在子命令内做）
    res = _try_load_architecture(cfg)
    if res is not None:
        print(f"\n[arch] {res.architecture} → tier={res.tier} | plan.family={res.plan.family if res.plan else None} | accepted={res.accepted}")
        if not res.accepted:
            print(f"[arch] reason: {res.reason}", file=sys.stderr)

    # 资源门控
    wd, _ = _apply_resources(cfg)
    try:
        mode = args.mode or "serve"
        if mode == "serve":
            from ccut.api_server import run_server  # type: ignore

            return run_server(cfg, watchdog=wd)
        if mode == "chat":
            from ccut.tui_chat import run_tui  # type: ignore

            return run_tui(cfg)
        if mode == "bench":
            from ccut.tui_chat import run_bench  # type: ignore

            return run_bench(cfg)
        print(f"未知 mode: {mode}", file=sys.stderr)
        return 2
    finally:
        if wd is not None:
            wd.stop()


if __name__ == "__main__":
    raise SystemExit(main())
