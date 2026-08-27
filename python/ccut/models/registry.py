"""ccut.models.registry — 架构账本查找（R8：零静默「未处理」）。

账本 = ``registry_table.json``（P0-5 由 sync_vllm_registry 从 vLLM registry 机械生成，
入 git）+ 族模板目录 ``families/*.json``（L0 可表达家族）。

查找流程 :func:`resolve_architecture`::

    config.json.architectures
      → 账本查 tier（L0/L1/L2）
      → L0：族模板匹配（family 字段 → families/<family>.json）→ 返回 L0 计划
      → L1：返回 transformers 兜底计划（L1 架构可用，三大机制不生效）
      → L2：显式拒绝（带账本 reason，含 vLLM 移除版本/OOT 插件理由）

``--arch-tier`` 语义（§4）：
- ``auto``（默认）：L0 优先，缺族模板则 L1 兜底（L2 恒拒）；
- ``strict``：仅 L0，L1 架构显式报错（列出可用家族）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "RegistryEntry",
    "FamilyPlan",
    "ArchResolution",
    "load_registry_table",
    "load_family_templates",
    "resolve_architecture",
    "list_families",
    "list_l0_architectures",
]

_FAMILY_DIR = Path(__file__).parent / "families"
_TABLE_PATH = Path(__file__).parent / "registry_table.json"


@dataclass(frozen=True)
class RegistryEntry:
    """账本单条目。"""

    architecture: str
    tier: str  # L0 | L1 | L2
    family: str | None
    module: str | None
    vllm_class: str | None
    task: str
    reason: str
    text_side: dict | None = None  # 多模态包装器 → 文本侧条目（arch/tier/family）


@dataclass(frozen=True)
class FamilyPlan:
    """族模板（L0 数据流声明，families/<family>.json 载入）。"""

    family: str
    data: dict

    @property
    def layer_kind(self) -> str:
        """层模板类型：``hybrid_gdn``（linear+full 交替）/ ``all_full`` / ``all_linear``。"""
        return str(self.data.get("layer_kind", "all_full"))

    @property
    def expert_tensor_pattern(self) -> str:
        return str(self.data.get("expert_tensor_pattern", ""))

    @property
    def tensor_prefix(self) -> str:
        return str(self.data.get("tensor_prefix", "model."))


@dataclass(frozen=True)
class ArchResolution:
    """架构解析结果：tier + 执行计划 + 拒绝理由（L2）。"""

    architecture: str
    tier: str
    plan: FamilyPlan | None
    reason: str

    @property
    def accepted(self) -> bool:
        return self.tier in ("L0", "L1")


def load_registry_table(path: str | Path | None = None) -> dict:
    """载入 registry_table.json（缺失时显式报错——提示先跑 sync 工具）。"""
    p = Path(path) if path else _TABLE_PATH
    if not p.exists():
        raise FileNotFoundError(
            f"架构账本缺失: {p}。先运行: python -m ccut.tools.sync_vllm_registry"
        )
    with open(p, "rb") as fh:
        return json.load(fh)


def load_family_templates(family_dir: str | Path | None = None) -> dict[str, FamilyPlan]:
    """载入 families/*.json → {family: FamilyPlan}。"""
    d = Path(family_dir) if family_dir else _FAMILY_DIR
    out: dict[str, FamilyPlan] = {}
    for f in sorted(d.glob("*.json")):
        with open(f, "rb") as fh:
            data = json.load(fh)
        out[str(data.get("family", f.stem))] = FamilyPlan(str(data.get("family", f.stem)), data)
    return out


def _entry_from_dict(arch: str, d: dict) -> RegistryEntry:
    return RegistryEntry(
        architecture=arch,
        tier=str(d.get("tier", "L1")),
        family=d.get("family"),
        module=d.get("module"),
        vllm_class=d.get("vllm_class"),
        task=str(d.get("task", "text_generation")),
        reason=str(d.get("reason", "")),
        text_side=d.get("text_side"),
    )


def _resolve_with_text_side(
    architecture: str,
    entry: RegistryEntry,
    tier_mode: str,
    families: dict[str, FamilyPlan],
) -> ArchResolution | None:
    """多模态包装器条目：文本侧存在时按 text_side 递归解析（L0 优先）。

    返回 None 表示无 text_side（走常规流程）。
    """
    ts = entry.text_side
    if not ts or tier_mode.casefold() == "strict" and False:
        return None
    ts_arch = str(ts.get("arch", ""))
    ts_tier = str(ts.get("tier", "L1"))
    ts_family = ts.get("family")
    if ts_tier == "L0":
        plan = families.get(str(ts_family or ""))
        if plan is None:
            if tier_mode.casefold() == "strict":
                return ArchResolution(
                    architecture, "L2", None,
                    f"文本侧家族 {ts_family!r} 无族模板——strict 模式拒绝。可用家族: {sorted(families)}",
                )
            return ArchResolution(
                architecture, "L1", None,
                f"文本侧家族 {ts_family!r} 族模板未落地 → auto 降级 L1（visual 侧禁用）",
            )
        return ArchResolution(
            architecture, "L0", plan,
            f"多模态包装器：文本侧按 {ts_arch}（家族 {ts_family}）走 L0 原生快速路径；visual 侧禁用",
        )
    return ArchResolution(
        architecture, "L1", None,
        f"多模态包装器：文本侧 {ts_arch} 为 L1 → 整体 L1 transformers 兜底（visual 侧禁用）",
    )


def resolve_architecture(
    architecture: str,
    tier_mode: str = "auto",
    table: dict | None = None,
    families: dict[str, FamilyPlan] | None = None,
) -> ArchResolution:
    """config 架构名 → 解析（零静默：未知名显式 L2 拒绝）。

    - 账本缺失该架构 → ``L2``（reason=「不在 vLLM 账本内」）——**不猜**；
    - L0 但族模板缺失（族模板未落地）→ auto 降级 L1 / strict 报错；
    - L2 → 恒拒（带账本 reason）。
    """
    table = table or load_registry_table()
    families = families if families is not None else load_family_templates()
    archs: dict = table.get("architectures", {})
    d = archs.get(architecture)
    if d is None:
        return ArchResolution(
            architecture=architecture, tier="L2", plan=None,
            reason=f"架构 {architecture!r} 不在 vLLM 架构账本（{table.get('total')} 条）内——显式拒绝，不静默处理",
        )
    entry = _entry_from_dict(architecture, d)
    ts_res = _resolve_with_text_side(architecture, entry, tier_mode, families)
    if ts_res is not None:
        return ts_res
    if entry.tier == "L2":
        return ArchResolution(architecture, "L2", None, entry.reason)
    if entry.tier == "L0":
        plan = families.get(entry.family or "")
        if plan is None:
            if tier_mode.casefold() == "strict":
                return ArchResolution(
                    architecture, "L2", None,
                    f"L0 家族 {entry.family!r} 无族模板（families/ 缺 {entry.family}.json）"
                    f"——strict 模式拒绝。可用家族: {sorted(families)}",
                )
            return ArchResolution(
                architecture, "L1", None,
                f"L0 家族 {entry.family!r} 族模板未落地 → auto 降级 L1 transformers 兜底（{entry.reason}）",
            )
        return ArchResolution(architecture, "L0", plan, entry.reason)
    # L1
    return ArchResolution(
        architecture, "L1", None,
        entry.reason + "（L1：transformers AutoModel 运行，三大机制不生效）",
    )


def list_families(family_dir: str | Path | None = None) -> list[str]:
    return sorted(load_family_templates(family_dir))


def list_l0_architectures(table: dict | None = None) -> list[str]:
    table = table or load_registry_table()
    return sorted(
        a for a, d in table.get("architectures", {}).items() if d.get("tier") == "L0"
    )
