"""ccut.tools.sync_vllm_registry — vLLM registry → registry_table.json 同步工具（P0-5 / R8）。

机械解析本地 vLLM 克隆的 ``model_executor/models/registry.py``（**ast 解析，不 import
vllm**——避免拉 torch 运行时），生成架构账本 ``ccut/models/registry_table.json``（入 git）：
每个架构**恰好归入一层**——

- **L0** 原生快速路径：标准 decoder 家族（有族模板可表达）+ 文本生成任务；
- **L1** transformers 兜底层：vLLM 的 ``_TRANSFORMERS_BACKEND_MODELS`` /
  ``_TRANSFORMERS_SUPPORTED_MODELS``，或非标准家族的架构（功能可用，三大机制不生效）；
- **L2** 显式不支持：``_PREVIOUSLY_SUPPORTED_MODELS``（vLLM 已移除，带版本号）与
  ``_OOT_SUPPORTED_MODELS``（out-of-tree 插件）——**L2 条目均带理由字段**。

零静默「未处理」：``test_registry_coverage.py`` 断言条目数 = vLLM 解析数、无架构落层外。

用法::

    python -m ccut.tools.sync_vllm_registry [--vllm-registry PATH] [--out PATH]
"""

from __future__ import annotations

import ast
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

__all__ = ["parse_registry", "infer_family", "build_table", "main"]

# ---------------------------------------------------------------------------
# 族推断（模块名 → 族模板）
# ---------------------------------------------------------------------------

#: 标准 decoder 家族（可被通用组装器 + 声明式族模板表达）→ L0 候选。
#: 族模板文件在 ccut/models/families/<family>.json（#8 落地）。
L0_FAMILY_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("gdn-hybrid", ("qwen3_5_moe", "qwen3_5", "qwen3_next")),
    ("kimi", ("kimi_k3", "kimi_k25", "kimi_k2")),
    ("deepseek-mla", ("deepseek_v2", "deepseek_v3", "deepseek_v4", "deepseek2", "deepseek3", "deepseek")),
    ("glm-dsa", ("glm_moe_dsa", "glm_moe", "glm4_moe", "glm4")),
    ("minimax", ("minimax_m3", "minimax_m2", "minimax_vl", "minimax")),
    ("hy", ("hy_v3", "hy3", "hy_v")),
    ("longcat", ("longcat",)),
    ("mistral", ("mistral3", "mistral4", "mixtral", "mistral")),
    ("qwen3", ("qwen3",)),
    ("qwen2", ("qwen2", "qwen", "q_wen")),
    ("llama", ("llama4", "llama3", "llama2", "llama", "olmo")),
    ("gemma", ("gemma3n", "gemma3", "gemma2", "gemma", "olmo2")),
    ("phi", ("phi4", "phi3", "phi2", "phi1", "phi")),
    ("gpt2", ("gpt2", "gpt_neox", "gpt_oss")),
    ("bloom", ("bloom",)),
    ("mpt", ("mpt",)),
    ("stablelm", ("stablelm", "stable_lm")),
    ("falcon", ("falcon",)),
    ("baichuan", ("baichuan",)),
    ("chatglm", ("chatglm", "chatglm4")),
    ("internlm", ("internlm2", "internlm3", "internlm")),
    ("seed_oss", ("seed_oss", "seed")),
    ("ernie", ("ernie",)),
    ("apertus", ("apertus",)),
    ("bailing", ("bailing_moe", "bailing")),
    ("arcee", ("arcee",)),
    ("afmoe", ("afmoe",)),
    ("granite", ("granite",)),
    ("jais", ("jais",)),
    ("jamba", ("jamba",)),
    ("olmoe", ("olmoe",)),
    ("nemotron", ("nemotron",)),
    ("plamo", ("plamo",)),
    ("smaug", ("smaug",)),
    ("smollm", ("smollm",)),
    ("soli2", ("soli2",)),
    ("gpt_oss", ("gpt_oss",)),
)

#: vLLM registry 子字典 → 任务类型。
DICT_TASK_MAP = {
    "_TEXT_GENERATION_MODELS": "text_generation",
    "_EMBEDDING_MODELS": "embedding",
    "_LATE_INTERACTION_MODELS": "late_interaction",
    "_REWARD_MODELS": "reward",
    "_TOKEN_CLASSIFICATION_MODELS": "token_classification",
    "_SEQUENCE_CLASSIFICATION_MODELS": "sequence_classification",
    "_MULTIMODAL_MODELS": "multimodal",
    "_SPECULATIVE_DECODING_MODELS": "speculative_decoding",
    "_TRANSFORMERS_SUPPORTED_MODELS": "transformers_supported",
    "_TRANSFORMERS_BACKEND_MODELS": "transformers_backend",
}


def infer_family(module: str) -> str | None:
    """模块名 → 族模板名（最长前缀规则优先）。"""
    low = module.casefold()
    best: tuple[int, str] | None = None
    for family, prefixes in L0_FAMILY_RULES:
        for prefix in prefixes:
            if low.startswith(prefix) or prefix in low:
                rank = len(prefix)
                if best is None or rank > best[0]:
                    best = (rank, family)
    return best[1] if best else None


# ---------------------------------------------------------------------------
# ast 解析
# ---------------------------------------------------------------------------


def _const_str(node: ast.AST) -> str | None:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def _pair_value(node: ast.AST) -> tuple[str, str] | None:
    """``("module", "Class")`` 元组 → (module, class)。"""
    if isinstance(node, ast.Tuple) and len(node.elts) == 2:
        a, b = (_const_str(e) for e in node.elts)
        if a and b:
            return a, b
    return None


def _flatten_merged_dict(
    dict_node: ast.Dict,
    all_dicts: dict[str, ast.Dict],
    resolved: dict[str, dict[str, tuple[str, str]]] | None = None,
) -> dict[str, tuple[str, str]]:
    """展开 ``{**_A, **_B, "X": ("m","C"), ...}`` 合并字典。"""
    resolved = resolved or {}
    out: dict[str, tuple[str, str]] = {}
    for key, val in zip(dict_node.keys, dict_node.values):
        # 字典字面量里 ``**expr`` 的 key 为 None（AST 约定）
        if key is None:
            if isinstance(val, ast.Name):
                sub_name = val.id
                if sub_name not in all_dicts and sub_name not in resolved:
                    raise ValueError(f"无法展开 **{sub_name}（未定义）")
                sub = resolved.get(sub_name)
                if sub is None:
                    sub = _pair_dict(all_dicts[sub_name], resolved)
                    resolved[sub_name] = sub
            elif isinstance(val, ast.DictComp):
                sub = _eval_inline_dict_comp(val, resolved)
            else:
                raise ValueError(f"无法展开 **{ast.unparse(val)}")
            out.update(sub)
            continue
        name = _const_str(key)
        pair = _pair_value(val)
        if name is None or pair is None:
            raise ValueError(f"无法解析条目 {ast.unparse(key)}: {ast.unparse(val)}")
        out[name] = pair
    return out


def _plain_str_dict(dict_node: ast.Dict) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, val in zip(dict_node.keys, dict_node.values):
        k, v = _const_str(key), _const_str(val)
        if k is None or v is None:
            raise ValueError(f"无法解析 {ast.unparse(key)}")
        out[k] = v
    return out


def _pair_dict(dict_node: ast.Dict, resolved: dict[str, dict[str, tuple[str, str]]]) -> dict[str, tuple[str, str]]:
    """解析子字典（value 为 ``("module", "Class")`` 元组），支持内联 ``**{推导式}`` 展开。

    registry 229 行特例（`_EMBEDDING_MODELS` 内）::

        **{k: (mod, arch) for k, (mod, arch) in _TEXT_GENERATION_MODELS.items()
           if arch == "LlamaForCausalLM"}

    展开规则：从已解析的源字典里筛出 value 第二元 == 常量 的条目。
    """
    out: dict[str, tuple[str, str]] = {}
    for key, val in zip(dict_node.keys, dict_node.values):
        # 字典字面量里 ``**expr`` 的 key 为 None（AST 约定）
        if key is None:
            if not isinstance(val, ast.DictComp):
                raise ValueError(f"不支持的内联展开: {ast.unparse(val)}")
            out.update(_eval_inline_dict_comp(val, resolved))
            continue
        k = _const_str(key)
        pair = _pair_value(val)
        if k is None or pair is None:
            raise ValueError(f"无法解析 {ast.unparse(key)}: {ast.unparse(val)}")
        out[k] = pair
    return out


def _eval_inline_dict_comp(comp: ast.DictComp, resolved: dict[str, dict[str, tuple[str, str]]]) -> dict[str, tuple[str, str]]:
    gen = comp.generators[0] if comp.generators else None
    if (
        gen is None
        or not isinstance(gen.iter, ast.Call)
        or not isinstance(gen.iter.func, ast.Attribute)
        or gen.iter.func.attr != "items"
        or not isinstance(gen.iter.func.value, ast.Name)
    ):
        raise ValueError(f"不支持的内联展开 iter: {ast.unparse(comp)}")
    src_name = gen.iter.func.value.id
    if src_name not in resolved:
        raise ValueError(f"内联展开引用的 {src_name} 未先定义（registry 应保证定义顺序）")
    # 过滤条件：value 内元素 == 常量（如 arch == "LlamaForCausalLM"）
    cond_val: str | None = None
    for node in gen.ifs:
        if not (isinstance(node, ast.Compare) and len(node.ops) == 1 and isinstance(node.ops[0], ast.Eq)):
            raise ValueError(f"不支持的内联展开条件: {ast.unparse(node)}")
        for operand in (node.left, *node.comparators):
            if isinstance(operand, ast.Constant) and isinstance(operand.value, str):
                cond_val = operand.value
    out: dict[str, tuple[str, str]] = {}
    for k, pair in resolved[src_name].items():
        if cond_val is None or pair[1] == cond_val:
            out[k] = pair
    return out


_STR_DICTS = {"_PREVIOUSLY_SUPPORTED_MODELS", "_OOT_SUPPORTED_MODELS"}


def parse_registry(registry_path: str | Path) -> dict:
    """ast 解析 registry.py → 分类型字典集合（不 import vllm）。

    返回::

        {
          "_VLLM_MODELS": {arch: (module, class)},            # 已展开 **_ 子字典
          "_TEXT_GENERATION_MODELS": {arch: (module, class)}, # 各子字典原始成员
          ...
          "_PREVIOUSLY_SUPPORTED_MODELS": {arch: version},
          "_OOT_SUPPORTED_MODELS": {arch: url},
        }
    """
    src = Path(registry_path).read_text(encoding="utf-8")
    tree = ast.parse(src)
    dicts: dict[str, ast.Dict] = {}
    for node in tree.body:  # 保持定义顺序（内联展开依赖先定义）
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id.startswith("_")
            and isinstance(node.value, ast.Dict)
        ):
            dicts[node.targets[0].id] = node.value

    if "_VLLM_MODELS" not in dicts:
        raise ValueError("registry.py 中找不到 _VLLM_MODELS")
    result: dict[str, object] = {}
    resolved: dict[str, dict[str, tuple[str, str]]] = {}
    for name, d in dicts.items():
        if name == "_VLLM_MODELS":
            result[name] = _flatten_merged_dict(d, dicts, resolved)
        elif name in _STR_DICTS:
            result[name] = _plain_str_dict(d)
        else:
            pairs = _pair_dict(d, resolved)
            resolved[name] = pairs
            result[name] = pairs
    return result


# ---------------------------------------------------------------------------
# 账本生成
# ---------------------------------------------------------------------------


def _detect_vllm_version(repo_root: Path) -> str:
    for rel in ("VERSION", "vllm/VERSION"):
        p = repo_root / rel
        if p.exists():
            return p.read_text(encoding="utf-8").strip().splitlines()[0].strip()
    return "unknown"


def build_table(registry_path: str | Path, repo_root: Path | None = None) -> dict:
    """解析 + 逐条归层 → registry_table 结构。"""
    registry_path = Path(registry_path)
    repo_root = repo_root or registry_path.resolve().parents[4]  # .../vllm/vllm/model_executor/models/registry.py
    parsed = parse_registry(registry_path)
    vllm_models: dict[str, tuple[str, str]] = parsed["_VLLM_MODELS"]  # type: ignore[assignment]
    previously: dict[str, str] = parsed["_PREVIOUSLY_SUPPORTED_MODELS"]  # type: ignore[assignment]
    oot: dict[str, str] = parsed["_OOT_SUPPORTED_MODELS"]  # type: ignore[assignment]

    # 各子字典的原始成员（用于任务标注；解析时已展开内联推导式）
    dict_members: dict[str, set[str]] = {}
    for dict_name in DICT_TASK_MAP:
        if dict_name in parsed and isinstance(parsed[dict_name], dict):
            dict_members[dict_name] = set(parsed[dict_name].keys())

    architectures: dict[str, dict] = {}

    def add_entry(arch: str, *, tier: str, family: str | None, module: str | None, vllm_class: str | None, task: str, reason: str) -> None:
        if arch in architectures:
            # 重复条目（个别架构同时出现在多字典）：保留先归层者，记录别名
            architectures[arch]["aliases"].append({"task": task, "module": module, "vllm_class": vllm_class})
            return
        architectures[arch] = {
            "tier": tier,
            "family": family,
            "module": module,
            "vllm_class": vllm_class,
            "task": task,
            "reason": reason,
            "aliases": [],
        }

    # 1) _VLLM_MODELS 全量：先按子字典任务标注，再归层
    for arch, (module, vllm_class) in vllm_models.items():
        task = "text_generation"
        for dict_name, members in dict_members.items():
            if arch in members:
                task = DICT_TASK_MAP[dict_name]
                break
        family = infer_family(module)
        if dict_name if False else task in ("transformers_backend", "transformers_supported"):
            add_entry(arch, tier="L1", family=family, module=module, vllm_class=vllm_class, task=task,
                      reason="vLLM transformers 后端条目 → 本引擎 L1 兜底层（transformers AutoModel 运行）")
        elif task == "text_generation" and family is not None:
            add_entry(arch, tier="L0", family=family, module=module, vllm_class=vllm_class, task=task,
                      reason=f"标准 decoder 家族 {family}（族模板声明式表达，通用组装器构建）")
        elif task == "text_generation":
            add_entry(arch, tier="L1", family=None, module=module, vllm_class=vllm_class, task=task,
                      reason="无可表达族模板 → L1 transformers 兜底（P1.5 可评估升级 L0）")
        else:
            add_entry(arch, tier="L1", family=family, module=module, vllm_class=vllm_class, task=task,
                      reason=f"任务类型 {task}（非文本生成主路径）→ L1 兜底/专用头（P1.5 评估）")

    # 2) L2：vLLM 已移除 + OOT 插件（带理由）
    for arch, version in previously.items():
        add_entry(arch, tier="L2", family=None, module=None, vllm_class=None, task="removed",
                  reason=f"vLLM {version} 已移除该架构 → 显式不支持（建议迁移至后继架构）")
    for arch, url in oot.items():
        add_entry(arch, tier="L2", family=None, module=None, vllm_class=None, task="out_of_tree",
                  reason=f"out-of-tree 插件架构（{url}）→ 显式不支持")

    # 2.5) text_side：多模态包装器（*ForConditionalGeneration）→ 同 module 的文本侧 CausalLM 条目
    #      （Ornith checkpoint 即此情形：文本侧 = Qwen3_5 家族 L0，visual 侧禁用）
    l0_by_module: dict[str, str] = {}
    for a, i in architectures.items():
        if i["tier"] == "L0" and i["task"] == "text_generation" and i["module"]:
            l0_by_module.setdefault(i["module"], a)
    for a, i in architectures.items():
        if i["tier"] == "L1" and i["task"] == "multimodal" and i["module"] in l0_by_module:
            i["text_side"] = {
                "arch": l0_by_module[i["module"]],
                "tier": "L0",
                "family": architectures[l0_by_module[i["module"]]]["family"],
            }
            i["reason"] += f"（多模态包装器：文本侧 {l0_by_module[i['module']]} 可走 L0，visual 侧禁用）"

    tier_counts = {"L0": 0, "L1": 0, "L2": 0}
    for info in architectures.values():
        tier_counts[info["tier"]] += 1

    return {
        "version": 1,
        "vllm_registry": str(registry_path),
        "vllm_version": _detect_vllm_version(repo_root),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "total": len(architectures),
        "vllm_registry_count": len(vllm_models),
        "tier_counts": tier_counts,
        "task_counts": _task_count(architectures),
        "family_counts": _family_count(architectures),
        "architectures": architectures,
    }


def _flatten_dict_raw(registry_path: Path, dict_name: str) -> dict[str, tuple[str, str]]:
    src = registry_path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == dict_name
            and isinstance(node.value, ast.Dict)
        ):
            out: dict[str, tuple[str, str]] = {}
            for key, val in zip(node.value.keys, node.value.values):
                k = _const_str(key)
                pair = _pair_value(val)
                if k and pair:
                    out[k] = pair
            return out
    return {}


def _task_count(architectures: dict[str, dict]) -> dict[str, int]:
    out: dict[str, int] = {}
    for info in architectures.values():
        out[info["task"]] = out.get(info["task"], 0) + 1
    return dict(sorted(out.items()))


def _family_count(architectures: dict[str, dict]) -> dict[str, int]:
    out: dict[str, int] = {}
    for info in architectures.values():
        fam = info["family"] or "-"
        out[fam] = out.get(fam, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    repo = Path(__file__).resolve().parents[3]  # python/ccut/tools/x.py → Engine/
    registry = repo / "vllm" / "vllm" / "model_executor" / "models" / "registry.py"
    out = repo / "python" / "ccut" / "models" / "registry_table.json"
    i = 0
    while i < len(argv):
        if argv[i] == "--vllm-registry":
            registry = Path(argv[i + 1]); i += 2
        elif argv[i] == "--out":
            out = Path(argv[i + 1]); i += 2
        else:
            print(f"未知参数 {argv[i]}（支持 --vllm-registry PATH / --out PATH）")
            return 2
    if not registry.exists():
        print(f"registry 不存在: {registry}（先 git clone 本地 vLLM 参考实现）")
        return 1
    table = build_table(registry, registry.parents[3])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(table, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"registry_table 已生成: {out}")
    print(f"  vLLM {table['vllm_version']} | 总架构 {table['total']}（registry {table['vllm_registry_count']} 条）")
    print(f"  层级: {table['tier_counts']}")
    print(f"  任务: {table['task_counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
