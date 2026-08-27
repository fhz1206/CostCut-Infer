"""ccut.experts.index — 专家清单扫描（shard 头解析）+ 落盘缓存 + 校验。

设计（§3.2 / P0-4）：
- 启动时扫描全部主 shard 的 safetensors 头（只读 8B+JSON，≈秒级），建立
  ``(layer, expert) → {shard, offset, len, scale...}`` 清单，落盘到
  ``expert_index_cache``（默认 ``.kv_cache/expert_index.json``）。
- 校验：张量计数与 manifest（model.safetensors.index.json 的 weight_map）一致，
  专家覆盖 ``num_layers × num_experts`` 无遗漏。
- 清单按**族模板**的张量名正则解析（张量名 = 架构间主要差异，§3.4）：
  ``expert_tensor_pattern`` 含 ``{layer}`` / ``{expert}`` / ``{proj}`` 占位符。

清单缓存格式（JSON，可入 git 调试）::

    {
      "version": 1,
      "model_path": "<abs>",
      "weight_map_hash": "<sha256 of weight_map json>",
      "pattern": "<expert_tensor_pattern>",
      "tensors_total": 62565,
      "entries": {
        "3:0": {"shard": "model-00001-of-00016.safetensors", "gate": [off,len,...], ...}
      }
    }
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from ccut.io_.safetensors_io import SafetensorsFile, iter_shards, load_index

__all__ = ["ExpertEntry", "ExpertIndex", "build_expert_index", "load_expert_index", "parse_expert_pattern"]


def parse_expert_pattern(pattern: str) -> re.Pattern[str]:
    """把族模板的 ``expert_tensor_pattern`` 转成正则。

    ``{layer}`` / ``{expert}`` 是数字捕获组，``{proj}`` 是非数字捕获组（gate/up/down）。
    接受两种写法（均 casefold 无关）：
    - 纯文本（推荐，族模板 JSON 用）：``model.language_model.layers.{layer}.mlp.experts.{expert}.{proj}_proj.weight``，
      内部 ``re.escape``；
    - 已转义的正则（含反斜杠）：直接使用，不再 escape。

    注意：真实权重名带 ``.weight`` / ``.weight_scale`` 后缀（FP8 per-channel scale），
    模式应止于 ``{proj}_proj`` 前缀匹配（用 ``^...`` 锚定开头、末尾不加 ``$``），
    由调用方按后缀区分 weight 与 scale 段。
    """
    if "\\" in pattern:
        regex = pattern
    else:
        regex = re.escape(pattern)
    regex = regex.replace(r"\{layer\}", r"(?P<layer>\d+)")
    regex = regex.replace(r"\{expert\}", r"(?P<expert>\d+)")
    regex = regex.replace(r"\{proj\}", r"(?P<proj>[a-z_]+)")
    return re.compile("^" + regex)


@dataclass(frozen=True)
class ExpertEntry:
    """单个专家的权重段清单（数据区相对偏移 + shape）。"""

    shard: str  # shard 文件名
    segments: dict[str, tuple[int, int, int, str]]  # key → (offset, length, numel, dtype)
    shapes: dict[str, tuple[int, ...]] = field(default_factory=dict)  # key → shape

    def bytes_total(self) -> int:
        return sum(seg[1] for seg in self.segments.values())


@dataclass
class ExpertIndex:
    """全模型专家清单。"""

    model_path: str
    pattern: str
    num_layers: int
    num_experts: int
    top_k: int
    entries: dict[tuple[int, int], ExpertEntry] = field(default_factory=dict)
    tensors_total: int = 0  # manifest 张量总数（校验用）
    build_ms: float = 0.0

    def get(self, layer: int, expert: int) -> ExpertEntry | None:
        return self.entries.get((layer, expert))

    def coverage(self) -> tuple[int, int]:
        """已覆盖 (layer, expert) 数 / 期望总数。"""
        return len(self.entries), self.num_layers * self.num_experts

    def verify(self) -> None:
        have, expect = self.coverage()
        if have != expect:
            raise ValueError(f"专家清单不完整：{have}/{expect}（layers={self.num_layers}×experts={self.num_experts}）")

    def to_dict(self) -> dict:
        return {
            "version": 2,
            "model_path": self.model_path,
            "pattern": self.pattern,
            "num_layers": self.num_layers,
            "num_experts": self.num_experts,
            "top_k": self.top_k,
            "tensors_total": self.tensors_total,
            "entries": {
                f"{l}:{e}": {
                    "shard": v.shard,
                    "segments": {k: list(s) for k, s in v.segments.items()},
                    "shapes": {k: list(s) for k, s in v.shapes.items()},
                }
                for (l, e), v in self.entries.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ExpertIndex":
        entries: dict[tuple[int, int], ExpertEntry] = {}
        for key, v in data["entries"].items():
            l, e = (int(x) for x in key.split(":"))
            entries[(l, e)] = ExpertEntry(
                shard=v["shard"],
                segments={k: tuple(s) for k, s in v["segments"].items()},
                shapes={k: tuple(s) for k, s in v.get("shapes", {}).items()},
            )
        return cls(
            model_path=data["model_path"],
            pattern=data["pattern"],
            num_layers=data["num_layers"],
            num_experts=data["num_experts"],
            top_k=data.get("top_k", 8),
            entries=entries,
            tensors_total=data.get("tensors_total", 0),
        )


def _weight_map_hash(weight_map: dict) -> str:
    blob = json.dumps(weight_map, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def build_expert_index(
    model_dir: str | Path,
    pattern: str,
    num_layers: int,
    num_experts: int,
    top_k: int = 8,
    cache_path: str | Path | None = None,
    use_cache: bool = True,
    verbose: bool = False,
) -> ExpertIndex:
    """扫描 shard 头建专家清单；命中缓存（weight_map_hash 一致）则直接读缓存。"""
    model_dir = Path(model_dir)
    t0 = time.perf_counter()
    info = load_index(model_dir)
    weight_map = info["weight_map"]
    map_hash = _weight_map_hash(weight_map)
    tensors_total = len(weight_map)

    cache_file = Path(cache_path) if cache_path else None
    if use_cache and cache_file and cache_file.exists():
        try:
            with open(cache_file, "rb") as fh:
                data = json.load(fh)
            if (
                data.get("version") == 1
                and data.get("pattern") == pattern
                and data.get("num_layers") == num_layers
                and data.get("num_experts") == num_experts
            ):
                cached_hash = _weight_map_hash_from_cache(data, model_dir)
                if cached_hash == map_hash:
                    idx = ExpertIndex.from_dict(data)
                    idx.verify()
                    if verbose:
                        print(f"[expert-index] 缓存命中 {cache_file}（{tensors_total} 张量，{len(idx.entries)} 专家）")
                    return idx
        except (json.JSONDecodeError, KeyError, ValueError):
            pass  # 缓存损坏 → 重建

    rx = parse_expert_pattern(pattern)
    entries: dict[tuple[int, int], dict] = {}
    # shard 文件名 → 头信息（只解析头，不读数据；保留 TensorView 以取真实 shape）
    shard_tensors: dict[str, dict[str, dict]] = {}
    for shard_path in iter_shards(model_dir):
        shard_name = shard_path.name
        with SafetensorsFile(shard_path) as sf:
            for name in sf.names:
                v = sf.tensor(name)
                shard_tensors.setdefault(shard_name, {})[name] = {
                    "offset": v.offset,
                    "length": v.length,
                    "numel": v.numel,
                    "dtype": v.dtype,
                    "shape": tuple(v.shape),
                }
        if verbose:
            print(f"[expert-index] 头解析 {shard_name}（{len(shard_tensors[shard_name])} 张量）")

    for shard_name, tensors in shard_tensors.items():
        for name, seg in tensors.items():
            m = rx.match(name)
            if not m:
                continue
            layer = int(m.group("layer"))
            expert = int(m.group("expert"))
            proj = m.group("proj")
            # 后缀区分 weight 与 weight_scale（FP8 per-channel scale 独立张量）
            suffix = name[m.end():] or ".weight"
            seg_key = f"{proj}{suffix}"
            entry = entries.setdefault((layer, expert), {"shard": shard_name, "segments": {}, "shapes": {}})
            entry["segments"][seg_key] = (
                seg["offset"],
                seg["length"],
                seg["numel"],
                seg["dtype"],
            )
            # shape 来自 shard 头（真实 2D 形状，scale 段也带 shape）
            entry["shapes"][seg_key] = seg["shape"]

    idx = ExpertIndex(
        model_path=str(model_dir.resolve()),
        pattern=pattern,
        num_layers=num_layers,
        num_experts=num_experts,
        top_k=top_k,
        entries={
            k: ExpertEntry(v["shard"], v["segments"], v.get("shapes", {}))
            for k, v in entries.items()
        },
        tensors_total=tensors_total,
        build_ms=round((time.perf_counter() - t0) * 1000, 1),
    )
    idx.verify()

    if cache_file:
        try:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_file, "w", encoding="utf-8") as fh:
                json.dump(idx.to_dict(), fh)
            if verbose:
                print(f"[expert-index] 清单已落盘 {cache_file}（{idx.build_ms}ms）")
        except OSError:
            pass  # 缓存失败不阻塞
    return idx


def _weight_map_hash_from_cache(data: dict, model_dir: Path) -> str:
    """缓存里保存 model_path；重建 weight_map 校验需重读 index.json（廉价）。"""
    info = load_index(model_dir)
    return _weight_map_hash(info["weight_map"])


def load_expert_index(
    model_dir: str | Path,
    pattern: str,
    num_layers: int,
    num_experts: int,
    top_k: int = 8,
    cache_path: str | Path | None = None,
    verbose: bool = False,
) -> ExpertIndex:
    """加载（缓存优先）或构建专家清单。"""
    return build_expert_index(
        model_dir,
        pattern=pattern,
        num_layers=num_layers,
        num_experts=num_experts,
        top_k=top_k,
        cache_path=cache_path,
        use_cache=True,
        verbose=verbose,
    )
