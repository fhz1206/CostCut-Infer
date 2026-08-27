"""ccut.config — 统一配置层（R5）。

三源合并，优先级 CLI > toml > 环境变量（``CCUT_`` 前缀）> 内置默认。
所有键在合并前统一 ``casefold()``：``--Temperature=0.95``、``--temperature==0.95``
（宽松解析把第二个 ``=`` 视为值的一部分剥掉）、toml ``[sampling] Temperature = 0.95`` 等价。
全部走 schema 声明的类型与取值域校验，非法值启动即报错并列出全部合法键。

本文件是参数的**单一事实来源**：``--list-params`` 输出由 :func:`render_params_table`
运行时动态生成，避免文档漂移。
"""

from __future__ import annotations

import json
import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "ParamSpec",
    "SCHEMA",
    "Config",
    "ConfigError",
    "parse_cli_args",
    "load_toml_overrides",
    "merge_config",
    "render_params_table",
    "resolve_bytes",
]

# ---------------------------------------------------------------------------
# 参数 schema（§6 全参数表）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParamSpec:
    """单个参数的声明：类型 / 默认值 / 取值域 / 说明。"""

    section: str
    name: str
    ptype: str  # str | int | float | bool | list | any
    default: Any
    choices: tuple[str, ...] = ()
    min_value: float | None = None
    max_value: float | None = None
    help: str = ""


def _P(section, name, ptype, default, help, **kw):  # noqa: A002 - 内部构造器
    return ParamSpec(section, name, ptype, default, help=help, **kw)


SCHEMA: dict[str, dict[str, ParamSpec]] = {
    "model": {
        "model_path": _P("model", "model_path", "str", "./models/Ornith-1.5-35B-A3B-MTP-FP8", "权重目录"),
        "dtype": _P("model", "dtype", "str", "auto", "auto 从 config 探测", choices=("auto", "bf16", "fp16", "fp32")),
        "max_model_len": _P("model", "max_model_len", "int", 32768, "最大上下文（上限 262144）", min_value=8, max_value=262144),
        "arch_tier": _P("model", "arch_tier", "str", "auto", "R8：auto=L0 优先、缺则 L1 兜底；strict=仅 L0", choices=("auto", "strict")),
        "enable_vision": _P("model", "enable_vision", "bool", False, "Ornith ViT 路径（P7，CPU 慢）"),
        "trust_remote_code": _P("model", "trust_remote_code", "bool", False, "信任模型目录内远程代码"),
    },
    "engine": {
        "max_num_seqs": _P("engine", "max_num_seqs", "int", 8, "并发请求上限", min_value=1),
        "max_num_batched_tokens": _P("engine", "max_num_batched_tokens", "int", 8192, "每步 token 预算（prefill+decode 共享）", min_value=1),
        "chunked_prefill_size": _P("engine", "chunked_prefill_size", "int", 4096, "prefill 分块", min_value=1),
        "enable_prefix_caching": _P("engine", "enable_prefix_caching", "bool", True, "前缀块复用"),
        "scheduler_delay_factor": _P("engine", "scheduler_delay_factor", "float", 0.0, "参考 vLLM 语义", min_value=0.0),
        "num_scheduler_threads": _P("engine", "num_scheduler_threads", "int", 1, "调度线程数", min_value=1),
        "num_worker_threads": _P("engine", "num_worker_threads", "int", 4, "算子线程数", min_value=1),
        "tensor_parallel_size": _P("engine", "tensor_parallel_size", "int", 1, "CPU 单机版：≠1 显式报错", min_value=1),
        "pipeline_parallel_size": _P("engine", "pipeline_parallel_size", "int", 1, "CPU 单机版：≠1 显式报错", min_value=1),
        "log_level": _P("engine", "log_level", "str", "INFO", "日志级别", choices=("DEBUG", "INFO", "WARNING", "ERROR")),
        "seed": _P("engine", "seed", "int", None, "全局随机种子"),
        "mem_budget_gb": _P("engine", "mem_budget_gb", "float", 2.5, "R9：进程私有内存预算（GB）；启动时反推 L1/激活规模并打印预算表", min_value=0.5, max_value=64.0),
    },
    "kv_cache": {
        "kv_l1_bytes": _P("kv_cache", "kv_l1_bytes", "int", 1073741824, "L1 内存块池大小（R9 默认 1GB；范围 512MB~4GB）", min_value=536870912, max_value=4294967296),
        "kv_l2_dir": _P("kv_cache", "kv_l2_dir", "str", "./.kv_cache", "L2 磁盘目录（相对 python/）"),
        "kv_l2_max_bytes": _P("kv_cache", "kv_l2_max_bytes", "int", 68719476736, "L2 上限（可远超 RAM）", min_value=67108864),
        "kv_policy": _P("kv_cache", "kv_policy", "str", "hybrid", "hybrid / disk_first（RAM 极省）/ memory_first", choices=("hybrid", "disk_first", "memory_first")),
        "kv_block_size": _P("kv_cache", "kv_block_size", "int", 16, "token/块", min_value=1, max_value=256),
        "kv_evict_high_water": _P("kv_cache", "kv_evict_high_water", "float", 0.8, "L1 水位触发下沉", min_value=0.0, max_value=1.0),
        "kv_evict_low_water": _P("kv_cache", "kv_evict_low_water", "float", 0.6, "L1 水位触发回收", min_value=0.0, max_value=1.0),
        "kv_hot_window_steps": _P("kv_cache", "kv_hot_window_steps", "int", 16, "冷判定窗口（步）", min_value=1),
        "kv_l2_compression": _P("kv_cache", "kv_l2_compression", "str", "none", "none / lz4（若装）", choices=("none", "lz4")),
        "kv_cache_ttl_seconds": _P("kv_cache", "kv_cache_ttl_seconds", "int", 0, "0=不过期；会话级 .kvdb 清理", min_value=0),
    },
    "experts": {
        "expert_residency": _P("experts", "expert_residency", "str", "zero", "zero（零驻留，默认）/ page_cache_only", choices=("zero", "page_cache_only")),
        "expert_ring_slots": _P("experts", "expert_ring_slots", "int", 2, "每层 ring buffer 槽位数", min_value=1, max_value=8),
        "expert_index_cache": _P("experts", "expert_index_cache", "str", "./.kv_cache/expert_index.json", "专家清单缓存"),
        "expert_verify_crc": _P("experts", "expert_verify_crc", "bool", False, "读后抽样 CRC 校验（调试用）"),
    },
    "weights": {
        "weight_streaming": _P("weights", "weight_streaming", "str", "auto", "auto（§3.5 阈值 auto 切换）/ on / off", choices=("auto", "on", "off")),
        "weight_ring_layers": _P("weights", "weight_ring_layers", "int", 2, "WeightRing 槽位数（1 更省内存）", min_value=1, max_value=8),
        "weight_stream_chunk": _P("weights", "weight_stream_chunk", "str", "auto", "auto=单层 > 槽位 50% 自动 sublayer", choices=("auto", "layer", "sublayer")),
        "weight_stream_bandwidth_log": _P("weights", "weight_stream_bandwidth_log", "bool", True, "每步打印静态权重读取带宽 / 预计 token/s"),
    },
    "quant": {
        "quantization": _P("quant", "quantization", "str", "auto", "auto=按 checkpoint quantization_config；可指定在线简写（fp8_per_token / int8_per_channel_weight_only / mxfp8 / nvfp4_per_token 等）", choices=("auto", "fp8_per_token", "int8_per_channel_weight_only", "mxfp8", "nvfp4_per_token", "int8_per_token", "bf16")),
        "quant_ignore": _P("quant", "quant_ignore", "list", [], "在线量化时的 ignore 层名正则列表"),
        "fp8_compute_mode": _P("quant", "fp8_compute_mode", "str", "w8a16", "w8a16（默认精确反量化）/ w8a8（CPU 对照基准）", choices=("w8a16", "w8a8")),
        "kv_cache_dtype": _P("quant", "kv_cache_dtype", "str", "auto", "auto（checkpoint kv_cache_scheme）/ bf16 / fp8（KV 容量减半）", choices=("auto", "bf16", "fp8")),
    },
    "resources": {
        "resource_pct": _P("resources", "resource_pct", "int", 50, "R11 全局资源限制比例（%）：CPU/内存/IO 同时生效", min_value=1, max_value=100),
        "resource_cpu_pct": _P("resources", "resource_cpu_pct", "str", "auto", "auto=resource_pct；CPU 线程预算比例（%）"),
        "resource_mem_pct": _P("resources", "resource_mem_pct", "str", "auto", "auto=resource_pct；内存私有 RSS 硬上限比例（%）"),
        "resource_io_pct": _P("resources", "resource_io_pct", "str", "auto", "auto=resource_pct；磁盘 IO 带宽限比例（%）"),
        "resource_monitor_interval": _P("resources", "resource_monitor_interval", "float", 1.0, "看门狗采样间隔（秒）", min_value=0.05),
        "resource_throttle": _P("resources", "resource_throttle", "str", "auto", "auto（超限自限）/ warn（仅告警）/ off（仅打印预算表）", choices=("auto", "warn", "off")),
    },
    "pipeline": {
        "pipeline_depth": _P("pipeline", "pipeline_depth", "int", 3, "三层流水线：计算 N ‖ 预取 N+1 ‖ 投机 N+2", min_value=2, max_value=4),
        "prefetch_layers_ahead": _P("pipeline", "prefetch_layers_ahead", "int", 2, "投机预取提前层数", min_value=1, max_value=4),
        "prefetch_mode": _P("pipeline", "prefetch_mode", "str", "auto", "auto / avx2 / off（AVX2 预取指令开关）", choices=("auto", "avx2", "off")),
        "expert_readahead_mb": _P("pipeline", "expert_readahead_mb", "int", 128, "顺序读 readahead 提示量（MB）", min_value=0),
        "speculative_route_history": _P("pipeline", "speculative_route_history", "int", 4, "投机路由历史窗口（步）", min_value=0),
        "pipeline_metrics": _P("pipeline", "pipeline_metrics", "bool", True, "输出 overlap_ratio 等时序指标"),
    },
    "sampling": {
        "temperature": _P("sampling", "temperature", "float", 1.0, "采样温度", min_value=0.0),
        "min_p": _P("sampling", "min_p", "float", 0.0, "min-p 采样", min_value=0.0, max_value=1.0),
        "top_p": _P("sampling", "top_p", "float", 0.95, "nucleus 采样", min_value=0.0, max_value=1.0),
        "typical_p": _P("sampling", "typical_p", "float", 1.0, "typical 采样", min_value=0.0, max_value=1.0),
        "top_k": _P("sampling", "top_k", "int", 20, "top-k 采样（1≈greedy，0=禁用 top-k）", min_value=0, max_value=250000),
        "repetition_penalty": _P("sampling", "repetition_penalty", "float", 1.0, "重复惩罚", min_value=0.0),
        "presence_penalty": _P("sampling", "presence_penalty", "float", 0.0, "存在惩罚", min_value=-2.0, max_value=2.0),
        "frequency_penalty": _P("sampling", "frequency_penalty", "float", 0.0, "频率惩罚", min_value=-2.0, max_value=2.0),
        "length_penalty": _P("sampling", "length_penalty", "float", 1.0, "长度惩罚（ITL）", min_value=0.0),
        "early_stopping": _P("sampling", "early_stopping", "bool", False, "best_of>1 时提前停止"),
        "n": _P("sampling", "n", "int", 1, "每请求生成序列数", min_value=1),
        "best_of": _P("sampling", "best_of", "int", 1, "候选数", min_value=1),
        "max_tokens": _P("sampling", "max_tokens", "int", None, "最大生成 token（null=max_model_len-输入）"),
        "max_completion_tokens": _P("sampling", "max_completion_tokens", "int", None, "最大完成 token（含思考）"),
        "stop": _P("sampling", "stop", "list", [], "停止字符串列表"),
        "stop_token_ids": _P("sampling", "stop_token_ids", "list", [], "停止 token id 列表"),
        "seed": _P("sampling", "seed", "int", None, "采样随机种子（null=继承 engine.seed）"),
        "ignore_eos": _P("sampling", "ignore_eos", "bool", False, "忽略 EOS 继续生成"),
        "logprobs": _P("sampling", "logprobs", "int", None, "每 token top-k logprobs（null=关闭）"),
        "prompt_logprobs": _P("sampling", "prompt_logprobs", "int", None, "prompt 阶段 logprobs（null=关闭）"),
        "guided_json": _P("sampling", "guided_json", "any", None, "JSON schema 约束解码"),
        "guided_regex": _P("sampling", "guided_regex", "str", None, "正则约束解码"),
        "guided_grammar": _P("sampling", "guided_grammar", "str", None, "EBNF/LLM grammar 约束解码"),
        "detokenize": _P("sampling", "detokenize", "bool", True, "是否增量 detokenize"),
    },
    "spec_decode": {
        "enable_mtp": _P("spec_decode", "enable_mtp", "bool", True, "用模型自带 1 层 MTP"),
        "mtp_draft_tokens": _P("spec_decode", "mtp_draft_tokens", "int", 1, "每步草稿 token 数（0=关闭）", min_value=0),
        "enable_ngram": _P("spec_decode", "enable_ngram", "bool", False, "ngram 备选 proposer"),
        "ngram_window": _P("spec_decode", "ngram_window", "int", 8, "ngram 窗口", min_value=2),
    },
    "api": {
        "host": _P("api", "host", "str", "0.0.0.0", "监听地址"),
        "port": _P("api", "port", "int", 8000, "监听端口", min_value=1, max_value=65535),
        "api_key": _P("api", "api_key", "str", None, "可选鉴钥"),
        "sse_heartbeat_seconds": _P("api", "sse_heartbeat_seconds", "float", 15.0, "流式心跳间隔（秒）", min_value=1.0),
        "cors": _P("api", "cors", "bool", True, "开启 CORS"),
    },
}


class ConfigError(Exception):
    """配置错误（非法键/非法值/取值域越界）。"""

    def __init__(self, message: str, valid_keys: list[str] | None = None):
        super().__init__(message)
        self.valid_keys = valid_keys


def _valid_key_lines() -> list[str]:
    lines = []
    for section in SCHEMA:
        for key in SCHEMA[section]:
            lines.append(f"  [{section}] {key}")
    return lines


# ---------------------------------------------------------------------------
# 值转换（string → schema 类型）
# ---------------------------------------------------------------------------

_TRUE_WORDS = {"true", "1", "yes", "on"}
_FALSE_WORDS = {"false", "0", "no", "off"}
_NULL_WORDS = {"", "null", "none", "nil", "~"}


def resolve_bytes(value: Any) -> int:
    """把 ``1gb`` / ``512mb`` / ``1048576`` 之类的输入解析成字节数（大小写不敏感）。"""
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    s = str(value).strip().casefold()
    if not s:
        raise ConfigError(f"空数值: {value!r}")
    multipliers = {"kb": 1024, "mb": 1024**2, "gb": 1024**3, "tb": 1024**4}
    for suffix, mult in multipliers.items():
        if s.endswith(suffix):
            return int(float(s[: -len(suffix)]) * mult)
    return int(float(s))


def convert_value(spec: ParamSpec, raw: Any) -> Any:
    """按 schema 类型把原始值（str/原生）转成最终类型。"""
    if spec.ptype == "any":
        return raw
    if raw is None:
        return None  # null 默认值（seed / api_key / guided_* 等）
    if isinstance(raw, str):
        s = raw.strip()
        if spec.ptype == "bool":
            low = s.casefold()
            if low in _TRUE_WORDS:
                return True
            if low in _FALSE_WORDS:
                return False
            raise ConfigError(f"{spec.section}.{spec.name}: 无法解析布尔值 {raw!r}")
        if s.casefold() in _NULL_WORDS:
            if spec.ptype in ("str", "any"):
                return None if s.casefold() in _NULL_WORDS and s else s
            return None
        try:
            if spec.ptype == "int":
                return resolve_bytes(s)
            if spec.ptype == "float":
                return float(s)
        except ValueError:
            raise ConfigError(f"{spec.section}.{spec.name}: 无法解析 {spec.ptype} 值 {raw!r}") from None
        if spec.ptype == "list":
            if s.startswith(("[", "{")):
                try:
                    parsed = json.loads(s)
                except json.JSONDecodeError as exc:
                    raise ConfigError(f"{spec.section}.{spec.name}: 列表解析失败 {raw!r}") from exc
                if isinstance(parsed, list):
                    return [str(x) if spec.ptype == "list" and not isinstance(x, (list, dict)) else x for x in parsed]
                raise ConfigError(f"{spec.section}.{spec.name}: 期望 JSON 数组 {raw!r}")
            return [item.strip() for item in s.split(",") if item.strip()]
        return s
    if spec.ptype == "bool":
        if isinstance(raw, bool):
            return raw
        raise ConfigError(f"{spec.section}.{spec.name}: 期望布尔值 {raw!r}")
    if spec.ptype == "int":
        if isinstance(raw, bool):
            return int(raw)
        if isinstance(raw, int):
            return raw
        return int(raw)
    if spec.ptype == "float":
        return float(raw)
    if spec.ptype == "list":
        if isinstance(raw, list):
            return raw
        return [raw]
    if spec.ptype == "str":
        return None if raw is None else str(raw)
    return raw


def validate_value(spec: ParamSpec, value: Any) -> None:
    """取值域校验：choices / min / max。"""
    if value is None:
        return
    if spec.choices:
        allowed = spec.choices
        if spec.ptype == "str" and all(c.casefold() == c for c in allowed):
            value = str(value).casefold()
        elif spec.ptype == "str":
            value = str(value).casefold()
        if value not in allowed and str(value).casefold() not in {c.casefold() for c in allowed}:
            raise ConfigError(
                f"{spec.section}.{spec.name}: 取值 {value!r} 不在合法集合 {list(allowed)} 内"
            )
    if spec.ptype in ("int", "float") and isinstance(value, (int, float)):
        if spec.min_value is not None and value < spec.min_value:
            raise ConfigError(f"{spec.section}.{spec.name}: {value} 小于下限 {spec.min_value}")
        if spec.max_value is not None and value > spec.max_value:
            raise ConfigError(f"{spec.section}.{spec.name}: {value} 大于上限 {spec.max_value}")


# ---------------------------------------------------------------------------
# CLI 宽松解析
# ---------------------------------------------------------------------------

SPECIAL_FLAGS = {
    "--list-params",
    "--list-architectures",
    "--info",
    "--version",
    "--help",
    "-h",
}


def _normalize_key(token: str) -> str | None:
    """``--Temperature`` / ``--kv-l1-bytes`` → ``kv_l1_bytes``。

    统一 casefold + 连字符归一为下划线：``--KV-L1-Bytes`` 与 ``--kv_l1_bytes`` 等价。
    """
    key = token[2:]
    if "=" in key:
        key = key.split("=", 1)[0]
    return key.casefold().replace("-", "_")


def parse_cli_args(argv: list[str]) -> tuple[dict[str, dict[str, str]], list[str], list[str]]:
    """宽松解析命令行。

    返回 ``(overrides, positional, special_flags)``：
    - ``overrides``: ``{section: {casefolded_key: 原始字符串值}}``（值保持原始字符串，
      由 schema 统一转换）；
    - ``positional``: 位置参数（如 prompt 文本）；
    - ``special_flags``: ``--list-params`` 等特殊开关。

    支持 ``--key=value``、``--key value``、``--key==value``（第二个 ``=`` 剥掉一个）。
    """
    overrides: dict[str, dict[str, str]] = {}
    positional: list[str] = []
    special: list[str] = []

    def find_section(key: str) -> str | None:
        for section, params in SCHEMA.items():
            if key in params:
                return section
        return None

    i = 0
    while i < len(argv):
        token = argv[i]
        if token in SPECIAL_FLAGS:
            special.append(token)
            i += 1
            continue
        if token.startswith("--"):
            key = _normalize_key(token)
            value: str | None = None
            if "=" in token[2:]:
                # --key=value / --key==value（宽松：剥掉多余的一个 =）
                after_eq = token.split("=", 1)[1]
                value = after_eq[1:] if after_eq.startswith("=") else after_eq
            elif i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                value = argv[i + 1]
                i += 1
            else:
                value = "true"  # 裸 flag 视为布尔开关
            section = find_section(key)
            if section is None:
                raise ConfigError(
                    f"未知参数 --{key}。全部合法键：\n" + "\n".join(_valid_key_lines())
                )
            overrides.setdefault(section, {})[key] = value
        elif token.startswith("-") and len(token) > 1:
            raise ConfigError(f"不支持的单字符选项 {token!r}（请使用 --key=value）")
        else:
            positional.append(token)
        i += 1
    return overrides, positional, special


def load_toml_overrides(toml_path: str | Path) -> dict[str, dict[str, Any]]:
    """读取 toml 配置：分节 + 键全部 casefold。缺失文件返回空。"""
    path = Path(toml_path)
    if not path.exists():
        return {}
    with open(path, "rb") as fh:
        data = tomllib.load(fh)
    result: dict[str, dict[str, Any]] = {}
    for section, params in data.items():
        if not isinstance(params, dict):
            raise ConfigError(f"toml 分节 [{section}] 必须是表（table）")
        result[str(section).casefold()] = {str(k).casefold(): v for k, v in params.items()}
    return result


def _env_overrides(env: dict[str, str] | None = None) -> dict[str, dict[str, Any]]:
    """环境变量源：优先 ``CCUT_<SECTION>__<KEY>``，回退 ``CCUT_<KEY>``。"""
    env = os.environ if env is None else env
    result: dict[str, dict[str, Any]] = {}
    for section, params in SCHEMA.items():
        for key in params:
            scoped = f"CCUT_{section.upper()}__{key.upper()}"
            plain = f"CCUT_{key.upper()}"
            value = env.get(scoped, env.get(plain))
            if value is not None:
                result.setdefault(section, {})[key] = value
    return result


# ---------------------------------------------------------------------------
# 合并与 Config 对象
# ---------------------------------------------------------------------------


def merge_config(
    cli_overrides: dict[str, dict[str, Any]] | None = None,
    toml_overrides: dict[str, dict[str, Any]] | None = None,
    env: dict[str, str] | None = None,
) -> dict[str, dict[str, Any]]:
    """三源合并：默认 < env < toml < CLI。返回 ``{section: {key: 已转换值}}``。

    ``env=None`` 时从 :data:`os.environ` 实时读取（支持测试 ``monkeypatch`` 注入）；
    显式 ``env={}`` 表示「无环境变量源」（用于隔离测试）。
    """
    if env is None:
        env = os.environ
    merged: dict[str, dict[str, Any]] = {}
    for section, params in SCHEMA.items():
        merged[section] = {}
        for key, spec in params.items():
            raw: Any = spec.default
            if _env_lookup(env, section, key) is not None:
                raw = _env_lookup(env, section, key)
            if toml_overrides and section in toml_overrides and key in toml_overrides[section]:
                raw = toml_overrides[section][key]
            if cli_overrides and section in cli_overrides and key in cli_overrides[section]:
                raw = cli_overrides[section][key]
            value = convert_value(spec, raw)
            validate_value(spec, value)
            merged[section][key] = value
    return merged


def _env_lookup(env: dict[str, str], section: str, key: str) -> Any:
    scoped = f"CCUT_{section.upper()}__{key.upper()}"
    plain = f"CCUT_{key.upper()}"
    # 返回第一个**存在**的值（即便为空字符串也算——空字符串由 convert_value 转 None）
    if scoped in env:
        return env[scoped]
    if plain in env:
        return env[plain]
    return None


class Config:
    """合并后的只读配置对象（按 casefold 键访问）。"""

    def __init__(self, values: dict[str, dict[str, Any]]):
        self._values = {
            str(s).casefold(): {str(k).casefold(): v for k, v in params.items()}
            for s, params in values.items()
        }

    def get(self, key: str, section: str) -> Any:
        key = key.casefold()
        section = section.casefold()
        try:
            return self._values[section][key]
        except KeyError:
            raise ConfigError(f"未知参数 [{section}] {key}", _valid_key_lines()) from None

    def section(self, section: str) -> dict[str, Any]:
        return dict(self._values[str(section).casefold()])

    def sections(self) -> list[str]:
        return list(self._values)

    def as_dict(self) -> dict[str, dict[str, Any]]:
        return {s: dict(p) for s, p in self._values.items()}

    def cross_validate(self) -> None:
        """跨字段校验（单字段由 schema 完成）。"""
        e = self._values
        if e["engine"]["tensor_parallel_size"] != 1 or e["engine"]["pipeline_parallel_size"] != 1:
            raise ConfigError("本引擎为 CPU 单机版，不支持多卡：tensor_parallel_size / pipeline_parallel_size 必须为 1")
        if e["engine"]["chunked_prefill_size"] > e["engine"]["max_num_batched_tokens"]:
            raise ConfigError("chunked_prefill_size 不能大于 max_num_batched_tokens")
        if e["kv_cache"]["kv_evict_low_water"] >= e["kv_cache"]["kv_evict_high_water"]:
            raise ConfigError("kv_evict_low_water 必须小于 kv_evict_high_water")
        if e["sampling"]["n"] > e["engine"]["max_num_seqs"] * 4:
            raise ConfigError("sampling.n 过大（> max_num_seqs×4），请调整")
        if e["resources"]["resource_throttle"] != "auto" and e["resources"]["resource_pct"] != 50:
            pass  # 允许组合，不报错
        # 分资源 pct 覆盖值校验
        for name in ("resource_cpu_pct", "resource_mem_pct", "resource_io_pct"):
            v = e["resources"][name]
            if v != "auto":
                iv = convert_value(ParamSpec("resources", name, "int", 0, min_value=1, max_value=100), v)
                validate_value(ParamSpec("resources", name, "int", 0, min_value=1, max_value=100), iv)

    @classmethod
    def build(
        cls,
        argv: list[str] | None = None,
        toml_path: str | Path | None = None,
        env: dict[str, str] | None = None,
    ) -> "Config":
        """从 CLI + toml + env 构建完整配置（含跨字段校验）。"""
        cli: dict[str, dict[str, Any]] = {}
        if argv:
            cli, _, _ = parse_cli_args(argv)
        toml = load_toml_overrides(toml_path) if toml_path else {}
        values = merge_config(cli, toml, env)
        cfg = cls(values)
        cfg.cross_validate()
        return cfg


# ---------------------------------------------------------------------------
# --list-params 输出（运行时动态生成，单一事实来源）
# ---------------------------------------------------------------------------


def render_params_table() -> str:
    """生成完整参数表文本（供 ``--list-params``）。"""
    lines = ["CostCut-Infer 参数表（CLI > toml > env[CCUT_…] > 默认；大小写不敏感）", ""]
    for section, params in SCHEMA.items():
        lines.append(f"[{section}]")
        for key, spec in params.items():
            default = spec.default
            if isinstance(default, float) and default.is_integer():
                default = int(default)
            domain = ""
            if spec.choices:
                domain = "choices=" + "|".join(spec.choices)
            elif spec.min_value is not None or spec.max_value is not None:
                lo = spec.min_value if spec.min_value is not None else "-∞"
                hi = spec.max_value if spec.max_value is not None else "+∞"
                domain = f"range=[{lo}, {hi}]"
            dstr = json.dumps(default, ensure_ascii=False) if isinstance(default, (list, dict)) else repr(default)
            lines.append(f"  {key:<28} {spec.ptype:<6} default={dstr:<24} {domain}  # {spec.help}")
        lines.append("")
    lines.append("环境变量命名：CCUT_<KEY>（如 CCUT_TEMPERATURE）或 CCUT_<SECTION>__<KEY>（如 CCUT_KV_CACHE__KV_L1_BYTES，优先级更高）。")
    return "\n".join(lines)
