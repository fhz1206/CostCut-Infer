"""ccut.quant.kv — KV cache 量化（§3.6-4，对齐 vLLM kv_cache_dtype + BaseKVCacheMethod）。

``kv_cache_dtype=auto|bf16|fp8``：
- auto：从 checkpoint ``kv_cache_scheme`` 读（Ornith = null → BF16）；
- fp8：per-token 对称量化 + 每 token 1B scale（E4M3 动态 scale 字节）——
  **KV 块 L1/L2 字节减半**（R1 协同：同样 1GB L1 装 2× token；
  256K 上下文从 5.1GB → 2.6GB）；
- 量化/反量化发生在 KV 块**写入/读回**的 numba 路径里，注意力计算仍 BF16。

块字节公式参数化（§3.4-7 KV 预算泛化）::

    bf16: bytes/token = 2(K/V) × num_kv_heads × head_dim × 2B
    fp8 : bytes/token = 2 × num_kv_heads × head_dim × 1B + 2 × num_kv_heads × head_dim × 1B(scale)
          ≈ bf16 的 1/2（scale 开销 1B/元素×2 侧，净减半保留）
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ccut.quant import kernels

__all__ = [
    "KVQuantMode",
    "resolve_kv_dtype",
    "kv_bytes_per_token",
    "quantize_kv_token",
    "dequantize_kv_token",
]


class KVQuantMode:
    BF16 = "bf16"
    FP8 = "fp8"


def resolve_kv_dtype(model_dir: str | Path | None, requested: str = "auto") -> str:
    """kv_cache_dtype 参数 → 最终模式（auto 从 checkpoint kv_cache_scheme 读）。"""
    requested = (requested or "auto").casefold()
    if requested in (KVQuantMode.BF16, KVQuantMode.FP8):
        return requested
    if requested == "auto":
        if model_dir is None:
            return KVQuantMode.BF16
        cfg_path = Path(model_dir) / "config.json"
        if cfg_path.exists():
            with open(cfg_path, "rb") as fh:
                cfg = json.load(fh)
            text_cfg = cfg.get("text_config", cfg)
            scheme = text_cfg.get("kv_cache_scheme") or cfg.get("kv_cache_scheme")
            if scheme and "fp8" in str(scheme).casefold():
                return KVQuantMode.FP8
        return KVQuantMode.BF16
    raise ValueError(f"未知 kv_cache_dtype: {requested!r}（auto/bf16/fp8）")


def kv_bytes_per_token(num_kv_heads: int, head_dim: int, mode: str) -> int:
    """每 token 每 full_attn 层的 KV 字节（块池公式参数化入口）。"""
    if mode == KVQuantMode.BF16:
        return 2 * num_kv_heads * head_dim * 2
    if mode == KVQuantMode.FP8:
        # K/V 各 1B/元素 + 各 1B/token 的 per-token scale（摊到每头）
        return 2 * num_kv_heads * head_dim * 1 + 2 * num_kv_heads
    raise ValueError(f"未知 KV 量化模式 {mode!r}")


def quantize_kv_token(kv: np.ndarray, out: np.ndarray, scales: np.ndarray) -> None:
    """per-token KV 量化（BF16 → FP8 + scale）。

    ``kv``: [2(K/V), heads, seq, dim] float32（已 dequant 的 BF16 值）；
    ``out``: [2, heads, seq, dim] uint8（FP8 码）；
    ``scales``: [2, heads, seq] uint8（E4M3 编码的 per-token scale）。
    """
    sides, heads, seq, dim = kv.shape
    for side in range(sides):
        for h in range(heads):
            for t in range(seq):
                x = kv[side, h, t]
                amax = float(np.abs(x).max())
                if amax <= 0:
                    out[side, h, t] = 0
                    scales[side, h, t] = 0
                    continue
                s = amax / 448.0
                q = np.clip(x / s, -448.0, 448.0)
                out[side, h, t] = kernels.float32_to_fp8_e4m3(q)
                scales[side, h, t] = int(kernels.float32_to_fp8_e4m3(np.array([s], dtype=np.float32))[0])


def dequantize_kv_token(fp8_kv: np.ndarray, scales: np.ndarray, out: np.ndarray) -> None:
    """FP8 KV 读回 → float32（注意力计算前）。语义为 quantize 的逆（含量化误差）。"""
    sides, heads, seq, dim = fp8_kv.shape
    for side in range(sides):
        for h in range(heads):
            for t in range(seq):
                sc = float(kernels.fp8_e4m3_to_float32(np.array([scales[side, h, t]], dtype=np.uint8))[0])
                vals = kernels.fp8_e4m3_to_float32(fp8_kv[side, h, t].astype(np.uint8))
                out[side, h, t] = vals * sc if sc > 0 else 0.0
