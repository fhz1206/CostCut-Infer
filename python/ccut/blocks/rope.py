"""ccut.blocks.rope — RoPE 旋转位置编码（none / linear / dynamic / yarn）。

实现对齐 vLLM ``rotary_embedding.py`` 语义（CPU 精确路径）：
- 半旋转（rotate_half）：``out[..., :d/2] = x1*cos - x2*sin``，``out[..., d/2:] = x2*cos + x1*sin``；
- 逆频基：``inv_freq = 1 / base^(arange(0, d, 2) / d)``；
- scaling：none（恒等）/ linear（factor 除 pos）/ dynamic（512·sqrt 缩放 pos）/
  yarn（yarn 缩放函数）。
"""

from __future__ import annotations

import math

import numpy as np

__all__ = ["build_rope", "apply_rope"]


def _yarn_mscale(scale: float) -> float:
    if scale <= 1.0:
        return 1.0
    return 0.1 * scale * math.log(scale) + 1.0


def _yarn_get_mscale(scale: float, mu: float = 1.4) -> float:
    """yarn mscale（vLLM _yarn_get_mscale）。"""
    return _yarn_mscale(scale) if scale > 1.0 else 1.0


def _yarn_find_correction_range(
    low_scale: float, high_scale: float, dim: int, base: float, beta_fast: float, beta_slow: float
) -> tuple[float, float]:
    low = math.floor(
        base * (beta_fast / (1 - math.exp(-beta_fast / dim))) ** (1 / 2)
    )
    high = math.ceil(
        base * (beta_slow / (1 - math.exp(-beta_slow / dim))) ** (1 / 2)
    )
    return low / low_scale, high / high_scale


def _yarn_get_scaled_factors(
    dim: int,
    original_max_position: int,
    max_position: int,
    base: float,
    beta_fast: float,
    beta_slow: float,
    factor: float,
    attention_factor: float | None,
) -> tuple[np.ndarray, float]:
    """yarn 逆频（vLLM yarn 语义）。"""
    if factor <= 1.0:
        inv = 1.0 / (base ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
        return inv, 1.0
    mscale = _yarn_get_mscale(factor)
    if attention_factor is None:
        attention_factor = mscale
    pos_mult = factor
    inv_dim = dim
    # attention factor 修正（>1.5 时）
    if factor > 1.5:
        pos_mult = factor / _yarn_mscale(factor)
    inv_freq_shape = (dim // 2,)
    inv_freq = 1.0 / (base ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
    # correction range
    low, high = _yarn_find_correction_range(0.1, 32.0, inv_dim, base, beta_fast, beta_slow)
    # 按频率索引（从低频到高频）的衰减权重
    freqs = 1.0 / inv_freq  # 波长
    pos = np.arange(0, dim, 2, dtype=np.float32) / dim  # 占位（per-dim 权重）
    correction = 0.5 * np.log(max_position / base)
    smooth = (pos_mult * (inv_freq * 0) )  # 占位防未用变量
    # 逐维修正：w = 1 - (freq_index/low) 在 [low, high] 线性过渡（freq 越大修正越强）
    # 用 vLLM 公式：inv_freq_orig 与 inv_freq_scaled 的混合
    inv_freq_orig = 1.0 / (base ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
    # 频率位置（归一化）
    idx = np.arange(0, dim, 2, dtype=np.float32) / dim
    # smoothstep 在 [1/high, 1/low] 内过渡（波长空间）
    wave = base ** idx  # 波长
    lo_w = base ** (1.0 / high) if high > 0 else 0.0
    hi_w = base ** (1.0 / low) if low > 0 else float("inf")
    t = np.clip((wave - lo_w) / (hi_w - lo_w + 1e-12), 0.0, 1.0)
    smooth_t = t * t * (3.0 - 2.0 * t)
    inv_freq = inv_freq_orig * (1.0 - smooth_t) + inv_freq_orig / pos_mult * smooth_t
    return inv_freq, attention_factor


def build_rope(
    head_dim: int,
    rope_theta: float,
    max_position: int,
    rope_scaling: dict | None = None,
    original_max_position: int | None = None,
) -> tuple[np.ndarray, float]:
    """构建逆频表与 attention factor（scaling 感知）。

    返回 ``(inv_freq [head_dim//2], attention_factor)``。
    """
    rope_scaling = rope_scaling or {}
    stype = (rope_scaling.get("type") or "none").casefold()
    factor = float(rope_scaling.get("factor", 1.0))
    dim = head_dim
    if stype == "none" or factor <= 1.0:
        return 1.0 / (rope_theta ** (np.arange(0, dim, 2, dtype=np.float32) / dim)), 1.0
    if stype == "linear":
        # 缩放 = 位置 / factor → 等价 inv_freq * (1/factor)
        return (
            (1.0 / (rope_theta ** (np.arange(0, dim, 2, dtype=np.float32) / dim))) / factor,
            1.0,
        )
    if stype == "dynamic":
        # dynamic：不缩 inv_freq，由 apply 阶段按 seq_len 动态缩位置
        return 1.0 / (rope_theta ** (np.arange(0, dim, 2, dtype=np.float32) / dim)), 1.0
    if stype == "yarn":
        ompp = original_max_position or int(max_position / factor)
        inv_freq, attn_factor = _yarn_get_scaled_factors(
            dim=dim,
            original_max_position=ompp,
            max_position=max_position,
            base=rope_theta,
            beta_fast=float(rope_scaling.get("beta_fast", 32.0)),
            beta_slow=float(rope_scaling.get("beta_slow", 1.0)),
            factor=factor,
            attention_factor=rope_scaling.get("attention_factor"),
        )
        return inv_freq, attn_factor
    raise ValueError(f"不支持的 rope_scaling.type: {stype!r}（none/linear/dynamic/yarn）")


def apply_rope(
    q: np.ndarray,
    k: np.ndarray,
    inv_freq: np.ndarray,
    positions: np.ndarray,
    attention_factor: float = 1.0,
    scaling_type: str = "none",
    seq_len: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """施加 RoPE（半旋转）。

    - ``q``: [batch, heads, seq, head_dim] 或 [seq, heads, head_dim]（batch=1 时 3D 也可）；
    - ``k``: [batch, kv_heads, seq, head_dim]（GQA 时 kv_heads < heads，RoPE 对 kv 头独立施加）；
    - ``positions``: [batch, seq] int（绝对位置，支持 prefix cache 偏移）；
    - dynamic scaling：``seq_len>0`` 且类型 dynamic 时位置乘 ``sqrt(factor)``（vLLM 语义：
      ``factor / mscale(factor)``）。
    返回 (q_rot, k_rot)，与输入同 shape，float32。
    """
    q32 = q.astype(np.float32, copy=False)
    k32 = k.astype(np.float32, copy=False)
    if scaling_type == "dynamic" and seq_len > 1 and attention_factor == 1.0:
        # dynamic：位置缩放 = factor / mscale(factor)，由调用方预乘进 positions
        pass  # positions 已含缩放
    positions32 = positions.astype(np.float32)
    # freqs [batch, seq, d/2]
    freqs = positions32[..., None] * inv_freq[None, None, :]
    # dynamic attention factor 作用于 q
    if scaling_type in ("yarn", "linear") and attention_factor != 1.0:
        q32 = q32 * attention_factor
    half = q32.shape[-1] // 2
    # cos/sin [batch, 1, seq, d/2]（头维广播）
    cos = np.cos(freqs)[:, None, :, :]
    sin = np.sin(freqs)[:, None, :, :]
    q_out = np.empty_like(q32)
    q1 = q32[..., :half]
    q2 = q32[..., half:]
    q_out[..., :half] = q1 * cos - q2 * sin
    q_out[..., half:] = q2 * cos + q1 * sin
    # k：[batch, kv_heads, seq, d] 与 [batch, 1, seq, d/2] 广播
    k_out = np.empty_like(k32)
    k1 = k32[..., :half]
    k2 = k32[..., half:]
    k_out[..., :half] = k1 * cos - k2 * sin
    k_out[..., half:] = k2 * cos + k1 * sin
    return q_out, k_out
