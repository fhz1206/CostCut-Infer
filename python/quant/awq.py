"""AWQ 反量化算法（int4 列打包 + AWQ 非标准列序还原）。"""
from __future__ import annotations

from numpy import ndarray

from quant.config import QuantConfig
from quant.unpack import _OUT_DTYPES, _dequant_formula, _unpack_int4_colwise
from engine.registry import register_quant_method

__all__ = ["dequantize_awq"]


@register_quant_method("awq")
def _handle_awq(qweight, qzeros, scales, cfg: QuantConfig, dtype: str = "float32") -> ndarray:
    """注册表处理器：quant_method == "awq" 时经 registry 分发。"""
    return dequantize_awq(qweight, qzeros, scales, cfg.group_size, dtype)


def dequantize_awq(qweight: ndarray, qzeros: ndarray, scales: ndarray,
                   group_size: int = 32, dtype: str = "float32") -> ndarray:
    """AWQ 4bit 反量化：返回 [out, in] 权重矩阵。

    含 AWQ 非标准列序还原（vLLM _REVERSE_AWQ_PACK_ORDER / gptqmodel reverse_awq_order），
    与本仓库 Qwen3.6-35B-A3B-AWQ-4bit 权重布局一致。
    """
    out_dtype = _OUT_DTYPES[dtype]
    w = _unpack_int4_colwise(qweight)          # [out, in]（列序已还原）
    if qzeros is None:                          # 对称量化（无 qzeros）
        return _dequant_formula(w, None, scales, group_size, sym=True, out_dtype=out_dtype)
    z = _unpack_int4_colwise(qzeros)           # [groups, in]
    return _dequant_formula(w, z, scales, group_size, sym=False, out_dtype=out_dtype)
