"""GPTQ / 通用列打包反量化（线性序，bits 任意，对称/非对称）。"""
from __future__ import annotations

from numpy import ndarray

from liteengine.quant.config import QuantConfig
from liteengine.quant.unpack import _OUT_DTYPES, _dequant_formula, _unpack_colwise
from liteengine.registry import register_quant_method

__all__ = ["dequantize_gptq"]


@register_quant_method("gptq")
def _handle_gptq(qweight, qzeros, scales, cfg: QuantConfig, dtype: str = "float32") -> ndarray:
    """注册表处理器：quant_method == "gptq"/"gptq_v2" 等线性序时经 registry 分发。"""
    return dequantize_gptq(qweight, qzeros, scales, cfg, dtype)


def dequantize_gptq(qweight: ndarray, qzeros: ndarray | None, scales: ndarray,
                    cfg: QuantConfig, dtype: str = "float32") -> ndarray:
    """GPTQ / 通用列打包反量化（线性序，无 AWQ 列序重排）。

    - ``bits``：2/4/8（列打包解包）
    - ``sym``：无 qzeros，直接 ``q * scale``
    - ``group_size``：沿 out 维分组
    """
    out_dtype = _OUT_DTYPES[dtype]
    unpack = lambda t: _unpack_colwise(t, cfg.bits)          # noqa: E731
    w = unpack(qweight)
    gs = cfg.group_size if cfg.group_size and cfg.group_size > 0 else w.shape[0]
    if cfg.sym:
        return _dequant_formula(w, None, scales, gs, sym=True, out_dtype=out_dtype)
    z = unpack(qzeros)
    return _dequant_formula(w, z, scales, gs, sym=False, out_dtype=out_dtype)
