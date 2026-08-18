"""通用反量化分发：按 QuantConfig.quant_method 经注册表选择 AWQ / GPTQ 路径。"""
from __future__ import annotations

from numpy import ndarray

from liteengine.quant.awq import dequantize_awq        # noqa: F401（触发 awq 处理器注册）
from liteengine.quant.config import QuantConfig
from liteengine.quant.fp8 import dequantize_fp8        # noqa: F401（触发 fp8/e4m3/e5m2 注册）
from liteengine.quant.gptq import dequantize_gptq      # noqa: F401（触发 gptq 处理器注册）
from liteengine.quant.nvfp4 import dequantize_nvfp4    # noqa: F401（触发 nvfp4 处理器注册）
from liteengine.registry import get_quant_method

__all__ = ["dequantize"]


def dequantize(qweight, qzeros, scales, cfg: QuantConfig, dtype: str = "float32") -> ndarray:
    """通用反量化：列打包 × 对称/非对称 × bits × group_size。

    按 ``cfg.quant_method`` 经注册表分发（liteengine.registry）：
    - ``awq``：AWQ 列序还原路径（真实列序）
    - 其它 / 未知方法：回退 GPTQ 线性序（gptq/gptq_v2/...）
    - 外部新增量化方法：``@register_quant_method("my_method")`` 接入
    """
    handler = get_quant_method(cfg.quant_method) or get_quant_method("gptq")
    return handler(qweight, qzeros, scales, cfg, dtype)
