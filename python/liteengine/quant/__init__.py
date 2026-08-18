"""量化子包：按算法拆分（config / unpack / awq / gptq / dequantize）。

对外 API 与旧 ``liteengine.quant.py`` 保持兼容：
``QuantConfig`` / ``load_quant_config`` / ``dequantize`` / ``dequantize_awq`` /
``_unpack_colwise`` / ``_unpack_int4_colwise`` 导入路径不变。
"""
from liteengine.quant.awq import dequantize_awq
from liteengine.quant.config import QuantConfig, load_quant_config
from liteengine.quant.dequantize import dequantize
from liteengine.quant.fp8 import dequantize_fp8, e4m3_to_f32, e5m2_to_f32
from liteengine.quant.gptq import dequantize_gptq
from liteengine.quant.nvfp4 import dequantize_nvfp4, e2m1_to_f32
from liteengine.quant.unpack import (_OUT_DTYPES, _REVERSE_AWQ_PACK_ORDER,
                                     _unpack_colwise, _unpack_int4_colwise)

__all__ = ["QuantConfig", "load_quant_config", "dequantize", "dequantize_awq",
           "dequantize_gptq", "dequantize_fp8", "dequantize_nvfp4",
           "e4m3_to_f32", "e5m2_to_f32", "e2m1_to_f32",
           "_unpack_colwise", "_unpack_int4_colwise",
           "_OUT_DTYPES", "_REVERSE_AWQ_PACK_ORDER"]
