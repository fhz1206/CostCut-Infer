"""ccut.quant — 量化子系统（§3.6，对齐 vLLM layers/quantization 结构）。

- spec：QuantKey/ScaleDesc/GroupShape 声明式规格（移植 vLLM QuantKey）。
- method：QuantizeMethodBase / QuantizationConfig / get_quant_method 层分发。
- registry：QUANTIZATION_METHODS 注册表 + checkpoint quant_method 解析。
- kernels：numba 内核（FP8/INT8/MX 转换、group broadcast、融合 GEMM、VNNI）。
- fp8 / compressed_tensors / int8 / mx / weight_only / online / kv：各格式方法。
"""
