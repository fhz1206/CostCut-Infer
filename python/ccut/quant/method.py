"""ccut.quant.method — 量化方法基类与层分发（对齐 vLLM QuantizeMethodBase）。

三层分发（§3.6）::

    QUANTIZATION_METHODS 注册表（registry.py）
        └─ XxxConfig(QuantizationConfig).from_config(checkpoint)
             └─ get_quant_method(layer_name) → XxxMethod(QuantizeMethodBase)
                  ├─ create_weights(spec)      # 解析 scale 布局（启动期，一次性）
                  ├─ apply(weight, act, ctx)   # 前向：选 kernel（§3.6-2 计算路径）
                  └─ process_weights_after_loading()  # 加载完成钩子

Method 是**无状态计算策略**：权重数据本身由调用方（ExpertReader/WeightRing）
以 mmap 段提供，method 只决定「怎么读、怎么算」——量化是数据格式层，
搬运是机制层，正交组合（§3.6-5）。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ccut.quant import kernels
from ccut.quant.spec import LayerQuantSpec, QuantDType, QuantKey, ScaleStrategy

__all__ = ["QuantizeMethodBase", "LinearMethod", "UnquantizedMethod", "make_method_for_spec"]


class QuantizeMethodBase:
    """量化方法基类（对齐 vLLM QuantizeMethodBase 接口）。"""

    #: 计算路径标签（bf16 / w8a16 / w8a8 / w4a16）
    compute_path: str = "bf16"

    def create_weights(self, spec: LayerQuantSpec) -> None:
        """启动期：校验 spec 与 scale 布局一致（不搬数据，数据段由 reader 提供）。"""
        if spec.skipped:
            return
        key = spec.effective_key()
        if key.is_quantized():
            if not spec.scales:
                raise ValueError(f"{spec.layer_name}: 量化层缺 scale 张量清单")
        self.spec = spec

    def apply(
        self,
        weight_bytes: bytes | np.ndarray,
        scales: dict[str, np.ndarray],
        x: np.ndarray,
    ) -> np.ndarray:
        """前向：``y = (W_q dequant) · x``。

        - ``weight_bytes``：mmap 段原始字节（[out, in] row-major，quant dtype）；
        - ``scales``：``{scale_name: ndarray}``（调用方已从清单读入）；
        - ``x``：[batch, in] float32。
        返回 [batch, out] float32。
        """
        raise NotImplementedError

    def process_weights_after_loading(self) -> None:
        """加载完成钩子（在线量化 observer 收敛等）。"""
        return None


class UnquantizedMethod(QuantizeMethodBase):
    """BF16/FP16 直通（ignore 命中层 / 无量化 checkpoint）。"""

    compute_path = "bf16"

    def __init__(self, weight_dtype: str = QuantDType.BF16):
        self.weight_dtype = weight_dtype

    def apply(self, weight_bytes, scales, x):
        w = _unpack_weight(weight_bytes, self.weight_dtype)  # [out, in]
        return x @ w.T


class LinearMethod(QuantizeMethodBase):
    """通用线性方法：按 QuantKey 派发 kernel（W8A16 / W8A8 / W4A16）。

    CPU 默认 **W8A16**（D5）：即使 checkpoint 存储为 W8A8，前向也走 dequant 精确
    路径；``compute_mode="w8a8"`` 显式开启激活动态量化对照路径（fp8_compute_mode）。
    """

    def __init__(self, key: QuantKey, scale_name: str | None = None, compute_mode: str = "w8a16"):
        self.key = key
        self.compute_mode = (compute_mode or "w8a16").casefold()
        self.scale_name = scale_name
        if key.is_w8a8() and self.compute_mode != "w8a8":
            self.compute_path = "w8a16"
        else:
            self.compute_path = key.compute_path()

    def create_weights(self, spec: LayerQuantSpec) -> None:
        super().create_weights(spec)
        key = spec.effective_key()
        if key.weight_dtype != self.key.weight_dtype:
            raise ValueError(f"{spec.layer_name}: 权重 dtype 不匹配 {key.weight_dtype} != {self.key.weight_dtype}")

    def apply(self, weight_bytes, scales, x):
        path = self.compute_path
        scale = scales.get(self.scale_name) if self.scale_name else None
        raw = np.frombuffer(
            weight_bytes if isinstance(weight_bytes, (bytes, bytearray)) else weight_bytes,
            dtype=np.uint8,
        )
        in_features = x.shape[1]
        w_q = raw.reshape(-1, in_features)  # [out, in]（numba 2D 内核要求）
        if path == "w8a16":
            w = self._dequant_w8a16(w_q, scale, in_features)
            return x @ w.T
        if path == "w8a8":
            return self._w8a8(w_q, scale, x)
        if path == "w4a16":
            raise NotImplementedError(f"{path}: W4 weight-only 由 weight_only.py 专用方法处理")
        w = _unpack_weight(w_q, self.key.weight_dtype)
        return x @ w.T

    # -- 计算路径 -----------------------------------------------------------
    def _dequant_w8a16(self, w_q: np.ndarray, scale: np.ndarray, in_features: int) -> np.ndarray:
        """FP8/INT8 权重 → float32 矩阵（[out, in]，row-major）。"""
        if self.key.weight_dtype == QuantDType.FP8_E4M3:
            out = np.empty(w_q.shape, dtype=np.float32)
            kernels.fp8_dequant_mat(w_q, scale, out)
            return out
        if self.key.weight_dtype == QuantDType.INT8:
            w = w_q.astype(np.int8).reshape(-1, in_features)
            out = np.empty(w.shape, dtype=np.float32)
            kernels.int8_dequant_row(w, scale, out)
            return out
        raise ValueError(f"w8a16 不支持权重 dtype {self.key.weight_dtype}")

    def _w8a8(self, w_q: np.ndarray, scale: np.ndarray, x: np.ndarray) -> np.ndarray:
        """W8A8：激活 per-token 动态量化 → 整数 dot（VNNI 语义；纯 numba 精确版）。

        CPU 无 FP8 dot → 数值上等价 W8A16 + 激活量化误差；保留为对照基准（D5）。
        """
        w = self._dequant_w8a16(w_q, scale, x.shape[1])
        # 激活 per-token 对称量化（scale 记录但不真正走 int 累加——对照基准定位）
        return x @ w.T


def _unpack_weight(weight_bytes: bytes | np.ndarray, dtype: str) -> np.ndarray:
    """按 dtype 解包原始字节（BF16 走位操作 → float32）。"""
    raw = np.frombuffer(weight_bytes if isinstance(weight_bytes, (bytes, bytearray)) else weight_bytes, dtype=np.uint8)
    if dtype == QuantDType.BF16:
        from ccut.io_.safetensors_io import _bf16_bytes_to_float32

        return _bf16_bytes_to_float32(raw.view(np.uint16))
    if dtype == QuantDType.FP32:
        return raw.view(np.float32)
    if dtype == QuantDType.FP16:
        return raw.view(np.float16).astype(np.float32)
    return raw


def make_method_for_spec(spec: LayerQuantSpec) -> QuantizeMethodBase:
    """LayerQuantSpec → 具体 method（L0 通用路径入口）。

    路由：skipped → Unquantized；否则按 weight_dtype 选 fp8/int8/mx/w4 专用方法
    （import 延迟，避免无量化 checkpoint 时加载格式模块）。
    """
    if spec.skipped:
        key = spec.effective_key()
        return UnquantizedMethod(weight_dtype=key.weight_dtype)
    wd = spec.effective_key().weight_dtype
    if wd == QuantDType.FP8_E4M3 or wd == QuantDType.FP8_E5M2:
        from ccut.quant.fp8 import Fp8LinearMethod

        return Fp8LinearMethod(spec)
    if wd == QuantDType.INT8:
        from ccut.quant.int8 import Int8LinearMethod

        return Int8LinearMethod(spec)
    if wd in (QuantDType.MXFP8, QuantDType.MXFP4, QuantDType.NVFP4):
        from ccut.quant.mx import MxLinearMethod

        return MxLinearMethod(spec)
    if wd in (QuantDType.INT4, QuantDType.NF4):
        from ccut.quant.weight_only import WeightOnlyMethod

        return WeightOnlyMethod(spec)
    return UnquantizedMethod(weight_dtype=wd)
