"""量化配置：QuantConfig + 从模型 config.json 读取。"""
from __future__ import annotations

from dataclasses import dataclass
from json import load

__all__ = ["QuantConfig", "load_quant_config"]


@dataclass
class QuantConfig:
    """量化配置（模型 config.json 的 quantization_config 解析）。

    支持：quant_method（awq/gptq/gptq_v2/...）、bits（2/4/8）、group_size、
    sym（对称，无 qzeros）、desc_act（激活序分组，g_idx）。
    """

    quant_method: str = "awq"
    bits: int = 4
    group_size: int = 32
    sym: bool = False
    desc_act: bool = False
    dtype: str = ""                   # fp8 变体：fp8_e4m3 / fp8_e5m2（空 = 默认 e4m3）

    @classmethod
    def from_dict(cls, qcfg: dict | None) -> "QuantConfig":
        if not qcfg:
            return cls()
        sym = bool(qcfg.get("sym", False))
        if "zero_point" in qcfg:                     # AWQ 的 zero_point 字段（True=非对称）
            sym = not bool(qcfg.get("zero_point"))
        return cls(
            quant_method=str(qcfg.get("quant_method", "awq")).lower(),
            bits=int(qcfg.get("bits", 4)),
            group_size=int(qcfg.get("group_size", 32)),
            sym=sym,
            desc_act=bool(qcfg.get("desc_act", False)),
            dtype=str(qcfg.get("dtype", "")).lower(),
        )


def load_quant_config(model_dir: str) -> QuantConfig:
    """读取模型目录 config.json 的 quantization_config。"""
    try:
        with open(f"{model_dir}/config.json", "r", encoding="utf-8") as f:
            return QuantConfig.from_dict(load(f).get("quantization_config"))
    except Exception:
        return QuantConfig()
