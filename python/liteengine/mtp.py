"""MTP（Multi-Token Prediction）模块（DeepSeek-V3.2 风格）。

k 个 MTP 模块 = 共享嵌入 + 投影（eh_proj）+ 附加 Transformer 层 + 共享输出头。
前向：输入 = 当前 token 表示 + 上一深度 token 嵌入（enorm/hnorm 归一化 + eh_proj 投影合并）
→ 附加 DecoderLayer → 输出表示（投机解码的草稿——多 token 预测，DeepSeek-V3 的 MTP 目标）。
权重前缀约定（engine.py/lazy_loader.py 默认跳过 mtp.*——加载时按需读取）。
"""
from __future__ import annotations

from torch import Tensor
from torch.nn.functional import linear

from liteengine.layer import DecoderLayer
from liteengine.loader import WeightStore
from liteengine.moe import torch_weight
from liteengine.core.norm import rms_norm

__all__ = ["MtpModule"]


class MtpModule:
    """单个 MTP 模块：enorm/hnorm（RMSNorm）+ eh_proj（降维投影）+ 附加 DecoderLayer。"""

    def __init__(self, store: WeightStore, prefix: str, cfg: dict, layer_idx: int):
        self.prefix = prefix
        self.enorm_w = torch_weight(store, f"{prefix}.enorm.weight")
        self.hnorm_w = torch_weight(store, f"{prefix}.hnorm.weight")
        self.eh_proj = torch_weight(store, f"{prefix}.eh_proj.weight")   # (hidden, mtp_hidden)
        self.eps = float(cfg.get("rms_norm_eps", 1e-6))
        # 附加 Transformer 层（与主模型层结构相同——复用 DecoderLayer）
        self.layer = DecoderLayer(store, layer_idx, cfg)

    def forward(self, h: Tensor, embed_prev: Tensor) -> Tensor:
        """输入 = 当前表示 + 上一深度 token 嵌入（enorm/hnorm 归一化 + eh_proj 投影合并）。

        ``embed_prev`` 为上一预测深度的 token 嵌入（多 token 预测链）；附加层的
        cos/sin/mask 忽略（MTP 附加层为线性注意力场景）。
        """
        h_n = rms_norm(h, self.enorm_w, self.eps)
        e_n = rms_norm(embed_prev, self.hnorm_w, self.eps)
        h_in = h_n + linear(e_n, self.eh_proj)
        return self.layer(h_in)

    def draft(self, h: Tensor, embed_prev: Tensor, lm_head) -> Tensor:
        """投机草稿：输出表示 → 共享输出头（主模型 lm_head）→ logits（多 token 预测）。

        ``lm_head`` 与主模型共享（DeepSeek-V3 的 shared_head 约定）；多 token 预测链
        由多个 MTP 模块顺序拼接（第 k 个模块的 embed_prev = 第 k-1 个预测 token 的嵌入）。
        """
        return lm_head(self.forward(h, embed_prev))
