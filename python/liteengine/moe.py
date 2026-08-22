"""MoE 路由 + AWQ 量化专家 + shared expert（对照 transformers Qwen3_5MoeSparseMoeBlock）。

量化专家：仅对激活专家按需反量化（M2 正确性优先，逐专家计算；
M4 将替换为专家流式换入，避免 256 个专家全部驻留内存）。
"""
from __future__ import annotations

import torch
from torch import Tensor
from torch.nn.functional import linear, silu, softmax

from liteengine.cache import ExpertCache
from liteengine.quant import QuantConfig, dequantize

__all__ = ["TopKRouter", "QuantizedExperts", "MergedExperts", "MLP",
           "SparseMoeBlock", "torch_weight", "torch_weight_native"]


def torch_weight(store, name: str, dtype: torch.dtype = torch.float32) -> Tensor:
    """WeightStore 普通权重 → torch 张量（CPU 统一 fp32 计算）。"""
    return torch.from_numpy(store.get(name)).to(dtype)


def torch_weight_native(store, name: str) -> Tensor:
    """按原 dtype 加载权重（精度适配：fp16 保留为 float16——省内存；量化整数张量转 float32）。"""
    t = torch.from_numpy(store.get(name))
    if t.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        return t.to(dtype=torch.float32)
    return t


class TopKRouter:
    """Qwen3_5MoeTopKRouter：linear → softmax(fp32) → topk(top_k) → 归一化。"""

    def __init__(self, weight: Tensor, top_k: int):
        self.weight = weight  # (num_experts, hidden)
        self.top_k = top_k

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        flat = x.reshape(-1, x.shape[-1])
        logits = linear(flat, self.weight)
        probs = softmax(logits, dim=-1, dtype=torch.float32)
        scores, indices = probs.topk(self.top_k, dim=-1)
        scores = scores / scores.sum(dim=-1, keepdim=True)
        return scores.to(logits.dtype), indices

    __call__ = forward


class QuantizedExperts:
    """num_experts 个 AWQ 4bit 专家：gate/up/down 按激活专家反量化后 SwiGLU。

    Args:
        store: WeightStore（提供 .get(name) → numpy）。
        prefix: 专家张量名前缀，如 ``model.language_model.layers.0.mlp.experts``。
        num_experts: 专家数量（256）。
        group_size: AWQ 分组（32）。
    """

    def __init__(self, store, prefix: str, num_experts: int, group_size: int = 32,
                 cache: ExpertCache | None = None, layer_idx: int = 0,
                 quant_cfg: QuantConfig | None = None, compute_dtype: str = "float32",
                 expert_parallel: bool = False, fused_matmul: bool = False):
        self.store = store
        self.prefix = prefix
        self.num_experts = num_experts
        self.group_size = group_size
        self.layer_idx = layer_idx
        self.quant_cfg = quant_cfg or QuantConfig(quant_method="awq", bits=4, group_size=group_size)
        self.compute_dtype = compute_dtype            # 反量化输出精度（fp32 默认 / fp16 可选）
        self.expert_parallel = expert_parallel        # 专家多线程并行（本机实测慢——默认关）
        self.fused_matmul = fused_matmul              # int4 融合 matmul（engine.toml [device]/[inference] 开关）
        # 反量化专家缓存：跨层共享（全局条目上限，LRU），避免重复读盘与反量化。
        # 未传入时使用独立缓存（测试直构场景）。
        self._cache = cache if cache is not None else ExpertCache()

    def _dequant(self, expert_idx: int, proj: str) -> Tensor:
        key = (self.layer_idx, expert_idx)
        entry = self._cache.get(key)
        if entry is None:
            entry = {
                p: torch.from_numpy(dequantize(
                    self.store.get(f"{self.prefix}.{expert_idx}.{p}.qweight"),
                    self.store.get(f"{self.prefix}.{expert_idx}.{p}.qzeros"),
                    self.store.get(f"{self.prefix}.{expert_idx}.{p}.scales"),
                    self.quant_cfg,
                    dtype=self.compute_dtype,
                ))
                for p in ("gate_proj", "up_proj", "down_proj")
            }
            self._cache.put(key, entry)
        return entry[proj]

    def clear_cache(self) -> None:
        """清空专家缓存（会话切换 / 内存回收时调用）。"""
        self._cache.clear()

    def _matmul_int4_fused(self, x, qweight, qzeros, scales, out_dim, in_dim, group_size):
        """T-MAC 风格融合 int4 matmul：反量化按 group 块即时融入 matmul（避免全量反量化
        物化到内存再读取的双重往返——镜像 llama.cpp Q4 的块级思路）。

        x: (B, in_dim)；qweight/qzeros: int32 打包；scales: (groups, in_dim) float32。
        返回 (B, out_dim) fp32。
        """
        import numpy as np
        from liteengine.quant.unpack import _unpack_int4_colwise
        w = _unpack_int4_colwise(np.ascontiguousarray(qweight))   # (rows, cols) int8
        # 转置存储兼容：真实模型专家权重为 [in, out] 转置——out_dim 等于列数时转置为 (out, in)
        if w.shape[0] != out_dim and w.shape[1] == out_dim:
            w = w.T
        z = _unpack_int4_colwise(np.ascontiguousarray(qzeros))    # (groups, in) int8
        xf = x.float()
        out = torch.empty((x.shape[0], out_dim), dtype=torch.float32)
        groups = (out_dim + group_size - 1) // group_size
        for g in range(groups):
            rows = slice(g * group_size, min((g + 1) * group_size, out_dim))
            wg = (torch.from_numpy(w[rows]).to(dtype=xf.dtype)
                  - torch.from_numpy(z[g]).to(dtype=xf.dtype)) * \
                 torch.from_numpy(scales[g].astype(np.float32))
            out[:, rows] = xf @ wg.T
        return out

    def forward(self, x: Tensor, indices: Tensor, weights: Tensor) -> Tensor:
        """x: (seq, hidden)；indices/weights: (seq, top_k)。返回 (seq, hidden)。

        注意：本模型量化专家权重以 [in, out] 转置存储（对照真实张量
        gate_proj.qweight [2048, 64] → 反量化 [hidden=2048, interm=512]），
        因此 matmul 前需转置回 [out, in] 的 linear 惯例。
        """
        final = torch.zeros_like(x)
        experts = torch.unique(indices).tolist()

        def _expert(e: int):
            e = int(e)
            pos = (indices == e).nonzero()          # (n, 2)：[:, 0]=token, [:, 1]=topk 位
            tok_idx, k_idx = pos[:, 0], pos[:, 1]
            cur = x[tok_idx]
            # 融合 int4 matmul（engine.toml [device].fused_matmul / [inference].int4_fused_matmul 开关）：
            # 开启时用 _matmul_int4_fused（T-MAC 风格块级——合成 [out,in] 实测 1.89x；
            # 真实模型 [in,out] 转置存储的 z/scales 适配为后续——默认关走 _dequant 正确性优先）
            if self.fused_matmul:
                qw_g = self.store.get(f"{self.prefix}.{e}.gate_proj.qweight")
                qw_u = self.store.get(f"{self.prefix}.{e}.up_proj.qweight")
                gate = self._matmul_int4_fused(
                    cur, qw_g,
                    self.store.get(f"{self.prefix}.{e}.gate_proj.qzeros"),
                    self.store.get(f"{self.prefix}.{e}.gate_proj.scales"),
                    min(qw_g.shape[0], qw_g.shape[1] * 8), cur.shape[1], self.quant_cfg.group_size)
                up = self._matmul_int4_fused(
                    cur, qw_u,
                    self.store.get(f"{self.prefix}.{e}.up_proj.qzeros"),
                    self.store.get(f"{self.prefix}.{e}.up_proj.scales"),
                    min(qw_u.shape[0], qw_u.shape[1] * 8), cur.shape[1], self.quant_cfg.group_size)
                out = silu(gate) * up
                qw_d = self.store.get(f"{self.prefix}.{e}.down_proj.qweight")
                out = self._matmul_int4_fused(
                    out, qw_d,
                    self.store.get(f"{self.prefix}.{e}.down_proj.qzeros"),
                    self.store.get(f"{self.prefix}.{e}.down_proj.scales"),
                    x.shape[1], out.shape[1], self.quant_cfg.group_size)
            else:
                gate = linear(cur, self._dequant(e, "gate_proj").T)
                up = linear(cur, self._dequant(e, "up_proj").T)
                out = silu(gate) * up
                out = linear(out, self._dequant(e, "down_proj").T)
            return tok_idx, out * weights[tok_idx, k_idx][:, None]

        # 开关（engine.toml [inference].expert_parallel）：默认串行（本机实测最优）；
        # 开启时多线程并行（他机/多核可能受益——本机 BLAS 竞争实测慢 ~5 倍）
        if self.expert_parallel and len(experts) > 1:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=min(len(experts), 4)) as pool:
                for tok_idx, contrib in pool.map(_expert, experts):
                    final.index_add_(0, tok_idx, contrib)
        else:
            for e in experts:
                tok_idx, contrib = _expert(e)
                final.index_add_(0, tok_idx, contrib)
        return final

    __call__ = forward


class MergedExperts:
    """非量化合并专家（Mixtral / Qwen3-MoE 格式）：
    ``gate_up_proj`` [E, 2*inter, hidden] + ``down_proj`` [E, hidden, inter]，标准 [out, in] 序
    （与 Qwen3.5 的量化分离存储 / [in, out] 转置约定不同）。
    """

    def __init__(self, store, prefix: str, num_experts: int):
        self.num_experts = num_experts
        self.gate_up = torch.from_numpy(store.get(f"{prefix}.gate_up_proj.weight")).float()
        self.down = torch.from_numpy(store.get(f"{prefix}.down_proj.weight")).float()

    def forward(self, x: Tensor, indices: Tensor, weights: Tensor) -> Tensor:
        """x: (seq, hidden)；indices/weights: (seq, top_k)。返回 (seq, hidden)。"""
        final = torch.zeros_like(x)
        for e in torch.unique(indices).tolist():
            e = int(e)
            pos = (indices == e).nonzero()          # (n, 2)：[:, 0]=token, [:, 1]=topk 位
            tok_idx, k_idx = pos[:, 0], pos[:, 1]
            cur = x[tok_idx]
            gate, up = linear(cur, self.gate_up[e]).chunk(2, dim=-1)
            out = silu(gate) * up
            out = linear(out, self.down[e])
            final.index_add_(0, tok_idx, out * weights[tok_idx, k_idx][:, None])
        return final

    __call__ = forward


class MLP:
    """SwiGLU MLP（shared_expert 用）：``silu(gate(x)) * up(x) → down``。"""

    def __init__(self, gate_w: Tensor, up_w: Tensor, down_w: Tensor):
        self.gate_w, self.up_w, self.down_w = gate_w, up_w, down_w

    def forward(self, x: Tensor) -> Tensor:
        return linear(silu(linear(x, self.gate_w)) * linear(x, self.up_w), self.down_w)

    __call__ = forward


class DenseBlock:
    """稠密 MLP 块（无路由，Llama 家族 / 通用回退的稠密模型）：
    与 SparseMoeBlock 同接口（forward(x) → (rows, hidden)）。"""

    def __init__(self, mlp: MLP):
        self.mlp = mlp

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(x)

    __call__ = forward


class SparseMoeBlock:
    """MoE 块：router + 专家 +（可选）共享专家（sigmoid 门控）。

    共享专家可选：Mixtral / DeepSeek 等无共享专家的模型传 ``shared_mlp=None`` 即跳过。
    """

    def __init__(self, router: TopKRouter, experts,
                 shared_mlp: MLP | None = None, shared_gate_w: Tensor | None = None):
        self.router = router
        self.experts = experts
        self.shared_mlp = shared_mlp
        self.shared_gate_w = shared_gate_w

    def forward(self, x: Tensor) -> Tensor:
        shape = x.shape
        xr = x.reshape(-1, x.shape[-1])
        scores, indices = self.router(xr)
        out = self.experts(xr, indices, scores)
        if self.shared_mlp is not None:
            shared = torch.sigmoid(linear(xr, self.shared_gate_w)) * self.shared_mlp(xr)
            out = out + shared
        return out.reshape(shape)

    __call__ = forward


# ---- 注册表：内置专家/MLP 构建器（layer 按格式名查找；外部组件可新增注册）----

from liteengine.registry import register_moe_format


@register_moe_format("quantized_separate")
def _build_quantized(store, prefix, moe, num_experts, expert_cache, layer_idx, quant_cfg,
                     compute_dtype="float32", expert_parallel=False):
    return QuantizedExperts(store, f"{prefix}.experts", num_experts,
                            cache=expert_cache, layer_idx=layer_idx, quant_cfg=quant_cfg,
                            compute_dtype=compute_dtype, expert_parallel=expert_parallel)


@register_moe_format("merged_plain")
def _build_merged(store, prefix, moe, num_experts, expert_cache, layer_idx, quant_cfg,
                  compute_dtype="float32", expert_parallel=False):
    return MergedExperts(store, f"{prefix}.experts", num_experts)


@register_moe_format("dense_mlp")
def _build_dense(store, prefix, moe, num_experts, expert_cache, layer_idx, quant_cfg,
                 compute_dtype="float32", expert_parallel=False):
    return DenseBlock(MLP(
        torch_weight(store, f"{prefix}.gate_proj.weight"),
        torch_weight(store, f"{prefix}.up_proj.weight"),
        torch_weight(store, f"{prefix}.down_proj.weight"),
    ))
