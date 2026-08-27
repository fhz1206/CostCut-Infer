"""ccut.models.generic — 通用组装器（ModelSpec + blocks + quant → 层前向）。

零手写模型代码（§3.4-3）：本模块按 :class:`ModelSpec.layer_templates` 逐层查
blocks（attn_gdn / attn_gqa / moe / norm / rope / heads）+ quant method，
组装成可运行的层前向。**权重不驻留**——每个投影的字节段由
:class:`WeightReader`（weights/stream.py）按 (layer, proj) 提供，method.apply
内做 dequant+matmul（R10 协同：层进 ring 时权重段就绪）。

数据流（Ornith 单步，decode 单 token）::

    h = embed(token)
    for layer in layer_templates:
        h = input_layernorm(h)
        h = attn_block(h)          # linear_attn（GDN 递归）或 full_attn（GQA+KV 块池）
        h += h 残差
        h2 = post_attention_layernorm(h)
        h += moe(h2)               # 路由 top-k → 专家（R2 流式）+ 共享专家
    h = final_norm(h) → lm_head → logits

本文件提供**层前向函数工厂**（纯函数，输入输出约定见各函数 docstring）；
引擎（engine.py）负责把 WeightReader 的段供给接进这些工厂。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ccut.blocks.attn_gdn import GDNState, gdn_step, short_conv1d
from ccut.blocks.attn_gqa import build_causal_mask, gqa_attention_fast
from ccut.blocks.heads import LmHead, EmbedTable
from ccut.blocks.moe import expert_ffn, shared_expert_add, topk_softmax
from ccut.blocks.norm import rms_norm
from ccut.blocks.rope import apply_rope, build_rope
from ccut.models.spec import ModelSpec
from ccut.quant import kernels as qk
from ccut.quant.method import make_method_for_spec
from ccut.quant.registry import resolve_checkpoint_quant
from ccut.quant.spec import LayerQuantSpec

__all__ = [
    "WeightReader",
    "LinearProj",
    "AttentionContext",
    "build_linear_projs",
    "linear_attn_forward",
    "full_attn_forward",
    "moe_forward",
    "model_forward_layer",
]


class WeightReader:
    """权重段供给接口（R10 WeightRing 的抽象；测试用 :class:`FakeWeightReader`）。

    - ``segment(layer, tensor_key) -> bytes``：返回该张量的**原始字节**（quant dtype）；
    - ``shape(layer, tensor_key) -> (rows, cols, dtype)``；
    - ``scales(layer, tensor_key) -> np.ndarray``：scale 张量（若该层量化）。

    引擎实现（weights/stream.py）：ring buffer 槽位命中 → 直接返回槽内字节；
    未命中 → mmap 读（阻塞，触发 prefetch）。
    """

    def segment(self, layer: int, tensor_key: str) -> bytes:
        raise NotImplementedError

    def shape(self, layer: int, tensor_key: str) -> tuple[int, int, str]:
        raise NotImplementedError

    def scales(self, layer: int, tensor_key: str) -> np.ndarray | None:
        return None


class FakeWeightReader(WeightReader):
    """测试/小模型用：从常驻 dict 取段（验证数值路径用，非性能路径）。"""

    def __init__(self, table: dict[tuple[int, str], bytes], shapes: dict[tuple[int, str], tuple[int, int, str]]):
        self._table = table
        self._shapes = shapes

    def segment(self, layer: int, tensor_key: str) -> bytes:
        return self._table[(layer, tensor_key)]

    def shape(self, layer: int, tensor_key: str) -> tuple[int, int, str]:
        return self._shapes[(layer, tensor_key)]

    def scales(self, layer: int, tensor_key: str) -> np.ndarray | None:
        key = (layer, tensor_key + ".weight_scale")
        if key in self._table:
            from ccut.io_.safetensors_io import _resolve_dtype

            raw = np.frombuffer(self._table[key], dtype=np.uint8)
            return raw.view(_resolve_dtype("F32"))
        return None


@dataclass
class LinearProj:
    """一个线性投影（已 dequant 语义的 apply 闭包）。

    ``apply(x [seq, in]) -> [seq, out]``；权重字节来自 reader，dequant 结果
    按 (layer, proj) 缓存（ring 槽位有效期内不重读——R10 协同）。
    """

    layer: int
    name: str  # 张量名（如 self_attn.q_proj）
    out_dim: int
    in_dim: int
    reader: WeightReader
    quant_cfg: object  # QuantizationConfig（或 None=无量化）

    def __init__(self, layer: int, name: str, out_dim: int, in_dim: int, reader: WeightReader, quant_cfg):
        self.layer = layer
        self.name = name
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.reader = reader
        self.quant_cfg = quant_cfg
        self._method = None
        self._w_cache: np.ndarray | None = None
        self._spec = LayerQuantSpec(
            layer_name=f"layer{layer}.{name}",
            weight_key=_default_key(quant_cfg),
            quant_method=str(getattr(quant_cfg, "name", "none")),
        )

    def _build_method(self):
        if self._method is None:
            spec = self._spec
            if self.quant_cfg is not None and not self.quant_cfg.is_layer_skipped(spec.layer_name) if hasattr(self.quant_cfg, "is_layer_skipped") else False:
                pass
            self._method = make_method_for_spec(spec)
            self._method.create_weights(spec)
        return self._method

    def dequant_weight(self) -> np.ndarray:
        """读段 → dequant → [out, in] float32（带缓存）。"""
        if self._w_cache is not None:
            return self._w_cache
        raw = self.reader.segment(self.layer, self.name)
        if self.quant_cfg is not None:
            spec = self.quant_cfg.get_layer_spec(self.name)
            method = make_method_for_spec(spec)
            method.create_weights(spec)
            if not spec.skipped:
                scales = {}
                for sd in spec.scales:
                    s = self.reader.scales(self.layer, self.name)
                    if s is not None:
                        scales[sd.name.split(".")[-1]] = s
                # dequant 到 float32（W8A16 语义）
                w = _dequant_to_f32(method, raw, scales, self.in_dim)
                self._w_cache = w
                return w
        self._w_cache = _unpack_f32(raw, self.out_dim, self.in_dim)
        return self._w_cache

    def apply(self, x: np.ndarray) -> np.ndarray:
        w = self.dequant_weight()
        return x @ w.T


def _default_key(quant_cfg) -> LayerQuantSpec:
    from ccut.quant.spec import QuantKey

    return LayerQuantSpec("tmp", QuantKey(name="tmp"))


def _unpack_f32(raw: bytes, out_dim: int, in_dim: int) -> np.ndarray:
    """BF16 字节 → [out, in] float32。"""
    from ccut.io_.safetensors_io import _bf16_bytes_to_float32

    u16 = np.frombuffer(raw, dtype=np.uint16)
    return _bf16_bytes_to_float32(u16).reshape(out_dim, in_dim)


def _dequant_to_f32(method, raw: bytes, scales: dict, in_dim: int) -> np.ndarray:
    """量化段 → [out, in] float32（按 method 的权重 dtype 分派）。"""
    from ccut.quant.spec import QuantDType

    out_dim = len(raw) // _DTYPE_BYTES.get(method.spec.effective_key().weight_dtype, 1) // in_dim
    wd = method.spec.effective_key().weight_dtype
    if wd in (QuantDType.FP8_E4M3, QuantDType.FP8_E5M2):
        u8 = np.frombuffer(raw, dtype=np.uint8).reshape(out_dim, in_dim)
        scale = _pick_scale(scales, out_dim, in_dim)
        out = np.empty((out_dim, in_dim), dtype=np.float32)
        qk.fp8_dequant_mat(u8, scale, out)
        return out
    if wd == QuantDType.INT8:
        i8 = np.frombuffer(raw, dtype=np.int8).reshape(out_dim, in_dim)
        scale = _pick_scale(scales, out_dim, in_dim)
        out = np.empty((out_dim, in_dim), dtype=np.float32)
        qk.int8_dequant_row(i8, scale, out)
        return out
    if wd in (QuantDType.BF16, QuantDType.FP16):
        return _unpack_f32(raw, out_dim, in_dim)
    raise NotImplementedError(f"generic 组装暂不支持权重 dtype {wd}（走 weight_only/mx 专用路径）")


_DTYPE_BYTES = {
    "bf16": 2, "fp16": 2, "fp32": 4, "int8": 1,
    "fp8_e4m3": 1, "fp8_e5m2": 1, "int4": 0.5, "nf4": 0.5,
}


def _pick_scale(scales: dict, out_dim: int, in_dim: int) -> np.ndarray:
    """从 scales dict 挑 per-channel scale（长度=out_dim 优先）。"""
    for v in scales.values():
        v = np.asarray(v, dtype=np.float32).reshape(-1)
        if v.shape[0] == out_dim:
            return v
        if v.shape[0] == in_dim:
            return v
    if scales:
        return next(iter(scales.values())).astype(np.float32).reshape(-1)
    return np.ones(out_dim, dtype=np.float32)


def build_linear_projs(
    layer: int,
    reader: WeightReader,
    quant_cfg,
    prefix: str,
    names: dict[str, tuple[int, int]],
) -> dict[str, LinearProj]:
    """批量构建某层投影：``names = {"q_proj": (out, in), ...}`` → 张量名 ``{prefix}{n}``。"""
    out: dict[str, LinearProj] = {}
    for n, (o, i) in names.items():
        out[n] = LinearProj(layer, f"{prefix}{n}", o, i, reader, quant_cfg)
    return out


@dataclass
class AttentionContext:
    """跨步持久状态（引擎持有，按请求管理）。"""

    inv_freq: np.ndarray
    attn_factor: float
    gdn_states: dict[int, GDNState]  # 请求 id → GDN 状态（linear_attn 层）
    start_pos: int = 0

    def next(self) -> int:
        self.start_pos += 1
        return self.start_pos - 1


def linear_attn_forward(
    ctx: AttentionContext,
    h: np.ndarray,
    projs: dict[str, LinearProj],
    gdn: GDNState,
    a_log: np.ndarray,
    dt_bias: np.ndarray,
    conv_weight: np.ndarray,
    norm_w: np.ndarray,
    kernel_size: int,
    out_dim: int,
) -> np.ndarray:
    """Ornith linear_attn 层前向（GDN 递归，decode 单 token）。

    投影布局（checkpoint 实测）：
    - ``in_proj_qkv``: [hidden, (q+k+v) 头×维]（q16×128 + k16×128 + v32×128 = 16384 列）
    - ``in_proj_z``: [hidden, v_heads×v_dim]（门控）
    - ``in_proj_b``: [hidden, v_heads]（beta）
    - ``in_proj_a``: [hidden, v_heads]（a → softplus 衰减，+ dt_bias）
    - ``conv1d``: 对 qkv 融合流 causal conv（kernel=4）
    - ``norm``: GDN 状态归一化（RMSNorm over k_dim）
    - ``out_proj``: [v_heads×v_dim, hidden]
    """
    hidden = h.shape[1]
    qkv = projs["in_proj_qkv"].apply(h)
    z = projs["in_proj_z"].apply(h)
    b = projs["in_proj_b"].apply(h)
    a = projs["in_proj_a"].apply(h) + dt_bias
    # conv1d（decode 单 token：仅本步 + 历史 3 步——历史由 ctx 缓存，简化版仅本步）
    qkv_c = short_conv1d(qkv, conv_weight, kernel_size) if qkv.ndim == 2 else qkv
    # 切头
    qk_dim = 16 * 128
    v_dim = 32 * 128
    q = qkv_c[:, :qk_dim].reshape(-1, 16, 128)
    k = qkv_c[:, qk_dim : 2 * qk_dim].reshape(-1, 16, 128)
    v = qkv_c[:, 2 * qk_dim : 2 * qk_dim + v_dim].reshape(-1, 32, 128)
    # GDN 递推（每 v-head；k 头按 g=2 映射）
    seq = q.shape[0]
    out = np.empty((seq, 32, 128), dtype=np.float32)
    a_seq = a  # [seq, 32]
    b_seq = b  # [seq, 32]
    for t in range(seq):
        for hv in range(32):
            hk = hv // 2
            gdn_step(gdn.states[hv], q[t, hk], k[t, hk], v[t, hv], float(a_seq[t, hv]), float(b_seq[t, hv]), float(a_log[hv]), out[t, hv])
    # norm（GDN 输出归一化）+ 门控 silu(z)
    o = out.reshape(seq, -1)
    o = rms_norm(o, norm_w)
    o = o * _silu(z)
    return projs["out_proj"].apply(o)


def _silu(x: np.ndarray) -> np.ndarray:
    return (x * np.sigmoid(x)).astype(np.float32) if hasattr(np, "sigmoid") else x / (1.0 + np.exp(-x))


def full_attn_forward(
    ctx: AttentionContext,
    h: np.ndarray,
    projs: dict[str, LinearProj],
    kv_blocks,  # KVBlockPool 视图（R1）：.write(k, v) / .gather() -> [kv_heads, kv_len, d]
    q_norm_w: np.ndarray | None,
    k_norm_w: np.ndarray | None,
    softcap: float | None,
    heads: int,
    kv_heads: int,
    head_dim: int,
    seq_len: int,
) -> np.ndarray:
    """Ornith full_attn 层前向（GQA + qk_norm + KV 块池，decode 单 token / prefill 批量）。

    投影：q/k/v/o_proj（qk_norm 对 q/k 做 RMSNorm，Ornith attn 层实测）；
    RoPE（half-rotate，ctx.inv_freq）；KV 写块池 + gather 历史。
    """
    q = projs["q_proj"].apply(h)  # [seq, heads*d]
    k = projs["k_proj"].apply(h)
    v = projs["v_proj"].apply(h)
    seq = q.shape[0]
    q = q.reshape(seq, heads, head_dim)
    k = k.reshape(seq, kv_heads, head_dim)
    v = v.reshape(seq, kv_heads, head_dim)
    if q_norm_w is not None:
        q = rms_norm(q.reshape(seq, -1), q_norm_w).reshape(seq, heads, head_dim)
    if k_norm_w is not None:
        k = rms_norm(k.reshape(seq, -1), k_norm_w).reshape(seq, kv_heads, head_dim)
    # RoPE
    pos = np.arange(seq, dtype=np.int64) + ctx.start_pos - (seq - 1)  # 本批起始绝对位置
    pos = pos[None, :]
    q4 = q[None].astype(np.float32)
    k4 = k[None].astype(np.float32)
    qr, kr = apply_rope(q4, k4, ctx.inv_freq, pos, ctx.attn_factor)
    # KV 块池：写 + gather 历史
    kv_blocks.write(kr[0], vr := v.astype(np.float32)[None])
    k_all, v_all = kv_blocks.gather()  # [kv_heads, kv_len, d]
    mask = build_causal_mask(seq, ctx.start_pos - (seq - 1))
    o = gqa_attention_fast(qr, k_all[None], v_all[None], mask[:, None] if mask.shape[0] == seq else mask, scale=None, )
    # mask 形状对齐 [seq, kv_len]
    return projs["o_proj"].apply(o[0].reshape(seq, -1))


def moe_forward(
    h: np.ndarray,
    gate_w: np.ndarray,
    reader: WeightReader,
    layer: int,
    moe_spec,
    shared_projs: dict[str, LinearProj] | None,
) -> np.ndarray:
    """MoE 层前向（路由 top-k → 专家流式 → 加权融合 → 共享专家残差）。

    专家权重经 ``reader.segment(layer, "experts.{eid}.{proj}")`` 现取（R2）；
    本函数是数据流入口——实际段供给由 engine 的 prefetch 流水线保证就绪。
    """
    logits = h @ gate_w.T
    weights, ids = topk_softmax(logits, moe_spec.top_k, moe_spec.norm_topk_prob)
    seq, hidden = h.shape
    out = np.zeros((seq, hidden), dtype=np.float32)
    inter = moe_spec.intermediate_size
    # 按专家聚合（同专家 token 一批）
    by_eid: dict[int, list[int]] = {}
    for t in range(seq):
        for e in range(moe_spec.top_k):
            by_eid.setdefault(int(ids[t, e]), []).append(t)
    for eid, toks in by_eid.items():
        wg = _expert_seg_f32(reader, layer, eid, "gate_proj", hidden, inter)
        wu = _expert_seg_f32(reader, layer, eid, "up_proj", hidden, inter)
        wd = _expert_seg_f32(reader, layer, eid, "down_proj", inter, hidden)
        batch = h[np.array(toks, dtype=np.int64)]
        y = expert_ffn(batch, wg, wu, wd)
        for i, t in enumerate(toks):
            out[t] += weights[t, list(ids[t]).index(eid)] * y[i]
    # 共享专家
    if shared_projs is not None and moe_spec.has_shared_expert:
        s = shared_projs["gate_proj"].apply(h)
        su = shared_projs["up_proj"].apply(h)
        inter_out = np.empty(s.shape, dtype=np.float32)
        qk.silu_mul_fused(s, su, inter_out)
        out += shared_projs["down_proj"].apply(inter_out)
    return out


def _expert_seg_f32(reader: WeightReader, layer: int, eid: int, proj: str, out_dim: int, in_dim: int) -> np.ndarray:
    """专家投影段 → [out, in] float32（FP8 per-channel 或 BF16 直通）。"""
    key = f"experts.{eid}.{proj}_proj"
    raw = reader.segment(layer, key)
    scale = reader.scales(layer, key)
    if scale is not None:
        scale = np.asarray(scale, dtype=np.float32).reshape(-1)
        u8 = np.frombuffer(raw, dtype=np.uint8).reshape(out_dim, in_dim)
        out = np.empty((out_dim, in_dim), dtype=np.float32)
        qk.fp8_dequant_mat(u8, scale if scale.shape[0] == out_dim else _pick_scale({0: scale}, out_dim, in_dim), out)
        return out
    return _unpack_f32(raw, out_dim, in_dim)


def model_forward_layer(
    layer_idx: int,
    h: np.ndarray,
    reader: WeightReader,
    spec: ModelSpec,
    quant_cfg,
    ctx: AttentionContext,
    kv_pool_by_layer: dict[int, object],
    layer_state: dict,
) -> np.ndarray:
    """单层完整前向（input_layernorm → attn → 残差 → post_norm → MoE → 残差）。

    工厂入口：引擎逐层调用；``layer_state`` 由 :func:`init_layer_state` 预建
    （投影 + GDN 状态 + KV 池引用），避免每步重建。
    """
    template = spec.layer_templates[layer_idx]
    pre = layer_state[f"{layer_idx}.input_layernorm"]
    h_norm = rms_norm(h, pre)
    if template.type == "linear_attn":
        attn = linear_attn_forward(
            ctx, h_norm, layer_state[f"{layer_idx}.projs"],
            layer_state[f"{layer_idx}.gdn"], layer_state[f"{layer_idx}.a_log"],
            layer_state[f"{layer_idx}.dt_bias"], layer_state[f"{layer_idx}.conv"],
            layer_state[f"{layer_idx}.gdn_norm"], spec.gdn_conv_kernel_dim or 4,
            spec.hidden_size,
        )
    else:
        attn = full_attn_forward(
            ctx, h_norm, layer_state[f"{layer_idx}.projs"],
            kv_pool_by_layer[layer_idx],
            layer_state.get(f"{layer_idx}.q_norm"), layer_state.get(f"{layer_idx}.k_norm"),
            spec.attn_logit_softcapping, spec.num_attention_heads,
            spec.num_key_value_heads, spec.head_dim, h.shape[0],
        )
    h = h + attn
    post = layer_state[f"{layer_idx}.post_attention_layernorm"]
    h2 = rms_norm(h, post)
    if spec.moe is not None:
        h = h + moe_forward(
            h2, layer_state[f"{layer_idx}.gate_w"], reader, layer_idx, spec.moe,
            layer_state.get(f"{layer_idx}.shared"),
        )
    else:
        h = h + layer_state[f"{layer_idx}.ffn"](h2)
    return h
