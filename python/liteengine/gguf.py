"""GGUF 模型读取与反量化（llama.cpp GGUF 格式，纯标准库 + numpy）。

- 解析：头（magic/version/tensor_count/metadata_kv_count）+ 元数据 KV（含值）+ 张量索引，
  张量数据惰性 seek 读取。
- 反量化支持：F32 / F16 / Q4_0 / Q4_1 / Q5_0 / Q5_1 / Q8_0（公式按 ggml 约定）。
- K 系列（Q2_K/Q3_K/Q4_K/Q5_K/Q6_K/Q8_K）布局复杂且本机无参考源码，
  从记忆重构风险高（避免输出乱码），加载时明确报错（记为后续工作）。
- 名称映射：ggml 命名（token_embd / blk.N.attn_q / ffn_gate 等）→ 引擎 HF 风格命名。
"""
from __future__ import annotations

from math import prod

from numpy import (asarray, float16, float32, frombuffer, int8, uint16, uint8, zeros)

# GGML 张量类型枚举（常用子集）
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q4_0 = 2
GGML_TYPE_Q4_1 = 3
GGML_TYPE_Q5_0 = 6
GGML_TYPE_Q5_1 = 7
GGML_TYPE_Q8_0 = 8

_TYPE_NAMES = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0"}

# 元数据值类型（GGUF v3）
_META_EMPTY = {0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12}   # 直接值（按表长读取）
_META_STR = 8
_META_ARRAY = 9


def _f16_to_f32(h: int) -> float:
    s = (h >> 15) & 1
    e = (h >> 10) & 0x1F
    m = h & 0x3FF
    if e == 0:
        v = m * 2.0 ** -24
    elif e == 31:
        return float("nan") if m else (0.0 if s else -0.0)
    else:
        v = (1.0 + m / 1024.0) * 2.0 ** (e - 15)
    return -v if s else v


class GGUFTensor:
    def __init__(self, name: str, shape: list[int], ggml_type: int, offset: int):
        self.name = name
        self.shape = shape          # GGUF 存储序（dims[0] = 最内维）
        self.ggml_type = ggml_type
        self.offset = offset

    @property
    def n_elements(self) -> int:
        return prod(self.shape)


class GGUFReader:
    """GGUF 文件解析：头 + 元数据（含值）+ 张量索引；张量数据惰性读取。"""

    def __init__(self, path: str):
        self._f = open(path, "rb")
        magic = self._f.read(4)
        if magic != b"GGUF":
            raise ValueError("不是 GGUF 文件（magic 非 GGUF）")
        self.version = int.from_bytes(self._f.read(4), "little")
        self.tensor_count = int.from_bytes(self._f.read(8), "little")
        self.metadata_kv_count = int.from_bytes(self._f.read(8), "little")
        self.metadata: dict[str, object] = {}
        self._read_metadata()
        self.tensors: dict[str, GGUFTensor] = {}
        for _ in range(self.tensor_count):
            name = self._read_str()
            n_dims = int.from_bytes(self._f.read(4), "little")
            dims = [int.from_bytes(self._f.read(8), "little") for _ in range(n_dims)]
            gtype = int.from_bytes(self._f.read(4), "little")
            offset = int.from_bytes(self._f.read(8), "little")
            self.tensors[name] = GGUFTensor(name, dims, gtype, offset)
        self.data_start = self._f.tell()          # 数据区起点 = 张量索引之后

    def close(self) -> None:
        self._f.close()

    def _read_str(self) -> str:
        ln = int.from_bytes(self._f.read(8), "little")
        return self._f.read(ln).decode("utf-8")

    def _read_value(self, t: int):
        if t == _META_STR:
            return self._read_str()
        if t == _META_ARRAY:
            etype = int.from_bytes(self._f.read(4), "little")
            n = int.from_bytes(self._f.read(8), "little")
            return [self._read_value(etype) for _ in range(n)]
        sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
        raw = self._f.read(sizes[t])
        return int.from_bytes(raw, "little", signed=(t in (1, 3, 5, 11)))

    def _read_metadata(self) -> None:
        for _ in range(self.metadata_kv_count):
            key = self._read_str()
            t = int.from_bytes(self._f.read(4), "little")
            self.metadata[key] = self._read_value(t)

    # ---- 张量读取与反量化 ----

    def get_f32(self, name: str) -> object:
        """读取张量为 numpy f32（形状按 GGUF 存储序返回，行主序）。"""
        t = self.tensors[name]
        n = t.n_elements
        self._f.seek(self.data_start + t.offset)
        g = t.ggml_type
        if g == GGML_TYPE_F32:
            return frombuffer(self._f.read(n * 4), dtype=float32).reshape(t.shape)
        if g == GGML_TYPE_F16:
            u = frombuffer(self._f.read(n * 2), dtype=uint16)
            return asarray([_f16_to_f32(int(x)) for x in u], dtype=float32).reshape(t.shape)
        if g == GGML_TYPE_Q4_0:
            return self._dequant_q4_0(t)
        if g == GGML_TYPE_Q4_1:
            return self._dequant_q4_1(t)
        if g == GGML_TYPE_Q5_0:
            return self._dequant_q5_0(t)
        if g == GGML_TYPE_Q5_1:
            return self._dequant_q5_1(t)
        if g == GGML_TYPE_Q8_0:
            return self._dequant_q8_0(t)
        raise ValueError(f"不支持的 GGML 张量类型 {g}（{t.shape} 名称 {t.name}）；"
                         f"当前支持 {sorted(_TYPE_NAMES.values())}；K 系列（Q4_K/Q6_K 等）"
                         f"因本机无参考源码暂未实现")

    def _dequant_q4_0(self, t: GGUFTensor) -> object:
        # 块 32：2 字节 fp16 缩放 d + 16 字节 int4（低 4 位在前）。y = (q - 8) * d
        n = t.n_elements
        out = zeros(n, dtype=float32)
        raw = self._f.read(n // 32 * 18)
        for b in range(n // 32):
            d = _f16_to_f32(int.from_bytes(raw[b * 18:b * 18 + 2], "little"))
            qs = raw[b * 18 + 2:(b + 1) * 18]
            for j in range(32):
                q = (qs[j // 2] >> (4 * (j & 1))) & 0xF
                out[b * 32 + j] = (q - 8) * d
        return out.reshape(t.shape)

    def _dequant_q4_1(self, t: GGUFTensor) -> object:
        # 块 32：2 字节 d + 2 字节 m + 16 字节 int4。y = q * d + m
        n = t.n_elements
        out = zeros(n, dtype=float32)
        raw = self._f.read(n // 32 * 20)
        for b in range(n // 32):
            d = _f16_to_f32(int.from_bytes(raw[b * 20:b * 20 + 2], "little"))
            m = _f16_to_f32(int.from_bytes(raw[b * 20 + 2:b * 20 + 4], "little"))
            qs = raw[b * 20 + 4:(b + 1) * 20]
            for j in range(32):
                q = (qs[j // 2] >> (4 * (j & 1))) & 0xF
                out[b * 32 + j] = q * d + m
        return out.reshape(t.shape)

    def _dequant_q5_0(self, t: GGUFTensor) -> object:
        # 块 32：2 字节 d + 16 字节低 4 位 + 4 字节高 1 位。y = ((q | (qh<<4)) - 16) * d
        n = t.n_elements
        out = zeros(n, dtype=float32)
        raw = self._f.read(n // 32 * 22)
        for b in range(n // 32):
            d = _f16_to_f32(int.from_bytes(raw[b * 22:b * 22 + 2], "little"))
            ql = raw[b * 22 + 2:b * 22 + 18]
            qh = raw[b * 22 + 18:(b + 1) * 22]
            for j in range(32):
                q = (ql[j // 2] >> (4 * (j & 1))) & 0xF
                hi = (qh[j // 8] >> (j & 7)) & 1
                out[b * 32 + j] = ((q | (hi << 4)) - 16) * d
        return out.reshape(t.shape)

    def _dequant_q5_1(self, t: GGUFTensor) -> object:
        # 块 32：2 字节 d + 2 字节 m + 16 字节低 4 位 + 4 字节高 1 位。y = (q | (qh<<4)) * d + m
        n = t.n_elements
        out = zeros(n, dtype=float32)
        raw = self._f.read(n // 32 * 24)
        for b in range(n // 32):
            d = _f16_to_f32(int.from_bytes(raw[b * 24:b * 24 + 2], "little"))
            m = _f16_to_f32(int.from_bytes(raw[b * 24 + 2:b * 24 + 4], "little"))
            ql = raw[b * 24 + 4:b * 24 + 20]
            qh = raw[b * 24 + 20:(b + 1) * 24]
            for j in range(32):
                q = (ql[j // 2] >> (4 * (j & 1))) & 0xF
                hi = (qh[j // 8] >> (j & 7)) & 1
                out[b * 32 + j] = (q | (hi << 4)) * d + m
        return out.reshape(t.shape)

    def _dequant_q8_0(self, t: GGUFTensor) -> object:
        # 块 32：2 字节 d + 32 个 int8。y = q * d
        n = t.n_elements
        out = zeros(n, dtype=float32)
        raw = self._f.read(n // 32 * 34)
        for b in range(n // 32):
            d = _f16_to_f32(int.from_bytes(raw[b * 34:b * 34 + 2], "little"))
            for j in range(32):
                out[b * 32 + j] = int.from_bytes(
                    raw[b * 34 + 2 + j:b * 34 + 3 + j], "little", signed=True) * d
        return out.reshape(t.shape)


# ---- GGUF 元数据 → 引擎配置（Llama 家族 / Mixtral）----

def gguf_metadata_to_config(reader: GGUFReader) -> dict:
    """从 GGUF 元数据构造引擎配置（架构探测：llama/mistral → dense；mixtral → MoE）。"""
    md = reader.metadata
    arch = str(md.get("general.architecture", "llama")).lower()
    n_layer = int(md.get(f"{arch}.block_count", md.get("llama.block_count", 0)))
    head_count = int(md.get(f"{arch}.attention.head_count", 0))
    hidden = int(md.get(f"{arch}.embedding_length", 0))
    if not (n_layer and head_count and hidden):
        raise ValueError(f"GGUF 元数据缺少模型参数（architecture={arch}）")
    cfg = {
        "architectures": [f"{arch.title()}ForCausalLM"],
        "hidden_size": hidden,
        "num_hidden_layers": n_layer,
        "num_attention_heads": head_count,
        "num_key_value_heads": int(md.get(f"{arch}.attention.head_count_kv", head_count)),
        "vocab_size": int(md.get(f"{arch}.vocab_size", md.get("tokenizer.ggml.model", 32000) or 32000)),
        "rms_norm_eps": float(md.get(f"{arch}.attention.layer_norm_rms_epsilon", 1e-5)),
        "rope_theta": float(md.get(f"{arch}.rope.freq_base", 10000.0)),
        "intermediate_size": int(md.get(f"{arch}.feed_forward_length", 4 * hidden)),
    }
    expert_count = int(md.get(f"{arch}.expert_count", 0) or 0)
    if expert_count:
        cfg["num_local_experts"] = expert_count
        cfg["num_experts_per_tok"] = int(md.get(f"{arch}.expert_used_count", 2))
    return cfg


# ---- ggml 命名 → 引擎 HF 风格命名 ----

def gguf_name_to_hf(name: str) -> str | None:
    """映射 GGUF 张量名到引擎 HF 风格名（None = 无需映射的辅助张量）。"""
    if name == "token_embd.weight":
        return "model.embed_tokens.weight"
    if name == "output.weight":
        return "lm_head.weight"
    if name == "token_embd_norm.weight":
        return "model.norm.weight"
    if name == "output_norm.weight":
        return "model.norm.weight"
    if name.startswith("blk."):
        parts = name.split(".")
        layer, rest = parts[1], ".".join(parts[2:])
        base = f"model.layers.{layer}"
        table = {
            "attn_norm.weight": "input_layernorm.weight",
            "attn_q.weight": "self_attn.q_proj.weight",
            "attn_k.weight": "self_attn.k_proj.weight",
            "attn_v.weight": "self_attn.v_proj.weight",
            "attn_output.weight": "self_attn.o_proj.weight",
            "ffn_norm.weight": "post_attention_layernorm.weight",
            "ffn_gate.weight": "mlp.gate_proj.weight",
            "ffn_up.weight": "mlp.up_proj.weight",
            "ffn_down.weight": "mlp.down_proj.weight",
            "ffn_gate_inp.weight": "mlp.gate.weight",
        }
        if rest in table:
            return f"{base}.{table[rest]}"
        # MoE 专家：ffn_exps.N.w1/w3 → 合并 gate_up；w2 → down
        if rest.startswith("ffn_exps.") and rest.endswith(".weight"):
            exp_parts = rest.split(".")
            e, w = exp_parts[1], exp_parts[2]
            if w in ("w1", "w3"):
                return f"{base}.mlp.experts.gate_up_proj.{e}.weight"
            if w == "w2":
                return f"{base}.mlp.experts.down_proj.{e}.weight"
    return None


# ---- GGUFWeightStore：引擎 HF 风格命名 → ggml 命名适配 ----

def hf_to_gguf_name(name: str) -> str:
    """引擎 HF 风格命名 → ggml 命名（与 gguf_name_to_hf 互逆）。"""
    if name == "model.embed_tokens.weight":
        return "token_embd.weight"
    if name == "lm_head.weight":
        return "output.weight"
    if name == "model.norm.weight":
        return "output_norm.weight"
    if name.startswith("model.layers."):
        parts = name.split(".")
        layer, rest = parts[2], ".".join(parts[3:])
        base = f"blk.{layer}"
        table = {
            "input_layernorm.weight": "attn_norm.weight",
            "self_attn.q_proj.weight": "attn_q.weight",
            "self_attn.k_proj.weight": "attn_k.weight",
            "self_attn.v_proj.weight": "attn_v.weight",
            "self_attn.o_proj.weight": "attn_output.weight",
            "post_attention_layernorm.weight": "ffn_norm.weight",
            "mlp.gate_proj.weight": "ffn_gate.weight",
            "mlp.up_proj.weight": "ffn_up.weight",
            "mlp.down_proj.weight": "ffn_down.weight",
            "mlp.gate.weight": "ffn_gate_inp.weight",
        }
        if rest in table:
            return f"{base}.{table[rest]}"
        if rest.startswith("mlp.experts.down_proj."):
            return f"{base}.ffn_exps.{rest.rsplit('.', 2)[1]}.w2.weight"
        if rest.startswith("mlp.experts.gate_up_proj."):
            return f"{base}.ffn_exps.{rest.rsplit('.', 2)[1]}.w1.weight"
    return name


class GGUFWeightStore:
    """GGUF 权重适配：引擎 HF 风格命名 → ggml 命名（惰性读取 + 缓存）。

    特殊处理：
    - 形状：GGUF 存储序（最内维在前）→ 引擎 [out, in]（内存布局相同，仅 reshape）。
    - MoE：``mlp.experts.gate_up_proj.weight`` 由各专家 w1/w3 合并为 [E, 2i, h]。
    """

    def __init__(self, path: str):
        self.reader = GGUFReader(path)
        self.model_dir = path          # load_quant_config 读不到 config.json → 默认 QuantConfig
        md = self.reader.metadata
        arch = str(md.get("general.architecture", "llama")).lower()
        self.num_experts = int(md.get(f"{arch}.expert_count", 0) or 0)
        self._cache: dict[str, object] = {}

    def get(self, name: str) -> object:
        if name not in self._cache:
            self._cache[name] = self._load(name)
        return self._cache[name]

    def _load(self, name: str):
        # MoE 合并专家：mlp.experts.gate_up_proj.weight ← 各专家 w1（gate）+ w3（up）
        if name.endswith("experts.gate_up_proj.weight") and self.num_experts:
            import numpy as np
            layer = name.split(".")[2]
            w1s, w3s = [], []
            for e in range(self.num_experts):
                w1 = self.reader.get_f32(f"blk.{layer}.ffn_exps.{e}.w1.weight")
                w3 = self.reader.get_f32(f"blk.{layer}.ffn_exps.{e}.w3.weight")
                w1s.append(w1.reshape(list(reversed(w1.shape))))
                w3s.append(w3.reshape(list(reversed(w3.shape))))
            g1, g3 = np.stack(w1s), np.stack(w3s)      # [E, i, h]
            return np.concatenate([g1, g3], axis=1)    # [E, 2i, h]（gate 前半，up 后半）
        arr = self.reader.get_f32(hf_to_gguf_name(name))
        return arr.reshape(list(reversed(arr.shape)))  # ggml 序 → 引擎 [out, in]
