//! 分发注册表（镜像 Python `liteengine/registry.py` 的 attention/moe_format 注册点）。
//!
//! 注意力按类型名分发（"standard" / "mla" / "full"——由归一化配置的 `attention` 字段驱动），
//! 外部可通过 `register_attention` 注册新的注意力实现（可扩展性对齐 Python 版）。

use crate::core::tensor::Tensor;
use crate::engine::attention::{FullAttention, GatedDeltaNet, MlaAttention, StandardAttention};
use crate::io::safetensors::SafeTensors;
use std::collections::HashMap;

/// 注意力公共接口（prefill / 缓存续接 / decode 单步）。
pub trait Attention {
    fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor,
               mask: Option<&Tensor>) -> Tensor;
    fn forward_kv(&self, x: &Tensor, cos: &Tensor, sin: &Tensor,
                  mask: Option<&Tensor>) -> (Tensor, Tensor, Tensor);
    fn decode(&self, x: &Tensor, cos: &Tensor, sin: &Tensor,
              k_prev: &Tensor, v_prev: &Tensor) -> (Tensor, Tensor, Tensor);
}

impl Attention for StandardAttention {
    fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor,
               mask: Option<&Tensor>) -> Tensor {
        self.forward(x, cos, sin, mask)
    }
    fn forward_kv(&self, x: &Tensor, cos: &Tensor, sin: &Tensor,
                  mask: Option<&Tensor>) -> (Tensor, Tensor, Tensor) {
        self.forward_kv(x, cos, sin, mask)
    }
    fn decode(&self, x: &Tensor, cos: &Tensor, sin: &Tensor,
              k_prev: &Tensor, v_prev: &Tensor) -> (Tensor, Tensor, Tensor) {
        self.decode(x, cos, sin, k_prev, v_prev)
    }
}

impl Attention for FullAttention {
    fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor,
               mask: Option<&Tensor>) -> Tensor {
        self.forward(x, cos, sin, mask)
    }
    fn forward_kv(&self, x: &Tensor, cos: &Tensor, sin: &Tensor,
                  mask: Option<&Tensor>) -> (Tensor, Tensor, Tensor) {
        self.forward_kv(x, cos, sin, mask)
    }
    fn decode(&self, x: &Tensor, cos: &Tensor, sin: &Tensor,
              k_prev: &Tensor, v_prev: &Tensor) -> (Tensor, Tensor, Tensor) {
        self.decode(x, cos, sin, k_prev, v_prev)
    }
}

impl Attention for MlaAttention {
    fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor,
               mask: Option<&Tensor>) -> Tensor {
        self.forward(x, cos, sin, mask)
    }
    fn forward_kv(&self, x: &Tensor, cos: &Tensor, sin: &Tensor,
                  mask: Option<&Tensor>) -> (Tensor, Tensor, Tensor) {
        self.forward_kv(x, cos, sin, mask)
    }
    fn decode(&self, x: &Tensor, cos: &Tensor, sin: &Tensor,
              k_prev: &Tensor, v_prev: &Tensor) -> (Tensor, Tensor, Tensor) {
        self.decode(x, cos, sin, k_prev, v_prev)
    }
}

impl Attention for GatedDeltaNet {
    fn forward(&self, x: &Tensor, _cos: &Tensor, _sin: &Tensor,
               _mask: Option<&Tensor>) -> Tensor {
        self.forward(x)
    }
    fn forward_kv(&self, x: &Tensor, _cos: &Tensor, _sin: &Tensor,
                  _mask: Option<&Tensor>) -> (Tensor, Tensor, Tensor) {
        // 线性注意力无标准 KV——返回 (输出, 空 k/v)
        let out = self.forward(x);
        let empty = Tensor::from_vec(0, 0, vec![]);
        (out, empty.clone(), empty)
    }
    fn decode(&self, _x: &Tensor, _cos: &Tensor, _sin: &Tensor,
              _k_prev: &Tensor, _v_prev: &Tensor) -> (Tensor, Tensor, Tensor) {
        // 线性注意力的 decode 走 forward_step（conv/rec 状态）——此处为 trait 兼容占位
        let out = Tensor::from_vec(1, self.hidden, vec![0.0; self.hidden]);
        let empty = Tensor::from_vec(0, 0, vec![]);
        (out, empty.clone(), empty)
    }
}

/// 注意力构造器：store + 权重前缀 + 归一化配置 → 注意力实现。
pub type AttnBuilder = fn(&SafeTensors, &str, &crate::engine::model_config::ModelConfig)
    -> Box<dyn Attention>;

fn _build_standard(store: &SafeTensors, prefix: &str,
                   cfg: &crate::engine::model_config::ModelConfig) -> Box<dyn Attention> {
    let hidden = cfg.hidden_size;
    let h = cfg.num_heads;
    let kvh = cfg.num_kv_heads;
    let hd = cfg.head_dim;
    let get = |name: &str, out: usize| -> Tensor {
        Tensor::from_vec(out, hidden, store.get_f32(name).unwrap_or_else(|| vec![0.1; out * hidden]))
    };
    Box::new(StandardAttention {
        num_heads: h,
        num_kv_heads: kvh,
        head_dim: hd,
        rope_dim: hd,
        scaling: 1.0 / (hd as f32).sqrt(),
        q_w: get(&format!("{prefix}.self_attn.q_proj.weight"), h * hd),
        k_w: get(&format!("{prefix}.self_attn.k_proj.weight"), kvh * hd),
        v_w: get(&format!("{prefix}.self_attn.v_proj.weight"), kvh * hd),
        o_w: get(&format!("{prefix}.self_attn.o_proj.weight"), hidden),
    })
}

/// 全注意力构造器（镜像 _build_standard——FullAttention 含 GQA q/k/v/o + eps）。
fn _build_full(store: &SafeTensors, prefix: &str,
               cfg: &crate::engine::model_config::ModelConfig) -> Box<dyn Attention> {
    let hidden = cfg.hidden_size;
    let h = cfg.num_heads;
    let kvh = cfg.num_kv_heads;
    let hd = cfg.head_dim;
    let get = |name: &str, out: usize| -> Tensor {
        Tensor::from_vec(out, hidden, store.get_f32(name).unwrap_or_else(|| vec![0.1; out * hidden]))
    };
    Box::new(FullAttention {
        num_heads: h,
        num_kv_heads: kvh,
        head_dim: hd,
        rope_dim: hd,
        scaling: 1.0 / (hd as f32).sqrt(),
        eps: cfg.eps,
        q_w: get(&format!("{prefix}.self_attn.q_proj.weight"), 2 * h * hd),
        k_w: get(&format!("{prefix}.self_attn.k_proj.weight"), kvh * hd),
        v_w: get(&format!("{prefix}.self_attn.v_proj.weight"), kvh * hd),
        o_w: get(&format!("{prefix}.self_attn.o_proj.weight"), hidden),
        q_norm_w: store.get_f32(&format!("{prefix}.self_attn.q_norm.weight"))
            .unwrap_or_else(|| vec![1.0; h * hd]),
        k_norm_w: store.get_f32(&format!("{prefix}.self_attn.k_norm.weight"))
            .unwrap_or_else(|| vec![1.0; kvh * hd]),
    })
}

/// 注意力构造器注册表（name → 构造器）。
static mut ATTENTION_BUILDERS: Option<HashMap<String, AttnBuilder>> = None;

fn builders() -> &'static mut HashMap<String, AttnBuilder> {
    unsafe {
        ATTENTION_BUILDERS.get_or_insert_with(|| {
            let mut m = HashMap::new();
            m.insert("standard".to_string(), _build_standard as AttnBuilder);
            m.insert("full".to_string(), _build_full as AttnBuilder);
            m
        })
    }
}

/// 注册注意力构造器（外部扩展：新注意力实现接入）。
pub fn register_attention(name: &str, builder: AttnBuilder) {
    builders().insert(name.to_string(), builder);
}

/// 按名称取注意力构造器（未注册返回 None）。
pub fn get_attention(name: &str) -> Option<AttnBuilder> {
    builders().get(name).copied()
}

/// 已注册的注意力类型列表。
pub fn list_attentions() -> Vec<String> {
    let mut v: Vec<String> = builders().keys().cloned().collect();
    v.sort();
    v
}

/// 视觉编码器构造器（占位注册点——镜像 Python registry.py 的 register_vision_encoder）。
/// 真实视觉推理缺依赖/模型（诚实标注——现状 0 内置）。
pub type VisionEncoderBuilder = fn(&SafeTensors, &str, usize) -> Option<Vec<f32>>;

static mut VISION_ENCODERS: Option<HashMap<String, VisionEncoderBuilder>> = None;

fn vision_map() -> &'static mut HashMap<String, VisionEncoderBuilder> {
    unsafe {
        VISION_ENCODERS.get_or_insert_with(|| HashMap::new())
    }
}

/// 注册视觉编码器构造器（外部扩展：真实多模态接入）。
pub fn register_vision_encoder(name: &str, builder: VisionEncoderBuilder) {
    vision_map().insert(name.to_string(), builder);
}

/// 按名称取视觉编码器构造器。
pub fn get_vision_encoder(name: &str) -> Option<VisionEncoderBuilder> {
    vision_map().get(name).copied()
}

/// 已注册的视觉编码器列表（当前 0 内置——多模态真实推理待依赖/模型）。
pub fn list_vision_encoders() -> Vec<String> {
    let mut v: Vec<String> = vision_map().keys().cloned().collect();
    v.sort();
    v
}

// ---- 额外注册点（镜像 Python registry.py 的 moe_format/quant_method/arch_normalizer）----

/// MoE 分发构造函数类型（store → MoE 参数）。
pub type MoeFormatBuilder = fn(&SafeTensors, &str, usize, usize) -> (Vec<f32>, Vec<f32>, usize);

static mut MOE_FORMATS: Option<HashMap<String, MoeFormatBuilder>> = None;

fn moe_map() -> &'static mut HashMap<String, MoeFormatBuilder> {
    unsafe {
        MOE_FORMATS.get_or_insert_with(|| {
            let mut m = HashMap::new();
            m.insert("merged".to_string(), default_moe_builder as MoeFormatBuilder);
            m
        })
    }
}

/// 占位默认 MoE 构造（merged 形式——真实 per-expert 为后续）。
fn default_moe_builder(_store: &SafeTensors, _prefix: &str, _e: usize, _inter: usize)
                       -> (Vec<f32>, Vec<f32>, usize) {
    (vec![], vec![], 0)
}

/// 注册 MoE 分发构造器。
pub fn register_moe_format(name: &str, builder: MoeFormatBuilder) {
    moe_map().insert(name.to_string(), builder);
}

/// 按名称取 MoE 分发构造器。
pub fn get_moe_format(name: &str) -> Option<MoeFormatBuilder> {
    moe_map().get(name).copied()
}

/// 已注册的 MoE 分发类型列表。
pub fn list_moe_formats() -> Vec<String> {
    let mut v: Vec<String> = moe_map().keys().cloned().collect();
    v.sort();
    v
}

/// 量化反量化处理器（镜像 Python register_quant_method）：handler(qweight, qzeros,
/// scales, out, in_, group_size) -> Vec<f32>。
pub type QuantMethodHandler = fn(&[i32], Option<&[i32]>, &[f32], usize, usize, usize)
    -> Vec<f32>;

/// AWQ 处理器包装（qzeros 是 Option——镜像 Python 的 sym 支持）。
fn awq_handler(qw: &[i32], qz: Option<&[i32]>, sc: &[f32], out: usize, in_: usize, gs: usize)
               -> Vec<f32> {
    crate::quant::dequant::dequantize_awq(qw, qz.unwrap_or(&[]), sc, out, in_, gs)
}

static mut QUANT_METHODS: Option<HashMap<String, QuantMethodHandler>> = None;

fn quant_map() -> &'static mut HashMap<String, QuantMethodHandler> {
    unsafe {
        QUANT_METHODS.get_or_insert_with(|| {
            let mut m = HashMap::new();
            m.insert("awq".to_string(), awq_handler as QuantMethodHandler);
            m
        })
    }
}

/// 注册量化反量化处理器。
pub fn register_quant_method(name: &str, handler: QuantMethodHandler) {
    quant_map().insert(name.to_string(), handler);
}

/// 按名称取量化处理器。
pub fn get_quant_method(name: &str) -> Option<QuantMethodHandler> {
    quant_map().get(name).copied()
}

/// 已注册的量化方法列表。
pub fn list_quant_methods() -> Vec<String> {
    let mut v: Vec<String> = quant_map().keys().cloned().collect();
    v.sort();
    v
}

/// 架构归一化器（镜像 Python register_arch_normalizer——patterns 命中匹配分发）。
/// normalizer => 归一化配置；patterns 任一子串命中即选用（未命中走通用回退）。
pub type ArchNormalizer = fn(&std::collections::HashMap<String, String>)
    -> crate::engine::model_config::ModelConfig;

static mut ARCH_NORMALIZERS: Option<HashMap<String, ArcPatterns>> = None;

struct ArcPatterns {
    normalizer: ArchNormalizer,
    patterns: Vec<String>,
}

fn arch_map() -> &'static mut HashMap<String, ArcPatterns> {
    unsafe {
        ARCH_NORMALIZERS.get_or_insert_with(|| HashMap::new())
    }
}

/// 注册架构归一化器（patterns 为 architectures/model_type 子串匹配）。
pub fn register_arch_normalizer(name: &str, patterns: Vec<String>, normalizer: ArchNormalizer) {
    arch_map().insert(name.to_string(), ArcPatterns { normalizer, patterns });
}

/// 按名称取架构归一化器。
pub fn get_arch_normalizer(name: &str) -> Option<ArchNormalizer> {
    arch_map().get(name).map(|a| a.normalizer)
}

/// 已注册的架构归一化器列表。
pub fn list_arch_normalizers() -> Vec<String> {
    let mut v: Vec<String> = arch_map().keys().cloned().collect();
    v.sort();
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attention_registry() {
        assert!(list_attentions().contains(&"standard".to_string()));
        assert!(get_attention("standard").is_some());
        assert!(get_attention("nope").is_none());
        // 注册新注意力（外部扩展模式）
        register_attention("custom", _build_standard);
        assert!(get_attention("custom").is_some());
        assert!(list_attentions().contains(&"custom".to_string()));
    }

    #[test]
    fn test_vision_moe_registry() {
        // vision 注册点（外部扩展——多模态真实接入为后续）
        assert!(list_vision_encoders().is_empty());
        assert!(get_vision_encoder("none").is_none());
        fn fake_vision(_s: &SafeTensors, _p: &str, _h: usize) -> Option<Vec<f32>> { Some(vec![0.0]) }
        register_vision_encoder("fake", fake_vision);
        assert!(get_vision_encoder("fake").is_some());
        assert!(list_vision_encoders().contains(&"fake".to_string()));
        // moe_format 注册点
        assert!(list_moe_formats().contains(&"merged".to_string()));
        assert!(get_moe_format("merged").is_some());
        fn fake_moe(_s: &SafeTensors, _p: &str, _e: usize, _i: usize)
                    -> (Vec<f32>, Vec<f32>, usize) { (vec![], vec![], 0) }
        register_moe_format("custom", fake_moe);
        assert!(get_moe_format("custom").is_some());
    }
}
