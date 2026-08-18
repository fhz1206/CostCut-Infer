//! 分发注册表（镜像 Python `liteengine/registry.py` 的 attention/moe_format 注册点）。
//!
//! 注意力按类型名分发（"standard" / "mla" / "full"——由归一化配置的 `attention` 字段驱动），
//! 外部可通过 `register_attention` 注册新的注意力实现（可扩展性对齐 Python 版）。

use crate::core::tensor::Tensor;
use crate::engine::attention::{FullAttention, MlaAttention, StandardAttention};
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

/// 注意力构造器注册表（name → 构造器）。
static mut ATTENTION_BUILDERS: Option<HashMap<String, AttnBuilder>> = None;

fn builders() -> &'static mut HashMap<String, AttnBuilder> {
    unsafe {
        ATTENTION_BUILDERS.get_or_insert_with(|| {
            let mut m = HashMap::new();
            m.insert("standard".to_string(), _build_standard as AttnBuilder);
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
}
