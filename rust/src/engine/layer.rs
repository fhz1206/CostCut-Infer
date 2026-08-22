//! DecoderLayer：input_norm → attention → +residual → post_norm → MoE（+可选共享）→ +residual。
use crate::engine::attention::StandardAttention;
use crate::engine::moe::{MLP, MergedExperts, TopKRouter};
use crate::core::norm::{rms_norm, rms_norm_add};
use crate::core::tensor::Tensor;

/// 共享专家门控：`sigmoid(x @ gate_w^T)` 广播到每行（返回 (rows, hidden) 每行同值）。
fn shared_gate(x: &Tensor, gate_w: &Tensor) -> Tensor {
    let mut data = vec![0.0f32; x.rows * x.cols];
    for i in 0..x.rows {
        let mut dot = 0.0f32;
        for j in 0..x.cols {
            dot += x.get(i, j) * gate_w.get(0, j);
        }
        let s = 1.0 / (1.0 + (-dot).exp());
        for j in 0..x.cols {
            data[i * x.cols + j] = s;
        }
    }
    Tensor::from_vec(x.rows, x.cols, data)
}

/// 单层前向。
pub struct DecoderLayer {
    pub eps: f32,
    pub input_norm_w: Vec<f32>,
    pub post_norm_w: Vec<f32>,
    pub attn: Box<dyn crate::engine::registry::Attention>,
    pub router: TopKRouter,
    pub experts: MergedExperts,
    pub shared: Option<(MLP, Tensor)>,   // (共享 MLP, 共享门控权重 (1, hidden))，可选
    pub dense_mlp: Option<MLP>,          // 稠密 MLP（通用回退 / Llama 家族）；Some 时无路由无共享
}

impl DecoderLayer {
    /// 真实模型层组装（from_real 用）：attn（trait 分发）+ MoE + norm（权重由 store 预载）。
    pub fn new_real(layer_idx: usize, cfg: &crate::engine::model_config::ModelConfig,
                    attn: Box<dyn crate::engine::registry::Attention>,
                    router_w: Tensor, gate_up: Vec<f32>, down: Vec<f32>, num_exp: usize)
                    -> DecoderLayer {
        let hidden = cfg.hidden_size;
        let _ = layer_idx;
        DecoderLayer {
            eps: cfg.eps,
            input_norm_w: vec![1.0; hidden],
            post_norm_w: vec![1.0; hidden],
            attn,
            router: TopKRouter { weight: router_w, top_k: 1 },
            experts: MergedExperts {
                num_experts: num_exp, intermediate: cfg.moe_intermediate, hidden,
                gate_up, down, gate_up_f16: None, down_f16: None,
                gate_up_bf16: None, down_bf16: None,
            },
            shared: None,
            dense_mlp: None,
        }
    }
    /// prefill：返回 (L, hidden)。
    pub fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor,
                   mask: Option<&Tensor>) -> Tensor {
        self.forward_kv(x, cos, sin, mask).0
    }

    /// prefill 且返回每层 k/v（供缓存填充）。
    pub fn forward_kv(&self, x: &Tensor, cos: &Tensor, sin: &Tensor,
                      mask: Option<&Tensor>) -> (Tensor, Tensor, Tensor) {
        let residual = x.clone();
        let h = rms_norm(x, &self.input_norm_w, self.eps);
        println!("[layer] enter attn rows={} type={}", x.rows,
                 std::any::type_name_of_val(&*self.attn));
        let ta = std::time::Instant::now();
        let (attn_out, k, v) = self.attn.forward_kv(&h, cos, sin, mask);
        if x.rows >= 3 {
            println!("[layer] attn {:.3}s", ta.elapsed().as_secs_f64());
        }
        let tm = std::time::Instant::now();
        let (h, h_pre) = rms_norm_add(&attn_out, &residual, &self.post_norm_w, self.eps);
        let out = h_pre.add(&self.mlp_forward(&h));
        if x.rows >= 3 {
            println!("[layer] attn {:.3}s mlp {:.3}s",
                     ta.elapsed().as_secs_f64(), tm.elapsed().as_secs_f64());
        }
        (out, k, v)
    }

    /// decode：x (1, hidden)；k_prev/v_prev 缓存续接。返回 (out, new_k, new_v)。
    pub fn decode(&self, x: &Tensor, cos: &Tensor, sin: &Tensor,
                  k_prev: &Tensor, v_prev: &Tensor) -> (Tensor, Tensor, Tensor) {
        let residual = x.clone();
        let h = rms_norm(x, &self.input_norm_w, self.eps);
        let (attn_out, k, v) = self.attn.decode(&h, cos, sin, k_prev, v_prev);
        // 残差 + post_norm 融合（镜像 Python rms_norm_add；差异报告 #8）
        let (h, h_pre) = rms_norm_add(&attn_out, &residual, &self.post_norm_w, self.eps);
        (h_pre.add(&self.mlp_forward(&h)), k, v)
    }

    /// 稠密/MoE 分发：dense_mlp 存在时直接 MLP（无路由），否则走 MoE。
    fn mlp_forward(&self, h: &Tensor) -> Tensor {
        match &self.dense_mlp {
            Some(mlp) => mlp.forward(h),
            None => self.moe_forward(h),
        }
    }

    fn moe_forward(&self, h: &Tensor) -> Tensor {
        let (indices, weights) = self.router.forward(h);
        let out = self.experts.forward(h, &indices, &weights);
        match &self.shared {
            Some((mlp, gate_w)) => {
                out.add(&shared_gate(h, gate_w).elementwise_mul(&mlp.forward(h)))
            }
            None => out,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_shape() {
        let (hidden, h, kvh, hd, e, inter) = (16usize, 4usize, 2usize, 4usize, 4usize, 8usize);
        let layer = DecoderLayer {
            eps: 1e-6,
            input_norm_w: vec![1.0; hidden],
            post_norm_w: vec![1.0; hidden],
            attn: Box::new(StandardAttention {
                num_heads: h, num_kv_heads: kvh, head_dim: hd, rope_dim: 4, scaling: 0.5,
                q_w: Tensor::from_vec(h * hd, hidden, vec![0.1; h * hd * hidden]),
                k_w: Tensor::from_vec(kvh * hd, hidden, vec![0.1; kvh * hd * hidden]),
                v_w: Tensor::from_vec(kvh * hd, hidden, vec![0.1; kvh * hd * hidden]),
                o_w: Tensor::from_vec(hidden, h * hd, vec![0.1; hidden * h * hd]),
            }),
            router: TopKRouter {
                weight: Tensor::from_vec(e, hidden, vec![0.1; e * hidden]), top_k: 2,
            },
            experts: MergedExperts {
                num_experts: e, intermediate: inter, hidden,
                gate_up: vec![0.1; e * 2 * inter * hidden],
                down: vec![0.1; e * hidden * inter],
                gate_up_f16: None, down_f16: None,
                gate_up_bf16: None, down_bf16: None,
            },
            shared: None,
            dense_mlp: None,
        };
        let x = Tensor::from_vec(4, hidden, vec![0.5; 4 * hidden]);
        let cos = Tensor::from_vec(4, 4, vec![1.0; 16]);
        let sin = Tensor::from_vec(4, 4, vec![0.0; 16]);
        let out = layer.forward(&x, &cos, &sin, Some(&crate::engine::attention::causal_mask(4)));
        assert_eq!((out.rows, out.cols), (4, hidden));
        assert!(out.max_abs().is_finite());
    }

    #[test]
    fn test_dense_layer_shape() {
        // 稠密 MLP 层（Llama 家族 / 通用回退）：dense_mlp 存在时无路由。
        let (hidden, h, kvh, hd, inter) = (16usize, 4usize, 2usize, 4usize, 32usize);
        let layer = DecoderLayer {
            eps: 1e-6,
            input_norm_w: vec![1.0; hidden],
            post_norm_w: vec![1.0; hidden],
            attn: Box::new(StandardAttention {
                num_heads: h, num_kv_heads: kvh, head_dim: hd, rope_dim: 4, scaling: 0.5,
                q_w: Tensor::from_vec(h * hd, hidden, vec![0.1; h * hd * hidden]),
                k_w: Tensor::from_vec(kvh * hd, hidden, vec![0.1; kvh * hd * hidden]),
                v_w: Tensor::from_vec(kvh * hd, hidden, vec![0.1; kvh * hd * hidden]),
                o_w: Tensor::from_vec(hidden, h * hd, vec![0.1; hidden * h * hd]),
            }),
            router: TopKRouter {
                weight: Tensor::from_vec(1, hidden, vec![0.1; hidden]), top_k: 1,
            },
            experts: MergedExperts {
                num_experts: 1, intermediate: inter, hidden,
                gate_up: vec![0.1; 2 * inter * hidden],
                down: vec![0.1; hidden * inter],
                gate_up_f16: None, down_f16: None,
                gate_up_bf16: None, down_bf16: None,
            },
            shared: None,
            dense_mlp: Some(MLP {
                gate_w: Tensor::from_vec(inter, hidden, vec![0.1; inter * hidden]),
                up_w: Tensor::from_vec(inter, hidden, vec![0.1; inter * hidden]),
                down_w: Tensor::from_vec(hidden, inter, vec![0.1; hidden * inter]),
            }),
        };
        let x = Tensor::from_vec(4, hidden, vec![0.5; 4 * hidden]);
        let cos = Tensor::from_vec(4, 4, vec![1.0; 16]);
        let sin = Tensor::from_vec(4, 4, vec![0.0; 16]);
        let out = layer.forward(&x, &cos, &sin, Some(&crate::engine::attention::causal_mask(4)));
        assert_eq!((out.rows, out.cols), (4, hidden));
        assert!(out.max_abs().is_finite());
    }
}
