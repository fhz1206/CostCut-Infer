//! MTP 多 token 预测（镜像 Python `liteengine/mtp.py`——enorm/hnorm + eh_proj + 附加层 + 共享输出头）。
//!
//! 多 token 预测链：第 k 个模块的 `embed_prev` = 第 k-1 个预测 token 的嵌入；
//! 输出头与主模型共享（DeepSeek-V3 的 shared_head 约定）。

use crate::core::tensor::Tensor;

/// 单个 MTP 模块：enorm/hnorm（RMSNorm）+ eh_proj（投影）+ 附加 DecoderLayer。
pub struct MtpModule {
    pub eps: f32,
    pub hidden: usize,
    pub enorm_w: Vec<f32>,
    pub hnorm_w: Vec<f32>,
    pub eh_proj: Tensor,   // (hidden, hidden)——投影合并（mtp_hidden 简化等于 hidden）
    pub layer: crate::engine::layer::DecoderLayer,
}

fn rms_norm_vec(x: &Tensor, w: &[f32], eps: f32) -> Tensor {
    let mean_sq = x.data.iter().map(|v| v * v).sum::<f32>() / x.data.len() as f32;
    let inv = 1.0 / (mean_sq + eps).sqrt();
    let mut out = x.clone();
    for i in 0..out.data.len() {
        out.data[i] *= inv * w[i % w.len().max(1)];
    }
    out
}

impl MtpModule {
    /// 构造（权重前缀 "{prefix}.enorm/hnorm/eh_proj" + 附加层权重）。
    pub fn new(store: &crate::io::safetensors::SafeTensors, prefix: &str,
               cfg: &crate::engine::model_config::ModelConfig, layer_idx: usize) -> MtpModule {
        let hidden = cfg.hidden_size;
        let get = |name: &str, out: usize| -> Tensor {
            let mut d = store.get_f32(name).unwrap_or_else(|| vec![0.0; out * hidden]);
            if d.len() != out * hidden {
                d.truncate(out * hidden);
                d.resize(out * hidden, 0.0);
            }
            Tensor::from_vec(out, hidden, d)
        };
        let eh = store.get_f32(&format!("{prefix}.eh_proj.weight")).unwrap_or_default();
        // mtp_hidden 简化等于 hidden（eh_proj 视为 (hidden, hidden) 方阵）
        let eh_proj = Tensor::from_vec(hidden, hidden,
            if eh.len() >= hidden * hidden { eh[..hidden * hidden].to_vec() } else {
                let mut e = eh;
                e.resize(hidden * hidden, 0.0);
                e
            });
        // 附加 DecoderLayer（MTP 层权重——前缀 "{prefix}.layers.{layer_idx}"——简化构造）
        let lp = format!("{prefix}.layers.{layer_idx}");
        let attn: Box<dyn crate::engine::registry::Attention> = {
            // 简化：标准注意力（占位——真实 MTP 层权重接入为后续）
            let hd = hidden / 8;
            Box::new(crate::engine::attention::StandardAttention {
                num_heads: 8, num_kv_heads: 2, head_dim: hd, rope_dim: hd,
                scaling: (hd as f32).powf(-0.5),
                q_w: get(&format!("{lp}.self_attn.q_proj.weight"), hidden),
                k_w: get(&format!("{lp}.self_attn.k_proj.weight"), hidden / 4),
                v_w: get(&format!("{lp}.self_attn.v_proj.weight"), hidden / 4),
                o_w: get(&format!("{lp}.self_attn.o_proj.weight"), hidden),
            })
        };
        let inter = cfg.moe_intermediate;
        let layer = crate::engine::layer::DecoderLayer {
            eps: cfg.eps,
            input_norm_w: store.get_f32(&format!("{lp}.input_layernorm.weight"))
                .unwrap_or_else(|| vec![1.0; hidden]),
            post_norm_w: store.get_f32(&format!("{lp}.post_attention_layernorm.weight"))
                .unwrap_or_else(|| vec![1.0; hidden]),
            attn,
            router: crate::engine::moe::TopKRouter {
                weight: get(&format!("{lp}.mlp.gate.weight"), 1), top_k: 1 },
            experts: crate::engine::moe::MergedExperts {
                num_experts: 1, intermediate: inter, hidden,
                gate_up: vec![0.0; 2 * inter * hidden],
                down: vec![0.0; hidden * inter],
                gate_up_f16: None, down_f16: None,
                gate_up_bf16: None, down_bf16: None,
            },
            shared: None,
            dense_mlp: None,
        };
        MtpModule {
            eps: cfg.eps,
            hidden,
            enorm_w: store.get_f32(&format!("{prefix}.enorm.weight"))
                .unwrap_or_else(|| vec![1.0; hidden]),
            hnorm_w: store.get_f32(&format!("{prefix}.hnorm.weight"))
                .unwrap_or_else(|| vec![1.0; hidden]),
            eh_proj,
            layer,
        }
    }

    /// 前向：h_n + e_n（投影合并）→ 附加层（镜像 Python MtpModule.forward）。
    pub fn forward(&self, h: &Tensor, embed_prev: &Tensor) -> Tensor {
        let h_n = rms_norm_vec(h, &self.enorm_w, self.eps);
        let e_n = rms_norm_vec(embed_prev, &self.hnorm_w, self.eps);
        let e_proj = e_n.matmul(&self.eh_proj);   // (1, hidden)
        let mut h_in = h_n;
        for i in 0..h_in.data.len() {
            h_in.data[i] += e_proj.data[i];
        }
        // 附加层前向（cos/sin 占位——MTP 附加层线性注意力场景；mask 忽略）
        let cos = Tensor::from_vec(1, 8, vec![1.0; 8]);
        let sin = Tensor::from_vec(1, 8, vec![0.0; 8]);
        self.layer.forward(&h_in, &cos, &sin, None)
    }

    /// 投机草稿：输出表示 → 共享输出头（主模型 lm_head）→ logits。
    pub fn draft(&self, h: &Tensor, embed_prev: &Tensor,
                 lm_head: &Tensor) -> Tensor {
        let out = self.forward(h, embed_prev);
        out.matmul(&lm_head.transpose())   // (1, vocab)
    }
}
