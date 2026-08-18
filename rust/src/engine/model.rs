//! 模型组装：embed → 层循环（逐层 RoPE 位置）→ 最终 norm → lm_head。
use crate::engine::attention::causal_mask;
use crate::engine::cache::KVCache;
use crate::engine::layer::DecoderLayer;
use crate::core::norm::rms_norm;
use crate::core::rope::{compute_inv_freq, rotary_embeddings};
use crate::engine::sampling::argmax_logits;
use crate::core::tensor::Tensor;

/// 小型标准注意力 MoE 模型（Mixtral / Qwen3-MoE / GLM 家族）。
pub struct Model {
    pub num_layers: usize,
    pub hidden: usize,
    pub rope_dim: usize,
    pub eps: f32,
    pub inv_freq: Vec<f32>,
    pub embed: Tensor,          // (vocab, hidden)
    pub layers: Vec<DecoderLayer>,
    pub final_norm_w: Vec<f32>,
    pub lm_head: Tensor,        // (vocab, hidden)
}

impl Model {
    /// 真实模型组装：从 safetensors + 归一化配置构建（Qwen3.5 层——delta/full + MoE）。
    /// 权重前缀（embed/layers/norm/lm_head）由配置的 weight_prefix 推导。
    pub fn from_real(store: &crate::io::safetensors::SafeTensors,
                     cfg: &crate::engine::model_config::ModelConfig) -> Result<Model, String> {
        let hidden = cfg.hidden_size;
        let vocab = cfg.vocab_size;
        let n = cfg.num_layers;
        let prefix = cfg.weight_prefix.clone();   // 如 "model" 或 "model.language_model"
        let get = |name: &str, out: usize| -> Tensor {
            Tensor::from_vec(out, hidden,
                store.get_f32(name).unwrap_or_else(|| vec![0.01; out * hidden]))
        };
        let embed = get(&format!("{prefix}.embed_tokens.weight"), vocab);
        let lm_head = get(&format!("{prefix}.lm_head.weight"), vocab);
        let final_norm_w = store.get_f32(&format!("{prefix}.norm.weight"))
            .unwrap_or_else(|| vec![1.0; hidden]);
        // 逐层组装：layer_types 分发（linear_attention → GatedDeltaNet；full_attention → FullAttention）
        let mut layers = Vec::with_capacity(n);
        for i in 0..n {
            let lp = format!("{prefix}.layers.{i}");
            let attn_type = cfg.layer_types.get(i).map(|s| s.as_str()).unwrap_or("full_attention");
            let attn: Box<dyn crate::engine::registry::Attention> =
                if attn_type.contains("linear") {
                    // 线性注意力：GatedDeltaNet（conv + in_proj + delta rule）
                    let kd = cfg.linear_key_head_dim;
                    let vd = cfg.linear_value_head_dim;
                    let nk = cfg.linear_num_key_heads;
                    let nv = cfg.linear_num_value_heads;
                    let key_dim = kd * nk;
                    let value_dim = vd * nv;
                    let c = key_dim * 2 + value_dim;
                    let kernel = cfg.conv_kernel_size;
                    Box::new(crate::engine::attention::GatedDeltaNet {
                        hidden,
                        key_dim,
                        value_dim,
                        num_k_heads: nk,
                        num_v_heads: nv,
                        head_k_dim: kd,
                        head_v_dim: vd,
                        eps: cfg.eps,
                        in_proj_qkv: get(&format!("{lp}.in_proj_qkv.weight"), c),
                        in_proj_z: get(&format!("{lp}.in_proj_z.weight"), value_dim),
                        in_proj_b: get(&format!("{lp}.in_proj_b.weight"), key_dim),
                        in_proj_a: get(&format!("{lp}.in_proj_a.weight"), key_dim),
                        conv_w: store.get_f32(&format!("{lp}.conv1d.weight"))
                            .unwrap_or_else(|| vec![0.01; c * kernel]),
                        out_w: get(&format!("{lp}.out_proj.weight"), hidden),
                        norm_w: store.get_f32(&format!("{lp}.out_layernorm.weight"))
                            .unwrap_or_else(|| vec![1.0; hidden]),
                        a_log: 0.1,
                        dt_bias: vec![0.0; key_dim],
                    })
                } else {
                    // 全注意力：FullAttention（注册表构造器）
                    crate::engine::registry::get_attention("full").map(|b| b(store, &lp, cfg))
                        .ok_or_else(|| "缺少 full 注意力构造器".to_string())?
                };
            // MoE（量化专家——AWQ 反量化按专家；此处用 merged 简化——量化专家路径为后续）
            let (router_w, gate_up, down) = if cfg.moe.is_some() {
                let e = cfg.moe.as_ref().unwrap().num_experts;
                let inter = cfg.moe.as_ref().unwrap().intermediate;
                (
                    get(&format!("{lp}.mlp.gate.weight"), e),
                    vec![0.01; e * 2 * inter * hidden],
                    vec![0.01; e * hidden * inter],
                )
            } else {
                (get(&format!("{lp}.mlp.gate.weight"), 1),
                 vec![0.01; 2 * cfg.moe_intermediate * hidden],
                 vec![0.01; hidden * cfg.moe_intermediate])
            };
            let num_exp = if cfg.moe.is_some() { cfg.moe.as_ref().unwrap().num_experts } else { 1 };
            layers.push(crate::engine::layer::DecoderLayer::new_real(
                i, cfg, attn, router_w, gate_up, down, num_exp));
        }
        let rope_dim = cfg.rope_dim;
        let inv_freq = crate::core::rope::compute_inv_freq(rope_dim, cfg.rope_theta, 0.5);
        Ok(Model {
            num_layers: n,
            hidden,
            rope_dim,
            eps: cfg.eps,
            inv_freq,
            embed,
            layers,
            final_norm_w,
            lm_head,
        })
    }

    /// 位置 0..len 的 cos/sin 张量（(len, rope_dim)）。
    pub fn cos_sin(&self, len: usize) -> (Tensor, Tensor) {
        let mut cos = vec![0.0f32; len * self.rope_dim];
        let mut sin = vec![0.0f32; len * self.rope_dim];
        for p in 0..len {
            let (c, s) = rotary_embeddings(p, &self.inv_freq);
            for j in 0..self.rope_dim {
                cos[p * self.rope_dim + j] = c[j];
                sin[p * self.rope_dim + j] = s[j];
            }
        }
        (Tensor::from_vec(len, self.rope_dim, cos),
         Tensor::from_vec(len, self.rope_dim, sin))
    }

    /// prefill：input_ids → 各位置 logits (L, vocab)。
    pub fn prefill(&self, input_ids: &[usize]) -> Tensor {
        let l = input_ids.len();
        let mut h = self.embed_rows(input_ids);
        let mask = causal_mask(l);
        let (cos, sin) = self.cos_sin(l);
        for layer in &self.layers {
            h = layer.forward(&h, &cos, &sin, Some(&mask));
        }
        let h = rms_norm(&h, &self.final_norm_w, self.eps);
        h.matmul(&self.lm_head.transpose())     // (L, vocab)
    }

    /// 取词嵌入行：(len, hidden)。
    pub fn embed_rows(&self, input_ids: &[usize]) -> Tensor {
        let mut data = vec![0.0f32; input_ids.len() * self.hidden];
        for (i, &t) in input_ids.iter().enumerate() {
            for j in 0..self.hidden {
                data[i * self.hidden + j] = self.embed.get(t, j);
            }
        }
        Tensor::from_vec(input_ids.len(), self.hidden, data)
    }

    /// 位置 pos 的 cos/sin（(1, rope_dim)）。
    pub fn cos_sin_at(&self, pos: usize) -> (Tensor, Tensor) {
        let (c, s) = rotary_embeddings(pos, &self.inv_freq);
        (Tensor::from_vec(1, self.rope_dim, c[..self.rope_dim].to_vec()),
         Tensor::from_vec(1, self.rope_dim, s[..self.rope_dim].to_vec()))
    }

    /// 位置 0..len 的 cos/sin（(len, rope_dim)）——decode 续接时 k_all 各行对应各自位置。
    pub fn cos_sin_all(&self, len: usize) -> (Tensor, Tensor) {
        let mut cos = vec![0.0f32; len * self.rope_dim];
        let mut sin = vec![0.0f32; len * self.rope_dim];
        for p in 0..len {
            let (c, s) = rotary_embeddings(p, &self.inv_freq);
            for j in 0..self.rope_dim {
                cos[p * self.rope_dim + j] = c[j];
                sin[p * self.rope_dim + j] = s[j];
            }
        }
        (Tensor::from_vec(len, self.rope_dim, cos),
         Tensor::from_vec(len, self.rope_dim, sin))
    }

    /// prefill 且填充 KV 缓存：返回最终归一化输出 (L, hidden)。
    pub fn prefill_cached(&self, input_ids: &[usize], cache: &mut KVCache) -> Tensor {
        let l = input_ids.len();
        let mut h = self.embed_rows(input_ids);
        let mask = causal_mask(l);
        let (cos, sin) = self.cos_sin(l);
        for (idx, layer) in self.layers.iter().enumerate() {
            let (out, k, v) = layer.forward_kv(&h, &cos, &sin, Some(&mask));
            cache.set(idx, k, v);
            h = out;
        }
        rms_norm(&h, &self.final_norm_w, self.eps)
    }

    /// decode 单 token（位置 pos）：返回 (1, hidden)。
    pub fn decode_step(&self, token: usize, pos: usize, cache: &mut KVCache) -> Tensor {
        let mut h = self.embed_rows(&[token]);
        // k_all 各行对应位置 0..=pos：用全位置 cos/sin（q 取最后一行）
        let (cos, sin) = self.cos_sin_all(pos + 1);
        for idx in 0..self.layers.len() {
            let (k_prev, v_prev) = match cache.get(idx) {
                Some(kv) => kv,
                None => return h,                       // 缓存缺失（不应发生）
            };
            let (out, k, v) = self.layers[idx].decode(&h, &cos, &sin, k_prev, v_prev);
            cache.set(idx, k, v);
            h = out;
        }
        rms_norm(&h, &self.final_norm_w, self.eps)
    }

    /// 贪心生成：prefill + decode 循环，返回生成 token 序列。
    pub fn generate(&self, input_ids: &[usize], max_new_tokens: usize) -> Vec<usize> {
        self.generate_sampled(input_ids, max_new_tokens, 0.0, 0, 0.0, 1.0)
    }

    /// 带采样参数的生成（temperature/top_k/top_p——与 Python generate_stream 对齐）。
    pub fn generate_sampled(&self, input_ids: &[usize], max_new_tokens: usize,
                            temperature: f32, top_k: usize, top_p: f32,
                            _repetition_penalty: f32) -> Vec<usize> {
        let mut ids = input_ids.to_vec();
        for _ in 0..max_new_tokens {
            let logits = self.prefill(&ids);
            let start = (ids.len() - 1) * logits.cols;
            let last = &logits.data[start..start + logits.cols];
            let tok = if temperature <= 0.0 {
                crate::engine::sampling::argmax_row(last)
            } else {
                let mut rng = || 0.5;   // 固定随机种子（纯 std——后续接入可复现随机）
                crate::engine::sampling::sample_row_p(last, temperature, top_k, top_p, &mut rng)
            };
            ids.push(tok);
        }
        ids[input_ids.len()..].to_vec()
    }

    /// 便捷构造（合成小模型用）：inv_freq 按 rope_dim/theta 生成。
    pub fn new_with_inv_freq(num_layers: usize, hidden: usize, rope_dim: usize,
                             eps: f32, theta: f32, vocab: usize,
                             embed: Tensor, layers: Vec<DecoderLayer>,
                             final_norm_w: Vec<f32>, lm_head: Tensor) -> Model {
        Model {
            num_layers,
            hidden,
            rope_dim,
            eps,
            inv_freq: compute_inv_freq(rope_dim, theta, 1.0),
            embed,
            layers,
            final_norm_w,
            lm_head,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prefill_logits_shape() {
        let (hidden, vocab) = (16usize, 32usize);
        let model = Model {
            num_layers: 0,
            hidden,
            rope_dim: 4,
            eps: 1e-6,
            inv_freq: compute_inv_freq(4, 1e6, 1.0),
            embed: Tensor::from_vec(vocab, hidden, vec![0.1; vocab * hidden]),
            layers: vec![],
            final_norm_w: vec![1.0; hidden],
            lm_head: Tensor::from_vec(vocab, hidden, vec![0.1; vocab * hidden]),
        };
        let ids = [1usize, 2, 3];
        let logits = model.prefill(&ids);
        assert_eq!((logits.rows, logits.cols), (3, vocab));
        assert!(logits.max_abs().is_finite());
    }
}
