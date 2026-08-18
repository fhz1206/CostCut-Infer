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
        let mut cache = KVCache::new(self.num_layers);
        let h_full = self.prefill_cached(input_ids, &mut cache);    // (L, hidden)
        // 取最后位置（(1, hidden)）作为首个生成 token 的隐藏状态
        let mut h = Tensor::from_vec(
            1, h_full.cols,
            h_full.data[(h_full.rows - 1) * h_full.cols..].to_vec());
        let mut pos = input_ids.len();
        let mut out = Vec::new();
        for _ in 0..max_new_tokens {
            let logits = h.matmul(&self.lm_head.transpose());       // (1, vocab)
            let tok = argmax_logits(&logits);
            out.push(tok);
            h = self.decode_step(tok, pos, &mut cache);
            pos += 1;
        }
        out
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
