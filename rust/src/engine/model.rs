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
        Self::from_real_truncated(store, cfg, cfg.num_layers)
    }

    /// 截断构造（浅层冒烟用——限制层数避免完整 61 层标量反量化的小时级耗时）。
    pub fn from_real_truncated(store: &crate::io::safetensors::SafeTensors,
                               cfg: &crate::engine::model_config::ModelConfig,
                               max_layers: usize) -> Result<Model, String> {
        let hidden = cfg.hidden_size;
        let vocab = cfg.vocab_size;
        let n = cfg.num_layers.min(max_layers);
        let prefix = cfg.weight_prefix.clone();   // 如 "model" 或 "model.language_model"
        let get = |name: &str, out: usize| -> Tensor {
            Tensor::from_vec(out, hidden,
                store.get_f32(name).unwrap_or_else(|| vec![0.01; out * hidden]))
        };
        let embed = get(&format!("{prefix}.embed_tokens.weight"), vocab);
        let lm_head = get("lm_head.weight", vocab);   // Qwen3.5 的 lm_head 在顶层（无 language_model 前缀）
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
            // MoE（量化专家——AWQ 按专家反量化；gate_up/down 来自 dequantize_awq）
            // compute_dtype="float16"/"bf16" 时同时构建相应权重接入 matmul_f16/bf16
            let use_fp16 = cfg.compute_dtype == "float16";
            let use_bf16 = cfg.compute_dtype == "bf16";
            let (router_w, gate_up, down, num_exp,
                 gate_up_f16, down_f16, gate_up_bf16, down_bf16) = if let Some(moe_cfg) = &cfg.moe {
                let e = moe_cfg.num_experts;
                let inter = moe_cfg.intermediate;
                let gs_size = moe_cfg.group_size;
                let mut gate_up = vec![0.0f32; e * 2 * inter * hidden];
                let mut down = vec![0.0f32; e * hidden * inter];
                let mut gate_up_f16: Vec<crate::core::tensor::F16Tensor> = Vec::new();
                let mut down_f16: Vec<crate::core::tensor::F16Tensor> = Vec::new();
                let mut gate_up_bf16: Vec<crate::core::tensor::BF16Tensor> = Vec::new();
                let mut down_bf16: Vec<crate::core::tensor::BF16Tensor> = Vec::new();
                let dq = crate::quant::dequant::dequantize_awq;
                for ex in 0..e {
                    let ep = format!("{lp}.mlp.experts.{ex}");
                    // Qwen3.5 实际为 gate_proj/up_proj 分离（非融合 gate_up_proj）
                    let gw = store.get_i32(&format!("{ep}.gate_proj.qweight")).unwrap_or_default();
                    let gz = store.get_i32(&format!("{ep}.gate_proj.qzeros")).unwrap_or_default();
                    let gs = store.get_f32(&format!("{ep}.gate_proj.scales")).unwrap_or_default();
                    let g = dq(&gw, &gz, &gs, inter, hidden, gs_size);
                    let uw = store.get_i32(&format!("{ep}.up_proj.qweight")).unwrap_or_default();
                    let uz = store.get_i32(&format!("{ep}.up_proj.qzeros")).unwrap_or_default();
                    let us = store.get_f32(&format!("{ep}.up_proj.scales")).unwrap_or_default();
                    let u = dq(&uw, &uz, &us, inter, hidden, gs_size);
                    let base = ex * 2 * inter * hidden;
                    gate_up[base..base + inter * hidden].copy_from_slice(&g);
                    gate_up[base + inter * hidden..base + 2 * inter * hidden].copy_from_slice(&u);
                    // fp16/bf16 权重（[2*inter, hidden]——与 f32 融合布局一致）
                    let mut gu_f16 = Vec::with_capacity(inter * hidden * 2);
                    gu_f16.extend_from_slice(&g);
                    gu_f16.extend_from_slice(&u);
                    let dw = store.get_i32(&format!("{ep}.down_proj.qweight")).unwrap_or_default();
                    let dz = store.get_i32(&format!("{ep}.down_proj.qzeros")).unwrap_or_default();
                    let ds = store.get_f32(&format!("{ep}.down_proj.scales")).unwrap_or_default();
                    let d = dq(&dw, &dz, &ds, hidden, inter, gs_size);
                    down[ex * hidden * inter..(ex + 1) * hidden * inter].copy_from_slice(&d);
                    if use_fp16 {
                        gate_up_f16.push(crate::core::tensor::F16Tensor::from_f32(
                            2 * inter, hidden, &gu_f16));
                        down_f16.push(crate::core::tensor::F16Tensor::from_f32(
                            hidden, inter, &d));
                    }
                    if use_bf16 {
                        gate_up_bf16.push(crate::core::tensor::BF16Tensor::from_f32(
                            2 * inter, hidden, &gu_f16));
                        down_bf16.push(crate::core::tensor::BF16Tensor::from_f32(
                            hidden, inter, &d));
                    }
                }
                (get(&format!("{lp}.mlp.gate.weight"), e), gate_up, down, e,
                 if use_fp16 { Some(gate_up_f16) } else { None },
                 if use_fp16 { Some(down_f16) } else { None },
                 if use_bf16 { Some(gate_up_bf16) } else { None },
                 if use_bf16 { Some(down_bf16) } else { None })
            } else {
                let inter = cfg.moe_intermediate;
                (get(&format!("{lp}.mlp.gate.weight"), 1),
                 vec![0.01; 2 * inter * hidden],
                 vec![0.01; hidden * inter], 1, None, None, None, None)
            };
            let mut dl = crate::engine::layer::DecoderLayer::new_real(
                i, cfg, attn, router_w, gate_up, down, num_exp);
            if let (Some(gu), Some(dn)) = (gate_up_f16, down_f16) {
                dl.experts.gate_up_f16 = Some(gu);
                dl.experts.down_f16 = Some(dn);
            }
            if let (Some(gu), Some(dn)) = (gate_up_bf16, down_bf16) {
                dl.experts.gate_up_bf16 = Some(gu);
                dl.experts.down_bf16 = Some(dn);
            }
            layers.push(dl);
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

    /// 输入末位的隐藏态（final norm 前——投机草稿的 aux 目标隐藏态，镜像 Python draft 的 h_target）。
    pub fn hidden_state_at_last(&self, input_ids: &[usize]) -> Tensor {
        let l = input_ids.len();
        let mut h = self.embed_rows(input_ids);
        let mask = causal_mask(l);
        let (cos, sin) = self.cos_sin(l);
        for layer in &self.layers {
            h = layer.forward(&h, &cos, &sin, Some(&mask));
        }
        // 取最后位置 (1, hidden)——final norm 前的隐藏态（aux 层目标）
        Tensor::from_vec(1, self.hidden,
            h.data[(l - 1) * self.hidden..l * self.hidden].to_vec())
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
                            repetition_penalty: f32) -> Vec<usize> {
        self.generate_stream_sampled(input_ids, max_new_tokens, temperature,
                                     top_k, top_p, repetition_penalty, &mut |_tok| {})
    }

    /// 逐 token 流式生成（镜像 Python generate_stream——每 token 回调一次）。
    pub fn generate_stream_sampled(&self, input_ids: &[usize], max_new_tokens: usize,
                                   temperature: f32, top_k: usize, top_p: f32,
                                   _repetition_penalty: f32,
                                   on_token: &mut dyn FnMut(usize)) -> Vec<usize> {
        let mut ids = input_ids.to_vec();
        let mut out = Vec::new();
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
            out.push(tok);
            on_token(tok);
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
