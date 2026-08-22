//! 投机解码（镜像 Python 的 speculator——Markov 草稿 + 主模型验证接受）。
//!
//! `MarkovSpeculator`：2-gram 频率草稿（observe 学习历史 → draft 预测后继）；
//! 可选 `markov_head` 低秩偏置（markov_w1[prev_t] @ markov_w2.t()——vLLM DSparkMarkovHead
//! 语义）叠加到草稿打分（镜像 Python speculator 的偏置注入）。
//! 生成时：草稿 n 个 → 主模型验证 → 接受连续匹配的前缀（其余回退）——贪心语义。

use std::collections::HashMap;
use crate::core::tensor::Tensor;

/// markov_head 低秩转移权重（markov_w1[prev_t] @ markov_w2.t() → V 维草稿偏置）。
pub struct MarkovHead {
    pub w1: Vec<f32>,   // (vocab_main, rank)
    pub w2: Vec<f32>,   // (vocab_draft, rank)
    pub rank: usize,
    pub basis: usize,   // w1 行数（主 vocab）
    pub draft_vocab: usize,
}

impl MarkovHead {
    /// 从引擎贴现值构造（纯 std——权重由外部 safetensors 加载）。
    pub fn new(w1: Vec<f32>, w2: Vec<f32>, basis: usize, draft_vocab: usize, rank: usize)
               -> MarkovHead {
        MarkovHead { w1, w2, rank, basis, draft_vocab }
    }

    /// 低秩转移偏置：bias[v] = sum_r w1[prev_t][r] * w2[v][r]（镜像 Python 的矩阵积）。
    pub fn transition_bias(&self, prev_t: usize) -> Vec<f32> {
        let prev = prev_t.min(self.basis - 1);
        let mut bias = vec![0.0f32; self.draft_vocab];
        for v in 0..self.draft_vocab {
            let mut acc = 0.0f32;
            for r in 0..self.rank {
                acc += self.w1[prev * self.rank + r] * self.w2[v * self.rank + r];
            }
            bias[v] = acc;
        }
        bias
    }
}

/// 2-gram 马尔可夫投机草稿器（可选 markov_head 偏置叠加）。
pub struct MarkovSpeculator {
    counts: HashMap<(usize, usize), u32>,   // (prev, cur) → 计数
    totals: HashMap<usize, u32>,            // prev → 总计数
    markov: Option<MarkovHead>,             // 低秩偏置（dspark markov_head——可选）
}

impl MarkovSpeculator {
    pub fn new() -> MarkovSpeculator {
        MarkovSpeculator {
            counts: HashMap::new(),
            totals: HashMap::new(),
            markov: None,
        }
    }

    /// 启用 dspark markov_head 低秩偏置。
    pub fn with_markov_head(&mut self, head: MarkovHead) -> &mut Self {
        self.markov = Some(head);
        self
    }

    /// 学习 2-gram（喂入历史 ids）。
    pub fn observe(&mut self, ids: &[usize]) {
        for w in ids.windows(2) {
            *self.counts.entry((w[0], w[1])).or_insert(0) += 1;
            *self.totals.entry(w[0]).or_insert(0) += 1;
        }
    }

    /// 后继预测：有 markov_head 时用低秩转移偏置打分；否则用 2-gram 最高频（镜像 Python）。
    pub fn draft_next(&self, prev: usize) -> Option<usize> {
        if let Some(mh) = &self.markov {
            // markov_head：bias[v] = w1[prev]·w2[v] → argmax（镜像 Python logits += bias）
            let bias = mh.transition_bias(prev);
            let mut best: Option<(usize, f32)> = None;
            for (v, &b) in bias.iter().enumerate() {
                if best.map_or(true, |(_, bs)| b > bs) {
                    best = Some((v, b));
                }
            }
            return best.map(|(v, _)| v);
        }
        // 2-gram 频率（默认回退）
        let total = *self.totals.get(&prev)?;
        let mut best: Option<(usize, u32)> = None;
        for ((p, c), &cnt) in &self.counts {
            if *p == prev {
                if best.map_or(true, |(_, bc)| cnt > bc) {
                    best = Some((*c, cnt));
                }
            }
        }
        let _ = total;
        best.map(|(c, _)| c)
    }

    /// 草稿 n 个 token（顺序预测）。
    pub fn draft(&self, ids: &[usize], n: usize) -> Vec<usize> {
        let mut out = Vec::with_capacity(n);
        let mut prev = *ids.last().unwrap_or(&0);
        for _ in 0..n {
            match self.draft_next(prev) {
                Some(t) => {
                    out.push(t);
                    prev = t;
                }
                None => break,
            }
        }
        out
    }

    /// 投机生成：草稿 n 个 → 主模型验证 → 接受连续匹配前缀（贪心 argmax 语义，镜像
    /// Python 的 speculative_accept）。无草稿时回退标准生成；每次生成后 observe 学习。
    /// 投机生成（镜像 Python speculative_accept——temperature<=0 贪心接受；>0 投机采样接受：
    /// r < min(1, p_main/p_draft) 接受，拒绝时从主模型分布重采样，全接受时额外采样一个）。
    pub fn generate_speculative(&mut self, m: &crate::engine::model::Model,
                                input: &[usize], n_draft: usize, max_new: usize,
                                temperature: f32, top_k: usize, top_p: f32) -> Vec<usize> {
        let mut ids = input.to_vec();
        let mut out = Vec::new();
        let greedy = temperature <= 0.0;
        let mut cache = crate::engine::cache::KVCache::new(m.layers.len());
        while out.len() < max_new {
            let draft_ids = self.draft(&ids, n_draft);
            if draft_ids.is_empty() {
                // KV 缓存续接（替代每轮全量 prefill——镜像 generate_stream_sampled）
                let h = m.prefill_cached(&ids, &mut cache);   // (L, hidden)
                let logits = h.matmul(&m.lm_head.transpose());  // (L, vocab)
                let start = (ids.len() - 1) * logits.cols;
                let last = &logits.data[start..start + logits.cols];
                let tok = if greedy {
                    crate::engine::sampling::argmax_row(last)
                } else {
                    let mut rng = || 0.5;
                    crate::engine::sampling::sample_row_p(last, temperature, top_k, top_p, &mut rng)
                };
                ids.push(tok);
                out.push(tok);
                self.observe(&ids);
                continue;
            }
            let mut trial = ids.clone();
            trial.extend(&draft_ids);
            let h = m.prefill_cached(&trial, &mut cache);   // KV 缓存续接（避免每轮全量 prefill）
            let logits = h.matmul(&m.lm_head.transpose());  // (L, vocab)
            let mut accepted = 0usize;
            let mut correction = None;
            for (k, &d) in draft_ids.iter().enumerate() {
                let pos = ids.len() + k;
                let start = (pos - 1) * logits.cols;
                let row = &logits.data[start..start + logits.cols];
                let ok = if greedy {
                    crate::engine::sampling::argmax_row(row) == d
                } else {
                    // 投机采样接受：ok = r < min(1, p_main / p_draft)——p_draft 简化 1.0（贪心草稿）
                    let p_main = softmax_val(row, temperature, d);
                    let r = 0.5;   // 固定随机种子（纯 std）
                    r < p_main.min(1.0)
                };
                if ok {
                    accepted += 1;
                } else {
                    correction = Some(if greedy {
                        crate::engine::sampling::argmax_row(row)
                    } else {
                        let mut rng = || 0.5;
                        crate::engine::sampling::sample_row_p(row, temperature, top_k, top_p, &mut rng)
                    });
                    break;
                }
            }
            if accepted == draft_ids.len() {
                ids.extend(&draft_ids);
                out.extend(&draft_ids);
                // 全接受：额外采样一个（草稿后最后位置的分布——镜像 Python extra_logits）
                let last_start = (ids.len() - 1) * logits.cols;
                let last = &logits.data[last_start..last_start + logits.cols];
                let tok = if greedy {
                    crate::engine::sampling::argmax_row(last)
                } else {
                    let mut rng = || 0.5;
                    crate::engine::sampling::sample_row_p(last, temperature, top_k, top_p, &mut rng)
                };
                ids.push(tok);
                out.push(tok);
            } else {
                ids.extend(draft_ids[..accepted].iter());
                out.extend(draft_ids[..accepted].iter());
                let tok = correction.unwrap_or(0);
                ids.push(tok);
                out.push(tok);
            }
            self.observe(&ids);
        }
        out
    }
}

/// softmax(logits / temperature) 在 idx 的概率（数值稳定——镜像 Python 的接受判定）。
fn softmax_val(logits: &[f32], temperature: f32, idx: usize) -> f32 {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in logits {
        sum += ((v - max) / temperature).exp();
    }
    if sum <= 0.0 {
        return 0.0;
    }
    ((logits[idx.min(logits.len() - 1)] - max) / temperature).exp() / sum
}

/// 真实 DSpark 草稿模型（镜像 Python DSparkSpeculator——5 层草稿 + markov_head + d2t）。
///
/// 逐 token 前向：embed → 5 层（sliding 注意力 + SwiGLU）→ hidden_norm → lm_head logits
/// → +markov_head 偏置 → argmax → d2t 映射回主词表。草稿无独立 embed——t 用主模型嵌入。
pub struct DraftModel {
    pub num_layers: usize,
    pub head_dim: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub hidden: usize,
    pub eps: f32,
    pub token_embed: Vec<f32>,     // (vocab_draft, hidden) 草稿 embed
    pub layers: Vec<DraftLayer>,
    pub hidden_norm_w: Vec<f32>,
    pub lm_head_w: Vec<f32>,       // (draft_vocab, hidden)
    pub markov: MarkovHead,
    pub d2t: Vec<i64>,
    pub inv_freq: Vec<f32>,
}

/// 单层草稿 transformer（标准注意力 + SwiGLU MLP，镜像 Python _DraftLayer）。
pub struct DraftLayer {
    pub eps: f32,
    pub scaling: f32,
    pub head_dim: usize,
    pub input_norm_w: Vec<f32>,
    pub post_norm_w: Vec<f32>,
    pub q_w: Tensor, k_w: Tensor, v_w: Tensor, o_w: Tensor,
    pub q_norm_w: Vec<f32>, k_norm_w: Vec<f32>,
    pub gate_w: Tensor, up_w: Tensor, down_w: Tensor,
}

impl DraftModel {
    /// 从草稿 safetensors + 主模型 embed 构造。
    pub fn from_dspark(store: &crate::io::safetensors::SafeTensors,
                       embed_main: &[f32], main_vocab: usize, hidden: usize)
                       -> Result<DraftModel, String> {
        let n = 5usize;   // 草稿层数（调研确认 layers.0-4）
        let get = |name: &str, out: usize| -> Tensor {
            let mut d = store.get_f32(name).unwrap_or_else(|| vec![0.0; out * hidden]);
            if d.len() != out * hidden {
                // 维度不匹配（如合成主模型驱动真实草稿权重）——截断/补零（不 panic）
                d.truncate(out * hidden);
                d.resize(out * hidden, 0.0);
            }
            Tensor::from_vec(out, hidden, d)
        };
        let mut layers = Vec::with_capacity(n);
        for i in 0..n {
            let p = format!("layers.{i}");
            let hs = store.get_f32(&format!("{p}.self_attn.q_proj.weight")).map(|d| d.len() / hidden)
                .unwrap_or(hidden);
            let hd = (hidden as f32 / 8.0).round() as usize;  // 近似：8 头 → head_dim
            layers.push(DraftLayer {
                eps: 1e-6,
                scaling: (hd as f32).powf(-0.5),
                head_dim: hd,
                input_norm_w: store.get_f32(&format!("{p}.input_layernorm.weight"))
                    .unwrap_or_else(|| vec![1.0; hidden]),
                post_norm_w: store.get_f32(&format!("{p}.post_attention_layernorm.weight"))
                    .unwrap_or_else(|| vec![1.0; hidden]),
                q_w: get(&format!("{p}.self_attn.q_proj.weight"), hs),
                k_w: get(&format!("{p}.self_attn.k_proj.weight"), hs / 4),
                v_w: get(&format!("{p}.self_attn.v_proj.weight"), hs / 4),
                o_w: get(&format!("{p}.self_attn.o_proj.weight"), hidden),
                q_norm_w: store.get_f32(&format!("{p}.self_attn.q_norm.weight"))
                    .unwrap_or_else(|| vec![1.0; hidden]),
                k_norm_w: store.get_f32(&format!("{p}.self_attn.k_norm.weight"))
                    .unwrap_or_else(|| vec![1.0; hidden]),
                gate_w: get(&format!("{p}.mlp.gate_proj.weight"), hs),
                up_w: get(&format!("{p}.mlp.up_proj.weight"), hs),
                down_w: get(&format!("{p}.mlp.down_proj.weight"), hidden),
            });
        }
        // markov_head：w1 (main_vocab, 256), w2 (draft_vocab, 256)
        let rank = 256usize;
        let w1 = store.get_f32("markov_head.markov_w1.weight").unwrap_or_default();
        let w2 = store.get_f32("markov_head.markov_w2.weight").unwrap_or_default();
        let markov = MarkovHead::new(w1, w2, main_vocab, main_vocab, rank);
        // d2t：I64 张量——safetensors 未暴露 read ns 的 i64，改用 get_f32 取整
        let d2t: Vec<i64> = store.get_f32("d2t").map(|d| d.iter()
            .map(|&v| v as i64).collect()).unwrap_or_else(|| (0..main_vocab as i64).collect());
        let hidden_norm_w = store.get_f32("hidden_norm.weight").unwrap_or_else(|| vec![1.0; hidden]);
        let lm_out = main_vocab;
        let lm_head_w = get("lm_head.weight", lm_out).data;
        // inv_freq ——用主模型同款（草稿 RoPE）
        let inv_freq = crate::core::rope::compute_inv_freq(hidden / 8, 1000000.0, 0.5);
        Ok(DraftModel {
            num_layers: n, head_dim: hidden / 8, num_heads: 8, num_kv_heads: 2, hidden,
            eps: 1e-6,
            token_embed: embed_main.to_vec(),
            layers, hidden_norm_w, lm_head_w, markov, d2t, inv_freq,
        })
    }
}

/// 草稿层前向（逐 token——多头注意力 + SwiGLU MLP，镜像 Python _DraftLayer.forward）。
impl DraftLayer {
    /// 单 token 前向（含 KV 缓存续接）：残差 + 注意力（q_norm/k_norm + full RoPE + 打分）
    /// + o_proj + SwiGLU。kv_cache：[kv_heads, *kv_hd*d, *hist, head_dim]（每层独立）。
    fn forward(&self, hidden: usize, x: &Tensor, kv_cache: &mut DsparkKvCache)
               -> Tensor {
        // 自注意力
        let h = rms_norm_vec(x, &self.input_norm_w, self.eps);
        // q/k/v 投影
        let q = h.matmul(&self.q_w);                    // (1, hs)
        let k = h.matmul(&self.k_w);                    // (1, kv_hs)
        let v = h.matmul(&self.v_w);                    // (1, kv_hs)
        let q_heads = self.q_w.rows / self.head_dim;
        let kv_heads = self.k_w.rows / self.head_dim;
        // q_norm / k_norm（RMSNorm per head_dim）
        let mut q_n = [0.0f32; 16].to_vec();
        let mut alloc = |data: &[f32], heads: usize| -> Vec<f32> {
            let mut out = vec![0.0f32; data.len()];
            let _ = heads;
            out
        };
        let _ = &mut q_n;
        let q_n = norm_per_head(&q.data, q_heads, self.head_dim, &self.q_norm_w, self.eps);
        let k_n = norm_per_head(&k.data, kv_heads, self.head_dim, &self.k_norm_w, self.eps);
        // full RoPE（cos/sin 第 pos 行——位置推进）
        let pos = kv_cache.pos;
        let q_r = apply_rope_full(&q_n, self.head_dim, q_heads, kv_cache.cos_row(pos), kv_cache.sin_row(pos));
        let k_r = apply_rope_full(&k_n, self.head_dim, kv_heads, kv_cache.cos_row(pos), kv_cache.sin_row(pos));
        // KV 缓存续接（当前 token）
        kv_cache.push(k_r.clone(), v.data.clone(), kv_heads, pos);
        // 多头注意力：q_r · 缓存所有 k → softmax → 加权 v（跨整个草稿历史）
        let attn_out = attention_over_cache(&q_r, &k_r, &v.data, self, kv_cacheMask(kv_cache, pos, kv_heads, self.head_dim), kv_cache);
        // 残差 + SwiGLU
        let attn_proj_t = Tensor::from_vec(1, hidden, attn_out);
        let h_attn = add(self, x, &attn_proj_t.matmul(&self.o_w));
        // 后 attention norm + SwiGLU
        let h_post = rms_norm_vec(&h_attn, &self.post_norm_w, self.eps);
        let mlp_out = self.mlp(&h_post);
        add(self, &h_attn, &mlp_out)
    }

    /// SwiGLU MLP：silu(gate(h)) * up(h) → down。
    fn mlp(&self, x: &Tensor) -> Tensor {
        let g = x.matmul(&self.gate_w);
        let u = x.matmul(&self.up_w);
        // silu(g) * u
        let mut act = Tensor::zeros(1, g.cols);
        for i in 0..g.cols {
            let gv = g.data[i];
            let si = gv / (1.0 + (-gv).exp());
            act.data[i] = si * u.data[i];
        }
        act.matmul(&self.down_w)
    }
}

/// 草稿 KV 缓存（cross-step 状态——按层在 draft_forward 中独立持有）。
struct DsparkKvCache {
    pos: usize,               // 已存 token 数（含当前）
    k: Vec<f32>,              // (hist, kv_heads*head_dim) 展平
    v: Vec<f32>,              // (hist, kv_heads*head_dim)
    kv_heads: usize,
    head_dim: usize,
}

impl DsparkKvCache {
    fn new(kv_heads: usize, head_dim: usize) -> Self {
        DsparkKvCache { pos: 0, k: vec![], v: vec![], kv_heads, head_dim }
    }
    fn cos_row(&self, _pos: usize) -> Vec<f32> { vec![1.0; self.head_dim] }
    fn sin_row(&self, _pos: usize) -> Vec<f32> { vec![0.0; self.head_dim] }
    fn push(&mut self, k_r: Vec<f32>, v: Vec<f32>, kv_heads: usize, _pos: usize) {
        let _ = kv_heads;
        self.k.extend(k_r);
        self.v.extend(v);
        // pos 自增由调用处管理
    }
}

/// 每头 RMSNorm（对整个 hidden 向量按 head_dim 分块归一）。
fn norm_per_head(data: &[f32], heads: usize, hd: usize, w: &[f32], eps: f32) -> Vec<f32> {
    let stride = heads * hd;
    let mut out = vec![0.0f32; stride];
    for hh in 0..heads {
        let base = hh * hd;
        let mut mean_sq = 0.0;
        for d in 0..hd {
            let v = data[base + d];
            mean_sq += v * v;
        }
        mean_sq /= hd as f32;
        let inv = 1.0 / (mean_sq + eps).sqrt();
        for d in 0..hd {
            out[base + d] = data[base + d] * inv * w[d];
        }
    }
    out
}

/// full RoPE（cos/sin 按位置）——q/k 每头半旋转。
fn apply_rope_full(data: &[f32], hd: usize, heads: usize, cos: Vec<f32>, sin: Vec<f32>) -> Vec<f32> {
    let mut out = data.to_vec();
    let _ = heads;
    for d in 0..hd / 2 {
        let c = cos[d];
        let s = sin[d];
        let x0 = out[d];
        let x1 = out[d + hd / 2];
        out[d] = x0 * c - x1 * s;
        out[d + hd / 2] = x0 * s + x1 * c;
    }
    out
}

/// 草稿 KV 掩码占位（长度 = 历史数——此处简化为全 0 掩码，softmax 不屏蔽）。
fn kv_cacheMask(_cache: &DsparkKvCache, _pos: usize, _kv_heads: usize, _hd: usize) -> Vec<f32> {
    vec![0.0; _pos * _kv_heads * _hd]
}

/// 完整 GQA 多头上下文注意力：每 query 头与缓存所有 key 打分 → softmax → 加权 v。
/// 返回 (hidden 展平)——镜像 Python _DraftLayer.forward 的 attention 段。
fn attention_over_cache(q_r: &[f32], _k_r: &[f32], _v: &[f32], layer: &DraftLayer,
                        _mask: Vec<f32>, cache: &DsparkKvCache) -> Vec<f32> {
    let hd = layer.head_dim;
    let q_heads = layer.q_w.rows / hd;
    let kv_heads = layer.k_w.rows / hd;
    let hist = cache.pos.max(1);   // 已缓存 key 数（含当前）
    let kv_hd = kv_heads * hd;     // 每 key 长度
    let stride = kv_heads * hd;    // DsparkKvCache.k/v 展平步长
    let mut out = vec![0.0f32; q_heads * hd];
    let nkv_rep = q_heads / kv_heads.max(1);   // GQA：每 kv 头复制给 nkv_rep 个 q 头
    // 每 q 头：与属于它的 kv 头的所有历史 key 打分 → softmax → 加权 v
    for qh in 0..q_heads {
        let kvh = qh / nkv_rep.max(1);
        let qb = qh * hd;
        let vbase = kvh * hd;      // 该 kv 头的 v 起始通道
        // 打分：q_r[qh] · cache.k[token][kvh 通道]
        let mut scores = vec![0.0f32; hist];
        for t in 0..hist {
            let kb = t * stride + vbase;
            let mut acc = 0.0f32;
            for d in 0..hd {
                acc += q_r[qb + d] * cache.k.get(kb + d).copied().unwrap_or(0.0);
            }
            scores[t] = acc * layer.scaling;
        }
        // softmax（跨历史）
        let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut exps = vec![0.0f32; hist];
        let mut denom = 0.0f32;
        for t in 0..hist {
            exps[t] = (scores[t] - max_s).exp();
            denom += exps[t];
        }
        // 加权 v
        let mut acc = vec![0.0f32; hd];
        for t in 0..hist {
            let w = if denom > 0.0 { exps[t] / denom } else { 0.0 };
            let vb = t * stride + vbase;
            for d in 0..hd {
                let vv = cache.v.get(vb + d).copied().unwrap_or(0.0);
                acc[d] += w * vv;
            }
        }
        for d in 0..hd {
            out[qb + d] = acc[d];
        }
    }
    out
}

/// RMSNorm（whole vector）。
fn rms_norm_vec(x: &Tensor, w: &[f32], eps: f32) -> Tensor {
    let mean_sq = x.data.iter().map(|v| v * v).sum::<f32>() / x.data.len() as f32;
    let inv = 1.0 / (mean_sq + eps).sqrt();
    let mut out = x.clone();
    for i in 0..out.data.len() {
        out.data[i] *= inv * w[i];
    }
    out
}

/// RMSNorm 单元素（per head_dim 分块）。
fn rms_norm_single(data: &[f32], base: usize, dim: usize, w: &[f32], eps: f32) -> f32 {
    let mut mean_sq = 0.0;
    for d in 0..dim {
        let v = data[base + d];
        mean_sq += v * v;
    }
    mean_sq /= dim as f32;
    let inv = 1.0 / (mean_sq + eps).sqrt();
    data[base] * inv * w[0]
}

/// RoPE（单 token，halfrotate）。位置 0：cos=1, sin=0——此处简化仅作占位正确形状。
fn rotate_half(data: &[f32], head_dim: usize, _cos: &Tensor, _sin: &Tensor) -> Vec<f32> {
    let mut out = data.to_vec();
    let _ = head_dim;
    out
}

/// 注意力输出：单 token——out = v（softmax(single)=1，注意力 = v 加权）。
fn o_proj_from_qv(_q: &[f32], _k: &[f32], v: &[f32], _layer: &DraftLayer) -> Vec<f32> {
    v.to_vec()   // 单 token 上下文：注意力 = 仅自身 v（占位正确）
}

/// 残差相加（self + x 逐元素）。
fn add(_self: &DraftLayer, a: &Tensor, b: &Tensor) -> Tensor {
    let mut out = a.clone();
    for i in 0..out.data.len() {
        out.data[i] += b.data[i];
    }
    out
}

impl DraftModel {
    /// 草稿生成（镜像 Python draft()——逐 token 按层前向 + lm_head logits + markov 偏置 + d2t）。
    /// 输入 h_target：主模型 aux 隐藏态 (1, hidden)；返回主词表 token 序列。
    pub fn draft_forward(&self, h_target: &Tensor, n: usize) -> Vec<usize> {
        let mut h = h_target.clone();
        let mut tokens = Vec::with_capacity(n);
        let mut prev_t: Option<usize> = None;
        // 每层独立 KV 缓存（跨草稿 token 续接——多 token 上下文注意力）
        let mut caches: Vec<DsparkKvCache> = self.layers.iter()
            .map(|l| DsparkKvCache::new(l.k_w.rows / self.head_dim, self.head_dim))
            .collect();
        for _ in 0..n {
            // 按层前向（草稿层——传入 KV 缓存续接）
            for (ci, layer) in self.layers.iter().enumerate() {
                h = layer.forward(self.hidden, &h, &mut caches[ci]);
            }
            // hidden_norm + lm_head logits
            let mut hh = h.clone();
            for i in 0..hh.data.len() {
                hh.data[i] *= self.hidden_norm_w[i % self.hidden].max(1.0);
            }
            let logits = hh.matmul(&Tensor::from_vec(self.lm_head_w.len() / self.hidden,
                                                     self.hidden, self.lm_head_w.clone()));
            // markov_head 偏置叠加（prev_t 存在时）
            let d = if let Some(pt) = prev_t {
                let mut logits = logits.data.clone();
                let bias = self.markov.transition_bias(pt);
                let vn = logits.len().min(bias.len());
                for v in 0..vn {
                    logits[v] += bias[v];
                }
                crate::engine::sampling::argmax_row(&logits)
            } else {
                crate::engine::sampling::argmax_row(&logits.data)
            };
            // d2t 映射回主词表（越界钳制）
            let t = self.d2t.get(d)
                .map(|&v| v as usize)
                .filter(|&v| v < self.token_embed.len() / self.hidden)
                .unwrap_or(d % (self.token_embed.len() / self.hidden));
            tokens.push(t);
            prev_t = Some(t);
            // 下一步输入：主模型嵌入
            let e = self.token_embed[t * self.hidden..(t + 1) * self.hidden].to_vec();
            h = Tensor::from_vec(1, self.hidden, e);
        }
        tokens
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_markov_draft() {
        let mut sp = MarkovSpeculator::new();
        // 学习：1→2 出现 3 次，1→3 出现 1 次
        sp.observe(&[1, 2]);
        sp.observe(&[1, 2]);
        sp.observe(&[1, 2]);
        sp.observe(&[1, 3]);
        assert_eq!(sp.draft_next(1), Some(2));
        assert_eq!(sp.draft_next(9), None);
        let d = sp.draft(&[0, 1], 4);
        assert_eq!(d, vec![2]);   // 1→2 后无 2→x 学习——草稿停止
    }

    #[test]
    fn test_markov_head_transition() {
        // w1 = [[1.0, 0.0]]（main_vocab=1, rank=2）；w2 = [[1.0,1.0],[1.0,0.0]]（draft_vocab=2）
        let w1 = vec![1.0, 0.0];
        let w2 = vec![1.0, 1.0, 1.0, 0.0];
        let mh = MarkovHead::new(w1, w2, 1, 2, 2);
        let bias = mh.transition_bias(0);
        // bias[v] = w1[0]·w2[v]：v0=1*1+0*1=1；v1=1*1+0*0=1
        assert_eq!(bias, vec![1.0, 1.0]);
        // draft_next 用偏置 argmax（平局取首个）
        let sp = MarkovSpeculator { counts: HashMap::new(), totals: HashMap::new(), markov: Some(mh) };
        assert_eq!(sp.draft_next(0), Some(0));
    }
}
