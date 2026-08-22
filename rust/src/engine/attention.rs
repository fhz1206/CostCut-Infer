//! StandardAttention：标准 GQA 注意力（Mixtral / Qwen3-MoE / GLM 风格，无 gate）。
//!
//! q/k/v/o 投影 → 每头 RoPE（前 rope_dim 维）→ 因果注意力 → o_proj。
//! 纯 std 实现：2D 张量 + 每头循环（后续 M4 可并行化头维度）。
use crate::core::norm::rms_norm;
use crate::core::rope::{apply_rotary_rows, apply_rotary_rows_last};
use crate::core::tensor::Tensor;

/// 上三角因果掩码 (L, L)：j > i 处为 -inf。
pub fn causal_mask(len: usize) -> Tensor {
    let mut data = vec![0.0f32; len * len];
    for i in 0..len {
        for j in (i + 1)..len {
            data[i * len + j] = f32::NEG_INFINITY;
        }
    }
    Tensor::from_vec(len, len, data)
}

/// 从 (rows, cols_total) 张量中提取第 h 个头（列偏移 h*cols，宽 cols）。
fn slice_head(t: &Tensor, h: usize, cols: usize) -> Tensor {
    let mut data = vec![0.0f32; t.rows * cols];
    for i in 0..t.rows {
        for j in 0..cols {
            data[i * cols + j] = t.get(i, h * cols + j);
        }
    }
    Tensor::from_vec(t.rows, cols, data)
}

/// 标准 GQA 注意力。
pub struct StandardAttention {
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub rope_dim: usize,
    pub scaling: f32,
    pub q_w: Tensor,    // (num_heads*head_dim, hidden)
    pub k_w: Tensor,    // (num_kv_heads*head_dim, hidden)
    pub v_w: Tensor,    // (num_kv_heads*head_dim, hidden)
    pub o_w: Tensor,    // (hidden, num_heads*head_dim)
}

impl StandardAttention {
    /// prefill：x (L, hidden)；cos/sin (L, rope_dim)；mask (L, L) 或 None。返回 (L, hidden)。
    pub fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor,
                   mask: Option<&Tensor>) -> Tensor {
        self.forward_kv(x, cos, sin, mask).0
    }

    /// prefill 且返回每层 k/v（供缓存填充）：(out (L, hidden), k (L, H*hd), v (L, kvh*hd))。
    pub fn forward_kv(&self, x: &Tensor, cos: &Tensor, sin: &Tensor,
                      mask: Option<&Tensor>) -> (Tensor, Tensor, Tensor) {
        // q/k/v/o 权重为 [out, in] 约定：线性 = x @ w.T
        let q = x.matmul(&self.q_w.transpose());
        let k = x.matmul(&self.k_w.transpose());
        let v = x.matmul(&self.v_w.transpose());
        let out = self.attend(&q, &k, &v, cos, sin, mask, false);
        (out, k, v)
    }

    /// decode：x (1, hidden)；k_prev/v_prev 为缓存（(ctx, ...)）。返回 (out, new_k, new_v)。
    pub fn decode(&self, x: &Tensor, cos: &Tensor, sin: &Tensor,
                  k_prev: &Tensor, v_prev: &Tensor) -> (Tensor, Tensor, Tensor) {
        let k = x.matmul(&self.k_w.transpose());
        let v = x.matmul(&self.v_w.transpose());
        let k_all = crate::engine::cache::concat_rows(k_prev, &k);
        let v_all = crate::engine::cache::concat_rows(v_prev, &v);
        let q = x.matmul(&self.q_w.transpose());
        let out = self.attend(&q, &k_all, &v_all, cos, sin, None, true);
        (out, k_all, v_all)
    }

    /// 内部注意力（q/k/v 已投影）：每头 RoPE → 打分 → softmax → @v → o_proj。
    /// `q_single`：decode 时 q 为单行（用 cos/sin 最后一行），prefill 时逐行。
    fn attend(&self, q: &Tensor, k: &Tensor, v: &Tensor, cos: &Tensor, sin: &Tensor,
              mask: Option<&Tensor>, q_single: bool) -> Tensor {
        let rep = self.num_heads / self.num_kv_heads;
        let rows = q.rows;
        let mut outs = vec![0.0f32; rows * self.num_heads * self.head_dim];
        for h in 0..self.num_heads {
            let kh = h / rep;
            let qh = if q_single {
                apply_rotary_rows_last(&slice_head(q, h, self.head_dim), cos, sin, self.rope_dim)
            } else {
                apply_rotary_rows(&slice_head(q, h, self.head_dim), cos, sin, self.rope_dim)
            };
            let kh_r = apply_rotary_rows(&slice_head(k, kh, self.head_dim), cos, sin, self.rope_dim);
            let vh = slice_head(v, kh, self.head_dim);
            let scores = qh.matmul(&kh_r.transpose()).scale(self.scaling);   // (L, L)
            let scores = match mask {
                Some(m) => scores.add(m),
                None => scores,
            };
            let out_h = scores.softmax_rows().matmul(&vh);                    // (L, hd)
            for i in 0..rows {
                for j in 0..self.head_dim {
                    outs[i * self.num_heads * self.head_dim + h * self.head_dim + j] =
                        out_h.get(i, j);
                }
            }
        }
        let joined = Tensor::from_vec(rows, self.num_heads * self.head_dim, outs);
        joined.matmul(&self.o_w.transpose())
    }
}

/// FullAttention：Qwen3.5 的 gated full attention（q_proj 输出含 gate、q/k 带 RMSNorm）。
pub struct FullAttention {
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub rope_dim: usize,
    pub scaling: f32,
    pub eps: f32,
    pub q_w: Tensor,          // (2*H*hd, hidden)
    pub k_w: Tensor,          // (kvh*hd, hidden)
    pub v_w: Tensor,          // (kvh*hd, hidden)
    pub o_w: Tensor,          // (hidden, H*hd)
    pub q_norm_w: Vec<f32>,   // (H*hd,)
    pub k_norm_w: Vec<f32>,   // (kvh*hd,)
}

impl FullAttention {
    /// prefill：x (L, hidden)。返回 (L, hidden)。
    pub fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor,
                   mask: Option<&Tensor>) -> Tensor {
        self.forward_kv(x, cos, sin, mask).0
    }

    /// prefill 且返回每层 k/v（供缓存填充）——与 Python FullAttention.forward_kv 对应。
    pub fn forward_kv(&self, x: &Tensor, cos: &Tensor, sin: &Tensor,
                      mask: Option<&Tensor>) -> (Tensor, Tensor, Tensor) {
        println!("[FullAttn] x {}x{} q_w {}x{} k_w {}x{} v_w {}x{}",
                 x.rows, x.cols, self.q_w.rows, self.q_w.cols,
                 self.k_w.rows, self.k_w.cols, self.v_w.rows, self.v_w.cols);
        let qg = x.matmul(&self.q_w.transpose());                      // (L, 2*H*hd)
        let (query, gate) = qg.split_cols(self.num_heads * self.head_dim);
        let query = rms_norm(&query, &self.q_norm_w, self.eps);        // (L, H*hd)
        let k = rms_norm(&x.matmul(&self.k_w.transpose()), &self.k_norm_w, self.eps);
        let v = x.matmul(&self.v_w.transpose());
        let out = self.attend(&query, &k, &v, cos, sin, mask, false);
        // 输出乘 sigmoid(gate) → o_proj
        (out.elementwise_mul(&gate.sigmoid()).matmul(&self.o_w), k, v)
    }

    /// decode：x (1, hidden)；k_prev/v_prev 缓存续接。返回 (out, new_k, new_v)。
    pub fn decode(&self, x: &Tensor, cos: &Tensor, sin: &Tensor,
                  k_prev: &Tensor, v_prev: &Tensor) -> (Tensor, Tensor, Tensor) {
        let qg = x.matmul(&self.q_w.transpose());
        let (query, gate) = qg.split_cols(self.num_heads * self.head_dim);
        let query = rms_norm(&query, &self.q_norm_w, self.eps);
        let k = rms_norm(&x.matmul(&self.k_w.transpose()), &self.k_norm_w, self.eps);
        let v = x.matmul(&self.v_w.transpose());
        let k_all = crate::engine::cache::concat_rows(k_prev, &k);
        let v_all = crate::engine::cache::concat_rows(v_prev, &v);
        let out = self.attend(&query, &k_all, &v_all, cos, sin, None, true);
        (out.elementwise_mul(&gate.sigmoid()).matmul(&self.o_w), k_all, v_all)
    }

    /// 内部注意力（query/k/v 已归一化）：每头打分（与 StandardAttention 相同）。
    /// `q_single`：decode 时 q 为单行（用 cos/sin 最后一行），prefill 时逐行。
    fn attend(&self, q: &Tensor, k: &Tensor, v: &Tensor, cos: &Tensor, sin: &Tensor,
              mask: Option<&Tensor>, q_single: bool) -> Tensor {
        let rep = self.num_heads / self.num_kv_heads;
        let rows = q.rows;
        let mut outs = vec![0.0f32; rows * self.num_heads * self.head_dim];
        for h in 0..self.num_heads {
            let kh = h / rep;
            let qh = if q_single {
                apply_rotary_rows_last(&slice_head(q, h, self.head_dim), cos, sin, self.rope_dim)
            } else {
                apply_rotary_rows(&slice_head(q, h, self.head_dim), cos, sin, self.rope_dim)
            };
            let kh_r = apply_rotary_rows(&slice_head(k, kh, self.head_dim), cos, sin, self.rope_dim);
            let vh = slice_head(v, kh, self.head_dim);
            let scores = qh.matmul(&kh_r.transpose()).scale(self.scaling);
            let scores = match mask {
                Some(m) => scores.add(m),
                None => scores,
            };
            let out_h = scores.softmax_rows().matmul(&vh);
            for i in 0..rows {
                for j in 0..self.head_dim {
                    outs[i * self.num_heads * self.head_dim + h * self.head_dim + j] =
                        out_h.get(i, j);
                }
            }
        }
        Tensor::from_vec(rows, self.num_heads * self.head_dim, outs)
    }
}

/// 递归版 gated delta rule（镜像 Python recurrent_gated_delta_rule）——decode 单步。
///
/// q/k: (1, H*kd)，v: (1, H*vd)，g/beta: (1, H)；prev_state: (H*kd*vd) 或 None（零初始）。
/// 状态递推：``last = last*g.exp() + k ⊗ ((v - kv_mem)*beta)``；输出 ``core = sum_kd last*q``。
pub fn recurrent_delta_rule(q: &Tensor, k: &Tensor, v: &Tensor, g: &Tensor, beta: &Tensor,
                            prev_state: Option<&[f32]>, h: usize, kd: usize, vd: usize)
                            -> (Tensor, Vec<f32>) {
    assert_eq!(k.cols, h * kd);
    let mut state: Vec<f32> = match prev_state {
        Some(s) => s.to_vec(),
        None => vec![0.0f32; h * kd * vd],
    };
    let mut core = vec![0.0f32; h * vd];
    let scale = 1.0 / (kd as f32).sqrt();
    for hh in 0..h {
        // l2norm q/k（每头）
        let mut qn = 0.0f32;
        let mut kn = 0.0f32;
        for kk in 0..kd {
            qn += q.get(0, hh * kd + kk).powi(2);
            kn += k.get(0, hh * kd + kk).powi(2);
        }
        let (qinv, kinv) = (1.0 / qn.max(1e-12).sqrt(), 1.0 / kn.max(1e-12).sqrt());
        let st = &mut state[hh * kd * vd..(hh + 1) * kd * vd];
        let gt = g.get(0, hh).exp();
        let betat = beta.get(0, hh);
        // 先衰减（Python：last = last * g_t）——kv_mem 须用衰减后的状态
        for x in st.iter_mut() {
            *x *= gt;
        }
        // kv_mem (vd) = sum_kd last[kd, vd] * k[kd]（衰减后的 last）
        let mut kv_mem = vec![0.0f32; vd];
        for kk in 0..kd {
            let kk_v = k.get(0, hh * kd + kk) * kinv;
            for vv in 0..vd {
                kv_mem[vv] += st[kk * vd + vv] * kk_v;
            }
        }
        // delta = (v - kv_mem) * beta；last += k ⊗ delta（已衰减——不再乘 gt）
        for kk in 0..kd {
            let kk_v = k.get(0, hh * kd + kk) * kinv;
            for vv in 0..vd {
                let idx = kk * vd + vv;
                st[idx] += kk_v * ((v.get(0, hh * vd + vv) - kv_mem[vv]) * betat);
            }
        }
        // core (vd) = sum_kd last[kd, vd] * q[kd] * scale
        for vv in 0..vd {
            let mut acc = 0.0f32;
            for kk in 0..kd {
                acc += st[kk * vd + vv] * q.get(0, hh * kd + kk) * qinv;
            }
            core[hh * vd + vv] = acc * scale;
        }
    }
    (Tensor::from_vec(1, h * vd, core), state)
}

/// Chunk 版 gated delta rule（镜像 Python chunk_gated_delta_rule）——prefill。
/// q/k: (L, H*kd)，v: (L, H*vd)，g/beta: (L, H)；chunk_size = 64。
/// 返回 (L, H*vd) 输出（与逐 token recurrent 数学等价——chunk 内并行）。
pub fn chunk_delta_rule(q: &Tensor, k: &Tensor, v: &Tensor, g: &Tensor, beta: &Tensor,
                        h: usize, kd: usize, vd: usize) -> (Tensor, Vec<f32>) {
    let l = q.rows;
    let cs = 64usize;
    let pad = (cs - l % cs) % cs;
    let total = l + pad;
    let nc = total / cs;
    let scale = 1.0 / (kd as f32).sqrt();
    // l2norm 每 (行, 头) + 缩放（存到临时数组；pad 部分为 0——不影响输出）
    let mut qq = vec![0.0f32; total * h * kd];
    let mut kk2 = vec![0.0f32; total * h * kd];
    for i in 0..l {
        for hh in 0..h {
            let (mut qn, mut kn) = (0.0f32, 0.0f32);
            for d in 0..kd {
                qn += q.get(i, hh * kd + d).powi(2);
                kn += k.get(i, hh * kd + d).powi(2);
            }
            let (qi, ki) = (1.0 / qn.max(1e-12).sqrt(), 1.0 / kn.max(1e-12).sqrt());
            for d in 0..kd {
                qq[i * h * kd + hh * kd + d] = q.get(i, hh * kd + d) * qi * scale;
                kk2[i * h * kd + hh * kd + d] = k.get(i, hh * kd + d) * ki;
            }
        }
    }
    // v_beta / k_beta（pad 为 0）
    let mut vv = vec![0.0f32; total * h * vd];
    let mut kb = vec![0.0f32; total * h * kd];
    for i in 0..l {
        for hh in 0..h {
            let b = beta.get(i, hh);
            for d in 0..vd { vv[i * h * vd + hh * vd + d] = v.get(i, hh * vd + d) * b; }
            for d in 0..kd { kb[i * h * kd + hh * kd + d] = kk2[i * h * kd + hh * kd + d] * b; }
        }
    }
    let mut state = vec![0.0f32; h * kd * vd];
    let mut core = vec![0.0f32; total * h * vd];
    // 逐 chunk
    for c in 0..nc {
        for hh in 0..h {
            // chunk 内 g 累积 + 衰减掩码（tril(exp(g_i - g_j))）
            let mut gcum = vec![0.0f32; cs];
            let mut acc = 0.0f32;
            for t in 0..cs { acc += g.get(c * cs + t, hh); gcum[t] = acc; }
            let mut decay = vec![vec![0.0f32; cs]; cs];
            let mut attn_intra = vec![vec![0.0f32; cs]; cs];
            for i in 0..cs {
                for j in 0..cs {
                    decay[i][j] = if i >= j { (gcum[i] - gcum[j]).exp() } else { 0.0 };
                }
                // 严格下三角（不含对角线——Python masked_fill 上三角含对角线为 0，再加 eye）
                for j in 0..i {
                    let mut s = 0.0f32;
                    for d in 0..kd {
                        s += kb[(c * cs + i) * h * kd + hh * kd + d]
                            * kk2[(c * cs + j) * h * kd + hh * kd + d];
                    }
                    attn_intra[i][j] = -s * decay[i][j];
                }
            }
            // 行递推修正：attn[i, :i] += row_orig*(sub)（Python 用克隆的原始 row）
            for i in 1..cs {
                let row_orig: Vec<f32> = (0..i).map(|t| attn_intra[i][t]).collect();
                for j in 0..i {
                    let mut s = 0.0f32;
                    for t in 0..i {
                        s += row_orig[t] * attn_intra[t][j];
                    }
                    attn_intra[i][j] += s;
                }
            }
            for i in 0..cs { attn_intra[i][i] += 1.0; }
            // value = attn @ v_beta；k_cumdecay = attn @ (k_beta * g.exp())
            let mut val = vec![0.0f32; cs * vd];
            let mut kcum = vec![0.0f32; cs * kd];
            for i in 0..cs {
                for d in 0..vd {
                    let mut s = 0.0f32;
                    for t in 0..cs { s += attn_intra[i][t] * vv[(c * cs + t) * h * vd + hh * vd + d]; }
                    val[i * vd + d] = s;
                }
                for d in 0..kd {
                    let mut s = 0.0f32;
                    for t in 0..cs {
                        s += attn_intra[i][t] * kb[(c * cs + t) * h * kd + hh * kd + d] * gcum[t].exp();
                    }
                    kcum[i * kd + d] = s;
                }
            }
            // 逐 chunk 递推（c==0 时 state 为零初始——真实续接用 initial_state，此处合成路径）
            let mut qrow = vec![0.0f32; cs * kd];
            let mut krow = vec![0.0f32; cs * kd];
            let mut grow = vec![0.0f32; cs];
            for t in 0..cs {
                grow[t] = gcum[t].exp();
                for d in 0..kd {
                    qrow[t * kd + d] = qq[(c * cs + t) * h * kd + hh * kd + d];
                    krow[t * kd + d] = kk2[(c * cs + t) * h * kd + hh * kd + d];
                }
            }
            // v_new = val - kcum @ state；core = (q*grow)@state + q@(decay-掩码)@v_new；state 更新
            let mut v_new = vec![0.0f32; cs * vd];
            for t in 0..cs {
                for d in 0..vd {
                    let mut s = 0.0f32;
                    for kk in 0..kd { s += kcum[t * kd + kk] * state[hh * kd * vd + kk * vd + d]; }
                    v_new[t * vd + d] = val[t * vd + d] - s;
                }
            }
            let mut out = vec![0.0f32; cs * vd];
            for t in 0..cs {
                for d in 0..vd {
                    let mut s1 = 0.0f32;
                    for kk in 0..kd { s1 += qrow[t * kd + kk] * grow[t] * state[hh * kd * vd + kk * vd + d]; }
                    let mut s2 = 0.0f32;
                    for u in 0..cs {
                        for kk in 0..kd {
                            s2 += qrow[t * kd + kk] * decay[t][u] * krow[u * kd + kk] * v_new[u * vd + d];
                        }
                    }
                    out[t * vd + d] = s1 + s2;
                }
            }
            // last = last*g[-1].exp() + (k*(g[-1]-g).exp())^T @ v_new
            let g_last = gcum[cs - 1].exp();
            let mut new_state = vec![0.0f32; kd * vd];
            for kk in 0..kd {
                for d in 0..vd {
                    let mut s = 0.0f32;
                    for t in 0..cs {
                        s += krow[t * kd + kk] * (g_last - gcum[t]).exp() * v_new[t * vd + d];
                    }
                    new_state[kk * vd + d] = state[hh * kd * vd + kk * vd + d] * g_last + s;
                }
            }
            for i in 0..kd * vd { state[hh * kd * vd + i] = new_state[i]; }
            for t in 0..cs {
                for d in 0..vd { core[(c * cs + t) * h * vd + hh * vd + d] = out[t * vd + d]; }
            }
        }
    }
    // 截断 pad（手动行截断——Tensor 无 split_rows）
    let out = Tensor::from_vec(total, h * vd, core);
    let keep = Tensor::from_vec(l, h * vd, out.data[..l * h * vd].to_vec());
    (keep, state)
}
/// q/k/v 经低秩投影压缩（q_a→q_b / kv_a→kv_b），RoPE 仅用于解耦的 rope 部分，
/// 注意力分数 = q_nope@k_nope + q_rope@k_rope。
/// 线性层（向量）：out = x @ w^T（x (1, hidden) → out (out_dim)）。
fn linear_vec(x: &Tensor, w: &Tensor, out_dim: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; out_dim];
    for i in 0..out_dim {
        let mut acc = 0.0f32;
        for k in 0..x.cols {
            acc += x.get(0, k) * w.get(i, k);
        }
        out[i] = acc;
    }
    out
}

/// GatedDeltaNet：Qwen3.5 的线性注意力层（conv1d + in_proj + delta rule + gated norm）。
///
/// decode（forward_step）：causal conv1d 续接 + q/k/v/z/b/a 投影 → beta/g → recurrent
/// delta rule → 输出乘 silu(z)（gated norm）→ out_proj。
/// 注：prefill（forward）当前走逐 token recurrent（与 chunk 数学等价——chunk 加速为后续）。
pub struct GatedDeltaNet {
    pub hidden: usize,
    pub key_dim: usize,
    pub value_dim: usize,
    pub num_k_heads: usize,
    pub num_v_heads: usize,
    pub head_k_dim: usize,
    pub head_v_dim: usize,
    pub eps: f32,
    pub in_proj_qkv: Tensor,   // (key_dim*2+value_dim, hidden)
    pub in_proj_z: Tensor,
    pub in_proj_b: Tensor,
    pub in_proj_a: Tensor,
    pub conv_w: Vec<f32>,      // (C, kernel)
    pub out_w: Tensor,
    pub norm_w: Vec<f32>,
    pub a_log: f32,
    pub dt_bias: Vec<f32>,
}

impl GatedDeltaNet {
    /// decode 单步：x (1, hidden)；conv_state 续接；rec_state (h_k*kd*vd)。
    pub fn forward_step(&self, x: &Tensor, conv_state: Option<&[f32]>,
                        rec_state: Option<&[f32]>) -> (Tensor, Vec<f32>, Vec<f32>) {
        let t0 = std::time::Instant::now();
        println!("[fs0] enter kd={} c={}", self.key_dim, self.key_dim * 2 + self.value_dim);
        let c = self.key_dim * 2 + self.value_dim;
        let kernel = self.conv_w.len() / c;
        // 因果 conv1d（single token：kernel 个窗口，不足补零）
        let have = kernel - 1;
        let mut conv_in = vec![0.0f32; c * kernel];
        if let Some(st) = conv_state {
            for t in 0..have {
                for ch in 0..c {
                    conv_in[t * c + ch] = st[t * c + ch];
                }
            }
        }
        println!("[fs] conv {:.3}s", t0.elapsed().as_secs_f64());
        let qkv_pre = linear_vec(x, &self.in_proj_qkv, c);   // (c)
        for ch in 0..c {
            conv_in[(kernel - 1) * c + ch] = qkv_pre[ch];
        }
        println!("[fs] qkv {:.3}s", t0.elapsed().as_secs_f64());
        // conv1d + silu
        let mut qkv = vec![0.0f32; c];
        for ch in 0..c {
            let mut acc = 0.0f32;
            for kk in 0..kernel {
                acc += conv_in[kk * c + ch] * self.conv_w[ch * kernel + kk];
            }
            qkv[ch] = acc / (1.0 + (-acc).exp());
        }
        // 新 conv 状态 = pre-conv 序列末尾 kernel-1 个值（滑动窗口）
        let mut new_conv = vec![0.0f32; have * c];
        for t in 0..have {
            let idx = if t < have { (have - 1 - t) * c } else { 0 };
            for ch in 0..c {
                new_conv[t * c + ch] = conv_in[idx + ch];
            }
        }
        // 拆分 q/k/v
        let kd = self.key_dim / self.num_k_heads;
        let vd = self.value_dim / self.num_v_heads;
        let (mut q, mut k, mut v) = (vec![0.0f32; self.key_dim], vec![0.0f32; self.key_dim],
                                     vec![0.0f32; self.value_dim]);
        for i in 0..self.key_dim { q[i] = qkv[i]; k[i] = qkv[self.key_dim + i]; }
        for i in 0..self.value_dim { v[i] = qkv[self.key_dim * 2 + i]; }
        // z/b/a → beta/g
        let z = linear_vec(x, &self.in_proj_z, self.value_dim);
        let b = linear_vec(x, &self.in_proj_b, self.key_dim);
        let a = linear_vec(x, &self.in_proj_a, self.key_dim);
        let mut beta = vec![0.0f32; self.key_dim];
        let mut g = vec![0.0f32; self.key_dim];
        for i in 0..self.key_dim {
            beta[i] = 1.0 / (1.0 + (-b[i]).exp());
            let sp = (a[i] + self.dt_bias[i]).max(0.0).ln_1p();
            g[i] = -self.a_log.exp() * sp;
        }
        // recurrent delta rule（每头）
        let qt = Tensor::from_vec(1, self.key_dim, q);
        let kt = Tensor::from_vec(1, self.key_dim, k);
        let vt = Tensor::from_vec(1, self.value_dim, v);
        let gt = Tensor::from_vec(1, self.key_dim, g);
        let bt = Tensor::from_vec(1, self.key_dim, beta);
        println!("[fs] pre-rec {:.3}s", t0.elapsed().as_secs_f64());
        let (core, new_rec) = recurrent_delta_rule(
            &qt, &kt, &vt, &gt, &bt, rec_state, self.num_k_heads, kd, vd);
        // gated norm：core * silu(z) → out_proj
        let mut gated = vec![0.0f32; self.value_dim];
        for i in 0..self.value_dim {
            gated[i] = core.get(0, i) * (z[i] / (1.0 + (-z[i]).exp()));
        }
        let out = linear_vec(&Tensor::from_vec(1, self.value_dim, gated), &self.out_w, self.hidden);
        println!("[fs] 耗时 {:.3}s", t0.elapsed().as_secs_f64());
        (Tensor::from_vec(1, self.hidden, out), new_conv, new_rec)
    }

    /// prefill：逐 token recurrent（与 chunk 数学等价；chunk 加速为后续）。
    pub fn forward(&self, x: &Tensor) -> Tensor {
        println!("[GDN] forward enter rows={}", x.rows);
        let l = x.rows;
        let mut conv: Option<Vec<f32>> = None;
        let mut rec: Option<Vec<f32>> = None;
        let mut outs = vec![0.0f32; l * self.hidden];
        for t in 0..l {
            let xt = Tensor::from_vec(1, self.hidden,
                (0..self.hidden).map(|d| x.get(t, d)).collect());
            let (o, nc, nr) = self.forward_step(&xt, conv.as_deref(), rec.as_deref());
            conv = Some(nc);
            rec = Some(nr);
            for d in 0..self.hidden { outs[t * self.hidden + d] = o.get(0, d); }
        }
        Tensor::from_vec(l, self.hidden, outs)
    }
}

pub struct MlaAttention {
    pub num_heads: usize,
    pub kv_lora_rank: usize,
    pub q_lora_rank: usize,
    pub qk_rope_head_dim: usize,
    pub v_head_dim: usize,
    pub qk_nope_head_dim: usize,
    pub q_head_dim: usize,
    pub eps: f32,
    pub scaling: f32,
    pub q_a_w: Tensor,          // (q_lora, hidden)
    pub q_b_w: Tensor,          // (H*qh, q_lora)
    pub kv_a_w: Tensor,         // (kv_lora+rope, hidden)
    pub kv_b_w: Tensor,         // (H*(nope+v), kv_lora)
    pub o_w: Tensor,            // (hidden, H*v_head)
    pub q_norm_w: Vec<f32>,     // (q_lora,)
    pub k_norm_w: Vec<f32>,     // (rope,)
}

impl MlaAttention {
    /// prefill：x (L, hidden)。返回 (L, hidden)。
    pub fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor,
                   mask: Option<&Tensor>) -> Tensor {
        self.forward_kv(x, cos, sin, mask).0
    }

    /// prefill 且返回每层 k/v（供缓存填充）：k = [k_nope | k_rope] 列拼接，v = v 部分。
    pub fn forward_kv(&self, x: &Tensor, cos: &Tensor, sin: &Tensor,
                      mask: Option<&Tensor>) -> (Tensor, Tensor, Tensor) {
        let q_latent = rms_norm(&x.matmul(&self.q_a_w.transpose()), &self.q_norm_w, self.eps);
        let q = q_latent.matmul(&self.q_b_w.transpose());               // (L, H*qh)
        let kv = x.matmul(&self.kv_a_w.transpose());                    // (L, kv_lora+rope)
        let (kv_latent, k_rope) = kv.split_cols(self.kv_lora_rank);
        let k_rope = rms_norm(&k_rope, &self.k_norm_w, self.eps);
        let kv_out = kv_latent.matmul(&self.kv_b_w.transpose());        // (L, H*(nope+v))
        let mut outs = vec![0.0f32; x.rows * self.num_heads * self.v_head_dim];
        for h in 0..self.num_heads {
            let qh = slice_head(&q, h, self.q_head_dim);
            let (q_nope, q_rope) = qh.split_cols(self.qk_nope_head_dim);
            let q_rope_r = apply_rotary_rows(&q_rope, cos, sin, self.qk_rope_head_dim);
            let hkv = slice_head(&kv_out, h, self.qk_nope_head_dim + self.v_head_dim);
            let (k_nope, vh) = hkv.split_cols(self.qk_nope_head_dim);
            let k_rope_r = apply_rotary_rows(&k_rope, cos, sin, self.qk_rope_head_dim);
            let scores = q_nope.matmul(&k_nope.transpose())
                .add(&q_rope_r.matmul(&k_rope_r.transpose()))
                .scale(self.scaling);
            let scores = match mask {
                Some(m) => scores.add(m),
                None => scores,
            };
            let out_h = scores.softmax_rows().matmul(&vh);
            for i in 0..x.rows {
                for j in 0..self.v_head_dim {
                    outs[i * self.num_heads * self.v_head_dim + h * self.v_head_dim + j] =
                        out_h.get(i, j);
                }
            }
        }
        let out = Tensor::from_vec(x.rows, self.num_heads * self.v_head_dim, outs)
            .matmul(&self.o_w.transpose());
        let (k_nope_all, v_all) = kv_out.split_cols(self.num_heads * self.qk_nope_head_dim);
        (out, k_nope_all.concat_cols(&k_rope), v_all)
    }

    /// decode：x (1, hidden)；k_prev/v_prev 缓存续接（k = [k_nope | k_rope]）。
    pub fn decode(&self, x: &Tensor, cos: &Tensor, sin: &Tensor,
                  k_prev: &Tensor, v_prev: &Tensor) -> (Tensor, Tensor, Tensor) {
        let q_latent = rms_norm(&x.matmul(&self.q_a_w.transpose()), &self.q_norm_w, self.eps);
        let q = q_latent.matmul(&self.q_b_w.transpose());
        let kv = x.matmul(&self.kv_a_w.transpose());
        let (kv_latent, k_rope_new) = kv.split_cols(self.kv_lora_rank);
        let k_rope_new = rms_norm(&k_rope_new, &self.k_norm_w, self.eps);
        let kv_out = kv_latent.matmul(&self.kv_b_w.transpose());
        let (k_nope_new, v_new) = kv_out.split_cols(self.num_heads * self.qk_nope_head_dim);
        let k_new = k_nope_new.concat_cols(&k_rope_new);
        let k_all = crate::engine::cache::concat_rows(k_prev, &k_new);
        let v_all = crate::engine::cache::concat_rows(v_prev, &v_new);
        // 每头解耦打分：q_nope vs k_nope（前 H*nope 列）+ q_rope vs 共享 k_rope（最后 rope 列）
        let rope_start = self.num_heads * self.qk_nope_head_dim;
        let (_, k_rope_slice) = k_all.split_cols(rope_start);
        let k_rope_r = apply_rotary_rows(&k_rope_slice, cos, sin, self.qk_rope_head_dim);
        let mut outs = vec![0.0f32; self.num_heads * self.v_head_dim];
        for h in 0..self.num_heads {
            let qh = slice_head(&q, h, self.q_head_dim);
            let (q_nope, q_rope) = qh.split_cols(self.qk_nope_head_dim);
            let q_rope_r = apply_rotary_rows_last(&q_rope, cos, sin, self.qk_rope_head_dim);
            let k_nope = slice_head(&k_all, h, self.qk_nope_head_dim);
            let vh = slice_head(&v_all, h, self.v_head_dim);
            let scores = q_nope.matmul(&k_nope.transpose())
                .add(&q_rope_r.matmul(&k_rope_r.transpose()))
                .scale(self.scaling);
            let out_h = scores.softmax_rows().matmul(&vh);
            for j in 0..self.v_head_dim {
                outs[h * self.v_head_dim + j] = out_h.get(0, j);
            }
        }
        (Tensor::from_vec(1, self.num_heads * self.v_head_dim, outs)
            .matmul(&self.o_w.transpose()), k_all, v_all)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attention_shape() {
        let (hidden, h, kvh, hd) = (16usize, 4usize, 2usize, 4usize);
        let attn = StandardAttention {
            num_heads: h,
            num_kv_heads: kvh,
            head_dim: hd,
            rope_dim: 4,
            scaling: 0.5,
            q_w: Tensor::from_vec(h * hd, hidden, vec![0.1; h * hd * hidden]),
            k_w: Tensor::from_vec(kvh * hd, hidden, vec![0.1; kvh * hd * hidden]),
            v_w: Tensor::from_vec(kvh * hd, hidden, vec![0.1; kvh * hd * hidden]),
            o_w: Tensor::from_vec(hidden, h * hd, vec![0.1; hidden * h * hd]),
        };
        let x = Tensor::from_vec(4, hidden, vec![0.5; 4 * hidden]);
        let cos = Tensor::from_vec(4, hd, vec![1.0; 4 * hd]);
        let sin = Tensor::from_vec(4, hd, vec![0.0; 4 * hd]);
        let out = attn.forward(&x, &cos, &sin, Some(&causal_mask(4)));
        assert_eq!((out.rows, out.cols), (4, hidden));
        assert!(out.max_abs().is_finite());
    }

    #[test]
    fn test_full_attention_shape() {
        let (hidden, h, kvh, hd) = (16usize, 4usize, 2usize, 4usize);
        let attn = FullAttention {
            num_heads: h, num_kv_heads: kvh, head_dim: hd, rope_dim: 4,
            scaling: 0.5, eps: 1e-6,
            q_w: Tensor::from_vec(2 * h * hd, hidden, vec![0.1; 2 * h * hd * hidden]),
            k_w: Tensor::from_vec(kvh * hd, hidden, vec![0.1; kvh * hd * hidden]),
            v_w: Tensor::from_vec(kvh * hd, hidden, vec![0.1; kvh * hd * hidden]),
            o_w: Tensor::from_vec(hidden, h * hd, vec![0.1; hidden * h * hd]),
            q_norm_w: vec![1.0; h * hd],
            k_norm_w: vec![1.0; kvh * hd],
        };
        let x = Tensor::from_vec(4, hidden, vec![0.5; 4 * hidden]);
        let cos = Tensor::from_vec(4, hd, vec![1.0; 4 * hd]);
        let sin = Tensor::from_vec(4, hd, vec![0.0; 4 * hd]);
        let out = attn.forward(&x, &cos, &sin, Some(&causal_mask(4)));
        assert_eq!((out.rows, out.cols), (4, hidden));
        assert!(out.max_abs().is_finite());
    }

    #[test]
    fn test_mla_attention_shape() {
        let (hidden, h) = (32usize, 4usize);
        let (kv_l, q_l, rope_d, v_d) = (16usize, 16usize, 8usize, 16usize);
        let nope = v_d;
        let qh = nope + rope_d;
        let attn = MlaAttention {
            num_heads: h, kv_lora_rank: kv_l, q_lora_rank: q_l,
            qk_rope_head_dim: rope_d, v_head_dim: v_d, qk_nope_head_dim: nope, q_head_dim: qh,
            eps: 1e-6, scaling: (nope as f32).powf(-0.5),
            q_a_w: Tensor::from_vec(q_l, hidden, vec![0.1; q_l * hidden]),
            q_b_w: Tensor::from_vec(h * qh, q_l, vec![0.1; h * qh * q_l]),
            kv_a_w: Tensor::from_vec(kv_l + rope_d, hidden, vec![0.1; (kv_l + rope_d) * hidden]),
            kv_b_w: Tensor::from_vec(h * (nope + v_d), kv_l, vec![0.1; h * (nope + v_d) * kv_l]),
            o_w: Tensor::from_vec(hidden, h * v_d, vec![0.1; hidden * h * v_d]),
            q_norm_w: vec![1.0; q_l],
            k_norm_w: vec![1.0; rope_d],
        };
        let x = Tensor::from_vec(4, hidden, vec![0.5; 4 * hidden]);
        let cos = Tensor::from_vec(4, rope_d, vec![1.0; 4 * rope_d]);
        let sin = Tensor::from_vec(4, rope_d, vec![0.0; 4 * rope_d]);
        let out = attn.forward(&x, &cos, &sin, Some(&causal_mask(4)));
        assert_eq!((out.rows, out.cols), (4, hidden));
        assert!(out.max_abs().is_finite());
    }

    #[test]
    fn test_full_attention_decode() {
        // FullAttention.decode：kv 续接 + gate 输出（q 用最后一行位置）
        let (hidden, h, kvh, hd) = (16usize, 4usize, 2usize, 4usize);
        let attn = FullAttention {
            num_heads: h, num_kv_heads: kvh, head_dim: hd, rope_dim: 4,
            scaling: 0.5, eps: 1e-6,
            q_w: Tensor::from_vec(2 * h * hd, hidden, vec![0.1; 2 * h * hd * hidden]),
            k_w: Tensor::from_vec(kvh * hd, hidden, vec![0.1; kvh * hd * hidden]),
            v_w: Tensor::from_vec(kvh * hd, hidden, vec![0.1; kvh * hd * hidden]),
            o_w: Tensor::from_vec(hidden, h * hd, vec![0.1; hidden * h * hd]),
            q_norm_w: vec![1.0; h * hd],
            k_norm_w: vec![1.0; kvh * hd],
        };
        let kv_k = Tensor::from_vec(4, kvh * hd, vec![0.1; 4 * kvh * hd]);
        let kv_v = Tensor::from_vec(4, kvh * hd, vec![0.1; 4 * kvh * hd]);
        let x1 = Tensor::from_vec(1, hidden, vec![0.5; hidden]);
        let cos = Tensor::from_vec(5, hd, vec![1.0; 5 * hd]);    // 全位置（pos+1）
        let sin = Tensor::from_vec(5, hd, vec![0.0; 5 * hd]);
        let (out, k_all, v_all) = attn.decode(&x1, &cos, &sin, &kv_k, &kv_v);
        assert_eq!(out.rows, 1);
        assert_eq!(k_all.rows, 5);
        assert_eq!(v_all.rows, 5);
        assert!(out.max_abs().is_finite());
    }

    #[test]
    fn test_delta_rule_chunk_vs_recurrent() {
        // 合成：L=64（一个 chunk），h=2, kd=4, vd=4——chunk 与逐 token recurrent 数学等价
        let (l, h, kd, vd) = (64usize, 2usize, 4usize, 4usize);
        let q = Tensor::from_vec(l, h * kd,
            (0..l * h * kd).map(|i| ((i % 9) as f32) * 0.05).collect());
        let k = Tensor::from_vec(l, h * kd,
            (0..l * h * kd).map(|i| ((i % 7) as f32) * 0.04).collect());
        let v = Tensor::from_vec(l, h * vd,
            (0..l * h * vd).map(|i| ((i % 5) as f32) * 0.06).collect());
        let g = Tensor::from_vec(l, h,
            (0..l * h).map(|i| ((i % 3) as f32) * -0.01).collect());
        let beta = Tensor::from_vec(l, h, vec![0.8f32; l * h]);
        let (c_out, _cs) = chunk_delta_rule(&q, &k, &v, &g, &beta, h, kd, vd);
        // recurrent 逐 token（state 续接）
        let mut state: Option<Vec<f32>> = None;
        let mut r_out = vec![0.0f32; l * h * vd];
        for i in 0..l {
            let q1 = Tensor::from_vec(1, h * kd, (0..h * kd).map(|d| q.get(i, d)).collect());
            let k1 = Tensor::from_vec(1, h * kd, (0..h * kd).map(|d| k.get(i, d)).collect());
            let v1 = Tensor::from_vec(1, h * vd, (0..h * vd).map(|d| v.get(i, d)).collect());
            let g1 = Tensor::from_vec(1, h, (0..h).map(|d| g.get(i, d)).collect());
            let b1 = Tensor::from_vec(1, h, (0..h).map(|d| beta.get(i, d)).collect());
            let (c, s) = recurrent_delta_rule(&q1, &k1, &v1, &g1, &b1,
                                              state.as_deref(), h, kd, vd);
            state = Some(s);
            for d in 0..h * vd { r_out[i * h * vd + d] = c.get(0, d); }
        }
        // 诊断：逐 token 误差（首 token=chunk 内基座；后续=状态递推）
        let mut per_tok = vec![0.0f32; l];
        for i in 0..l {
            per_tok[i] = (0..h * vd)
                .map(|d| (c_out.data[i * h * vd + d] - r_out[i * h * vd + d]).abs())
                .fold(0.0f32, f32::max);
        }
        let first5: Vec<String> = per_tok[..5].iter().map(|v| format!("{:.2e}", v)).collect();
        let last5: Vec<String> = per_tok[l - 5..].iter().map(|v| format!("{:.2e}", v)).collect();
        let (mi, mv) = per_tok.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap();
        println!("per-token err 前5: {} | 后5: {} | 最大 t={mi} err={mv:.2e}",
                 first5.join(" "), last5.join(" "));
        let max_err = (0..l * h * vd)
            .map(|i| (c_out.data[i] - r_out[i]).abs())
            .fold(0.0f32, f32::max);
        assert!(max_err < 1e-4, "chunk 与 recurrent 不等价, max_err={max_err}");
    }

    #[test]
    fn test_mla_attention_decode() {
        // MlaAttention.decode：k = [k_nope | k_rope] 续接 + 解耦打分
        let (hidden, h) = (32usize, 4usize);
        let (kv_l, q_l, rope_d, v_d) = (16usize, 16usize, 8usize, 16usize);
        let nope = v_d;
        let qh = nope + rope_d;
        let attn = MlaAttention {
            num_heads: h, kv_lora_rank: kv_l, q_lora_rank: q_l,
            qk_rope_head_dim: rope_d, v_head_dim: v_d, qk_nope_head_dim: nope, q_head_dim: qh,
            eps: 1e-6, scaling: (nope as f32).powf(-0.5),
            q_a_w: Tensor::from_vec(q_l, hidden, vec![0.1; q_l * hidden]),
            q_b_w: Tensor::from_vec(h * qh, q_l, vec![0.1; h * qh * q_l]),
            kv_a_w: Tensor::from_vec(kv_l + rope_d, hidden, vec![0.1; (kv_l + rope_d) * hidden]),
            kv_b_w: Tensor::from_vec(h * (nope + v_d), kv_l, vec![0.1; h * (nope + v_d) * kv_l]),
            o_w: Tensor::from_vec(hidden, h * v_d, vec![0.1; hidden * h * v_d]),
            q_norm_w: vec![1.0; q_l],
            k_norm_w: vec![1.0; rope_d],
        };
        let kv_k = Tensor::from_vec(4, h * nope + rope_d, vec![0.1; 4 * (h * nope + rope_d)]);
        let kv_v = Tensor::from_vec(4, h * v_d, vec![0.1; 4 * h * v_d]);
        let x1 = Tensor::from_vec(1, hidden, vec![0.5; hidden]);
        let cos = Tensor::from_vec(5, rope_d, vec![1.0; 5 * rope_d]);
        let sin = Tensor::from_vec(5, rope_d, vec![0.0; 5 * rope_d]);
        let (out, k_all, v_all) = attn.decode(&x1, &cos, &sin, &kv_k, &kv_v);
        assert_eq!(out.rows, 1);
        assert_eq!(k_all.rows, 5);
        assert_eq!(v_all.rows, 5);
        assert!(out.max_abs().is_finite());
    }
}
