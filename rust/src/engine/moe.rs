//! MoE 组件：TopKRouter（路由）+ MergedExperts（合并 plain 专家）+ MLP（共享/稠密）。
//!
//! 权重格式（Mixtral / Qwen3-MoE / GLM）：`gate_up_proj` [E, 2*inter, hidden] +
//! `down_proj` [E, hidden, inter]，标准 [out, in] 序。
use crate::core::tensor::{BF16Tensor, F16Tensor, Tensor};

/// Top-K 路由：linear → softmax(fp32) → topk → 归一化。
pub struct TopKRouter {
    pub weight: Tensor,   // (num_experts, hidden)
    pub top_k: usize,
}

impl TopKRouter {
    /// 返回 (每行 top-k 专家索引 [rows][top_k], 归一化权重 (rows, top_k))。
    pub fn forward(&self, x: &Tensor) -> (Vec<Vec<usize>>, Tensor) {
        let logits = x.matmul(&self.weight.transpose());
        let probs = logits.softmax_rows();
        let mut indices = Vec::with_capacity(x.rows);
        let mut weights = vec![0.0f32; x.rows * self.top_k];
        for i in 0..x.rows {
            let row = &probs.data[i * probs.cols..(i + 1) * probs.cols];
            let mut sorted: Vec<(usize, f32)> =
                (0..probs.cols).map(|j| (j, row[j])).collect();
            sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let sum: f32 = sorted[..self.top_k].iter().map(|(_, v)| *v).sum();
            let row_idx: Vec<usize> = sorted[..self.top_k].iter().map(|(j, _)| *j).collect();
            indices.push(row_idx);
            for k in 0..self.top_k {
                weights[i * self.top_k + k] = sorted[k].1 / sum;
            }
        }
        (indices, Tensor::from_vec(x.rows, self.top_k, weights))
    }
}

/// 合并 plain 专家（gate_up / down 为展平的 [E, *] 数据）。
pub struct MergedExperts {
    pub num_experts: usize,
    pub intermediate: usize,
    pub hidden: usize,
    pub gate_up: Vec<f32>,   // [E, 2*inter, hidden]
    pub down: Vec<f32>,      // [E, hidden, inter]
    pub gate_up_f16: Option<Vec<F16Tensor>>,   // compute_dtype="float16" 权重（可选）
    pub down_f16: Option<Vec<F16Tensor>>,
    pub gate_up_bf16: Option<Vec<BF16Tensor>>,  // compute_dtype="bf16" 权重（可选）
    pub down_bf16: Option<Vec<BF16Tensor>>,
}

impl MergedExperts {
    /// x: (rows, hidden)；indices/weights 来自路由。返回 (rows, hidden)。
    pub fn forward(&self, x: &Tensor, indices: &[Vec<usize>],
                   weights: &Tensor) -> Tensor {
        let gu_elems = 2 * self.intermediate * self.hidden;
        let down_elems = self.hidden * self.intermediate;
        let mut final_out = Tensor::zeros(x.rows, self.hidden);
        let use_f16 = self.gate_up_f16.is_some();
        let use_bf16 = self.gate_up_bf16.is_some();
        for i in 0..x.rows {
            let mut acc = vec![0.0f32; self.hidden];
            for k in 0..weights.cols {
                let e = indices[i][k];
                let w = weights.get(i, k);
                let xi = row_tensor(x, i);
                let fused = if use_f16 {
                    // fp16 权重路径（compute_dtype="float16"）
                    let gu_f16 = &self.gate_up_f16.as_ref().unwrap()[e];
                    xi.matmul_f16(gu_f16)                          // (1, 2*inter)
                } else if use_bf16 {
                    // bf16 权重路径（compute_dtype="bf16"）
                    let gu_bf16 = &self.gate_up_bf16.as_ref().unwrap()[e];
                    xi.matmul_bf16(gu_bf16)
                } else {
                    let gu = Tensor::from_vec(
                        2 * self.intermediate, self.hidden,
                        self.gate_up[e * gu_elems..(e + 1) * gu_elems].to_vec());
                    xi.matmul(&gu.transpose())
                };
                let mut gate = vec![0.0f32; self.intermediate];
                let mut up = vec![0.0f32; self.intermediate];
                for j in 0..self.intermediate {
                    gate[j] = fused.data[j];
                    up[j] = fused.data[self.intermediate + j];
                }
                let h = Tensor::from_vec(1, self.intermediate, gate).silu()
                    .elementwise_mul(&Tensor::from_vec(1, self.intermediate, up));
                let out_e = if use_f16 {
                    let down_f16 = &self.down_f16.as_ref().unwrap()[e];
                    h.matmul_f16(down_f16)                          // (1, hidden)
                } else if use_bf16 {
                    let down_bf16 = &self.down_bf16.as_ref().unwrap()[e];
                    h.matmul_bf16(down_bf16)
                } else {
                    let down_m = Tensor::from_vec(
                        self.hidden, self.intermediate,
                        self.down[e * down_elems..(e + 1) * down_elems].to_vec());
                    h.matmul(&down_m.transpose())
                };
                for j in 0..self.hidden {
                    acc[j] += out_e.get(0, j) * w;
                }
            }
            for j in 0..self.hidden {
                final_out.data[i * self.hidden + j] = acc[j];
            }
        }
        final_out
    }
}

/// SwiGLU MLP（共享专家 / 稠密层）：`silu(x@gate) * (x@up) → @down`。
pub struct MLP {
    pub gate_w: Tensor,   // (inter, hidden)
    pub up_w: Tensor,     // (inter, hidden)
    pub down_w: Tensor,   // (hidden, inter)
}

impl MLP {
    pub fn forward(&self, x: &Tensor) -> Tensor {
        // gate_w/up_w 为 [out=inter, in=hidden]：线性 = x @ w.T
        let h = x.matmul(&self.gate_w.transpose()).silu()
            .elementwise_mul(&x.matmul(&self.up_w.transpose()));
        h.matmul(&self.down_w.transpose())
    }
}

/// 取张量的第 i 行（1, cols）。
pub fn row_tensor(t: &Tensor, i: usize) -> Tensor {
    Tensor::from_vec(1, t.cols, t.data[i * t.cols..(i + 1) * t.cols].to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_router_weights_normalized() {
        let router = TopKRouter {
            weight: Tensor::from_vec(4, 8, vec![0.1; 32]),
            top_k: 2,
        };
        let x = Tensor::from_vec(3, 8, vec![0.5; 24]);
        let (indices, weights) = router.forward(&x);
        assert_eq!(indices.len(), 3);
        assert_eq!(weights.cols, 2);
        for i in 0..3 {
            let sum: f32 = (0..2).map(|k| weights.get(i, k)).sum();
            assert!((sum - 1.0).abs() < 1e-4);
        }
    }

    #[test]
    fn test_mlp_shape() {
        let (hidden, inter) = (8usize, 16usize);
        let mlp = MLP {
            gate_w: Tensor::from_vec(inter, hidden, vec![0.1; inter * hidden]),
            up_w: Tensor::from_vec(inter, hidden, vec![0.1; inter * hidden]),
            down_w: Tensor::from_vec(hidden, inter, vec![0.1; hidden * inter]),
        };
        let x = Tensor::from_vec(4, hidden, vec![0.5; 4 * hidden]);
        let out = mlp.forward(&x);
        assert_eq!((out.rows, out.cols), (4, hidden));
    }
}
