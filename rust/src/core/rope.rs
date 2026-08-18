//! 旋转位置编码（RoPE）：逆频率 → 每位置 cos/sin → 应用（2i, 2i+1 成对旋转）。
//!
//! 支持 partial rotary（Qwen3.5 的 partial_rotary_factor：仅旋转每头前 rope_dim 维）。
use crate::core::tensor::Tensor;

/// 计算逆频率（长度 = round(head_dim * partial) 的偶数维）。
pub fn compute_inv_freq(head_dim: usize, theta: f32, partial: f32) -> Vec<f32> {
    let dim = ((head_dim as f32 * partial) as usize) / 2 * 2;   // 取偶数
    (0..dim)
        .map(|i| {
            let exponent = 2.0 * (i as f32 / 2.0) / head_dim as f32;
            theta.powf(exponent).recip()
        })
        .collect()
}

/// 位置 pos 的 cos/sin（长度 = inv_freq.len()）。
pub fn rotary_embeddings(pos: usize, inv_freq: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let dim = inv_freq.len();
    let mut cos = vec![0.0f32; dim];
    let mut sin = vec![0.0f32; dim];
    for i in 0..dim {
        let angle = pos as f32 * inv_freq[i];
        cos[i] = angle.cos();
        sin[i] = angle.sin();
    }
    (cos, sin)
}

/// 应用旋转到每行前 `rope_dim` 维：
/// `(x[2i], x[2i+1]) = (x[2i]*cos - x[2i+1]*sin, x[2i]*sin + x[2i+1]*cos)`。
pub fn apply_rotary(x: &Tensor, cos: &[f32], sin: &[f32], rope_dim: usize) -> Tensor {
    let mut out = x.clone();
    let pairs = rope_dim / 2;
    for i in 0..x.rows {
        for p in 0..pairs {
            let j = 2 * p;
            let x0 = x.get(i, j);
            let x1 = x.get(i, j + 1);
            out.data[i * x.cols + j] = x0 * cos[j] - x1 * sin[j];
            out.data[i * x.cols + j + 1] = x0 * sin[j] + x1 * cos[j];
        }
    }
    out
}

/// 逐行应用旋转（prefill 用）：x (rows, cols)，cos/sin (rows, rope_dim)，每行用自己位置的旋转。
pub fn apply_rotary_rows(x: &Tensor, cos: &Tensor, sin: &Tensor, rope_dim: usize) -> Tensor {
    let mut out = x.clone();
    let pairs = rope_dim / 2;
    for i in 0..x.rows {
        for p in 0..pairs {
            let j = 2 * p;
            let x0 = x.get(i, j);
            let x1 = x.get(i, j + 1);
            let c = cos.get(i, j);
            let s = sin.get(i, j);
            out.data[i * x.cols + j] = x0 * c - x1 * s;
            out.data[i * x.cols + j + 1] = x0 * s + x1 * c;
        }
    }
    out
}

/// 单行 q 应用旋转（decode 用）：x (1, cols)，cos/sin 为全位置 (rows, rope_dim)，
/// 取**最后一行**（当前 token 的位置）。
pub fn apply_rotary_rows_last(x: &Tensor, cos: &Tensor, sin: &Tensor, rope_dim: usize) -> Tensor {
    let mut out = x.clone();
    let last = cos.rows - 1;
    let pairs = rope_dim / 2;
    for p in 0..pairs {
        let j = 2 * p;
        let x0 = x.get(0, j);
        let x1 = x.get(0, j + 1);
        let c = cos.get(last, j);
        let s = sin.get(last, j);
        out.data[j] = x0 * c - x1 * s;
        out.data[j + 1] = x0 * s + x1 * c;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_apply_rotary_identity_at_pos0() {
        let x = Tensor::from_vec(1, 4, vec![1.0, 0.0, 0.0, 1.0]);
        let inv = compute_inv_freq(4, 1e6, 1.0);
        let (cos, sin) = rotary_embeddings(0, &inv);   // 位置 0：cos=1, sin=0 → 恒等
        let r = apply_rotary(&x, &cos, &sin, 4);
        for j in 0..4 {
            assert!((r.get(0, j) - x.get(0, j)).abs() < 1e-5);
        }
    }

    #[test]
    fn test_apply_rotary_quarter_turn() {
        // 90° 旋转：(1,0) → (cos90, sin90) ≈ (0, 1)
        let x = Tensor::from_vec(1, 2, vec![1.0, 0.0]);
        let inv = vec![std::f32::consts::FRAC_PI_2 / 1.0f32];   // angle = pos * inv_freq
        let pos = 1usize;
        let (cos, sin) = rotary_embeddings(pos, &inv);          // angle = π/2
        let r = apply_rotary(&x, &cos, &sin, 2);
        assert!(r.get(0, 0).abs() < 1e-4);
        assert!((r.get(0, 1) - 1.0).abs() < 1e-4);
    }
}
