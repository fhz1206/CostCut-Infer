//! RMSNorm（Qwen / Mixtral / DeepSeek / GLM 通用）：
//! `out = x / sqrt(mean(x^2) + eps) * (1 + weight)`。
use crate::core::tensor::Tensor;

/// 逐行 RMSNorm：x (rows, cols)，weight 长度 = cols。
pub fn rms_norm(x: &Tensor, weight: &[f32], eps: f32) -> Tensor {
    assert_eq!(weight.len(), x.cols, "RMSNorm 权重长度须等于列数");
    let mut out = x.clone();
    for i in 0..x.rows {
        let row = &x.data[i * x.cols..(i + 1) * x.cols];
        let mean_sq = row.iter().map(|v| v * v).sum::<f32>() / x.cols as f32;
        let rms = (mean_sq + eps).sqrt();
        for j in 0..x.cols {
            out.data[i * x.cols + j] = row[j] / rms * (1.0 + weight[j]);
        }
    }
    out
}

/// 融合：残差相加与 RMSNorm 合并（镜像 Python `rms_norm_add`；差异报告 #8 算子融合）。
/// 返回 ``(norm(x + residual), x + residual)``——归一化结果 + 残差和（层最终输出用）。
pub fn rms_norm_add(x: &Tensor, residual: &Tensor, weight: &[f32], eps: f32)
                    -> (Tensor, Tensor) {
    let h = x.add(residual);
    let normed = rms_norm(&h, weight, eps);
    (normed, h)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rms_norm_scale_invariant() {
        let x = Tensor::from_vec(2, 4, vec![1.0, 2.0, 3.0, 4.0, 0.5, 1.0, 1.5, 2.0]);
        let w = vec![1.0f32; 4];
        let n = rms_norm(&x, &w, 1e-6);
        // 每行 RMS ≈ 1（weight=1 时 out = x / rms * 2）
        for i in 0..2 {
            let row = &n.data[i * 4..(i + 1) * 4];
            let rms2 = row.iter().map(|v| v * v).sum::<f32>() / 4.0;
            assert!((rms2.sqrt() - 2.0).abs() < 1e-4, "行 {i} RMS = {}", rms2.sqrt());
        }
    }

    #[test]
    fn test_rms_norm_add_equivalence() {
        let x = Tensor::from_vec(4, 8, vec![0.5; 32]);
        let r = Tensor::from_vec(4, 8, vec![0.3; 32]);
        let w = vec![1.0f32; 8];
        let (h, h_pre) = rms_norm_add(&x, &r, &w, 1e-6);
        // 等价：normed == rms_norm(x + residual)；h_pre == x + residual
        let expect = rms_norm(&x.add(&r), &w, 1e-6);
        let max_err = (0..h.data.len())
            .map(|i| (h.data[i] - expect.data[i]).abs())
            .fold(0.0f32, f32::max);
        assert!(max_err < 1e-5, "rms_norm_add 不等价, max_err={max_err}");
        let sum_err = (0..h_pre.data.len())
            .map(|i| (h_pre.data[i] - x.data[i] - r.data[i]).abs())
            .fold(0.0f32, f32::max);
        assert!(sum_err < 1e-5);
    }
}
