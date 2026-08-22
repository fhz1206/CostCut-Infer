//! 采样：贪心（argmax）/ 温度 softmax 采样（纯 std，无随机依赖——调用方提供 RNG）。
use crate::core::tensor::Tensor;

/// 行向量的 argmax（返回索引）。
pub fn argmax_row(data: &[f32]) -> usize {
    data.iter()
        .enumerate()
        // NaN 安全：partial_cmp 遇 NaN 返回 None——按 Equal 处理（避免 unwrap panic）
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// 温度 softmax 采样（返回索引）。`rng` 返回 [0,1) 均匀随机数。
pub fn sample_row(data: &[f32], temperature: f32, rng: &mut impl FnMut() -> f32) -> usize {
    let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = data.iter().map(|v| ((v - max) / temperature).exp()).collect();
    let sum: f32 = probs.iter().sum();
    for p in probs.iter_mut() {
        *p /= sum;
    }
    let mut acc = 0.0f32;
    let r = rng();
    for (i, p) in probs.iter().enumerate() {
        acc += p;
        if r <= acc {
            return i;
        }
    }
    probs.len() - 1
}

/// 采样：温度 + top_k + top_p（top_k=0 不限；top_p 核采样阈值——纯 std，调用方提供 RNG）。
pub fn sample_row_p(data: &[f32], temperature: f32, top_k: usize, top_p: f32,
                    rng: &mut impl FnMut() -> f32) -> usize {
    let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = data.iter().map(|v| ((v - max) / temperature).exp()).collect();
    let sum: f32 = probs.iter().sum();
    for p in probs.iter_mut() {
        *p /= sum;
    }
    if top_k > 0 && top_k < probs.len() {
        let mut idx: Vec<usize> = (0..probs.len()).collect();
        idx.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());
        let keep: std::collections::HashSet<usize> = idx[..top_k].iter().copied().collect();
        for (i, p) in probs.iter_mut().enumerate() {
            if !keep.contains(&i) {
                *p = 0.0;
            }
        }
        let s: f32 = probs.iter().sum();
        for p in probs.iter_mut() {
            *p /= s;
        }
    }
    if top_p > 0.0 && top_p < 1.0 {
        let mut idx: Vec<usize> = (0..probs.len()).collect();
        idx.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());
        let mut acc = 0.0f32;
        let mut cutoff = probs.len();
        for (n, &i) in idx.iter().enumerate() {
            acc += probs[i];
            if acc >= top_p {
                cutoff = n + 1;
                break;
            }
        }
        let keep: std::collections::HashSet<usize> = idx[..cutoff].iter().copied().collect();
        for (i, p) in probs.iter_mut().enumerate() {
            if !keep.contains(&i) {
                *p = 0.0;
            }
        }
        let s: f32 = probs.iter().sum();
        for p in probs.iter_mut() {
            *p /= s;
        }
    }
    let r = rng();
    let mut acc = 0.0f32;
    for (i, p) in probs.iter().enumerate() {
        acc += p;
        if r <= acc {
            return i;
        }
    }
    probs.len() - 1
}

/// 便捷包装：对 (1, vocab) 张量取 argmax。
pub fn argmax_logits(logits: &Tensor) -> usize {
    assert_eq!(logits.rows, 1);
    argmax_row(&logits.data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_argmax() {
        let logits = vec![0.1f32, 5.0, -1.0, 3.0];
        assert_eq!(argmax_row(&logits), 1);
    }

    #[test]
    fn test_sample_deterministic_with_zero_temp() {
        // 温度趋近 0 → 近似 argmax
        let logits = vec![0.1f32, 5.0, -1.0];
        let mut rng = || 0.9f32;
        assert_eq!(sample_row(&logits, 0.01, &mut rng), 1);
    }
}
