//! 采样：贪心（argmax）/ 温度 softmax 采样（纯 std，无随机依赖——调用方提供 RNG）。
use crate::core::tensor::Tensor;

/// 行向量的 argmax（返回索引）。
pub fn argmax_row(data: &[f32]) -> usize {
    data.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
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
