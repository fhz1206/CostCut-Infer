//! KV 缓存：每层缓存拼接后的 k/v（标准注意力 decode 续接用）。
use crate::core::tensor::Tensor;

/// 逐层 KV 缓存。
pub struct KVCache {
    pub k: Vec<Option<Tensor>>,   // 每层 (ctx, H*hd)
    pub v: Vec<Option<Tensor>>,   // 每层 (ctx, kvh*hd)
}

impl KVCache {
    pub fn new(layers: usize) -> Self {
        KVCache {
            k: (0..layers).map(|_| None).collect(),
            v: (0..layers).map(|_| None).collect(),
        }
    }

    /// 取第 layer 层的缓存（None = 尚未填充）。
    pub fn get(&self, layer: usize) -> Option<(&Tensor, &Tensor)> {
        match (&self.k[layer], &self.v[layer]) {
            (Some(k), Some(v)) => Some((k, v)),
            _ => None,
        }
    }

    /// 设置第 layer 层的缓存（prefill 填充 / decode 续接后更新）。
    pub fn set(&mut self, layer: usize, k: Tensor, v: Tensor) {
        self.k[layer] = Some(k);
        self.v[layer] = Some(v);
    }
}

/// 按行拼接两个张量（列数相同）。
pub fn concat_rows(a: &Tensor, b: &Tensor) -> Tensor {
    assert_eq!(a.cols, b.cols, "拼接张量列数须一致");
    let mut data = Vec::with_capacity((a.rows + b.rows) * a.cols);
    data.extend_from_slice(&a.data);
    data.extend_from_slice(&b.data);
    Tensor::from_vec(a.rows + b.rows, a.cols, data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_concat_rows() {
        let a = Tensor::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = Tensor::from_vec(1, 3, vec![7.0, 8.0, 9.0]);
        let c = concat_rows(&a, &b);
        assert_eq!((c.rows, c.cols), (3, 3));
        assert!((c.get(2, 2) - 9.0).abs() < 1e-6);
    }
}
