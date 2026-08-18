//! 最小张量：二维形状 + f32 数据 + 基础算子（matmul / softmax / 逐元素）。
//!
//! 纯 std 实现（无外部依赖，适配离线构建环境）；作为 Rust 版引擎的数学基础层。

/// 二维行主序 f32 张量。
#[derive(Clone, Debug)]
pub struct Tensor {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>,
}

impl Tensor {
    /// 全零张量。
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Tensor { rows, cols, data: vec![0.0; rows * cols] }
    }

    /// 从一维数据构造（长度须等于 rows*cols）。
    pub fn from_vec(rows: usize, cols: usize, data: Vec<f32>) -> Self {
        assert_eq!(data.len(), rows * cols, "数据长度与形状不匹配");
        Tensor { rows, cols, data }
    }

    /// 访问 (i, j) 元素。
    #[inline]
    pub fn get(&self, i: usize, j: usize) -> f32 {
        self.data[i * self.cols + j]
    }

    /// 矩阵乘法：self (m, k) × rhs (k, n) → (m, n)。
    /// 朴素三重循环（i-k-j 顺序利于缓存）。
    pub fn matmul(&self, rhs: &Tensor) -> Tensor {
        assert_eq!(self.cols, rhs.rows, "matmul 形状不匹配");
        let (m, k, n) = (self.rows, self.cols, rhs.cols);
        let mut out = Tensor::zeros(m, n);
        for i in 0..m {
            for kk in 0..k {
                let a = self.get(i, kk);
                if a == 0.0 {
                    continue;
                }
                let row = i * n;
                let b_row = kk * n;
                for j in 0..n {
                    out.data[row + j] += a * rhs.data[b_row + j];
                }
            }
        }
        out
    }

    /// 并行 matmul（std::thread::scope 按行分块，纯 std 无外部依赖）。
    pub fn matmul_par(&self, rhs: &Tensor, threads: usize) -> Tensor {
        assert_eq!(self.cols, rhs.rows, "matmul 形状不匹配");
        let (m, k, n) = (self.rows, self.cols, rhs.cols);
        let nthreads = threads.max(1).min(m);
        let blocks: Vec<Vec<f32>> = std::thread::scope(|s| {
            let mut handles = Vec::with_capacity(nthreads);
            for t in 0..nthreads {
                let row0 = t * m / nthreads;
                let row1 = (t + 1) * m / nthreads;
                let a = &self.data;
                let b = &rhs.data;
                handles.push(s.spawn(move || {
                    let mut block = vec![0.0f32; (row1 - row0) * n];
                    for i in row0..row1 {
                        let ai = &a[i * k..(i + 1) * k];
                        let oi = (i - row0) * n;
                        for (kk, &av) in ai.iter().enumerate() {
                            if av == 0.0 {
                                continue;
                            }
                            let b_row = &b[kk * n..(kk + 1) * n];
                            for j in 0..n {
                                block[oi + j] += av * b_row[j];
                            }
                        }
                    }
                    block
                }));
            }
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        let mut data = Vec::with_capacity(m * n);
        for block in blocks {
            data.extend_from_slice(&block);
        }
        Tensor::from_vec(m, n, data)
    }

    /// 行 softmax（沿列维），返回同形状张量。
    pub fn softmax_rows(&self) -> Tensor {
        let mut out = self.clone();
        for i in 0..self.rows {
            let row = &self.data[i * self.cols..(i + 1) * self.cols];
            let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for j in 0..self.cols {
                let e = (row[j] - max).exp();
                out.data[i * self.cols + j] = e;
                sum += e;
            }
            for j in 0..self.cols {
                out.data[i * self.cols + j] /= sum;
            }
        }
        out
    }

    /// 逐元素乘法（self 与 rhs 形状一致）。
    pub fn elementwise_mul(&self, rhs: &Tensor) -> Tensor {
        assert_eq!(self.rows, rhs.rows);
        assert_eq!(self.cols, rhs.cols);
        let data = self.data.iter().zip(rhs.data.iter()).map(|(a, b)| a * b).collect();
        Tensor { rows: self.rows, cols: self.cols, data }
    }

    /// 逐元素加（rhs 为同形状或可广播的行向量 (1, cols)）。
    pub fn add(&self, rhs: &Tensor) -> Tensor {
        assert_eq!(self.cols, rhs.cols);
        let mut data = self.data.clone();
        if rhs.rows == 1 {
            for i in 0..self.rows {
                for j in 0..self.cols {
                    data[i * self.cols + j] += rhs.data[j];
                }
            }
        } else {
            assert_eq!(self.rows, rhs.rows);
            for (a, b) in data.iter_mut().zip(rhs.data.iter()) {
                *a += b;
            }
        }
        Tensor { rows: self.rows, cols: self.cols, data }
    }

    /// AVX2/FMA 加速 matmul（std::arch，x86_64，纯 std 无依赖——docs 性能方向）。
    /// 每次 8 个输出元素 FMA 乘加（i-k-j 循环）；无 AVX2/FMA 时回退朴素 matmul。
    pub fn matmul_avx2(&self, rhs: &Tensor) -> Tensor {
        if std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma")
        {
            unsafe { self.avx2_fma_kernel(rhs) }
        } else {
            self.matmul(rhs)
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn avx2_fma_kernel(&self, rhs: &Tensor) -> Tensor {
        use std::arch::x86_64::*;
        let (m, k, n) = (self.rows, self.cols, rhs.cols);
        let mut out = vec![0.0f32; m * n];
        for i in 0..m {
            for kk in 0..k {
                let a = self.get(i, kk);
                if a == 0.0 {
                    continue;
                }
                let av = _mm256_set1_ps(a);
                let b_row = &rhs.data[kk * n..(kk + 1) * n];
                let mut j = 0;
                while j + 8 <= n {
                    let bv = _mm256_loadu_ps(b_row.as_ptr().add(j));
                    let ov = _mm256_loadu_ps(out.as_ptr().add(i * n + j));
                    let r = _mm256_fmadd_ps(av, bv, ov);
                    _mm256_storeu_ps(out.as_mut_ptr().add(i * n + j), r);
                    j += 8;
                }
                for jj in j..n {
                    out[i * n + jj] += a * b_row[jj];
                }
            }
        }
        Tensor::from_vec(m, n, out)
    }

    /// 分块缓存 matmul（docs 方向 B）：8×8 输出 tile + k 分块（cache blocking）。
    /// rhs 先转置为 (n, k) 行主序（列访问变连续行访问），8×8×8 块内三重循环。
    pub fn matmul_blocked(&self, rhs: &Tensor) -> Tensor {
        assert_eq!(self.cols, rhs.rows, "matmul 形状不匹配");
        let bt = rhs.transpose();
        let (m, k, n) = (self.rows, self.cols, rhs.cols);
        let mut out = vec![0.0f32; m * n];
        let t = 8usize;
        let mut i0 = 0;
        while i0 < m {
            let im = (i0 + t).min(m);
            let mut j0 = 0;
            while j0 < n {
                let jm = (j0 + t).min(n);
                let mut k0 = 0;
                while k0 < k {
                    let km = (k0 + t).min(k);
                    for i in i0..im {
                        for j in j0..jm {
                            let mut acc = 0.0f32;
                            for kk in k0..km {
                                acc += self.data[i * k + kk] * bt.data[j * k + kk];
                            }
                            out[i * n + j] += acc;
                        }
                    }
                    k0 = km;
                }
                j0 = jm;
            }
            i0 = im;
        }
        Tensor::from_vec(m, n, out)
    }

    /// 转置（(rows, cols) → (cols, rows)）。
    pub fn transpose(&self) -> Tensor {
        let mut data = vec![0.0f32; self.rows * self.cols];
        for i in 0..self.rows {
            for j in 0..self.cols {
                data[j * self.rows + i] = self.get(i, j);
            }
        }
        Tensor { rows: self.cols, cols: self.rows, data }
    }

    /// 乘标量。
    pub fn scale(&self, s: f32) -> Tensor {
        Tensor {
            rows: self.rows,
            cols: self.cols,
            data: self.data.iter().map(|v| v * s).collect(),
        }
    }

    /// SiLU 逐元素激活。
    pub fn silu(&self) -> Tensor {
        Tensor {
            rows: self.rows,
            cols: self.cols,
            data: self.data.iter().map(|v| v / (1.0 + (-v).exp())).collect(),
        }
    }

    /// Sigmoid 逐元素激活。
    pub fn sigmoid(&self) -> Tensor {
        Tensor {
            rows: self.rows,
            cols: self.cols,
            data: self.data.iter().map(|v| 1.0 / (1.0 + (-v).exp())).collect(),
        }
    }

    /// 按列切分：(rows, a+b) → ((rows, a), (rows, b))。
    pub fn split_cols(&self, a: usize) -> (Tensor, Tensor) {
        assert!(a <= self.cols, "split_cols 越界: {a} > {}", self.cols);
        let left = self.data[..self.rows * a].to_vec();
        let right = self.data[self.rows * a..].to_vec();
        (Tensor::from_vec(self.rows, a, left),
         Tensor::from_vec(self.rows, self.cols - a, right))
    }

    /// 按列拼接（行数相同）：(rows, a) + (rows, b) → (rows, a+b)。
    pub fn concat_cols(&self, rhs: &Tensor) -> Tensor {
        assert_eq!(self.rows, rhs.rows, "concat_cols 行数须一致");
        let mut data = Vec::with_capacity(self.rows * (self.cols + rhs.cols));
        for i in 0..self.rows {
            data.extend_from_slice(&self.data[i * self.cols..(i + 1) * self.cols]);
            data.extend_from_slice(&rhs.data[i * rhs.cols..(i + 1) * rhs.cols]);
        }
        Tensor::from_vec(self.rows, self.cols + rhs.cols, data)
    }

    /// 最大绝对值（数值 sanity 检查）。
    pub fn max_abs(&self) -> f32 {
        self.data.iter().fold(0.0f32, |acc, v| acc.max(v.abs()))
    }

    /// 均值（数值 sanity 检查）。
    pub fn mean(&self) -> f32 {
        self.data.iter().sum::<f32>() / self.data.len() as f32
    }

    /// 标准差（数值 sanity 检查）。
    pub fn std(&self) -> f32 {
        let m = self.mean();
        let var = self.data.iter().map(|v| (v - m) * (v - m)).sum::<f32>()
            / self.data.len() as f32;
        var.sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matmul_shape_and_value() {
        let a = Tensor::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = Tensor::from_vec(3, 2, vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
        let c = a.matmul(&b);
        assert_eq!((c.rows, c.cols), (2, 2));
        assert!((c.get(0, 0) - 4.0).abs() < 1e-5);
        assert!((c.get(1, 1) - 11.0).abs() < 1e-5);
    }

    #[test]
    fn test_softmax_sums_to_one() {
        let t = Tensor::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let s = t.softmax_rows();
        for i in 0..2 {
            let row_sum: f32 = (0..3).map(|j| s.get(i, j)).sum();
            assert!((row_sum - 1.0).abs() < 1e-4);
        }
    }

    #[test]
    fn test_matmul_avx2_equivalence() {
        // AVX2/FMA matmul == 朴素 matmul（权威等价）
        let a = Tensor::from_vec(64, 128,
            (0..64 * 128).map(|i| ((i % 13) as f32) * 0.01).collect());
        let b = Tensor::from_vec(128, 64,
            (0..128 * 64).map(|i| ((i % 7) as f32) * 0.01).collect());
        let c1 = a.matmul(&b);
        let c2 = a.matmul_avx2(&b);
        let max_err = (0..c1.data.len())
            .map(|i| (c1.data[i] - c2.data[i]).abs())
            .fold(0.0f32, f32::max);
        assert!(max_err < 1e-3, "AVX2 matmul 不等价, max_err={max_err}");
    }
}
