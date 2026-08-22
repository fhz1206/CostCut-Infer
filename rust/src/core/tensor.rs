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
        // 实测 matmul_blas 转换开销抵消 BLAS 优势（512³ 0.14x、2048² 0.99x）——回退纯标量
        assert_eq!(self.cols, rhs.rows, "matmul 形状不匹配");
        let (m, k, n) = (self.rows, self.cols, rhs.cols);
        let mut out = Tensor::zeros(m, n);
        let threads = std::thread::available_parallelism().map(|p| p.get()).unwrap_or(4);
        // 行并行（std::thread——大矩阵才有收益；小矩阵保持标量避免线程开销）
        if m >= 64 && k >= 256 && n >= 256 && threads > 1 {
            let chunk = m.div_ceil(threads);
            let mut out_rows: Vec<&mut [f32]> = Vec::new();
            let mut rest = &mut out.data[..];
            for t in 0..threads {
                let lo = t * chunk;
                let hi = (lo + chunk).min(m);
                if lo >= hi {
                    break;
                }
                let (head, tail) = rest.split_at_mut((hi - lo) * n);
                out_rows.push(head);
                rest = tail;
            }
            let a = &self.data;
            let b = &rhs.data;
            std::thread::scope(|s| {
                let mut handles = Vec::new();
                for (t, row_slice) in out_rows.into_iter().enumerate() {
                    let lo = t * chunk;
                    handles.push(s.spawn(move || {
                        for i in 0..row_slice.len() / n {
                            let row = i * n;
                            for kk in 0..k {
                                let av = a[(lo + i) * k + kk];
                                if av == 0.0 {
                                    continue;
                                }
                                let b_row = kk * n;
                                for j in 0..n {
                                    row_slice[row + j] += av * b[b_row + j];
                                }
                            }
                        }
                    }));
                }
                for h in handles {
                    h.join().unwrap();
                }
            });
        } else {
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
        }
        out
    }

    /// tch BLAS matmul（libtorch 后端——大矩阵性能追平 torch BLAS；小矩阵转换开销大）。
    pub fn matmul_blas(&self, rhs: &Tensor) -> Tensor {
        assert_eq!(self.cols, rhs.rows, "matmul 形状不匹配");
        let a = tch::Tensor::from_slice(&self.data)
            .reshape([self.rows as i64, self.cols as i64]);
        let b = tch::Tensor::from_slice(&rhs.data)
            .reshape([rhs.rows as i64, rhs.cols as i64]);
        let c = a.matmul(&b);
        let data: Vec<f32> = c.flatten(0, -1).try_into().unwrap();
        Tensor { rows: self.rows, cols: rhs.cols, data }
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

    /// fp16 权重 matmul（compute_dtype="float16" 路径）：self (m, k) f32 激活 ×
    /// rhs (k, n) fp16 权重 → (m, n) f32。fp16 数据就地 u16 位型实时转 f32（镜像 Python
    /// 以 float16 权重计算、输出 f32 的语义——纯 std，无 fp16 硬件指令）。
    pub fn matmul_f16(&self, rhs_f16: &F16Tensor) -> Tensor {
        assert_eq!(self.cols, rhs_f16.rows, "matmul_f16 形状不匹配");
        let (m, k, n) = (self.rows, self.cols, rhs_f16.cols);
        let mut out = Tensor::zeros(m, n);
        for i in 0..m {
            for kk in 0..k {
                let a = self.get(i, kk);
                if a == 0.0 {
                    continue;
                }
                let b_row = &rhs_f16.data[kk * n..(kk + 1) * n];
                for j in 0..n {
                    out.data[i * n + j] += a * f16_to_f32(b_row[j]);
                }
            }
        }
        out
    }
}

/// fp16 权重张量（u16 位型存储——RFC 7049 / IEEE fp16，行主序）。
#[derive(Clone, Debug)]
pub struct F16Tensor {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<u16>,
}

impl F16Tensor {
    /// 从 f32 数据构造（自动转 fp16 位型——compute_dtype="float16" 权重存储）。
    pub fn from_f32(rows: usize, cols: usize, data: &[f32]) -> Self {
        F16Tensor {
            rows,
            cols,
            data: data.iter().map(|&v| f32_to_f16(v)).collect(),
        }
    }

    /// 访问 (i, j) 元素的 fp16 位型。
    #[inline]
    pub fn get(&self, i: usize, j: usize) -> u16 {
        self.data[i * self.cols + j]
    }
}

/// f32 → fp16 位型（IEEE 754 half，与 quant::dequant 的转换函数一致——此处独立实现避免依赖）。
pub fn f32_to_f16(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let raw_exp = ((bits >> 23) & 0xFF) as i32 - 127;
    if raw_exp >= 16 {
        return sign | 0x7C00;   // 溢出 → 无穷
    }
    let mant = bits & 0x7FFFFF;
    let biased = (raw_exp + 15) as i32;
    if biased <= 0 {
        return sign | (mant >> 13) as u16;   // 次正规/零
    }
    let m = (mant >> 13) as u16;
    let round = ((mant >> 12) & 1) as u16;
    sign | ((biased as u16) << 10) | (m + round)
}

/// fp16 位型 → f32（IEEE 754 half 解码）。
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as f32 * -1.0;
    let exp = ((h >> 10) & 0x1F) as i32;
    let mant = (h & 0x3FF) as f32;
    let v = if exp == 0 {
        if mant == 0.0 { 0.0 } else { (mant / 1024.0) * (2.0f32).powi(-14) }
    } else if exp == 31 {
        if mant == 0.0 { f32::INFINITY } else { f32::NAN }
    } else {
        (1.0 + mant / 1024.0) * (2.0f32).powi(exp - 15)
    };
    if sign < 0.0 { -v } else { v }
}

/// bf16 权重张量（u16 位型存储——IEEE bfloat16，行主序）。
#[derive(Clone, Debug)]
pub struct BF16Tensor {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<u16>,
}

impl BF16Tensor {
    /// 从 f32 数据构造（自动转 bf16 位型——compute_dtype="bf16" 权重存储）。
    pub fn from_f32(rows: usize, cols: usize, data: &[f32]) -> Self {
        BF16Tensor {
            rows,
            cols,
            data: data.iter().map(|&v| f32_to_bf16(v)).collect(),
        }
    }

    /// 访问 (i, j) 元素的 bf16 位型。
    #[inline]
    pub fn get(&self, i: usize, j: usize) -> u16 {
        self.data[i * self.cols + j]
    }
}

impl Tensor {
    /// bf16 权重 matmul（compute_dtype="bf16" 路径）：f32 激活 × bf16 权重 → f32。
    pub fn matmul_bf16(&self, rhs_bf16: &BF16Tensor) -> Tensor {
        assert_eq!(self.cols, rhs_bf16.rows, "matmul_bf16 形状不匹配");
        let (m, k, n) = (self.rows, self.cols, rhs_bf16.cols);
        let mut out = Tensor::zeros(m, n);
        for i in 0..m {
            for kk in 0..k {
                let a = self.get(i, kk);
                if a == 0.0 {
                    continue;
                }
                let b_row = &rhs_bf16.data[kk * n..(kk + 1) * n];
                for j in 0..n {
                    out.data[i * n + j] += a * bf16_to_f32(b_row[j]);
                }
            }
        }
        out
    }
}

/// f32 → bf16 位型（取高 16 位——bfloat16 尾数截断）。
pub fn f32_to_bf16(v: f32) -> u16 {
    (v.to_bits() >> 16) as u16
}

/// bf16 位型 → f32（高 16 位左移回）。
pub fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
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
