//! AWQ int4 反量化（与 Python `liteengine/quant` 对应）。
//!
//! 权重布局（本仓库 Qwen3.6-35B-A3B-AWQ-4bit）：
//! - `qweight`: [out, in/8] int32，每个 int32 沿 in 维打包 8 个 int4（低位在前）
//! - `qzeros`:  [out/group_size, in/8] int32，同样打包；值域无符号 0..15
//! - `scales`:  [out/group_size, in] float16
//!
//! AWQ 非标准列序：int32 内 8 个 int4 按 [0,2,4,6,1,3,5,7]（AWQ_ORDER）存储，
//! 线性解包后须按逆序 [0,4,1,5,2,6,3,7]（vLLM `_REVERSE_AWQ_PACK_ORDER`）重排。
//! 公式：`dequant = (q - zp) * scale`，group 沿 out 维，每 group_size 行共享。

/// AWQ 列序还原（vLLM _REVERSE_AWQ_PACK_ORDER / gptqmodel reverse_awq_order）。
const REVERSE_AWQ_PACK_ORDER: [usize; 8] = [0, 4, 1, 5, 2, 6, 3, 7];

/// 字节 → 低/高 4 位查表（一次算好，避免逐字节移位）。
fn byte_lut() -> ([u8; 256], [u8; 256]) {
    let mut lo = [0u8; 256];
    let mut hi = [0u8; 256];
    for b in 0..256usize {
        lo[b] = (b & 0x0F) as u8;
        hi[b] = ((b >> 4) & 0x0F) as u8;
    }
    (lo, hi)
}

/// 列打包解包：把 `[rows, cols]` 的 int32 按列展开为 `[rows, cols*8]` 的 int4（含 AWQ 列序还原）。
pub fn unpack_int4_colwise(packed: &[i32], cols: usize) -> Vec<u8> {
    let (lo, hi) = byte_lut();
    let out_dim = cols * 8;
    let out_rows = packed.len() / cols;
    let mut out = vec![0u8; out_rows * out_dim];
    for r in 0..out_rows {
        for c in 0..cols {
            let v = packed[r * cols + c] as u32;
            let bytes = v.to_le_bytes();               // 4 字节，每字节 2 个 int4
            let base = r * out_dim + c * 8;
            for k in 0..4 {
                let b = bytes[k] as usize;
                out[base + 2 * k] = lo[b];             // 低 4 位 = int4 #(2k)
                out[base + 2 * k + 1] = hi[b];         // 高 4 位 = int4 #(2k+1)
            }
        }
    }
    // AWQ 列序还原：每个 int32 的 8 个槽位按逆序重排（否则专家权重列错乱）
    let mut reordered = vec![0u8; out_rows * out_dim];
    for r in 0..out_rows {
        for c in 0..cols {
            let base = r * out_dim + c * 8;
            for s in 0..8 {
                reordered[base + s] = out[base + REVERSE_AWQ_PACK_ORDER[s]];
            }
        }
    }
    reordered
}

/// AWQ 4bit 反量化：返回 `[out, in]` 的 f32 权重矩阵。
///
/// - `qweight`: `[out, in/8]` i32 打包权重
/// AWQ int4 反量化→matmul 融合内核（参考 llama.cpp 块级思路——按 group 块反量化即时乘加，
/// 避免全量反量化物化到内存再读取的双重往返）。x (B, in)；返回 (B, out)。
pub fn matmul_awq_fused(x: &[f32], qweight: &[i32], qzeros: &[i32], scales: &[f32],
                        out: usize, in_: usize, group_size: usize) -> Vec<f32> {
    assert!(in_ % 8 == 0, "in 维度须为 8 的倍数");
    let w = unpack_int4_colwise(qweight, in_ / 8);
    let z = unpack_int4_colwise(qzeros, in_ / 8);
    let groups = out / group_size;
    let b = x.len() / in_;
    let mut result = vec![0.0f32; b * out];
    let mut deq_row = vec![0.0f32; in_];   // 复用（避免每行分配）
    for g in 0..groups {
        let zg = g * in_;
        let r_hi = ((g + 1) * group_size).min(out);
        for r in g * group_size..r_hi {
            let wr = r * in_;
            let wrow = &w[wr..wr + in_];
            let zrow = &z[zg..zg + in_];
            let srow = &scales[zg..zg + in_];
            for c in 0..in_ {
                deq_row[c] = (wrow[c] as i32 - zrow[c] as i32) as f32 * srow[c];
            }
            for bi in 0..b {
                let xrow = &x[bi * in_..(bi + 1) * in_];
                let mut acc = 0.0f32;
                for c in 0..in_ {
                    acc += xrow[c] * deq_row[c];
                }
                result[bi * out + r] = acc;
            }
        }
    }
    result
}

/// - `qzeros`:  `[out/group_size, in/8]` i32 打包零点
/// - `scales`:  `[out/group_size, in]` f32 缩放
pub fn dequantize_awq(qweight: &[i32], qzeros: &[i32], scales: &[f32],
                      out: usize, in_: usize, group_size: usize) -> Vec<f32> {
    assert!(in_ % 8 == 0, "in 维度须为 8 的倍数");
    let w = unpack_int4_colwise(qweight, in_ / 8);
    let z = unpack_int4_colwise(qzeros, in_ / 8);
    let groups = out / group_size;
    let mut result = vec![0.0f32; out * in_];
    for r in 0..out {
        let g = r / group_size;
        let wr = r * in_;
        let zg = g * in_;
        // 循环局部化（编译器可自动向量化——显著快于逐元素重复索引）
        let wrow = &w[wr..wr + in_];
        let zrow = &z[zg..zg + in_];
        let srow = &scales[zg..zg + in_];
        let out_row = &mut result[wr..wr + in_];
        for c in 0..in_ {
            out_row[c] = (wrow[c] as i32 - zrow[c] as i32) as f32 * srow[c];
        }
    }
    result
}

/// f32 → fp16（IEEE 754 half——round-to-nearest，纯 std）。
/// 镜像 Python / torch 的 .to(torch.float16) 输出语义。
pub fn f32_to_f16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x7FFFFF;
    // 数值剥符号后的指数
    let raw_exp = ((bits >> 23) & 0xFF) as i32 - 127;
    if raw_exp >= 16 {
        // 溢出 → 无穷（或最大 fp16）
        return sign | 0x7C00;
    }
    let biased = (raw_exp + 15) as i32;
    if biased <= 0 {
        // 次正规 / 零
        return sign | (mant >> 13) as u16;   // 简化次正规
    }
    let m = (mant >> 13) as u16;
    let round = ((mant >> 12) & 1) as u16;
    sign | ((biased as u16) << 10) | (m + round)
}

/// 把 f32 数组转成 fp16 的 u16 位型数组（compute_dtype="float16" 的反量化输出）。
pub fn to_f16_bits(data: &[f32]) -> Vec<u16> {
    data.iter().map(|&v| f32_to_f16_bits(v)).collect()
}

/// int4 原生 matmul（差异报告 #3）：AWQ int4 权重**解量化融合进 matmul 循环**，
/// 避免反量化结果写回内存再读取的双重往返。
///
/// x: (rows, in)；qweight/qzeros/scales 同 `dequantize_awq`。返回 (rows, out)。
/// 每元素解包沿用 `unpack_int4_colwise` 的列序语义（含 AWQ 逆序还原）。
pub fn matmul_awq_int4(x: &crate::core::tensor::Tensor, qweight: &[i32], qzeros: &[i32],
                       scales: &[f32], out: usize, in_: usize,
                       group_size: usize) -> crate::core::tensor::Tensor {
    assert_eq!(x.cols, in_, "输入列数须等于权重 in 维");
    let groups = out / group_size;
    let mut result = crate::core::tensor::Tensor::zeros(x.rows, out);
    for i in 0..x.rows {
        for j in 0..out {
            let g = j / group_size;
            let zg = g * in_;
            let mut acc = 0.0f32;
            for k in 0..in_ {
                // AWQ 列序还原：最终位置 k ← 原始槽位 perm[k]
                let slot = REVERSE_AWQ_PACK_ORDER[k & 7];
                let packed = qweight[j * (in_ / 8) + k / 8] as u32;
                let raw = ((packed >> (8 * (slot / 2) + 4 * (slot & 1))) & 0xF) as i32;
                let zq = z_byte(qzeros, g, k, in_);
                let w_val = (raw - zq) as f32 * scales[zg + k];
                acc += x.get(i, k) * w_val;
            }
            result.data[i * out + j] = acc;
        }
    }
    result
}

/// qzeros 的 (g, k) 元素（列打包 + AWQ 逆序），与解包语义一致。
fn z_byte(qzeros: &[i32], g: usize, k: usize, in_: usize) -> i32 {
    let slot = REVERSE_AWQ_PACK_ORDER[k & 7];
    let packed = qzeros[g * (in_ / 8) + k / 8] as u32;
    ((packed >> (8 * (slot / 2) + 4 * (slot & 1))) & 0xF) as i32
}

// ---- FP8（E4M3 / E5M2）与 NVFP4（E2M1）浮点格式 ----

/// FP8 E4M3 字节 → f32（1 符号 + 4 指数偏置 7 + 3 尾数；最大 448，e=15 为 NaN）。
pub fn e4m3_to_f32(b: u8) -> f32 {
    let s = (b >> 7) & 1;
    let e = (b >> 3) & 0xF;
    let m = (b & 0x7) as f32;
    let v = if e == 0 {
        m * 2.0f32.powi(-6)
    } else if e == 15 {
        f32::NAN
    } else {
        (1.0 + m / 8.0) * 2.0f32.powi(e as i32 - 7)
    };
    if s == 1 { -v } else { v }
}

/// FP8 E5M2 字节 → f32（1 符号 + 5 指数偏置 15 + 2 尾数；最大 57344，e=31 为 inf/NaN）。
pub fn e5m2_to_f32(b: u8) -> f32 {
    let s = (b >> 7) & 1;
    let e = (b >> 2) & 0x1F;
    let m = (b & 0x3) as f32;
    let v = if e == 0 {
        m * 2.0f32.powi(-14)
    } else if e == 31 {
        if m == 0.0 { f32::INFINITY } else { f32::NAN }
    } else {
        (1.0 + m / 4.0) * 2.0f32.powi(e as i32 - 15)
    };
    if s == 1 { -v } else { v }
}

/// NVFP4 E2M1 4 位值 → f32（值集 {0, 0.5, 1, 1.5, 2, 3, 4, 6}±）。
pub fn e2m1_to_f32(q: u8) -> f32 {
    let s = (q >> 3) & 1;
    let e = (q >> 1) & 0x3;
    let m = (q & 0x1) as f32;
    let v = if e == 0 {
        m * 0.5
    } else {
        (1.0 + m / 2.0) * 2.0f32.powi(e as i32 - 1)
    };
    if s == 1 { -v } else { v }
}

/// FP8 权重反量化：qweight 为 u8 字节（E4M3/E5M2），乘每张量缩放。
pub fn dequantize_fp8(qweight: &[u8], scale: f32, e4m3: bool) -> Vec<f32> {
    qweight.iter().map(|&b| {
        let w = if e4m3 { e4m3_to_f32(b) } else { e5m2_to_f32(b) };
        w * scale
    }).collect()
}

/// NVFP4 权重反量化：``x = e2m1(q) * s_block * s_global``（每 block_size 个元素一块）。
pub fn dequantize_nvfp4(qweight: &[u8], s_block: &[f32], s_global: f32,
                        block_size: usize) -> Vec<f32> {
    let n = qweight.len() * 2;
    let mut out = vec![0.0f32; n];
    for i in 0..n {
        let q = if i % 2 == 0 { qweight[i / 2] & 0xF } else { (qweight[i / 2] >> 4) & 0xF };
        out[i] = e2m1_to_f32(q) * s_block[i / block_size] * s_global;
    }
    out
}

/// 通用列打包解包（线性序——bits 2/4/8——GPTQ 无 AWQ 列序重排）。
fn unpack_generic(packed: &[u8], bits: usize) -> Vec<i32> {
    let per_byte = 8 / bits;
    let mask = ((1u16 << bits) - 1) as u8;   // bits=8 时 1<<8 用 u16 避免溢出
    let mut out = vec![0i32; packed.len() * per_byte];
    for (i, &b) in packed.iter().enumerate() {
        for j in 0..per_byte {
            out[i * per_byte + j] = ((b >> (bits * j)) & mask) as i32;
        }
    }
    out
}

/// GPTQ 反量化（线性序列打包——bits 2/4/8；sym 无零点；group_size 沿 out 分组）。
pub fn dequantize_gptq(qweight: &[u8], qzeros: Option<&[u8]>, scales: &[f32],
                       out: usize, in_: usize, bits: usize, group_size: usize,
                       sym: bool) -> Vec<f32> {
    assert!(in_ * bits % 8 == 0, "in*bits 须为 8 的倍数");
    let w = unpack_generic(qweight, bits);                    // (out * in_)
    let z = qzeros.map(|qz| unpack_generic(qz, bits));        // (groups * in_)
    let groups = (out + group_size - 1) / group_size;
    let mut result = vec![0.0f32; out * in_];
    for r in 0..out {
        let g = r / group_size;
        for c in 0..in_ {
            let wv = w[r * in_ + c];
            let q = match &z {
                Some(z) => (wv - z[g * in_ + c]) as f32,
                None => wv as f32,
            };
            result[r * in_ + c] = q * scales[g * in_ + c];
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unpack_linear_order() {
        // 单个 int32：低 4 位 1、2、...、8（无 AWQ 重排时线性读取）
        // 0x87654321：字节 0x21 → lo=1, hi=2；字节 0x43 → lo=3, hi=4；...
        let packed = [0x87654321u32 as i32];
        let out = unpack_int4_colwise(&packed, 1);
        // 线性解包后（重排前）应为 [1,2,3,4,5,6,7,8]；AWQ 逆序 [0,4,1,5,2,6,3,7]
        // 重排后：槽位 0=原0, 1=原4, 2=原1, 3=原5, 4=原2, 5=原6, 6=原3, 7=原7
        let expected = [1u8, 5, 2, 6, 3, 7, 4, 8];
        assert_eq!(&out[..8], &expected[..]);
    }

    #[test]
    fn test_dequant_shape_and_finite() {
        let out = 8usize;
        let in_ = 8usize;
        let gs = 8usize;
        let qw = vec![0x12345678i32, 0x12345678, 0x12345678, 0x12345678,
                      0x12345678, 0x12345678, 0x12345678, 0x12345678];
        let qz = vec![0x00000000i32];
        let sc = vec![0.1f32; in_];
        let dq = dequantize_awq(&qw, &qz, &sc, out, in_, gs);
        assert_eq!(dq.len(), out * in_);
        assert!(dq.iter().all(|v| v.is_finite()));
        assert!(dq.iter().any(|v| *v != 0.0));
    }

    #[test]
    fn test_int4_matmul_equivalence() {
        // 融合 int4 matmul == 反量化后普通 matmul（权威等价）
        let (out, in_, gs, rows) = (8usize, 32usize, 8usize, 4usize);
        let qw: Vec<i32> = (0..out * in_ / 8).map(|i| 0x76543210 + i as i32).collect();
        let qz = vec![0x11111111i32; out / gs * in_ / 8];
        let sc: Vec<f32> = (0..out / gs * in_).map(|i| 0.01 + (i % 5) as f32 * 0.001).collect();
        let x = crate::core::tensor::Tensor::from_vec(rows, in_,
            (0..rows * in_).map(|i| ((i % 7) as f32) * 0.1).collect());
        // 两步：反量化 → matmul
        let w = dequantize_awq(&qw, &qz, &sc, out, in_, gs);
        let w_t = crate::core::tensor::Tensor::from_vec(out, in_, w);
        let expect = x.matmul(&w_t.transpose());
        // 融合：解量化内联进 matmul
        let got = matmul_awq_int4(&x, &qw, &qz, &sc, out, in_, gs);
        let max_err = (0..rows * out)
            .map(|i| (got.data[i] - expect.data[i]).abs())
            .fold(0.0f32, f32::max);
        assert!(max_err < 1e-4, "int4 融合 matmul 与两步不等价, max_err={max_err}");
    }

    #[test]
    fn test_fp8_bit_layouts() {
        // E4M3：0x38 = 1.0；0xC3 = -2.75
        assert!((e4m3_to_f32(0x38) - 1.0).abs() < 1e-6);
        assert!((e4m3_to_f32(0xC3) + 2.75).abs() < 1e-6);
        // E5M2：0x3C = 1.0；0xBC = -1.0；0xC0 = -2.0
        assert!((e5m2_to_f32(0x3C) - 1.0).abs() < 1e-6);
        assert!((e5m2_to_f32(0xBC) + 1.0).abs() < 1e-6);
        assert!((e5m2_to_f32(0xC0) + 2.0).abs() < 1e-6);
        // E2M1：5 = 3.0；8 = -0.0
        assert!((e2m1_to_f32(5) - 3.0).abs() < 1e-6);
        assert!(e2m1_to_f32(8) == -0.0);
    }

    #[test]
    fn test_dequantize_gptq_sym4() {
        // bits=4 线性序：0x12 0x34 → [2,1,4,3]（低 4 位在前，in=2 使 in*bits=8）；sym → q*scale
        let out = dequantize_gptq(&[0x12, 0x34], None, &[0.5, 0.5], 2, 2, 4, 2, true);
        assert_eq!(out.len(), 4);
        assert!((out[0] - 1.0).abs() < 1e-5, "{}", out[0]);   // 2 * 0.5
        assert!((out[1] - 0.5).abs() < 1e-5, "{}", out[1]);   // 1 * 0.5
    }

    #[test]
    fn test_dequantize_gptq_asym8() {
        // bits=8 非对称：0x7F=127；零点 0x01=1；scale 0.1 → (127-1)*0.1 = 12.6
        let out = dequantize_gptq(&[0x7F], Some(&[0x01]), &[0.1], 1, 1, 8, 1, false);
        assert!((out[0] - 12.6).abs() < 1e-4, "{}", out[0]);
    }

    #[test]
    fn test_f32_to_f16() {
        // 1.0 → fp16 0x3C00；2.0 → 0x4000；0.5 → 0x3800
        assert_eq!(f32_to_f16_bits(1.0), 0x3C00);
        assert_eq!(f32_to_f16_bits(2.0), 0x4000);
        assert_eq!(f32_to_f16_bits(0.5), 0x3800);
        assert_eq!(f32_to_f16_bits(-1.0), 0xBC00);
        // 大值溢出 → 无穷
        assert_eq!(f32_to_f16_bits(1e30), 0x7C00);
        // 数组转换
        let bits = to_f16_bits(&[1.0, 2.0]);
        assert_eq!(bits, vec![0x3C00, 0x4000]);
    }

    #[test]
    fn test_fp_nvfp4_dequant() {
        // NVFP4：8 字节 → 16 个 4-bit（低 4 位 1→0.5，高 4 位 5→3.0）；block 0.5 × global 2.0
        let qw = [0x51u8; 8];
        let out = dequantize_nvfp4(&qw, &[0.5], 2.0, 16);
        assert_eq!(out.len(), 16);
        assert!((out[0] - 0.5).abs() < 1e-5);
        assert!((out[1] - 3.0).abs() < 1e-5);
        // FP8 E5M2：0x3C = 1.0 × 2.0
        let fp8 = dequantize_fp8(&[0x3C], 2.0, false);
        assert!((fp8[0] - 2.0).abs() < 1e-5);
        // FP8 E4M3：0x38 = 1.0 × 3.0
        let fp8e = dequantize_fp8(&[0x38], 3.0, true);
        assert!((fp8e[0] - 3.0).abs() < 1e-5);
    }
}

