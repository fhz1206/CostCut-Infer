//! GGUF 解析与反量化（镜像 Python `liteengine/gguf.py`——非 K 系列）。
//!
//! 支持：头部/元数据解析 + 张量索引（惰性 seek）+ F32/F16/Q4_0 反量化。
//! Q4_1/Q5_x/Q8_0/K 系列为后续（需 ggml 参考——诚实标注）。

use std::collections::HashMap;
use std::fs;

/// 张量索引条目。
pub struct GgufTensor {
    pub name: String,
    pub shape: Vec<usize>,
    pub ggml_type: u32,
    pub offset: u64,
}

/// GGUF 读取器（惰性——权重按需反量化）。
pub struct GgufReader {
    data: Vec<u8>,
    pub tensors: Vec<GgufTensor>,
    tensor_map: HashMap<String, usize>,
    pub metadata: HashMap<String, String>,
}

/// 读取一个 KV：返回 (key, value 字符串, 下一位置)。镜像 Python _read_value 的值类型。
fn read_kv(data: &[u8], pos: usize) -> Result<(String, String, usize), String> {
    let (key, mut p) = read_string(data, pos)?;
    if p + 4 > data.len() {
        return Err("GGUF 值类型越界".into());
    }
    let t = u32::from_le_bytes(data[p..p + 4].try_into().unwrap());
    p += 4;
    match t {
        0 => Ok((key, data[p].to_string(), p + 1)),                    // u8
        1 => Ok((key, (data[p] as i8 as i32).to_string(), p + 1)),     // i8
        2 => Ok((key, u16::from_le_bytes(data[p..p + 2].try_into().unwrap()).to_string(), p + 2)),
        3 => Ok((key, i16::from_le_bytes(data[p..p + 2].try_into().unwrap()).to_string(), p + 2)),
        4 => Ok((key, u32::from_le_bytes(data[p..p + 4].try_into().unwrap()).to_string(), p + 4)),
        5 => Ok((key, i32::from_le_bytes(data[p..p + 4].try_into().unwrap()).to_string(), p + 4)),
        6 => Ok((key, f32::from_le_bytes(data[p..p + 4].try_into().unwrap()).to_string(), p + 4)),
        7 => Ok((key, (data[p] != 0).to_string(), p + 1)),              // bool
        8 => {
            let (v, np) = read_string(data, p)?;
            Ok((key, v, np))
        }
        9 => {   // array——跳过元素，值记为 "[array n]"
            let et = u32::from_le_bytes(data[p..p + 4].try_into().unwrap());
            let n = u64::from_le_bytes(data[p + 4..p + 12].try_into().unwrap()) as usize;
            let mut q = p + 12;
            for _ in 0..n {
                q = skip_elem(data, q, et)?;
            }
            Ok((key, format!("[array {n}]"), q))
        }
        10..=11 => Ok((key, u64::from_le_bytes(data[p..p + 8].try_into().unwrap()).to_string(), p + 8)),
        12 => Ok((key, f64::from_le_bytes(data[p..p + 8].try_into().unwrap()).to_string(), p + 8)),
        _ => Err(format!("未知 GGUF 值类型 {t}")),
    }
}

fn f16_to_f32(h: u16) -> f32 {
    let s = ((h >> 15) & 1) as u32;
    let e = ((h >> 10) & 0x1F) as u32;
    let m = (h & 0x3FF) as u32;
    let v = if e == 0 {
        if m == 0 { 0.0 } else { (m as f32) * 2.0f32.powi(-24) }
    } else if e == 31 {
        if m == 0 { f32::INFINITY } else { f32::NAN }
    } else {
        (1.0 + m as f32 / 1024.0) * 2.0f32.powi(e as i32 - 15)
    };
    if s == 1 { -v } else { v }
}

fn read_string(data: &[u8], pos: usize) -> Result<(String, usize), String> {
    if pos + 8 > data.len() {
        return Err("GGUF 字符串长度越界".into());
    }
    let len = u64::from_le_bytes(data[pos..pos + 8].try_into().unwrap()) as usize;
    let start = pos + 8;
    let end = start + len;
    if end > data.len() {
        return Err("GGUF 字符串越界".into());
    }
    Ok((String::from_utf8_lossy(&data[start..end]).to_string(), end))
}

fn skip_elem(data: &[u8], pos: usize, t: u32) -> Result<usize, String> {
    match t {
        0..=5 | 7 => Ok(pos + 4),          // u8/i8/u16/i16/u32/f32（含 bool）
        6 => read_string(data, pos).map(|(_, p)| p),   // f64
        8 => Ok(pos + 8),                  // string（key 已跳过——此处为 u64/int64）
        9..=10 => Ok(pos + 8),             // i64/u64
        _ => Err(format!("未知 GGUF 数组元素类型 {t}")),
    }
}

fn skip_value(data: &[u8], pos: usize) -> Result<usize, String> {
    let (_key, mut p) = read_string(data, pos)?;
    if p + 4 > data.len() {
        return Err("GGUF 值类型越界".into());
    }
    let t = u32::from_le_bytes(data[p..p + 4].try_into().unwrap());
    p += 4;
    match t {
        0 => Ok(p + 1),                    // u8
        1 => Ok(p + 1),                    // i8
        2 => Ok(p + 2),                    // u16
        3 => Ok(p + 2),                    // i16
        4..=6 => Ok(p + 4),                // u32/i32/f32
        7 => Ok(p + 1),                    // bool
        8 => read_string(data, p).map(|(_, np)| np),   // string
        9 => {                             // array
            if p + 12 > data.len() {
                return Err("GGUF 数组头越界".into());
            }
            let et = u32::from_le_bytes(data[p..p + 4].try_into().unwrap());
            let n = u64::from_le_bytes(data[p + 4..p + 12].try_into().unwrap()) as usize;
            let mut q = p + 12;
            for _ in 0..n {
                q = skip_elem(data, q, et)?;
            }
            Ok(q)
        }
        10..=11 => Ok(p + 8),              // u64/i64
        12 => Ok(p + 8),                   // f64
        _ => Err(format!("未知 GGUF 值类型 {t}")),
    }
}

impl GgufReader {
    /// 打开 GGUF 文件（头部 + 元数据 + 张量索引）。
    pub fn open(path: &str) -> Result<GgufReader, String> {
        let data = fs::read(path).map_err(|e| format!("读 GGUF 失败: {e}"))?;
        if data.len() < 24 || &data[0..4] != b"GGUF" {
            return Err("非 GGUF 文件".into());
        }
        let n_tensors = u64::from_le_bytes(data[8..16].try_into().unwrap()) as usize;
        let n_kv = u64::from_le_bytes(data[16..24].try_into().unwrap()) as usize;
        let mut pos = 24usize;
        let mut metadata = HashMap::new();
        for _ in 0..n_kv {
            let (k, v, np) = read_kv(&data, pos)?;
            metadata.insert(k, v);
            pos = np;
        }
        let mut tensors = Vec::with_capacity(n_tensors);
        let mut tensor_map = HashMap::new();
        for _ in 0..n_tensors {
            let (name, mut p) = read_string(&data, pos)?;
            if p + 4 > data.len() {
                return Err("GGUF 张量索引越界".into());
            }
            let n_dims = u32::from_le_bytes(data[p..p + 4].try_into().unwrap()) as usize;
            p += 4;
            let mut shape = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                if p + 8 > data.len() {
                    return Err("GGUF 形状越界".into());
                }
                shape.push(u64::from_le_bytes(data[p..p + 8].try_into().unwrap()) as usize);
                p += 8;
            }
            let ggml_type = u32::from_le_bytes(data[p..p + 4].try_into().unwrap());
            p += 4;
            let offset = u64::from_le_bytes(data[p..p + 8].try_into().unwrap());
            p += 8;
            pos = p;
            tensor_map.insert(name.clone(), tensors.len());
            tensors.push(GgufTensor { name, shape, ggml_type, offset });
        }
        Ok(GgufReader { data, tensors, tensor_map, metadata })
    }

    /// 从元数据构造配置（架构探测：llama/mistral → dense；mixtral → MoE——镜像 Python
    /// gguf_metadata_to_config；缺参返回 Err）。
    pub fn metadata_to_config(&self) -> Result<GgufConfig, String> {
        let arch = self.metadata.get("general.architecture")
            .map(|s| s.to_lowercase()).unwrap_or_else(|| "llama".to_string());
        let get = |k: &str, d: &str| -> String {
            self.metadata.get(k).cloned().unwrap_or_else(|| d.to_string())
        };
        let n_layer = get(&format!("{arch}.block_count"), "0").parse::<usize>().unwrap_or(0);
        let head_count = get(&format!("{arch}.attention.head_count"), "0").parse::<usize>().unwrap_or(0);
        let hidden = get(&format!("{arch}.embedding_length"), "0").parse::<usize>().unwrap_or(0);
        if n_layer == 0 || head_count == 0 || hidden == 0 {
            return Err(format!("GGUF 元数据缺少模型参数（architecture={arch}）"));
        }
        let kv_heads = get(&format!("{arch}.attention.head_count_kv"), "")
            .parse::<usize>().ok().filter(|&v| v > 0).unwrap_or(head_count);
        let intermediate = get(&format!("{arch}.feed_forward_length"), "")
            .parse::<usize>().ok().filter(|&v| v > 0).unwrap_or(4 * hidden);
        Ok(GgufConfig {
            arch: arch.clone(),
            n_layer,
            head_count,
            kv_heads,
            hidden,
            vocab: get(&format!("{arch}.vocab_size"), "32000").parse::<usize>().unwrap_or(32000),
            eps: get(&format!("{arch}.attention.layer_norm_rms_epsilon"), "1e-5")
                .parse::<f32>().unwrap_or(1e-5),
            theta: get(&format!("{arch}.rope.freq_base"), "10000.0")
                .parse::<f32>().unwrap_or(10000.0),
            intermediate,
            expert_count: get(&format!("{arch}.expert_count"), "0").parse::<usize>().unwrap_or(0),
            experts_per_tok: get(&format!("{arch}.expert_used_count"), "2")
                .parse::<usize>().unwrap_or(2),
        })
    }

    /// 按名称读取张量并反量化为 f32（F32/F16/Q4_0——其余类型返回 None）。
    pub fn get_f32(&self, name: &str) -> Option<Vec<f32>> {
        let idx = *self.tensor_map.get(name)?;
        let t = &self.tensors[idx];
        let n: usize = t.shape.iter().product();
        let start = t.offset as usize;
        if start >= self.data.len() {
            return None;
        }
        let raw = &self.data[start..];
        match t.ggml_type {
            0 => Some(raw.chunks_exact(4)
                .take(n)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect()),
            1 => Some(raw.chunks_exact(2)
                .take(n)
                .map(|c| f16_to_f32(u16::from_le_bytes(c.try_into().unwrap())))
                .collect()),
            2 => Some(dequant_q4_0(raw, n)),
            3 => Some(dequant_q4_1(raw, n)),
            6 => Some(dequant_q5_0(raw, n)),
            7 => Some(dequant_q5_1(raw, n)),
            8 => Some(dequant_q8_0(raw, n)),
            _ => None,
        }
    }
}

/// GGUF 元数据 → 配置结果（镜像 Python gguf_metadata_to_config 的返回字段）。
pub struct GgufConfig {
    pub arch: String,
    pub n_layer: usize,
    pub head_count: usize,
    pub kv_heads: usize,
    pub hidden: usize,
    pub vocab: usize,
    pub eps: f32,
    pub theta: f32,
    pub intermediate: usize,
    pub expert_count: usize,
    pub experts_per_tok: usize,
}

/// Q4_0 反量化（ggml 公式）：每 32 值一块——f16 scale + 16 字节（低 4 位前 16 值，高 4 位后 16 值）。
fn dequant_q4_0(data: &[u8], n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n];
    let mut pos = 0usize;
    let mut i = 0usize;
    while i < n && pos + 18 <= data.len() {
        let scale = f16_to_f32(u16::from_le_bytes(data[pos..pos + 2].try_into().unwrap()));
        pos += 2;
        for b in 0..16 {
            let q = data[pos + b];
            let lo = ((q & 0x0F) as i8 - 8) as f32 * scale;
            let hi = (((q >> 4) & 0x0F) as i8 - 8) as f32 * scale;
            if i < n {
                out[i] = lo;
            }
            if i + 1 < n {
                out[i + 1] = hi;
            }
            i += 2;
        }
        pos += 16;
    }
    out
}

/// Q4_1 反量化（ggml）：每 32 值一块——f16 d + f16 m + 16 字节 int4。y = q*d + m。
fn dequant_q4_1(data: &[u8], n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n];
    let mut pos = 0usize;
    let mut i = 0usize;
    while i < n && pos + 20 <= data.len() {
        let d = f16_to_f32(u16::from_le_bytes(data[pos..pos + 2].try_into().unwrap()));
        let m = f16_to_f32(u16::from_le_bytes(data[pos + 2..pos + 4].try_into().unwrap()));
        pos += 4;
        for b in 0..16 {
            let q = data[pos + b];
            let lo = (q & 0x0F) as f32 * d + m;
            let hi = ((q >> 4) & 0x0F) as f32 * d + m;
            if i < n { out[i] = lo; }
            if i + 1 < n { out[i + 1] = hi; }
            i += 2;
        }
        pos += 16;
    }
    out
}

/// Q5_0 反量化（ggml）：d + 16 字节低 4 位 + 4 字节高 1 位。y = ((q | (qh<<4)) - 16) * d。
fn dequant_q5_0(data: &[u8], n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n];
    let mut pos = 0usize;
    let mut i = 0usize;
    while i < n && pos + 22 <= data.len() {
        let d = f16_to_f32(u16::from_le_bytes(data[pos..pos + 2].try_into().unwrap()));
        pos += 2;
        let ql = &data[pos..pos + 16];
        let qh = &data[pos + 16..pos + 20];
        pos += 20;
        for j in 0..32 {
            let q = (ql[j / 2] >> (4 * (j & 1))) & 0x0F;
            let hi = (qh[j / 8] >> (j & 7)) & 1;
            let v = ((q | (hi << 4)) as i32 - 16) as f32 * d;
            if i + j < n { out[i + j] = v; }
        }
        i += 32;
    }
    out
}

/// Q5_1 反量化（ggml）：d + m + 16 字节低 4 位 + 4 字节高 1 位。y = (q | (qh<<4)) * d + m。
fn dequant_q5_1(data: &[u8], n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n];
    let mut pos = 0usize;
    let mut i = 0usize;
    while i < n && pos + 24 <= data.len() {
        let d = f16_to_f32(u16::from_le_bytes(data[pos..pos + 2].try_into().unwrap()));
        let m = f16_to_f32(u16::from_le_bytes(data[pos + 2..pos + 4].try_into().unwrap()));
        pos += 4;
        let ql = &data[pos..pos + 16];
        let qh = &data[pos + 16..pos + 20];
        pos += 20;
        for j in 0..32 {
            let q = (ql[j / 2] >> (4 * (j & 1))) & 0x0F;
            let hi = (qh[j / 8] >> (j & 7)) & 1;
            let v = (q | (hi << 4)) as f32 * d + m;
            if i + j < n { out[i + j] = v; }
        }
        i += 32;
    }
    out
}

/// Q8_0 反量化（ggml）：d + 32 个 int8。y = q * d。
fn dequant_q8_0(data: &[u8], n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n];
    let mut pos = 0usize;
    let mut i = 0usize;
    while i < n && pos + 34 <= data.len() {
        let d = f16_to_f32(u16::from_le_bytes(data[pos..pos + 2].try_into().unwrap()));
        pos += 2;
        for j in 0..32 {
            let q = data[pos + j] as i8 as i32;
            if i + j < n { out[i + j] = q as f32 * d; }
        }
        pos += 32;
        i += 32;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_mini_gguf() -> Vec<u8> {
        // 最小 GGUF：1 个 F16 张量 [2]（值 1.0, 2.0）
        let mut d = Vec::new();
        d.extend_from_slice(b"GGUF");
        d.extend_from_slice(&3u32.to_le_bytes());          // version
        d.extend_from_slice(&1u64.to_le_bytes());          // tensor 数
        d.extend_from_slice(&0u64.to_le_bytes());          // kv 数
        // 张量索引：name="w" shape=[2] type=F16 offset=0
        d.extend_from_slice(&1u64.to_le_bytes());
        d.extend_from_slice(b"w");
        d.extend_from_slice(&1u32.to_le_bytes());          // 1 维
        d.extend_from_slice(&2u64.to_le_bytes());          // shape[0]=2
        d.extend_from_slice(&1u32.to_le_bytes());          // F16
        d.extend_from_slice(&57u64.to_le_bytes());         // offset=数据位置（24头+33索引）
        // 数据：1.0f16, 2.0f16
        d.extend_from_slice(&(0x3C00u16).to_le_bytes());   // 1.0
        d.extend_from_slice(&(0x4000u16).to_le_bytes());   // 2.0
        d
    }

    #[test]
    fn test_gguf_open_and_f16() {
        let path = std::env::temp_dir().join("mini_test.gguf");
        fs::write(&path, build_mini_gguf()).unwrap();
        let g = GgufReader::open(path.to_str().unwrap()).expect("GGUF 打开失败");
        assert_eq!(g.tensors.len(), 1);
        assert_eq!(g.tensors[0].name, "w");
        let v = g.get_f32("w").expect("读取失败");
        assert_eq!(v.len(), 2);
        assert!((v[0] - 1.0).abs() < 1e-4);
        assert!((v[1] - 2.0).abs() < 1e-4);
        let _ = fs::remove_file(path);
    }
}
