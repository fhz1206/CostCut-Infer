//! safetensors 解析：8 字节长度头 + JSON 头 + 张量数据（纯 std）。
//!
//! 格式：`[8-byte header_len][header_json][tensor_data...]`，
//! 头为 JSON：`{"tensor_name": {"dtype": "F32|F16|BF16|I32", "shape": [...], "data_offsets": [b, e]}, ...}`。
//!
//! 本实现支持读取 I32 / F32 / F16 / BF16 张量并统一转换为 f32（引擎计算 dtype）。

use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};

/// 张量元数据。
#[derive(Clone, Debug)]
pub struct TensorMeta {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub offset: usize,
    pub length: usize,
}

/// 已打开的 safetensors 文件（惰性读取：仅解析头，张量按需 seek 读取，不整读大分片）。
pub struct SafeTensors {
    file: File,
    pub tensors: HashMap<String, TensorMeta>,
}

/// 简易 JSON 字段提取：把 `"key": value` 的顶层字段解析为字符串。
/// 仅支持扁平结构（safetensors 头与配置解析够用），不支持嵌套数组内对象。
pub fn extract_fields(json: &str) -> HashMap<String, String> {
    let mut out = HashMap::new();
    let bytes = json.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        // 找下一个 "key"
        if bytes[i] == b'"' {
            let key_start = i + 1;
            let mut j = key_start;
            while j < bytes.len() && bytes[j] != b'"' {
                j += 1;
            }
            let key = json[key_start..j].to_string();
            i = j + 1;
            // 跳过空白与冒号
            while i < bytes.len() && (bytes[i].is_ascii_whitespace() || bytes[i] == b':') {
                i += 1;
            }
            // 值：字符串 / 数字 / 数组 / 对象（只取首层）
            let (val, ni) = if i < bytes.len() && bytes[i] == b'"' {
                let v0 = i + 1;
                let mut j = v0;
                while j < bytes.len() && bytes[j] != b'"' {
                    j += 1;
                }
                (json[v0..j].to_string(), j + 1)
            } else {
                // 数字或数组/对象的原始文本：读到逗号或右花括号
                let v0 = i;
                let mut j = i;
                let mut depth = 0i32;
                while j < bytes.len() {
                    match bytes[j] {
                        b'{' | b'[' => depth += 1,
                        b'}' | b']' => {
                            depth -= 1;
                            if depth <= 0 && bytes[j] == b'}' {
                                j += 1;
                                break;
                            }
                        }
                        b',' if depth == 0 => break,
                        _ => {}
                    }
                    j += 1;
                }
                (json[v0..j].trim().to_string(), j)
            };
            out.insert(key, val);
            i = ni;
        } else {
            i += 1;
        }
    }
    out
}

impl SafeTensors {
    /// 打开并解析 safetensors 文件。
    pub fn open(path: &str) -> Result<Self, String> {
        let mut f = File::open(path).map_err(|e| format!("打开文件失败 {path}: {e}"))?;
        let mut header_len_buf = [0u8; 8];
        f.read_exact(&mut header_len_buf).map_err(|e| format!("读取头长度失败: {e}"))?;
        let header_len = u64::from_le_bytes(header_len_buf) as usize;
        let mut header_bytes = vec![0u8; header_len];
        f.read_exact(&mut header_bytes).map_err(|e| format!("读取头失败: {e}"))?;
        let header_json = String::from_utf8_lossy(&header_bytes).to_string();
        // 头部是嵌套对象：外层 key = 张量名，值 = 内层对象
        let mut tensors = HashMap::new();
        let bytes = header_json.as_bytes();
        let mut i = 0usize;
        while i < bytes.len() {
            if bytes[i] == b'"' {
                let ks = i + 1;
                let mut j = ks;
                while j < bytes.len() && bytes[j] != b'"' {
                    j += 1;
                }
                let name = header_json[ks..j].to_string();
                i = j + 1;
                while i < bytes.len() && (bytes[i].is_ascii_whitespace() || bytes[i] == b':') {
                    i += 1;
                }
                if i < bytes.len() && bytes[i] == b'{' {
                    // 提取内层对象的原始文本
                    let v0 = i + 1;
                    let mut depth = 1i32;
                    let mut j = i + 1;
                    while j < bytes.len() && depth > 0 {
                        match bytes[j] {
                            b'{' => depth += 1,
                            b'}' => depth -= 1,
                            _ => {}
                        }
                        j += 1;
                    }
                    let inner = &header_json[v0..j - 1];
                    let fields = extract_fields(inner);
                    if let (Some(dtype), Some(shape_raw), Some(offsets)) =
                        (fields.get("dtype"), fields.get("shape"), fields.get("data_offsets"))
                    {
                        let shape: Vec<usize> = shape_raw
                            .trim_matches(|c| c == '[' || c == ']')
                            .split(',')
                            .filter(|s| !s.trim().is_empty())
                            .map(|s| s.trim().parse().unwrap_or(0))
                            .collect();
                        let nums: Vec<usize> = offsets
                            .trim_matches(|c| c == '[' || c == ']')
                            .split(',')
                            .filter(|s| !s.trim().is_empty())
                            .map(|s| s.trim().parse().unwrap_or(0))
                            .collect();
                        if nums.len() == 2 {
                            let (b, e) = (nums[0], nums[1]);
                            tensors.insert(name, TensorMeta {
                                dtype: dtype.clone(),
                                shape,
                                offset: 8 + header_len + b,
                                length: e - b,
                            });
                        }
                    }
                    i = j;
                } else {
                    i += 1;
                }
            } else {
                i += 1;
            }
        }
        Ok(SafeTensors { file: f, tensors })
    }

    /// 按元数据 seek 读取张量原始字节。
    fn read_bytes(&self, meta: &TensorMeta) -> Option<Vec<u8>> {
        let mut f = &self.file;
        f.seek(SeekFrom::Start(meta.offset as u64)).ok()?;
        let mut buf = vec![0u8; meta.length];
        f.read_exact(&mut buf).ok()?;
        Some(buf)
    }

    /// 读取张量为 f32（I32/F32 直转，F16/BF16 转换）。
    pub fn get_f32(&self, name: &str) -> Option<Vec<f32>> {
        let meta = self.tensors.get(name)?;
        let slice = self.read_bytes(meta)?;
        let out = match meta.dtype.as_str() {
            "F32" => slice.chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect(),
            "I32" => slice.chunks_exact(4)
                .map(|c| i32::from_le_bytes(c.try_into().unwrap()) as f32)
                .collect(),
            "F16" => slice.chunks_exact(2)
                .map(|c| f16_to_f32(u16::from_le_bytes(c.try_into().unwrap())))
                .collect(),
            "BF16" => slice.chunks_exact(2)
                .map(|c| bf16_to_f32(u16::from_le_bytes(c.try_into().unwrap())))
                .collect(),
            "F8_E4M3" => slice.iter().map(|&b| crate::quant::dequant::e4m3_to_f32(b)).collect(),
            "F8_E5M2" => slice.iter().map(|&b| crate::quant::dequant::e5m2_to_f32(b)).collect(),
            other => return None,
        };
        Some(out)
    }

    /// 读取张量为 i32（qweight/qzeros 用）。
    pub fn get_i32(&self, name: &str) -> Option<Vec<i32>> {
        let meta = self.tensors.get(name)?;
        if meta.dtype != "I32" {
            return None;
        }
        let slice = self.read_bytes(meta)?;
        Some(slice.chunks_exact(4)
            .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
            .collect())
    }
}

/// IEEE 754 fp16 → f32。
fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1F) as u32;
    let frac = (h & 0x3FF) as u32;
    if exp == 0 {
        if frac == 0 {
            if sign == 1 { -0.0 } else { 0.0 }
        } else {
            let m = frac as f32 / 1024.0;
            if sign == 1 { -(m * 2.0f32.powi(-14)) } else { m * 2.0f32.powi(-14) }
        }
    } else if exp == 31 {
        f32::NAN
    } else {
        let m = 1.0 + frac as f32 / 1024.0;
        let v = m * 2.0f32.powi(exp as i32 - 15);
        if sign == 1 { -v } else { v }
    }
}

/// bf16 → f32（直接左移 16 位）。
fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f16_conversion() {
        // 1.0 = 0x3C00
        assert!((f16_to_f32(0x3C00) - 1.0).abs() < 1e-6);
        // -2.0 = 0xC000
        assert!((f16_to_f32(0xC000) + 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_bf16_conversion() {
        // 1.0 bf16 = 0x3F80
        assert!((bf16_to_f32(0x3F80) - 1.0).abs() < 1e-6);
    }
}
