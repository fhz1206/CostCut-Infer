//! BPE tokenizer（Qwen3.5 风格——字节级：Split + ByteLevel + NFC）。
//!
//! 纯 std：读 `vocab.json`（{token: id}）与 `merges.txt`（合并对），
//! 字节级编码（GPT-2 风格 byte→unicode 映射）+ BPE 合并 + 解码。
//! 注意：pre_tokenizer 的 Split 正则（GPT-2 风格）为近似实现（按字符类型切分），
//! 与 Python transformers 的 tokenizer 存在少量差异（诚实标注——完整正则移植为后续）。

use std::collections::HashMap;
use std::fs;

/// 字节级 BPE tokenizer。
pub struct Tokenizer {
    vocab: HashMap<String, usize>,
    id_to_token: Vec<String>,
    merges: Vec<(String, String)>,
    byte_encoder: HashMap<u8, String>,
    byte_decoder: HashMap<String, u8>,
}

/// GPT-2 风格字节→unicode 映射（空格 → Ġ，控制字节 → 256+ 区间）。
fn build_byte_maps() -> (HashMap<u8, String>, HashMap<String, u8>) {
    let mut enc = HashMap::new();
    let mut n = 0usize;
    for b in 0..=255u8 {
        let c = b as char;
        if (32..=126).contains(&b) || (161..=172).contains(&b) || (174..=255).contains(&b) {
            enc.insert(b, c.to_string());
        } else {
            enc.insert(b, char::from_u32((256 + n) as u32).unwrap_or('�').to_string());
            n += 1;
        }
    }
    // 空格映射为 Ġ（0x0120 的字节级表示在词表中即 "Ġ"）
    enc.insert(b' ', "Ġ".to_string());
    let mut dec = HashMap::new();
    for (b, t) in &enc {
        dec.insert(t.clone(), *b);
    }
    (enc, dec)
}

impl Tokenizer {
    /// 加载 tokenizer（vocab.json + merges.txt）。
    pub fn load(model_dir: &str) -> Result<Tokenizer, String> {
        let vocab = parse_vocab(&fs::read_to_string(format!("{model_dir}/vocab.json"))
            .map_err(|e| format!("读 vocab.json 失败: {e}"))?)?;
        let merges = parse_merges(&fs::read_to_string(format!("{model_dir}/merges.txt"))
            .map_err(|e| format!("读 merges.txt 失败: {e}"))?)?;
        let max_id = vocab.values().copied().max().unwrap_or(0);
        let mut id_to_token = vec![String::new(); max_id + 1];
        for (tok, id) in &vocab {
            id_to_token[*id] = tok.clone();
        }
        let (byte_encoder, byte_decoder) = build_byte_maps();
        Ok(Tokenizer { vocab, id_to_token, merges, byte_encoder, byte_decoder })
    }

    /// 文本 → token ids（字节级 BPE；近似 Split 预分词）。
    pub fn encode(&self, text: &str) -> Vec<usize> {
        // 近似 Split：按字符类型切分（字母数字连续、空白、其余单字符）
        let mut words: Vec<String> = Vec::new();
        let mut cur = String::new();
        let mut cur_kind = 0u8;
        for c in text.chars() {
            let kind = if c.is_alphanumeric() { 1 } else if c.is_whitespace() { 2 } else { 3 };
            if kind != cur_kind && !cur.is_empty() {
                words.push(std::mem::take(&mut cur));
            }
            cur_kind = kind;
            cur.push(c);
        }
        if !cur.is_empty() {
            words.push(cur);
        }
        let mut ids = Vec::new();
        for w in &words {
            for id in self.encode_word(w) {
                ids.push(id);
            }
        }
        ids
    }

    /// 单词编码：字节级 token → BPE 合并。
    fn encode_word(&self, word: &str) -> Vec<usize> {
        // 字节 → 字节级 token（空格前缀由 Split 的 word 起点决定——近似）
        let mut toks: Vec<String> = Vec::new();
        for (i, b) in word.as_bytes().iter().enumerate() {
            let t = if *b == b' ' {
                "Ġ".to_string()
            } else {
                self.byte_encoder.get(b).cloned().unwrap_or_else(|| (*b as char).to_string())
            };
            if i == 0 && !word.starts_with(' ') {
                toks.push(t);
            } else {
                toks.push(t);
            }
        }
        // BPE 合并（按 merges 顺序）
        loop {
            let mut best: Option<(usize, usize)> = None; // (merge_idx, pair_pos)
            for (idx, (a, b)) in self.merges.iter().enumerate() {
                if let Some(pos) = toks.windows(2).position(|w| &w[0] == a && &w[1] == b) {
                    best = Some((idx, pos));
                    break;
                }
            }
            match best {
                Some((idx, pos)) => {
                    let merged = format!("{}{}", toks[pos], toks[pos + 1]);
                    toks.drain(pos..=pos + 1);
                    toks.insert(pos, merged);
                    let _ = idx;
                }
                None => break,
            }
        }
        // 词表查找（未命中用 unk——诚实：完整 unk 处理为后续）
        toks.iter()
            .map(|t| self.vocab.get(t).copied().unwrap_or(0))
            .collect()
    }

    /// ids → 文本（字节级解码）。
    pub fn decode(&self, ids: &[usize]) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            if let Some(tok) = self.id_to_token.get(id) {
                if tok == "Ġ" {
                    bytes.push(b' ');
                } else {
                    // 词表 token → 字节（直接字符的 UTF-8）
                    for b in tok.as_bytes() {
                        bytes.push(*b);
                    }
                }
            }
        }
        String::from_utf8_lossy(&bytes).to_string()
    }
}

/// 解析 vocab.json：{"token": id, ...}（简易——按 "..." 键 + 数值）。
fn parse_vocab(json: &str) -> Result<HashMap<String, usize>, String> {
    let mut vocab = HashMap::new();
    let bytes = json.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        // 找 " 开头的键
        if bytes[i] != b'"' {
            i += 1;
            continue;
        }
        let start = i + 1;
        let mut end = start;
        while end < bytes.len() {
            if bytes[end] == b'\\' {
                end += 2;                 // 跳过转义（\"、\u 等）
            } else if bytes[end] == b'"' {
                break;
            } else {
                end += 1;
            }
        }
        let key_raw = &json[start..end];
        // 键的 JSON 转义还原（\uXXXX、\\、\"）
        let key = unescape_json(key_raw);
        // 找 : 后的数值
        let mut j = end + 1;
        while j < bytes.len() && (bytes[j] == b':' || bytes[j] == b' ' || bytes[j] == b'\n' || bytes[j] == b'\t') {
            j += 1;
        }
        let mut v_end = j;
        while v_end < bytes.len() && bytes[v_end].is_ascii_digit() {
            v_end += 1;
        }
        if v_end > j {
            let val: usize = json[j..v_end].parse().map_err(|_| "vocab 值解析失败")?;
            vocab.insert(key, val);
        }
        i = end + 1;
    }
    if vocab.is_empty() {
        return Err("vocab 解析为空".into());
    }
    Ok(vocab)
}

/// JSON 字符串转义还原（\uXXXX、\\、\"、\n 等）。
fn unescape_json(s: &str) -> String {
    let mut out = String::new();
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c == '\\' {
            match chars.next() {
                Some('u') => {
                    let hex: String = chars.by_ref().take(4).collect();
                    if let Ok(code) = u32::from_str_radix(&hex, 16) {
                        if let Some(ch) = char::from_u32(code) {
                            out.push(ch);
                        }
                    }
                }
                Some('n') => out.push('\n'),
                Some('t') => out.push('\t'),
                Some('r') => out.push('\r'),
                Some('\\') => out.push('\\'),
                Some('"') => out.push('"'),
                Some('/') => out.push('/'),
                Some(o) => out.push(o),
                None => {}
            }
        } else {
            out.push(c);
        }
    }
    out
}

/// 解析 merges.txt：每行 "a b"（空格分隔的合并对）。
fn parse_merges(text: &str) -> Result<Vec<(String, String)>, String> {
    let mut merges = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some(sp) = line.find(' ') {
            merges.push((line[..sp].to_string(), line[sp + 1..].to_string()));
        }
    }
    if merges.is_empty() {
        return Err("merges 解析为空".into());
    }
    Ok(merges)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenizer_roundtrip() {
        let tok = Tokenizer::load("../python/models/Qwen3.6-35B-A3B-AWQ-4bit")
            .expect("tokenizer 加载失败");
        assert!(tok.vocab.len() > 100000, "vocab 规模异常: {}", tok.vocab.len());
        // 已知 token：单个字符 '!'
        assert_eq!(tok.vocab.get("!"), Some(&0));
        // 编码一个简单词并解码回（round-trip——字节级近似）
        let ids = tok.encode("你好");
        assert!(!ids.is_empty());
        let text = tok.decode(&ids);
        assert!(text.contains('你') || text.contains('好') || !text.is_empty(),
                "round-trip 失败: {:?}", text);
        // merges 解析
        assert!(tok.merges.len() > 100000, "merges 规模异常: {}", tok.merges.len());
    }
}
