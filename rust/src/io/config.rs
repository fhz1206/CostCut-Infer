//! 轻量 TOML 行式解析（纯 std——无外部依赖，适配离线构建）。
//!
//! 仅解析本项目 engine.toml 需要的扁平结构：
//! - `[section]` 区块标记
//! - `key = value` 键值（'#' 后注释；value 可为字符串（去引号）/数值/布尔）
//! 字段访问：`get("inference", "temperature")`、`get_bool("inference", "speculate")` 等。

use std::fs;
use std::path::Path;

/// 解析后的配置：section → (key, value 字符串)。
#[derive(Default)]
pub struct TomlConfig {
    data: std::collections::HashMap<String, std::collections::HashMap<String, String>>,
}

impl TomlConfig {
    /// 从路径加载 engine.toml（路径缺失返回空配置）。
    pub fn load(path: &str) -> TomlConfig {
        let mut cfg = TomlConfig::default();
        if let Ok(text) = fs::read_to_string(path) {
            cfg.parse(&text);
        }
        cfg
    }

    /// 从文本解析。
    pub fn parse(&mut self, text: &str) {
        let mut section = String::new();
        for raw in text.lines() {
            let line = strip_comment(raw).trim().to_string();
            if line.is_empty() {
                continue;
            }
            if line.starts_with('[') && line.ends_with(']') {
                section = line[1..line.len() - 1].trim().to_string();
                continue;
            }
            if let Some(eq) = line.find('=') {
                let key = line[..eq].trim().to_string();
                let val = line[eq + 1..].trim().trim_matches('"').trim().to_string();
                self.data.entry(section.clone()).or_default().insert(key, val);
            }
        }
    }

    /// 取字符串值。
    pub fn get(&self, section: &str, key: &str) -> Option<&str> {
        self.data.get(section).and_then(|m| m.get(key)).map(|s| s.as_str())
    }

    /// 取布尔值（"true"/"false"，缺省 false）。
    pub fn get_bool(&self, section: &str, key: &str) -> bool {
        self.get(section, key).map(|v| v == "true" || v == "1").unwrap_or(false)
    }

    /// 取 f32 值（缺省 default）。
    pub fn get_f32(&self, section: &str, key: &str, default: f32) -> f32 {
        self.get(section, key).and_then(|v| v.parse().ok()).unwrap_or(default)
    }

    /// 取整数。
    pub fn get_int(&self, section: &str, key: &str, default: i64) -> i64 {
        self.get(section, key).and_then(|v| v.parse().ok()).unwrap_or(default)
    }

    /// 某区块存在。
    pub fn has_section(&self, section: &str) -> bool {
        self.data.contains_key(section)
    }

    /// 某区块内 key 存在。
    pub fn has(&self, section: &str, key: &str) -> bool {
        self.data.get(section).map_or(false, |m| m.contains_key(key))
    }
}

/// 去掉行内注释（'#' 不在引号内时截断——本项目值不含 #，简化为首个 '#' 截断）。
fn strip_comment(line: &str) -> &str {
    match line.find('#') {
        Some(i) => &line[..i],
        None => line,
    }
}

/// 便捷：解析 engine.toml（依次尝试常见位置——Rust 版 CWD：rust/ 或项目根或 python/）。
pub fn load_engine_toml() -> TomlConfig {
    for cand in ["engine.toml", "rust/src/engine.toml", "python/engine.toml"] {
        if Path::new(cand).exists() {
            return TomlConfig::load(cand);
        }
    }
    let _ = Path::new("");   // 无匹配——返回空配置
    TomlConfig::default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_toml_parse() {
        let text = r#"
[default]
model = "qwen"       # 默认
[inference]
temperature = 0.9
speculate = false
compute_dtype = "float32"
expert_parallel = true
[model]
name = "qwen"
dspark_model = "spec.dspark"
"#;
        let mut cfg = TomlConfig::default();
        cfg.parse(text);
        assert_eq!(cfg.get("default", "model"), Some("qwen"));
        assert!((cfg.get_f32("inference", "temperature", 1.0) - 0.9).abs() < 1e-4);
        assert!(!cfg.get_bool("inference", "speculate"));
        assert!(cfg.get_bool("inference", "expert_parallel"));
        assert_eq!(cfg.get("model", "dspark_model"), Some("spec.dspark"));
        assert_eq!(cfg.get("inference", "compute_dtype"), Some("float32"));
        // 缺省
        assert!(!cfg.get_bool("inference", "int4_fused_matmul"));
    }
}
