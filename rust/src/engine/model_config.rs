//! 模型配置归一化（镜像 Python `liteengine/model_config.py`）：
//! 架构探测（architectures / model_type 的原始文本子串）→ 引擎内部统一配置
//! + 通用回退（未知但非专属架构：标准解码器字段齐全且无线性注意力层）。
//!
//! 纯 std：JSON 字段提取复用 `safetensors::extract_fields`（扁平键值）。
use crate::io::safetensors::extract_fields;
use std::collections::HashMap;
use std::fs;

/// 归一化后的引擎内部配置（注意力 / MLP 分发数据）。
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub arch: String,            // 架构名（mixtral/qwen3_moe/glm_moe/deepseek_moe/.../generic）
    pub kind: String,            // 设备类型：cpu/gpu/npu/apu（CPU 默认——tch CUDA 可扩展）
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub eps: f32,
    pub rope_theta: f32,
    pub attention: String,       // "standard" | "mla" | "full_gated" | "linear_delta"
    pub moe: Option<MoeConfig>,  // None = 稠密 MLP（dense_mlp 路径）
    // 真实模型组装所需（Qwen3.5）
    pub vocab_size: usize,
    pub weight_prefix: String,
    pub layer_types: Vec<String>,
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    pub conv_kernel_size: usize,
    pub rope_dim: usize,
    pub moe_intermediate: usize,
    pub compute_dtype: String,   // "float32" / "float16" / "bf16"（fp16 权重路径）
}

/// MoE 规格。
#[derive(Debug, Clone)]
pub struct MoeConfig {
    pub num_experts: usize,
    pub top_k: usize,
    pub intermediate: usize,
    pub shared: bool,
    pub group_size: usize,
}

fn num(fields: &HashMap<String, String>, key: &str, default: usize) -> usize {
    fields.get(key).and_then(|v| v.trim().parse().ok()).unwrap_or(default)
}

fn fnum(fields: &HashMap<String, String>, key: &str, default: f32) -> f32 {
    fields.get(key).and_then(|v| v.trim().parse().ok()).unwrap_or(default)
}

/// 读取并归一化模型配置（架构探测 + 通用回退）。
/// 设备自动检测（tch 只支持 CPU/CUDA 两级——NPU/ROCm 无 tch 后端——诚实）。
pub fn detect_device() -> String {
    if tch::Cuda::is_available() {
        "gpu".to_string()
    } else {
        "cpu".to_string()
    }
}

pub fn load_model_config(model_dir: &str) -> Result<ModelConfig, String> {
    let text = fs::read_to_string(format!("{model_dir}/config.json"))
        .map_err(|e| format!("读取 config.json 失败: {e}"))?;
    let mut fields = extract_fields(&text);
    // Qwen3.5 等嵌套结构：核心字段在 text_config 子对象下——提取并合并到顶层
    if num(&fields, "hidden_size", 0) == 0 {
        if let Some(start) = text.find("\"text_config\"") {
            if let Some(open) = text[start..].find('{') {
                let open = start + open;
                let mut depth = 1i32;
                let mut close = open + 1;
                while close < text.len() && depth > 0 {
                    match text.as_bytes()[close] {
                        b'{' => depth += 1,
                        b'}' => depth -= 1,
                        _ => {}
                    }
                    close += 1;
                }
                if depth == 0 {
                    let inner = extract_fields(&text[open + 1..close - 1]);
                    for (k, v) in inner {
                        fields.entry(k).or_insert(v);
                    }
                }
            }
        }
    }
    let raw = text.to_lowercase();
    let hidden = num(&fields, "hidden_size", 0);
    let n_layers = num(&fields, "num_hidden_layers", 0);
    let heads = num(&fields, "num_attention_heads", 0);
    if hidden == 0 || n_layers == 0 || heads == 0 {
        return Err("非标准解码器结构（缺 hidden_size/num_hidden_layers/num_attention_heads）".into());
    }
    let kvh = num(&fields, "num_key_value_heads", heads);
    let head_dim = num(&fields, "head_dim", hidden / heads);
    let eps = fnum(&fields, "rms_norm_eps", 1e-5);
    let rope_theta = fnum(&fields, "rope_theta", 1e6);
    // 架构探测（按原始文本子串；顺序 = 优先级；未命中 → 通用回退）
    let kind = fields.get("kind").cloned().filter(|k| !k.is_empty())
        .unwrap_or_else(detect_device);  // 设备：cpu/gpu/npu/apu（"" 自动检测——tch 两级 CUDA→CPU）
    let (arch, attention) = if raw.contains("qwen3_5") {
        ("qwen3_5_moe".into(), "linear_delta".into())
    } else if raw.contains("mixtral") {
        ("mixtral".into(), "standard".into())
    } else if raw.contains("qwen3_moe") || raw.contains("qwen3moe") {
        ("qwen3_moe".into(), "standard".into())
    } else if raw.contains("glm4_moe") || raw.contains("glm_moe") {
        ("glm_moe".into(), "standard".into())
    } else if raw.contains("deepseek") || raw.contains("kimi") {
        ("deepseek_moe".into(), "mla".into())
    } else if raw.contains("dbrx") {
        ("dbrx".into(), "standard".into())
    } else if raw.contains("phi3_moe") || raw.contains("phi3moe") {
        ("phi3_moe".into(), "standard".into())
    } else {
        if raw.contains("linear_attention") {
            return Err("专属架构（含线性注意力层），不走通用回退".into());
        }
        ("generic".into(), "standard".into())
    };
    let n_exp = num(&fields, "num_local_experts",
                    num(&fields, "num_experts", num(&fields, "n_routed_experts", 0)));
    let moe = if n_exp > 0 {
        Some(MoeConfig {
            num_experts: n_exp,
            top_k: num(&fields, "num_experts_per_tok", 2),
            intermediate: num(&fields, "moe_intermediate_size",
                              num(&fields, "intermediate_size", 4 * hidden)),
            shared: num(&fields, "n_shared_experts", 0) > 0
                || num(&fields, "shared_expert_intermediate_size", 0) > 0,
            group_size: num(&fields, "group_size", 32),
        })
    } else {
        None                        // 稠密 MLP（Llama 家族 / 通用回退的稠密模型）
    };
    Ok(ModelConfig {
        arch, kind, hidden_size: hidden, num_layers: n_layers, num_heads: heads,
        num_kv_heads: kvh, head_dim, eps, rope_theta, attention, moe,
        vocab_size: num(&fields, "vocab_size", 0),
        weight_prefix: "model.language_model".to_string(),
        layer_types: vec![],                     // 由 raw 解析（简化——Qwen3.5 默认全 full）
        linear_key_head_dim: num(&fields, "linear_key_head_dim", 64),
        linear_value_head_dim: num(&fields, "linear_value_head_dim", 64),
        linear_num_key_heads: num(&fields, "linear_num_key_heads", 1),
        linear_num_value_heads: num(&fields, "linear_num_value_heads", 1),
        conv_kernel_size: num(&fields, "conv_kernel_size", 4),
        rope_dim: num(&fields, "rope_dim", head_dim),
        moe_intermediate: num(&fields, "moe_intermediate_size", 4 * hidden),
        compute_dtype: fields.get("compute_dtype")
            .cloned().unwrap_or_else(|| "float32".to_string()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::path::PathBuf;

    fn write_cfg(name: &str, body: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("rust_cfg_{name}"));
        fs::create_dir_all(&dir).unwrap();
        let mut f = fs::File::create(dir.join("config.json")).unwrap();
        f.write_all(body.as_bytes()).unwrap();
        dir
    }

    #[test]
    fn test_known_archs() {
        let mixtral = write_cfg("mixtral",
            r#"{"architectures": ["MixtralForCausalLM"], "model_type": "mixtral",
                "hidden_size": 4096, "num_hidden_layers": 32, "num_attention_heads": 32,
                "num_key_value_heads": 8, "vocab_size": 32000, "rms_norm_eps": 1e-5,
                "intermediate_size": 14336, "num_local_experts": 8, "num_experts_per_tok": 2}"#);
        let c = load_model_config(mixtral.to_str().unwrap()).unwrap();
        assert_eq!(c.arch, "mixtral");
        assert_eq!(c.attention, "standard");
        assert_eq!(c.moe.unwrap().num_experts, 8);

        let ds = write_cfg("deepseek",
            r#"{"model_type": "deepseek_v4", "architectures": ["DeepseekV4ForCausalLM"],
                "hidden_size": 7168, "num_hidden_layers": 61, "num_attention_heads": 128,
                "num_key_value_heads": 1, "n_routed_experts": 256, "n_shared_experts": 1,
                "moe_intermediate_size": 2048}"#);
        let c = load_model_config(ds.to_str().unwrap()).unwrap();
        assert_eq!(c.arch, "deepseek_moe");
        assert_eq!(c.attention, "mla");
        assert!(c.moe.unwrap().shared);
    }

    #[test]
    fn test_generic_fallback() {
        // 未知但非专属稠密架构 → generic + 稠密（moe=None）
        let dense = write_cfg("dense",
            r#"{"architectures": ["WeirdForCausalLM"], "model_type": "weird_dense",
                "hidden_size": 512, "num_hidden_layers": 8, "num_attention_heads": 8,
                "vocab_size": 32000, "intermediate_size": 2048}"#);
        let c = load_model_config(dense.to_str().unwrap()).unwrap();
        assert_eq!(c.arch, "generic");
        assert!(c.moe.is_none(), "稠密模型不应有 MoE 配置");

        // 未知 MoE → generic + MoE
        let moe = write_cfg("moe",
            r#"{"architectures": ["WeirdMoeForCausalLM"], "model_type": "weird_moe",
                "hidden_size": 1024, "num_hidden_layers": 12, "num_attention_heads": 16,
                "num_local_experts": 16, "num_experts_per_tok": 2}"#);
        let c = load_model_config(moe.to_str().unwrap()).unwrap();
        assert_eq!(c.arch, "generic");
        assert_eq!(c.moe.unwrap().num_experts, 16);
    }

    #[test]
    fn test_proprietary_rejected() {
        let lin = write_cfg("linear",
            r#"{"architectures": ["WeirdLinearForCausalLM"], "model_type": "weird_linear",
                "hidden_size": 512, "num_hidden_layers": 4, "num_attention_heads": 8,
                "layer_types": ["linear_attention", "full_attention"]}"#);
        assert!(load_model_config(lin.to_str().unwrap()).is_err());
    }
}
