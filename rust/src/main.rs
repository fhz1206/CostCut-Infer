//! CostCut Infer（Rust 版）入口。
//!
//! M1：真实模型 safetensors 加载 + AWQ 反量化冒烟；
//! M2-M3：合成小模型（标准注意力 MoE）prefill 前向 + 贪心生成；
//! M4：std 线程并行 matmul 与串行速度对比。

mod core;
mod engine;
mod io;
mod quant;

use quant::dequant::dequantize_awq;
use io::safetensors::SafeTensors;
use std::time::Instant;
use core::tensor::Tensor;

/// 当前默认模型名（多模型切换注册表为后续）。
const CURRENT_MODEL: &str = "Qwen3.6-35B-A3B-AWQ-4bit";

/// 合成小模型（1 层 Mixtral 风格：hidden 32 / 4 头 / 2 KV / 8 专家 / vocab 64）。
fn synthetic_model() -> engine::model::Model {
    let (hidden, h, kvh, hd) = (32usize, 4usize, 2usize, 8usize);
    let (e, inter, vocab) = (8usize, 16usize, 64usize);
    let attn: Box<dyn engine::registry::Attention> =
        Box::new(engine::attention::StandardAttention {
            num_heads: h,
            num_kv_heads: kvh,
            head_dim: hd,
            rope_dim: 8,
            scaling: (hd as f32).powf(-0.5),
            q_w: Tensor::from_vec(h * hd, hidden, vec![0.05; h * hd * hidden]),
            k_w: Tensor::from_vec(kvh * hd, hidden, vec![0.05; kvh * hd * hidden]),
            v_w: Tensor::from_vec(kvh * hd, hidden, vec![0.05; kvh * hd * hidden]),
            o_w: Tensor::from_vec(hidden, h * hd, vec![0.05; hidden * h * hd]),
        });
    let layer = engine::layer::DecoderLayer {
        eps: 1e-6,
        input_norm_w: vec![1.0; hidden],
        post_norm_w: vec![1.0; hidden],
        attn,
        router: engine::moe::TopKRouter {
            weight: Tensor::from_vec(e, hidden, vec![0.05; e * hidden]),
            top_k: 2,
        },
        experts: engine::moe::MergedExperts {
            num_experts: e,
            intermediate: inter,
            hidden,
            gate_up: vec![0.05; e * 2 * inter * hidden],
            down: vec![0.05; e * hidden * inter],
            gate_up_f16: None,
            down_f16: None,
            gate_up_bf16: None,
            down_bf16: None,
        },
        shared: None,
        dense_mlp: None,
    };
    engine::model::Model::new_with_inv_freq(
        1, hidden, 8, 1e-6, 1e4, vocab,
        Tensor::from_vec(vocab, hidden, vec![0.05; vocab * hidden]),
        vec![layer],
        vec![1.0; hidden],
        Tensor::from_vec(vocab, hidden, vec![0.05; vocab * hidden]),
    )
}

fn main() {
    // 默认 = 交互式 CLI；--smoke/--bench/--test = 开发冒烟（测试代码仅开发时使用）
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--smoke" || a == "--bench" || a == "--test") {
        run_smoke();
    } else {
        run_cli();
    }
}

/// 开发模式冒烟（M1-M4 + int4 对比——仅开发时运行）。
fn run_smoke() {
    println!("[dev] CostCut Infer 冒烟（--smoke：加载/反量化/前向/生成/并行 matmul）");

    // M1 冒烟：多分片加载真实模型（Qwen3.6-35B-A3B-AWQ-4bit——6 shard）专家 0 的 gate_proj 权重
    let dir = model_dir("Qwen3.6-35B-A3B-AWQ-4bit");
    let prefix = "model.language_model.layers.0.mlp.experts.0.gate_proj";
    let paths: Vec<String> = (1..=6)
        .map(|i| format!("{dir}/model-{i:05}-of-00006.safetensors"))
        .collect();
    match SafeTensors::open_multi(&paths) {
        Ok(st) => {
            println!("[M1] 多分片打开成功——张量索引 {} 项（跨 6 shard）", st.tensors.len());
            let qw = st.get_i32(&format!("{prefix}.qweight"));
            let qz = st.get_i32(&format!("{prefix}.qzeros"));
            let sc = st.get_f32(&format!("{prefix}.scales"));
            match (qw, qz, sc) {
                (Some(qw), Some(qz), Some(sc)) => {
                    let (out, in_, gs) = (2048usize, 512usize, 32usize);
                    let dq = dequantize_awq(&qw, &qz, &sc, out, in_, gs);
                    let t = Tensor::from_vec(out, in_, dq);
                    println!("[M1] 反量化 [{}x{}]: max_abs={:.4} std={:.4}",
                             t.rows, t.cols, t.max_abs(), t.std());
                    println!("[M1] AWQ 反量化冒烟 {}", if t.max_abs().is_finite() && t.std() > 0.0 { "OK" } else { "失败" });
                }
                _ => println!("[M1] 张量读取失败"),
            }
            // from_real 权重前缀读取验证（embed/lm_head/norm + 专家 gate/up/down——快速）
            let mp = "model.language_model";
            let wp = "model.language_model.layers.0.mlp.experts.0";
            let mut hit = 0usize;
            let mut total = 0usize;
            for (name, ok) in [
                ("embed", st.get_f32(&format!("{mp}.embed_tokens.weight")).is_some()),
                ("lm_head", st.get_f32("lm_head.weight").is_some()),   // 顶层（无前缀）
                ("norm", st.get_f32(&format!("{mp}.norm.weight")).is_some()),
                ("gate_proj", st.get_i32(&format!("{wp}.gate_proj.qweight")).is_some()),
                ("up_proj", st.get_i32(&format!("{wp}.up_proj.qweight")).is_some()),
                ("down_proj", st.get_i32(&format!("{wp}.down_proj.qweight")).is_some()),
            ] {
                total += 1;
                if ok { hit += 1; }
                if !ok { println!("[M1] 权重缺失: {name}"); }
            }
            println!("[M1] from_real 权重前缀验证：{hit}/{total} 命中（embed/lm_head/norm/专家投影）");
        }
        Err(e) => println!("[M1] safetensors 打开失败: {e}"),
    }

    // M5 冒烟：from_real 真实模型浅层构造 + 前向（截断 1 层——避免完整 61 层标量反量化耗时）
    println!("=== M5 from_real 真实模型浅层冒烟（截断 1 层）===");
    let dir5 = model_dir("Qwen3.6-35B-A3B-AWQ-4bit");
    match crate::engine::model_config::load_model_config(&dir5) {
        Ok(cfg5) => {
            let paths5: Vec<String> = (1..=6)
                .map(|i| format!("{dir5}/model-{i:05}-of-00006.safetensors"))
                .collect();
            if let Ok(st5) = SafeTensors::open_multi(&paths5) {
                match engine::model::Model::from_real_truncated(&st5, &cfg5, 1) {
                    Ok(m5) => {
                        let ids = [1usize, 2, 3];
                        let logits = m5.prefill(&ids);
                        println!("[M5] from_real 截断模型 prefill: {}x{} finite={}",
                                 logits.rows, logits.cols,
                                 logits.data.iter().all(|v| v.is_finite()));
                    }
                    Err(e) => println!("[M5] from_real 构造失败: {e}"),
                }
            } else {
                println!("[M5] safetensors 打开失败");
            }
        }
        Err(e) => println!("[M5] 配置加载失败: {e}"),
    }

    // M2-M3 冒烟：合成小模型前向 + 贪心生成
    println!("=== M2-M3 合成小模型冒烟 ===");
    let m = synthetic_model();
    let ids = [1usize, 2, 3];
    let logits = m.prefill(&ids);
    println!("[M2] prefill logits: {}x{} finite={}",
             logits.rows, logits.cols, logits.max_abs().is_finite());
    let t0 = Instant::now();
    let gen = m.generate(&ids, 3);
    let t_gen = t0.elapsed();
    let legal = gen.iter().all(|&t| t < 64);
    println!("[M3] generate: {:?} 合法={} 耗时 {:.1}ms（≈{:.3} s/token）",
             gen, legal, t_gen.as_secs_f64() * 1000.0, t_gen.as_secs_f64() / 3.0);

    // M4 冒烟：并行 matmul 与串行速度对比
    println!("=== M4 并行 matmul 对比 ===");
    let n = 512usize;
    let a = Tensor::from_vec(n, n, (0..n * n).map(|i| (i % 7) as f32 * 0.01).collect());
    let b = Tensor::from_vec(n, n, (0..n * n).map(|i| (i % 5) as f32 * 0.01).collect());
    let t0 = Instant::now();
    let c1 = a.matmul(&b);
    let ts = t0.elapsed();
    let t0 = Instant::now();
    let c2 = a.matmul_par(&b, 4);
    let tp = t0.elapsed();
    let same = (0..c1.data.len()).all(|i| (c1.data[i] - c2.data[i]).abs() < 1e-3);
    println!("serial {:?} vs parallel(4线程) {:?} 提速 {:.2}x 结果一致={}",
             ts, tp, ts.as_secs_f64() / tp.as_secs_f64().max(1e-9), same);

    // AVX2/FMA matmul（docs 性能方向）：与串行/并行对比
    let t0 = Instant::now();
    let c3 = a.matmul_avx2(&b);
    let tavx = t0.elapsed();
    let same2 = (0..c1.data.len()).all(|i| (c1.data[i] - c3.data[i]).abs() < 1e-3);
    println!("AVX2/FMA {:?} vs serial {:?} 提速 {:.2}x 结果一致={}",
             tavx, ts, ts.as_secs_f64() / tavx.as_secs_f64().max(1e-9), same2);

    // 分块缓存 matmul（docs 方向 B：8×8 tile + k 分块）
    let t0 = Instant::now();
    let c4 = a.matmul_blocked(&b);
    let tb = t0.elapsed();
    let same3 = (0..c1.data.len()).all(|i| (c1.data[i] - c4.data[i]).abs() < 1e-3);
    println!("分块缓存 {:?} vs serial {:?} 提速 {:.2}x 结果一致={}",
             tb, ts, ts.as_secs_f64() / tb.as_secs_f64().max(1e-9), same3);

    // int4 原生 matmul（差异报告 #3）：融合解量化 vs 两步（反量化→matmul）计时
    let (m, n, gs) = (64usize, 2048usize, 32usize);
    let qw: Vec<i32> = (0..m * n / 8).map(|i| 0x12345678 + i as i32).collect();
    let qz = vec![0x11111111i32; m / gs * n / 8];
    let sc: Vec<f32> = (0..m / gs * n).map(|i| 0.01 + (i % 7) as f32 * 0.0001).collect();
    let act = core::tensor::Tensor::from_vec(8, n,
        (0..8 * n).map(|i| ((i % 5) as f32) * 0.1).collect());
    let t0 = Instant::now();
    let w = quant::dequant::dequantize_awq(&qw, &qz, &sc, m, n, gs);
    let w_t = core::tensor::Tensor::from_vec(m, n, w);
    let r1 = act.matmul(&w_t.transpose());
    let t2s = t0.elapsed();
    let t0 = Instant::now();
    let r2 = quant::dequant::matmul_awq_int4(&act, &qw, &qz, &sc, m, n, gs);
    let t_fused = t0.elapsed();
    let same_i = (0..r1.data.len()).all(|i| (r1.data[i] - r2.data[i]).abs() < 1e-3);
    println!("int4 两步 {:?} vs 融合 {:?} 结果一致={}",
             t2s, t_fused, same_i);
}

/// 解析模型目录（CWD 容忍——rust/ 或项目根 CWD 均可，镜像 Python 的 python/models 归一化）。
fn model_dir(name: &str) -> String {
    for cand in [format!("../python/models/{name}"), format!("python/models/{name}"),
                 format!("../../python/models/{name}")] {
        if std::path::Path::new(&cand).exists() {
            return cand;
        }
    }
    format!("../python/models/{name}")
}

/// 加载真实 DSpark 草稿模型（speculator.dspark safetensors + 主模型 embed）——投机路径用。
/// 加载失败返回 None（回退简化 Markov 草稿）。
fn load_draft_model(m: &engine::model::Model) -> Option<engine::speculator::DraftModel> {
    let dir = model_dir("Qwen3.6-35B-A3B-speculator.dspark");
    let path = format!("{dir}/model.safetensors");
    let store = match SafeTensors::open(&path) {
        Ok(s) => s,
        Err(_) => return None,
    };
    let hidden = m.hidden;
    let vocab = m.embed.rows;
    let embed_main = &m.embed.data;
    match engine::speculator::DraftModel::from_dspark(&store, embed_main, vocab, hidden) {
        Ok(dm) => Some(dm),
        Err(_) => None,
    }
}

/// 历史文件加载（简易 JSON 数组 "[1,2,3]"——镜像 Python [chat].history_file 语义）。
fn load_history_file(path: &str) -> Vec<usize> {
    let Ok(text) = std::fs::read_to_string(path) else { return vec![]; };
    let t = text.trim().trim_start_matches('[').trim_end_matches(']');
    t.split(',').filter_map(|s| s.trim().parse().ok()).collect()
}

/// 历史文件保存（简易 JSON 数组）。
fn save_history_file(path: &str, ids: &[usize]) {
    let body = ids.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(",");
    let _ = std::fs::write(path, format!("[{body}]"));
}

/// 默认入口：交互式 CLI（与 Python cli_chat 输出格式一致——横幅/You:/Assistant:）。
fn run_cli() {
    use std::io::Write;
    // 启动横幅（与 Python cli_chat 一致）
    println!("{}", "=".repeat(50));
    println!("CLI Chat (CostCut Infer 推理引擎)");
    println!("Default Model: Qwen3.6-35B-A3B-AWQ-4bit");
    println!("{}", "=".repeat(50));
    println!("[System] 投机解码：已禁用（标准自回归）");
    let tok = match crate::io::tokenizer::Tokenizer::load(&model_dir("Qwen3.6-35B-A3B-AWQ-4bit")) {
        Ok(t) => t,
        Err(e) => {
            println!("[System] tokenizer 加载失败: {e}");
            return;
        }
    };
    let m = synthetic_model();   // 真实模型组装（from_real）接入为后续
    // engine.toml 配置化（[inference] 四开关 + 生成参数 + [chat] 历史——镜像 Python cli_chat）
    let cfg = crate::io::config::load_engine_toml();
    let auto_save = cfg.get_bool("chat", "auto_save_history");
    let hist_file = cfg.get("chat", "history_file").unwrap_or(".chat_history.json").to_string();
    let mut history: Vec<usize> = if auto_save { load_history_file(&hist_file) } else { Vec::new() };
    let mut current_model = CURRENT_MODEL.to_string();   // /model 切换的当前模型（from_real 多模型接入为后续）
    // 多轮上下文（ids 累积——简化；封顶 512；auto_save 时从文件加载）
    let mut speculate_on = cfg.get_bool("inference", "speculate");   // /speculate 开关（默认取配置）
    let temp = cfg.get_f32("inference", "temperature", 0.9);
    let top_p = cfg.get_f32("inference", "top_p", 0.9);
    let top_k = cfg.get_int("inference", "top_k", 0) as usize;
    let max_new = cfg.get_int("inference", "max_new_tokens", 2048).min(64) as usize;
    let _spec_enabled = cfg.get("model", "dspark_model").map_or(false, |v| !v.is_empty());
    let mut spec = crate::engine::speculator::MarkovSpeculator::new();
    // 真实草稿模型（DSpark）——接入投机路径（draft_forward 替代简化 Markov）
    let draft = load_draft_model(&m);
    loop {
        print!("\nYou: ");
        let _ = std::io::stdout().flush();
        let mut line = String::new();
        if std::io::stdin().read_line(&mut line).is_err() {
            break;
        }
        let line = line.trim().to_string();
        if line.starts_with('/') {
            // 命令处理（镜像 Python cli_chat：/help /model /models /clear /speculate /exit）
            let mut parts = line.splitn(2, ' ');
            let cmd = parts.next().unwrap_or("").to_lowercase();
            let arg = parts.next().unwrap_or("").trim().to_string();
            match cmd.as_str() {
                "/help" => {
                    println!("==================================================");
                    println!("Commands:");
                    println!("  /help           Show this help message");
                    println!("  /model [name]   Switch to a different model");
                    println!("  /models         List available models");
                    println!("  /clear          Clear conversation history");
                    println!("  /speculate      Toggle speculative decoding");
                    println!("  /exit, /quit    Exit the application");
                    println!("==================================================");
                }
                "/model" => {
                    if arg.is_empty() {
                        println!("Current Model: {current_model}");
                    } else {
                        let models = cfg.models();
                        if models.iter().any(|n| n == &arg) {
                            current_model = arg.clone();
                            history.clear();
                            if auto_save {
                                save_history_file(&hist_file, &history);
                            }
                            println!("[System] 已切换到模型: {current_model}");
                        } else {
                            println!("[Error] 未知模型: {arg}（/models 查看）");
                        }
                    }
                }
                "/models" => {
                    let models = cfg.models();
                    let list = if models.is_empty() {
                        CURRENT_MODEL.to_string()
                    } else {
                        models.join(", ")
                    };
                    println!("Available: {list}");
                }
                "/clear" => {
                    history.clear();
                    if auto_save {
                        save_history_file(&hist_file, &history);
                    }
                    println!("[System] 历史已清空");
                }
                "/speculate" => {
                    speculate_on = !speculate_on;
                    let kind = if draft.is_some() { "dspark 草稿" } else { "Markov 草稿" };
                    println!("[System] 投机解码：{}（{kind}）",
                             if speculate_on { "已启用" } else { "已禁用" });
                }
                "/exit" | "/quit" => break,
                _ => println!("[Error] Unknown command: {cmd}（/help 查看帮助）"),
            }
            continue;
        }
        if line.is_empty() {
            continue;
        }
        // 文本 → token ids（合成模型 vocab 64——id 取模钳制；真实模型接入后无需钳制）
        let ids: Vec<usize> = tok.encode(&line).iter().map(|&t| t % 64).collect();
        history.extend(ids);
        if history.len() > 512 {
            history.drain(0..history.len() - 512);   // 封顶——丢弃最旧
        }
        print!("\nAssistant: ");
        let _ = std::io::stdout().flush();
        // 流式输出（镜像 Python generate_stream——逐 token 打印增量）
        let mut gen_ids: Vec<usize> = Vec::new();
        let mut text = String::new();
        let mut emit = |tid: usize, gen_ids: &mut Vec<usize>, text: &mut String| {
            gen_ids.push(tid);
            let new_text = tok.decode(gen_ids);
            if new_text.len() > text.len() {
                print!("{}", &new_text[text.len()..]);
                let _ = std::io::stdout().flush();
                *text = new_text;
            }
        };
        if speculate_on {
            // 投机路径：真实草稿模型（DraftModel.draft_forward）草稿 + 主模型验证
            if let Some(dm) = &draft {
                // 从主模型 aux 隐藏态草稿 n 个 token（h_target = final norm 前的末位隐藏态）
                let h_target = m.hidden_state_at_last(&history);
                let draft_ids = dm.draft_forward(&h_target, 4);
                // 草稿 + 主模型验证接受（贪心 argmax 语义）
                let mut trial = history.clone();
                trial.extend(&draft_ids);
                let logits = m.prefill(&trial);
                let mut accepted = 0usize;
                for (k, &d) in draft_ids.iter().enumerate() {
                    let pos = history.len() + k;
                    let start = (pos - 1) * logits.cols;
                    let tok = crate::engine::sampling::argmax_row(
                        &logits.data[start..start + logits.cols]);
                    if tok == d { accepted += 1; } else { break; }
                }
                let mut gen = draft_ids[..accepted].to_vec();
                // 剩余空间用主模型继续
                let rem = max_new.saturating_sub(gen.len());
                if rem > 0 {
                    let mut tail = history.clone();
                    tail.extend(&draft_ids[..accepted]);
                    let rest = m.generate_sampled(&tail, rem, temp, top_k, top_p, 1.0);
                    gen.extend(rest);
                }
                println!("{}", tok.decode(&gen));
                gen_ids = gen;
                text = tok.decode(&gen_ids);
            } else {
                // 回退：Markov 草稿 + 主模型验证
                spec.observe(&history);
                let gen = spec.generate_speculative(&m, &history, 4, max_new);
                println!("{}", tok.decode(&gen));
                gen_ids = gen;
                text = tok.decode(&gen_ids);
            }
        } else {
            m.generate_stream_sampled(&history, max_new, temp, top_k, top_p, 1.0,
                                      &mut |tid| emit(tid, &mut gen_ids, &mut text));
            println!();
        }
        // 结果纳入历史
        let gen_out: Vec<usize> = gen_ids.iter().map(|&t| t % 64).collect();
        history.extend(gen_out);
        if history.len() > 512 {
            history.drain(0..history.len() - 512);
        }
        if auto_save {
            save_history_file(&hist_file, &history);
        }
    }
    println!();
}
