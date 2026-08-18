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

    // M1 冒烟：加载真实模型（Qwen3.6-35B-A3B-AWQ-4bit）专家 0 的 gate_proj 权重
    let shard = "../python/models/Qwen3.6-35B-A3B-AWQ-4bit/model-00001-of-00006.safetensors";
    let prefix = "model.language_model.layers.0.mlp.experts.0.gate_proj";
    match SafeTensors::open(shard) {
        Ok(st) => {
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
        }
        Err(e) => println!("[M1] safetensors 打开失败: {e}"),
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

/// 默认入口：交互式 CLI（与 Python cli_chat 输出格式一致——横幅/You:/Assistant:）。
fn run_cli() {
    use std::io::Write;
    // 启动横幅（与 Python cli_chat 一致）
    println!("{}", "=".repeat(50));
    println!("CLI Chat (CostCut Infer 推理引擎)");
    println!("Default Model: Qwen3.6-35B-A3B-AWQ-4bit");
    println!("{}", "=".repeat(50));
    println!("[System] 投机解码：已禁用（标准自回归）");
    let tok = match crate::io::tokenizer::Tokenizer::load(
        "../python/models/Qwen3.6-35B-A3B-AWQ-4bit") {
        Ok(t) => t,
        Err(e) => {
            println!("[System] tokenizer 加载失败: {e}");
            return;
        }
    };
    let m = synthetic_model();   // 真实模型组装（from_real）接入为后续
    loop {
        print!("\nYou: ");
        let _ = std::io::stdout().flush();
        let mut line = String::new();
        if std::io::stdin().read_line(&mut line).is_err() {
            break;
        }
        let line = line.trim().to_string();
        if line == "/exit" || line == "/quit" || line == "q" || line == "exit" {
            break;
        }
        if line.is_empty() {
            continue;
        }
        // 文本 → token ids（合成模型 vocab 64——id 取模钳制；真实模型接入后无需钳制）
        let ids: Vec<usize> = tok.encode(&line).iter().map(|&t| t % 64).collect();
        print!("\nAssistant: ");
        let _ = std::io::stdout().flush();
        let gen = m.generate_sampled(&ids, 16, 0.9, 0, 0.9, 1.0);
        println!("{}", tok.decode(&gen));
    }
    println!();
}
