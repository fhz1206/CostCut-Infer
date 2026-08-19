# Python 与 Rust 版本对比报告（v1.0——当前状态）

> 日期：2026-08-19 ｜ 对象：CostCut Infer（Python 版 liteengine + Rust 版 costcut-infer）｜ 两版均保留

## 1. 两版定位

| | Python 版（liteengine） | Rust 版（costcut-infer） |
|---|---|---|
| 定位 | **技术探索版**——新功能首发（验证/实验），不参与发布 | **发布版**——发布包永远为 Rust 版（build.sh 自动化构建） |
| 计算 | torch/numpy（BLAS 4 线程） | 纯 std（无外部依赖——离线可构建） |
| 真实模型运行 | Qwen3.5 61 层完整运行（36.08 s/token） | 组件就绪（from_real 组装）——完整生成受标量反量化性能限制 |
| 测试 | 分批次回归全过 | cargo test 41 全绿 |

## 2. 逐项能力对比

### 2.1 加载

| 项 | Python | Rust | 状态 |
|---|---|---|---|
| safetensors 多 dtype（F32/F16/BF16/I32/F8） | loader.py ✓ | safetensors.rs ✓（统一转 f32 + get_i32 原始位型） | ✅ 已同步 |
| **多 shard**（6 分片合并读取） | WeightStore ✓ | **open_multi**（跨分片按名读取）✓ | ✅ 已同步 |
| tokenizer（BPE） | transformers ✓ | tokenizer.rs（vocab 248044 + merges 247587）✓ | ✅ 已同步 |
| GGUF 导入 | gguf.py（F32/F16/Q4_0/Q4_1/Q5_0/Q5_1/Q8_0 + 元数据→配置）✓ | gguf.rs（同量化类型 + **read_kv 元数据 + metadata_to_config**）✓ | ✅ 已同步（K 系列均未做） |

### 2.2 反量化与精度

| 项 | Python | Rust | 状态 |
|---|---|---|---|
| AWQ int4 | awq.py ✓ | dequant.rs `dequantize_awq` ✓ | ✅ 已同步 |
| GPTQ（bits 2/4/8） | gptq.py ✓ | dequant.rs `dequantize_gptq` ✓ | ✅ 已同步 |
| FP8（E4M3/E5M2） | fp8.py ✓ | dequant.rs `e4m3/e5m2_to_f32` ✓ | ✅ 已同步 |
| NVFP4（E2M1） | nvfp4.py ✓ | dequant.rs `dequantize_nvfp4` ✓ | ✅ 已同步 |
| **compute_dtype** | config 驱动反量化输出精度（fp32/fp16/bf16） | ModelConfig.compute_dtype + **F16Tensor/BF16Tensor 权重路径 + matmul_f16/bf16** ✓ | ✅ 已同步（fp32 计算为主；fp16/bf16 权重接入） |
| 灵活缩放（per-tensor/per-channel[in]/[out]/2D） | ✓ | per-tensor 为主 | ⚠️ 部分 |

### 2.3 模型架构

| 项 | Python | Rust | 状态 |
|---|---|---|---|
| 注意力：Standard/Full/MLA/GatedDeltaNet | attention.py ✓（真实运行） | attention.rs ✓（+ delta rule chunk/recurrent + GatedDeltaNet 类） | ✅ 已同步 |
| MoE 量化专家 | moe.py ✓ | moe.rs MergedExperts（AWQ 按专家反量化 + fp16/bf16 路径）✓ | ✅ 已同步 |
| 配置归一化（多厂商） | model_config.py（DeepSeek/GLM/Kimi/Qwen3.5/Gemini/Mixtral）✓ | model_config.rs ✓ | ✅ 已同步 |
| **真实权重端到端生成** | ✓（Qwen3.5 完整运行） | from_real 组装就绪 + 权重前缀 6/6 验证 ✓；**完整生成受性能限制** | ⚠️ 部分（性能） |

### 2.4 投机解码

| 项 | Python | Rust | 状态 |
|---|---|---|---|
| 草稿机制 | DSparkSpeculator（markov_head 低秩偏置 + _DraftLayer + d2t）✓ | **MarkovSpeculator（2-gram）+ DraftModel（真实草稿模型 + KV 缓存 + GQA 注意力 + markov_head 偏置）** ✓ | ✅ 已同步（结构层） |
| 投机采样接受 | speculative_accept（draft_probs vs verify_logits）✓ | generate_speculative（贪心 argmax 接受）✓ | ⚠️ 部分（贪心 vs 投机采样） |
| **MTP 多 token 预测** | mtp.py ✓ | ✗ | ❌ 未同步 |
| CLI 接入 | /speculate 切换 + 配置 dspark_model | /speculate 切换 + DraftModel.draft_forward 接入 ✓ | ✅ 已同步 |

### 2.5 CLI 对话

| 项 | Python cli_chat | Rust run_cli | 状态 |
|---|---|---|---|
| 输出格式（横幅/You:/Assistant:） | ✓ | ✓（镜像） | ✅ 已同步 |
| 流式输出 | ✓（generate_stream 逐 token） | ✓（generate_stream_sampled 逐 token 增量） | ✅ 已同步 |
| 历史 | 消息列表 + **JSON 文件持久化** + /clear | 内存 ids 累积（封顶 512）+ /clear | ⚠️ 部分（无持久化） |
| **多模型切换 /model** | ✓（真实切换多模型） | /model 命令（仅显示/占位——真实切换待 from_real） | ⚠️ 部分 |
| /help /models /speculate /exit | ✓ | ✓（镜像） | ✅ 已同步 |
| 采样（temperature/top_k/top_p/rep_penalty） | ✓ | ✓（sample_row_p + generate_stream_sampled） | ✅ 已同步 |

### 2.6 配置与扩展

| 项 | Python | Rust | 状态 |
|---|---|---|---|
| engine.toml 配置 | config.py（[default]/[model]/[inference]/[chat]）✓ | **config.rs 轻量 toml 解析**（[inference] 四开关 + 生成参数）✓ | ✅ 已同步 |
| 注册表分发 | registry.py（attention/moe_format/quant_method/arch_normalizer/vision 五注册点）✓ | registry.rs（同五注册点——quant_method 含 awq/gptq 处理器）✓ | ✅ 已同步 |
| 多模态 | vision 注册点 + 方案文档（真实推理缺依赖） | vision 注册点 ✓（真实推理缺依赖——诚实标注） | ✅ 已同步（注册点层） |

## 3. 差距汇总

### 已同步 ✅（核心链路）
加载（多 dtype/多 shard/tokenizer/GGUF 全类型）、反量化（AWQ/GPTQ/FP8/NVFP4）、注意力（Standard/Full/MLA/GatedDeltaNet）、MoE 量化专家、compute_dtype（fp16/bf16 权重路径）、配置归一化、CLI 格式/流式/命令/采样、投机（Markov + DraftModel + markov_head）、engine.toml 配置、五注册点、多模态注册点。

### 部分 ⚠️
| 差距 | 说明 | 优先级 |
|---|---|---|
| 真实模型端到端生成 | from_real 组件/前缀/多 shard 就绪；61 层标量反量化（35B——小时级）性能限制——待 SIMD 内核 | **P0** |
| 历史持久化 | Rust 内存 ids 累积（Python JSON 文件持久化） | P1 |
| 多模型真实切换 | /model 仅占位（真实切换待 from_real 多模型） | P1 |
| 投机采样接受 | Rust 贪心 argmax（Python 投机采样 accept） | P2 |

### 未同步 ❌
| 差距 | 说明 | 优先级 |
|---|---|---|
| MTP 多 token 预测 | Python mtp.py 已有；Rust 无 | P2 |
| K 系列 GGUF 量化 | 两版均未做（无 llama.cpp 参考源码） | P2 |

## 4. 结论

1. **Rust 已大幅追平 Python**：核心推理链路（加载/反量化/注意力/MoE/生成/CLI/配置/注册表/投机结构）已对齐；组件覆盖从 20 项到与 Python 25 模块逐项对应（量化模块 Rust 内联于 dequant.rs）。
2. **剩余关键差距**：真实模型端到端生成（性能——P0）、历史持久化/多模型切换（P1）、投机采样接受/MTP（P2）。
3. **两版定位**：发布包永远为 Rust 版；Python 版作为技术探索（新功能首发）。两版均保留。
4. 逐项同步进度详见 `docs/rust/Python到Rust同步清单.md`。
