# Python 与 Rust 版本对比报告（v2.0——详细版）

> 日期：2026-08-19 ｜ 对象：CostCut Infer（Python 版 liteengine + Rust 版 costcut-infer）｜ 两版均保留
> 定位：发布包永远为 Rust 版（build.sh 自动化构建）；Python 版作为技术探索（新功能首发——稳定后同步 Rust）。

## 1. 两版总览

| 维度 | Python 版（liteengine） | Rust 版（costcut-infer） |
|---|---|---|
| 定位 | 技术探索版（新功能首发/验证），不参与发布 | 发布版（发布包永远为 Rust 版） |
| 计算 | torch/numpy（BLAS 4 线程） | 纯 std（无外部依赖——离线可构建） |
| 真实模型 | Qwen3.5 61 层完整运行（36.08 s/token） | from_real 组件就绪（截断浅层冒烟）——完整生成受标量反量化性能限制 |
| 测试 | 分批次回归全过 | cargo test 41 全绿 |
| 组件数 | 25 模块 | 17 文件（量化/加载内联） |

## 2. 模块级对应表（文件 ↔ 文件）

| Python 模块 | Rust 对应 | 状态 | 备注 |
|---|---|---|---|
| `attention.py` | `engine/attention.rs` | ✅ | Standard/Full/MLA/GatedDeltaNet + delta rule chunk/recurrent |
| `moe.py` | `engine/moe.rs` | ✅ | MergedExperts（AWQ 按专家反量化 + fp16/bf16 路径） |
| `layer.py` | `engine/layer.rs` | ✅ | DecoderLayer（trait 分发注意力 + 路由/专家） |
| `model.py` | `engine/model.rs` | ✅ | prefill/generate/from_real/from_real_truncated |
| `model_config.py` | `engine/model_config.rs` | ✅ | 配置归一化（多厂商）+ Qwen3.5 text_config 嵌套 |
| `config.py`（liteengine） | `io/config.rs` | ✅ | engine.toml 轻量解析（[model] 多块 + [inference] + [chat]） |
| `norm.py` | `core/norm.rs` | ✅ | RMSNorm |
| `rope.py` | `core/rope.rs` | ✅ | RoPE + inv_freq |
| `sampling.py` | `engine/sampling.rs` | ✅ | argmax/温度/top_k/top_p |
| `registry.py` | `engine/registry.rs` | ✅ | 五注册点（attention/moe_format/quant_method/arch_normalizer/vision） |
| `speculator.py` | `engine/speculator.rs` | ✅ | Markov + DraftModel（KV 缓存 + GQA + markov_head 偏置） |
| `cache.py` | `engine/cache.rs` | ✅ | KV 缓存 |
| `gguf.py` | `io/gguf.rs` | ✅ | F32/F16/Q4_0-Q8_0 + 元数据→配置（K 系列均未做） |
| `loader.py` | `io/safetensors.rs` | ✅ | 多 dtype + 多 shard（open_multi） |
| tokenizer（transformers） | `io/tokenizer.rs` | ✅ | BPE（vocab 248044 + merges 247587） |
| torch | `core/tensor.rs` | ✅ | 纯 std 张量 + matmul/matmul_f16/matmul_bf16 |
| `quant/awq.py` | `quant/dequant.rs`（内联） | ✅ | `dequantize_awq` |
| `quant/gptq.py` | `quant/dequant.rs`（内联） | ✅ | `dequantize_gptq`（bits 2/4/8） |
| `quant/fp8.py` | `quant/dequant.rs`（内联） | ✅ | E4M3/E5M2 |
| `quant/nvfp4.py` | `quant/dequant.rs`（内联） | ✅ | E2M1 |
| `quant/dequantize.py` + `unpack.py` | `quant/dequant.rs`（内联） | ✅ | 反量化分发 + 位解包 |
| `mtp.py` | — | ❌ | MTP 多 token 预测——Rust 无（P2） |
| `engine.py` + `lazy_loader.py` | `main.rs`（入口直接） | ⚠️ | 引擎门面/惰性加载——Rust 由 main.rs 直接承担 |

## 3. 功能级对比（详细子项）

### 3.1 加载
| 子项 | Python | Rust | 状态 |
|---|---|---|---|
| safetensors 多 dtype（F32/F16/BF16/I32/F8） | ✓ | ✓（统一转 f32 + get_i32 原始位型） | ✅ |
| 多 shard（6 分片合并读取） | ✓ WeightStore | ✓ open_multi（跨分片按名） | ✅ |
| tokenizer BPE（编码/解码） | ✓ transformers | ✓（vocab/merges/字节级） | ✅ |
| GGUF 量化类型 | F32/F16/Q4_0/Q4_1/Q5_0/Q5_1/Q8_0 | 同（镜像公式） | ✅ |
| GGUF 元数据→配置 | ✓ gguf_metadata_to_config | ✓ metadata_to_config（架构探测） | ✅ |
| GGUF 名称映射 | ✓ gguf_name_to_hf | ⚠️ 张量按名直读（映射表未做） | ⚠️ |
| 模型路径解析 | model_dir 归一化（python/models） | model_dir CWD 容忍（rust/ 或项目根） | ✅ |

### 3.2 反量化与精度
| 子项 | Python | Rust | 状态 |
|---|---|---|---|
| AWQ int4 | ✓ | ✓ | ✅ |
| GPTQ（bits 2/4/8 + sym/group） | ✓ | ✓ | ✅ |
| FP8（E4M3/E5M2） | ✓ | ✓ | ✅ |
| NVFP4（E2M1） | ✓ | ✓ | ✅ |
| compute_dtype（fp32/fp16/bf16 权重路径） | ✓ 反量化输出精度 | ✓ F16Tensor/BF16Tensor + matmul_f16/bf16 | ✅ |
| fp16/bf16 计算内核 | ✓ 全链路 | ⚠️ 权重路径已接入；f32 计算为主 | ⚠️ |
| 灵活缩放（per-tensor/per-channel[in]/[out]/2D） | ✓ | ⚠️ per-tensor 为主 | ⚠️ |

### 3.3 模型架构
| 子项 | Python | Rust | 状态 |
|---|---|---|---|
| Standard/Full/MLA/GatedDeltaNet 注意力 | ✓（真实运行） | ✓（+ delta rule chunk/recurrent） | ✅ |
| MoE 量化专家（AWQ 按专家反量化） | ✓ | ✓（+ fp16/bf16 权重） | ✅ |
| 配置归一化（多厂商：DeepSeek/GLM/Kimi/Qwen3.5/Gemini/Mixtral） | ✓ | ✓ | ✅ |
| Qwen3.5 配置解析（text_config 嵌套） | ✓ | ✓（本轮修复） | ✅ |
| 真实权重端到端生成 | ✓（61 层完整） | ⚠️ 截断浅层冒烟（1 层）；完整生成受性能限制 | ⚠️ P0 |

### 3.4 投机解码
| 子项 | Python | Rust | 状态 |
|---|---|---|---|
| 草稿机制（markov_head 低秩偏置） | ✓ dspark | ✓ MarkovHead（w1/w2 偏置） | ✅ |
| 真实草稿模型（按层前向 + KV 缓存 + GQA） | ✓ _DraftLayer | ✓ DraftModel（多 token 上下文注意力） | ✅ |
| d2t 映射 | ✓ | ✓（简化取整） | ✅ |
| 投机采样接受（draft_probs vs verify_logits） | ✓ | ⚠️ 贪心 argmax 接受 | ⚠️ P2 |
| MTP 多 token 预测 | ✓ mtp.py | ❌ | ❌ P2 |
| CLI 接入（/speculate + 草稿类型横幅） | ✓ | ✓（dspark 草稿/Markov 草稿动态） | ✅ |

### 3.5 CLI 对话
| 子项 | Python cli_chat | Rust run_cli | 状态 |
|---|---|---|---|
| 输出格式（横幅/You:/Assistant:） | ✓ | ✓（镜像） | ✅ |
| 流式输出（逐 token 增量） | ✓ generate_stream | ✓ generate_stream_sampled | ✅ |
| 历史（内存 + 文件持久化） | ✓ JSON 文件 + /clear | ✓ 简易 JSON 数组 + /clear + auto_save | ✅ |
| 多模型切换（/model <name>） | ✓（真实重建模型） | ✓（配置校验 + 切换 + 历史重置——真实模型待 from_real） | ✅ |
| /models /help /speculate /exit | ✓ | ✓ | ✅ |
| 采样（temperature/top_k/top_p/repetition_penalty） | ✓ | ✓（sample_row_p） | ✅ |

### 3.6 配置与扩展
| 子项 | Python | Rust | 状态 |
|---|---|---|---|
| engine.toml（[default]/[model] 多块/[inference]/[chat]） | ✓ | ✓（轻量解析 + [model] 多块收集 + [chat]） | ✅ |
| 注册表（attention/moe_format/quant_method/arch_normalizer/vision） | ✓ 装饰器 | ✓（函数式 + awq/gptq 处理器） | ✅ |
| 多模态（vision 注册点） | ✓（真实推理缺依赖） | ✓（注册点——真实推理缺依赖） | ✅ |

### 3.7 性能与测试
| 子项 | Python | Rust | 状态 |
|---|---|---|---|
| 合成模型速度 | torch BLAS | ~0.07 ms/token（纯 std） | ✅ Rust 快 |
| 真实模型速度 | 36.08 s/token（40 层） | 标量反量化受限（1 层截断冒烟——分钟级） | ⚠️ P0 |
| matmul 并行 | torch 4 线程 | std::thread 并行（512³ 2.56x） | ✅ |
| 单元测试 | 分批次回归全过 | cargo test 41 全绿 | ✅ |

## 4. 差距汇总

### 已同步 ✅（核心链路）
加载（多 dtype/多 shard/tokenizer/GGUF 全类型 + 元数据配置）、反量化（AWQ/GPTQ/FP8/NVFP4）、注意力（Standard/Full/MLA/GatedDeltaNet）、MoE 量化专家、compute_dtype（fp16/bf16 权重路径）、配置归一化（含 Qwen3.5 text_config）、CLI（格式/流式/历史/多模型/命令/采样）、投机（Markov + DraftModel + markov_head + CLI 接入）、engine.toml 配置（含 [model] 多块/[chat]）、五注册点、多模态注册点、模型路径解析。

### 部分 ⚠️
| 差距 | 说明 | 优先级 |
|---|---|---|
| 真实模型端到端生成 | 截断浅层冒烟（1 层）已通；完整 61 层生成受标量反量化性能限制（分钟级/层）——待 SIMD 内核 | **P0** |
| GGUF 名称映射 | Python gguf_name_to_hf；Rust 张量按名直读（映射表未做） | P1 |
| fp16/bf16 计算内核 | 权重路径已接入；f32 计算为主（完整 fp16/bf16 内核为 P2） | P2 |
| 灵活缩放（per-channel[in]/[out]/2D） | Rust per-tensor 为主 | P2 |
| 投机采样接受 | Rust 贪心 argmax（Python 投机采样 accept） | P2 |

### 未同步 ❌
| 差距 | 说明 | 优先级 |
|---|---|---|
| MTP 多 token 预测 | Python mtp.py 已有；Rust 无 | P2 |
| K 系列 GGUF 量化 | 两版均未做（无 llama.cpp 参考源码） | P2 |

## 5. 结论

1. **Rust 已大幅追平 Python**：12 个核心模块一一对应（attention/moe/layer/model/model_config/norm/rope/sampling/registry/speculator/cache/gguf），量化模块内联于 dequant.rs，CLI/配置/投机/注册表/多模态结构层全部对齐；cargo test 41 全绿。
2. **关键剩余差距**：真实模型端到端生成（P0——性能待 SIMD 内核）、GGUF 名称映射（P1）、fp16/bf16 内核/灵活缩放/投机采样接受（P2）、MTP（P2）。
3. **两版定位**：发布包永远为 Rust 版；Python 版技术探索（新功能首发）。两版均保留。
4. 逐项同步进度详见 `docs/rust/Python到Rust同步清单.md`。
