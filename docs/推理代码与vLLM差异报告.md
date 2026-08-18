# 推理代码与 vLLM 差异报告（v0.5——继续分析后的现状）

> 日期：2026-08-18 ｜ 对照：CostCut Infer（Python + Rust 双版本，CPU）vs vLLM（GPU）
> 说明：本地 vllm 克隆核心源码缺失，参考其测试覆盖（tests/models）+ 官方文档/内核文档（互联网）+ 本地实测。

## 1. 定位差异（本质）

| 维度 | vLLM | CostCut Infer |
|---|---|---|
| 硬件 | GPU（CUDA 内核：PagedAttention / flash-attn / MoE kernel / speculative） | CPU（torch BLAS + Rust 纯 std） |
| 定位 | 生产级推理服务（多卡/高吞吐/连续批处理） | 12GB 无 GPU 笔记本的本地推理（双版本：Python 扩展性优先 / Rust 性能优先） |
| 模型广度 | 大量模型注册（本地测试覆盖 deepseek_v32 / deepseek_v4_mega_moe / kimi_k3 / dspark_mla / multimodal 等） | MoE 系列配置归一化（DeepSeek/GLM/Kimi/Qwen3.5/Gemini/Mixtral 等——Dense 不做） |

## 2. 已对齐项（回顾——v0.4 状态）

| # | 差异点 | 状态 | 实测 |
|---|---|---|---|
| #1 KV 缓存管理 | Python 已预分配（kv_append——消除每步 torch.cat）；Rust 保持 concat_rows（owned-tensor 说明） | ✅ 有效 | 40 层 decode 61.8 → **36.08 s/token** |
| #8 算子融合 | Python + Rust 的 rms_norm_add（残差+norm 融合） | ✅ 有效 | 等价验证 ✓ |
| #5 MoE 专家调度 | 线程化实测慢 5 倍 → 回退串行 + 开关（expert_parallel） | ⚠️ 回退+开关 | 402.8 vs 81.3ms |
| #3 int4 权重计算 | Rust 融合 matmul 实测慢 2.7 倍 → 开关（int4_fused_matmul）；真 int4 原生需 SIMD 打包 | ⚠️ 回退+开关 | 3.04 vs 1.11ms |
| #4 dtype | 默认 fp32（CPU 实测最优）+ compute_dtype 开关（fp16/bf16 可选） | ⚠️ 配置化 | bf16 ~2% 无收益 |
| #6 投机解码 | dspark 5 层实测慢 → 开关（speculate）；vLLM 的 EAGLE-3（kimi_k3）为 CUDA 投机 | ⚠️ 配置化 | K=8 0.31x |

## 3. 新增差异（v0.4 后——本轮"继续"分析）

| 维度 | vLLM | CostCut Infer | 差异本质 |
|---|---|---|---|
| **多模态** | 完整多模态测试（generation / pooling / processing——vision/audio 推理） | vision 注册点占位（0 实现——方案见 docs/多模态适配方案.md） | **大差异**：vLLM 的视觉/音频端到端 vs 本项目的可扩展占位（缺依赖/模型） |
| **长上下文** | PagedAttention（GPU KV 分页——长上下文内存高效） | KV 预分配（CPU——4096-token decode **2.12 s/token**——注意力 O(ctx) 主导） | vLLM 的分页消除碎片；CPU 的注意力 O(ctx) 为固有 |
| **不同精度计算** | fp8/bf16 计算 + 多量化（fp8/fp4 等） | fp32 统一计算 + compute_dtype 可选；不同精度加载 ✓（torch_dtype/fp16 保留/FP8 缩放）；**计算内核未做** | 本项目的计算内核为后续（E 方向） |
| **多架构广度** | deepseek_v32 / deepseek_v4_mega_moe / kimi_k3（MLA+EAGLE-3）等专属实现 | 配置归一化 + 通用回退（MoE 系列；V4-Flash/DSA 索引器字段解析未实现） | 本项目的 MoE 适配广度受限（Dense 不做 + 索引器未实现） |
| **GGUF** | GGUF 模型支持 | Python GGUF ✓（解析/反量化/名称映射）；**Rust 无 GGUF** | Rust 的 GGUF 为待同步项（P0） |
| **Rust 双版本** | 无 Rust 版 | Python + Rust 双版本（纯 std 无依赖离线构建） | 本项目独有（性能/扩展性分工） |
| **插件注册表** | 模型/内核注册机制 | 五注册点（attention / moe_format / quant_method / arch_normalizer / vision） | 对齐（轻量对应） |
| **性能内核** | GPU 融合内核（PagedAttention/flash/MoE kernel） | CPU 实测：并行 matmul 2.56x；AVX2 ~1.0x / 分块 0.17x（负收益）；delta rule chunk/recurrent 移植（等价 0.0018 残余） | CPU 的 SIMD 探索需向量化打包内核 |

## 4. 实测差异（本轮新增数据）

- 长上下文：5 层 4096-token decode **2.12 s/token**（短上下文 2.01——注意力 O(ctx) 主导；KV 预分配无每步 concat）
- Rust SIMD 探索：分块缓存 matmul **0.17x** / AVX2 单循环 **~1.0x**（标量实现无 SIMD 时负收益——需向量化打包）
- Rust delta rule：chunk/recurrent 移植完成（等价 max_err 0.095→0.0018——对角线 bug 已修；残余差异待排查）
- 精度：不同精度加载（fp32/fp16/bf16/fp8/int4）验证 ✓；计算内核未做

## 5. 不可对齐项（理由）

| 项 | 理由 |
|---|---|
| PagedAttention / flash-attn / MoE kernel | GPU CUDA 内核——CPU 无对应（eager + BLAS 为 CPU 基线） |
| 多模态端到端推理 | 本项目缺视觉依赖/模型（CPU-only 评估见 docs/多模态适配方案.md） |
| EAGLE-3 投机（kimi_k3） | CUDA 投机——CPU 的 dspark 实测慢（开关化） |
| 连续批处理 / 多卡 | 单机本地推理定位——不适用 |
| torch.compile / Inductor | 本机无 MSVC——不可用 |

## 6. 结论

vLLM 与 CostCut Infer 的差异本质是**硬件环境（GPU vs CPU）与定位（生产级全模型 vs CPU MoE 双版本）**。
已对齐项（KV 预分配/算子融合）带来 40 层 ~42% 提速；实测回退项（专家并行/int4 融合/AVX2/分块）均开关化或记录。
剩余差距的 CPU 杠杆：int4/FP8 SIMD 打包内核（A 方向）与 Rust BLAS/真实模型接入（B/C 方向）——详见 docs/未来性能开发方向报告.md。
