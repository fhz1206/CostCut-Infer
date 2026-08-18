# Python → Rust 同步清单

> 日期：2026-08-18 ｜ 对象：CostCut Infer（Python 版 → Rust 版）
> 说明：Rust 版为纯 std 无依赖（离线可构建），同步按"功能等价 + 纯 std 可实现"筛选。

## 1. 已同步（Rust 已有）

| Python 功能 | Rust 对应 | 验证 |
|---|---|---|
| rms_norm_add（算子融合） | `core::norm::rms_norm_add` + layer 融合使用 | 等价测试 ✓ |
| FP8（E4M3/E5M2）/ NVFP4（E2M1）转换 | `quant::dequant`（e4m3/e5m2/e2m1 + dequantize_fp8/nvfp4） | 位布局 + 块反量化测试 ✓ |
| safetensors F8 dtype | `io::safetensors`（F8_E4M3/F8_E5M2 转换） | 测试 ✓ |
| FullAttention / MlaAttention | `engine::attention`（forward/forward_kv/decode） | 前向 + decode 冒烟 ✓ |
| dense_mlp 路径 | `engine::layer`（DecoderLayer.dense_mlp + mlp_forward） | 稠密层测试 ✓ |
| 配置归一化（多架构/通用回退） | `engine::model_config`（load_model_config + 注册表） | 12 项归一化测试 ✓ |
| int4 融合 matmul / AVX2 / 并行 matmul | `quant::dequant::matmul_awq_int4` / `core::tensor::matmul_avx2` / matmul_par | 等价 + 实测 ✓（均实测负收益，保留实现） |

## 2. 待同步（后续——按优先级）

| 优先级 | Python 功能 | Rust 同步内容 | 备注 |
|---|---|---|---|
| **P0** | 注册表（registry 四注册点 + vision 注册点） | Rust trait/枚举分发（attention/moe_format/quant_method/arch_normalizer/vision） | ✅ **已同步**：`engine/registry.rs`（Attention trait + 注册/获取/列表 + standard 构造器） |
| **P0** | GGUF 解析与反量化 | `io::gguf`（元数据/张量索引/名称映射/K 系列量化） | 需参考 llama.cpp k-quants 源码 |
| P1 | dspark 投机解码 | `engine::speculator`（草稿-验证-接受 + markov_head） | 5 层实测慢（开关化） |
| P1 | compute_dtype 配置 | Rust 反量化输出 dtype 参数（fp32 默认/fp16） | ⚠️ **已评估**：Rust 张量为 f32-only——fp16 输出需 fp16 张量类型/计算内核（归入 P2 不同精度计算）；Python 侧 compute_dtype 已配置化 |
| P1 | KV 预分配（kv_append） | 预分配 + 位置索引（owned-tensor 收益有限——评估后定） | ⚠️ **已评估**：Rust owned-tensor 切片即拷贝——预分配收益有限——**保持 concat_rows**（与 Python 的对齐结论一致） |
| P2 | 不同精度计算（fp16/bf16 原生） | 内核级 fp16 matmul（当前 f32 统一） | 视硬件 |
| P2 | 多模态（vision 注册点） | vision 编码器 trait（依赖/模型就绪后） | 见 docs/python/多模态适配方案.md |

## 3. 已评估不做同步（理由）

| Python 功能 | 理由 |
|---|---|
| numpy 向量化反量化（LRU 缓存） | Rust 标量实现 + 反量化缓存策略不同（纯 std） |
| torch 的 KV 预分配零拷贝切片 | Rust owned-tensor 切片即拷贝——预分配收益有限（保持 concat_rows） |

## 4. 执行建议

1. **P0：注册表**（trait 分发——与 model_config 注册表打通——可扩展性对齐）
2. **P0：GGUF**（真实 GGUF 模型接入——需 k-quants 参考）
3. **P1：投机 / compute_dtype / KV 预分配评估**（按需）
4. Rust 真实模型接入（Qwen3.5 delta rule）是以上各项的验证前提
