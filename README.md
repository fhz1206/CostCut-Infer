# CostCut Infer

> CPU-only MoE 大模型推理运行时（Python + Rust 双版本，均处于支持维护状态）
> 面向 12GB 内存的无 GPU 笔记本：配置驱动多 MoE 架构、多量化格式、离线可构建。

CostCut Infer 提供**两个独立版本**，定位互补、并行维护：

| 版本 | 定位 | 核心优势 |
|---|---|---|
| **Rust 版**（`rust/`） | **性能优先** | 纯 std 零依赖、离线可构建；并行 matmul（实测 2.56x）；int4/FP8 原生转换；编译零开销 |
| **Python 版**（`python/`） | **扩展性优先** | 插件注册表（注意力 / MoE 格式 / 量化方法 / 架构归一化四注册点即插即用）；真实模型推理（torch BLAS 内核）；投机解码 / GGUF / 多量化 |

两个版本均持续支持维护：**Rust 版持续追求性能**（BLAS/SIMD 内核、真实模型接入），
**Python 版持续追求扩展性**（新架构 / 新量化 / 新组件一个装饰器接入）。

## 特性

- **多 MoE 架构**（配置驱动，Dense 暂不纳入）：
  - DeepSeek-V3 / R1 / V4（MLA + 路由/共享专家 + 组路由 + 前 K 层 dense）
  - GLM-5（GlmMoeDsa：MLA + DSA 稀疏注意力）、GLM4-MoE
  - Kimi K2.5 / K2.6 / K3（MLA + MoE）
  - Qwen3.5-MoE（主模型：delta rule 线性注意力 + AWQ）
  - Gemini（稀疏 MoE：128 专家 / 8 激活 / 1 共享）
  - Mixtral 8x7B / Qwen3-MoE A3B / GLM4-MoE / DBRX / Phi3-MoE / 通用回退（未知但非专属架构自动适配）
- **多量化格式**：AWQ int4 / GPTQ / FP8（E4M3/E5M2）/ NVFP4（E2M1 + 块缩放）——注册表分发
- **Python 版独有**：dspark 投机解码、GGUF 模型推理、插件注册表、KV 缓存预分配、算子融合
- **Rust 版独有**：std 线程并行 matmul（2.56x）、纯 std 无依赖（离线 `cargo build --offline`）
- **内存友好**：mmap 惰性加载 + 专家反量化 LRU 缓存

## 目录结构

```
CostCut Infer/
  python/            Python 版（扩展性优先）
    cli_chat.py          CLI 入口（/help /model /models /speculate /clear /exit）
    liteengine/          引擎包（attention / moe / layer / model / model_config / quant /
                         registry / speculator / gguf / ...）
    tests/               unittest 测试（quant / config / moe_arch / gguf / registry / generate / ...）
    engine.toml          配置（[model] / [inference] / [chat]）
  rust/              Rust 版（性能优先）
    src/                 tensor / safetensors / dequant / norm / rope / attention /
                         moe / layer / model / model_config / cache / sampling
  models/            模型权重（Qwen3.6-35B-A3B-AWQ-4bit、speculator.dspark 等）
  docs/              文档（性能分析报告 / 性能优化方案 / 推理代码与vLLM差异报告）
```

## 快速开始

### Python 版（扩展性优先——真实模型推理）

```bash
pip install -r python/requirements.txt
python python/cli_chat.py                # 从项目根目录运行（模型路径按 CWD 解析）
```

启用投机解码：`engine.toml` 的 `[model]` 中设置 `dspark_model = "models/Qwen3.6-35B-A3B-speculator.dspark"`，
`/speculate` 可运行时切换。

### Rust 版（性能优先——纯 std）

```bash
cd rust
cargo build --offline                    # 离线可构建（无外部依赖）
cargo run --release --offline            # M1-M4 冒烟 + 性能对比（并行 matmul 2.56x）
cargo test --offline                     # 单元测试（29 项）
```

## 支持的架构与量化（配置驱动）

| 架构 | 注意力 | MoE 规格 | 状态 |
|---|---|---|---|
| Qwen3.5-MoE（主模型） | delta rule + gated full | 256 路由 + 共享（AWQ） | ✅ 完整运行 |
| DeepSeek-V3 / R1 / V4 | MLA（经典） | 256 路由 + 1 共享 + 组路由 | ✅ 配置 + 组件 |
| GLM-5（GlmMoeDsa） | MLA | 256 路由/top-8 + 1 共享 + DSA 索引器（字段解析） | ✅ 配置 + 组件 |
| GLM4-MoE | 标准 GQA | 64 路由/top-8 + 共享 | ✅ 配置 + 组件 |
| Kimi K2.5 / K2.6 / K3 | MLA（经典） | 256 路由 + 1 共享 | ✅ 配置 + 组件 |
| Gemini | 标准 GQA | 128 路由/8 激活 + 1 共享 | ✅ 配置 + 组件 |
| Mixtral / Qwen3-MoE / DBRX / Phi3-MoE | 标准 GQA | 无共享 / 共享 各异 | ✅ 配置 + 组件 |
| 未知非专属架构 | 标准 GQA（通用回退） | MoE / 稠密自动探测 | ✅ 自动适配 |

**量化**：AWQ int4 / GPTQ（bits 2/4/8）/ FP8（E4M3/E5M2）/ NVFP4（E2M1 + 块缩放）——注册表分发。

## 性能参考

- 实测数据与两版对比详见 `docs/性能分析报告.md`
- 优化方案与已实施项详见 `docs/性能优化方案.md`
- 推理代码与 vLLM 差异分析详见 `docs/推理代码与vLLM差异报告.md`

## 测试

```bash
# Python 版（从项目根目录，PYTHONPATH 指向 python/）
PYTHONPATH=python python -m unittest discover -s python/tests

# Rust 版
cd rust && cargo test --offline
```

## 限制与路线

- **Python 版**：主模型为 Qwen3.5-MoE（完整运行）；其它架构为配置 + 组件级（需真实权重端到端验证）；V4-Flash / DSA 索引器实现记后续
- **Rust 版**：当前为纯 std 功能验证（合成模型）；真实模型接入（delta rule + 量化分离专家）与 BLAS/SIMD 内核为性能路线核心
- 双版本均只适配 MoE 架构（Dense 暂不纳入）
