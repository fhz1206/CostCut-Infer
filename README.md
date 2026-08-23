<p align="center">
  <img src="https://raw.gitcode.com/fhz1206/CostCut-Infer/raw/main/CostCut-Infer-logo.png" alt="CostCut Infer" width="320"/>
</p>

> MoE 大模型推理运行时（Python + Rust 双版本，均处于支持维护状态）
> 支持 **CPU / GPU（CUDA）/ NPU（昇腾）/ APU（AMD ROCm）** 四类设备（**Python 自动检测**：CUDA→NPU→ROCm→CPU；**Rust 自动检测**：CUDA→CPU（tch 生态限制——NPU/APU 无后端）——两版均可显式配置或 `kind=""` 自动检测）；
> 面向 12GB 内存笔记本：配置驱动多 MoE 架构、多量化格式、离线可构建。

CostCut Infer 提供**两个独立版本**，定位互补、并行维护：

| 版本 | 定位 | 核心优势 |
|---|---|---|
| **Rust 版**（`rust/`） | **性能优化路径** | 9 依赖（tch/anyhow/rayon 等）+ 离线可构建；**lm_head 预转置（~8x）+ 转置权重 + 投机 KV 缓存**；int4/FP8 原生转换 |
| **Python 版**（`python/`） | **扩展性优先（新功能首发）** | 插件注册表（注意力 / MoE 格式 / 量化方法 / 架构归一化四注册点即插即用）；真实模型推理（torch BLAS 内核）；**设备自动检测（CUDA/NPU/ROCm/CPU）**；投机解码 / GGUF / 多量化 / 完整 YaRN |

两个版本均持续支持维护：**Rust 版持续追求性能**（BLAS/SIMD 内核、真实模型接入），
**Python 版持续追求扩展性**（新架构 / 新量化 / 新组件一个装饰器接入）。

> **重要**：Rust 版并非纯性能优先——**新功能会优先在 Python 版实现**（扩展性与验证更快），
> 稳定后按需同步到 Rust 版做性能优化。Rust 版定位为**既有功能的性能优化路径**，
> 而非新功能的首发版本；两版同步状态见 `docs/rust/Python到Rust同步清单.md`。
>
> **发布策略**：**发布包永远为 Rust 版**（`build.sh` 自动化构建 → `costcut-infer.exe`）；
> **Python 版仅作为技术探索用**（新功能验证/实验），不参与发布。

## 特性

- **多 MoE 架构**（配置驱动，Dense 暂不纳入）：
  - DeepSeek-V3 / R1 / V4（MLA + 路由/共享专家 + 组路由 + 前 K 层 dense）
  - GLM-5（GlmMoeDsa：MLA + DSA 稀疏注意力）、GLM4-MoE
  - Kimi K2.5 / K2.6 / K3（MLA + MoE）
  - Qwen3.5-MoE（主模型：delta rule 线性注意力 + AWQ）
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
  docs/              文档（total/ 双版共有 / python/ / rust/ —— 分版本归类）
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

## 支持的架构与量化（配置驱动——模型规格从各模型 config.json 读取）

| 架构 | 注意力 | 状态 |
|---|---|---|
| Qwen3.5-MoE（主模型） | delta rule + gated full | ✅ 完整运行 |
| DeepSeek-V3 / R1 / V4 | MLA（经典） | ✅ 配置 + 组件 |
| GLM-5（GlmMoeDsa） | MLA | ✅ 配置 + 组件 |
| GLM4-MoE | 标准 GQA | ✅ 配置 + 组件 |
| Kimi K2.5 / K2.6 / K3 | MLA（经典） | ✅ 配置 + 组件 |
| Mixtral / Qwen3-MoE / DBRX / Phi3-MoE | 标准 GQA | ✅ 配置 + 组件 |
| 未知非专属架构 | 标准 GQA（通用回退） | ✅ 自动适配（MoE / 稠密自动探测） |

> 各架构的路由数 / 层数 / 激活数等具体规格**由模型目录的 config.json 动态读取**（`model_config.py` 的架构归一化）——不在文档写死，避免与真实模型规格不符。

**量化**：AWQ int4 / GPTQ（bits 2/4/8）/ FP8（E4M3/E5M2）/ NVFP4（E2M1 + 块缩放）——注册表分发。

## 性能参考

- 实测数据与两版对比详见 `docs/total/性能分析报告.md`
> **vLLM 核心能力（已接入推理管线）**：PagedAttention（分页 KV 缓存）+ continuous batching（请求批处理）+ prefix caching（前缀复用）+ 量化内核（int4 融合 matmul——z 转置适配修复）；
> OpenAI 兼容 API 支持流式响应（`stream=true`——SSE）；参数用户可调：engine.toml `[paging]`（num_blocks/block_size）+ `[batching]`（enable）。
> 构建与发布：`./build.sh` 一条命令（Rust 构建 → v0.1.0_beta → Inno 打包 → setup/ 安装包——见 `docs/total/构建与发布指南.md`）。