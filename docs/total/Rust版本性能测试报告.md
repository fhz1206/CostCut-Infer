# Rust 版本性能测试报告（依赖启用后——64 位 mingw-w64 编译器）

> 日期：2026-08-19 ｜ 环境：Windows（Git Bash）x86_64-pc-windows-gnu ｜ 版本：release（target/release/costcut-infer.exe）
> 构建：依赖启用（candle-core/tokenizers/serde_json 等 8 个）——**64 位 mingw-w64（GCC 15.3——D:\mingw64\mingw-w64-gcc-15.3-stable-r45）**——RUSTFLAGS 绕行（getrandom rdrand + rust-lld + CC/CXX=x86_64-w64-mingw32-gcc/g++）；tch 暂禁用（需 libtorch 安装——~2GB）

## 1. 测试方法与数据来源

- 入口：`costcut-infer.exe --smoke`（开发冒烟——M1 真实加载/反量化 + M2/M3 合成前向/生成 + M4 并行 matmul 对比）
- 计时：各冒烟环节内部计时（release 优化——GCC 15.3 编译）

## 2. 实测数据（64 位 GCC 15.3 编译）

| 环节 | 项目 | 结果 | 说明 |
|---|---|---|---|
| **M1** | 多分片打开 | 张量索引 95427 项（跨 6 shard） | Qwen3.5-35B-A3B-AWQ-4bit |
| **M1** | AWQ 反量化 [2048x512] | max_abs=0.1282 std=0.0041 | 正确性验证 |
| **M2** | 合成 prefill | 3x64 logits finite=true | 前向正确性 |
| **M3** | 合成 generate（3 token） | **0.2ms**（≈0.000 s/token） | KV 缓存续接生成 |
| **M4** | 并行 matmul（512³） | serial 24.01ms vs parallel(4线程) 9.67ms——**2.48x** | **64 位编译器大幅提升（旧工具链 1.19x）** |
| **M4** | AVX2/FMA matmul | 19.14ms vs serial 24.01ms——**1.25x** | **转为正收益（旧工具链 0.93x 负收益）** |
| **M4** | 分块缓存 matmul | 105.5ms vs serial——**0.23x** | 仍为负收益 |

## 3. Python vs Rust 性能对比（合成口径）

| 项 | Python（torch 4 线程） | Rust（GCC 15.3） | 对比 |
|---|---|---|---|
| 合成 prefill matmul（3x64） | 20.53ms（torch 调用开销） | 0.2ms（3 token generate） | Rust 大幅领先（小规模） |
| 512³ matmul | **4.74ms（torch BLAS）** | serial 24.01ms / 并行 9.67ms | **Python BLAS 快 2-5x**（Rust 朴素标量无 BLAS） |
| 真实模型端到端 | 36.08 s/token（40 层） | 1 层标量反量化超时（P0） | Python 完整运行；Rust 受性能限制 |

## 4. 瓶颈与结论

1. **64 位现代编译器带来显著提升**：并行 matmul 1.19x → **2.48x**；AVX2 0.93x（负）→ **1.25x（正）**——编译器优化级别关键。
2. **合成模型 Rust 领先**（小规模 torch 调用开销大）；**大 matmul Python BLAS 领先 2-5x**（Rust 朴素标量——需 BLAS 内核追平）。
3. **真实模型端到端仍受标量反量化限制**（M5：1 层超时——P0 阻塞项——SIMD 打包内核前跳过）。
4. **tch 依赖暂禁用**（需 libtorch 安装——~2GB——LIBTORCH env；安装后启用 BLAS 追平 Python torch）。

## 5. 下一步（按优先级）

- **P0**：真实模型端到端——需 SIMD 打包内核（T-MAC/llama.cpp Q4 风格）或 libtorch（tch BLAS）
- **P1**：BLAS 内核接入（candle-core BLAS 或 tch/libtorch——大 matmul 追平 Python）
- **P2**：64 位编译器构建流程固化（CC/CXX/RUSTFLAGS 写入 build.sh）
