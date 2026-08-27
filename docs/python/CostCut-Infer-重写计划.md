# CostCut-Infer Python 推理引擎重写计划

> 版本：**v1.5**（2026-08-27，按用户反馈六次修订）｜ 状态：**待确认后开工**
> v1.5 变更：⑥ 新增 **资源限制系统（R11）**——CPU/内存/磁盘 IO 三类资源各限机器资源的可配比例，**默认 50%**：CPU 线程预算+优先级降档、内存 RSS 硬上限+看门狗五级自限、磁盘 IO 令牌桶限速；`--resource-pct 50` 全局一钮 + 分资源覆盖；Windows 无 cgroups → 合作式自治（诚实前提），§3.7。
> v1.4 变更：⑤ 新增 **量化子系统设计（§3.6）**——参考 vLLM `layers/quantization/` 体系（注册表 + QuantKey 声明式规格 + 每格式方法 + 在线量化 + KV 量化）：checkpoint 格式自动检测（compressed-tensors/FP8 主测）、CPU 量化内核矩阵（FP8 带宽收益 / INT8 VNNI 真加速）、在线量化（`--quantization fp8_per_token` 等，BF16 权重加载期量化）、KV cache 量化（fp8 减半容量）；§4 矩阵量化行扩展为 10 行分级；P8 扩 0.5 天。
> v1.3 变更：④ 新增 **R10 dense 模型层流式加载**——内存不足以容纳整个 dense 模型静态权重时 auto 切换到层流式（WeightRing，§3.5）：私有内存与模型总参数解耦，复用三层流水线预取阶段，不新增线程。
> v1.2 变更：③ **R8 升级为注册表级全覆盖**——对齐本地 vLLM 0.27.1 registry 全部 **434** 架构条目：L0 通用组装器原生 / L1 transformers 后端降级 / L2 显式清单，零静默未处理（§3.4 重写）。
> v1.1 变更：① 内存目标大幅下调——进程私有 RSS ≤ 2.5GB（权重全量 mmap 流式，embed/lm_head 逐行流式，§2.1）；② 新增 R8 架构适配（v1.2 已扩展为全覆盖）。
> 测试模型：`python/models/Ornith-1.5-35B-A3B-MTP-FP8`（≈41GB，35B 总参 / 3B 激活 MoE）
> 参考实现：本地克隆 `vllm/`（vLLM 0.27.1，v1 架构）
> 运行解释器：`C:/Users/ASUS/AppData/Local/Programs/Python/Python314/python.exe`（3.14.5）
> 配套文档：`docs/python/进度日志.md`（实时进度与困难，开发中持续更新）

---

## 0. 环境事实（已核查，非假设）

| 项 | 实测值 | 对设计的影响 |
|---|---|---|
| CPU | i7-1065G7，8 逻辑核，**AVX2 + AVX-512（含 VNNI/BF16）** | 算子用 numba JIT 出 AVX2/AVX512 代码；torch 2.13 CPU 自带 AVX512 GEMM |
| 内存 | 物理 11.7GB，可用约 8.2GB | **磁盘优先设计是硬需求**：41GB 模型 + KV 不可能常驻；40 层 MoE 专家必须零驻留 |
| 磁盘 | D 盘剩余 86.9GB，NTFS（mmap 大文件可用） | 专家权重直接 mmap 原 safetensors，不另做打包；KV 磁盘层另建文件 |
| 编译器 | 无 cl、无 gcc | **不能编译 C 扩展** → AVX2 路径 = numba JIT（SIMD 向量化）+ ctypes 调 Windows 系统 API；不引入需要编译的依赖 |
| 已装关键库 | torch 2.13.0+cpu、numpy 2.5.2、transformers 5.15.0（含 `qwen3_5_moe` 官方实现可对照）、vllm 0.27.1（仅参考）、numba 0.67.0、safetensors 0.8.0、fastapi/uvicorn、rich、orjson、tokenizers 0.22.2 | numba 是 AVX2 算子与流水线热点的唯一 JIT 手段；transformers/vllm 源码只读对照，不进运行时 |

### 0.1 模型结构事实（已从 config.json / index.json / safetensors 头核查）

- 架构：`Qwen3_5MoeForConditionalGeneration`（model_type `qwen3_5_moe`，与 Qwen3-Next 同族 hybrid 注意力）
- hidden 2048，**40 层**，`layer_types`：**30 层 linear_attention（Gated DeltaNet）+ 10 层 full_attention**（每第 4 层）
- **40 层全部是 MoE**：每层 256 专家、top-8、中间维 512；另有每层 1 个共享专家（中间维 512）
- 专家权重：**W8A8 FP8**（compressed-tensors `float-quantized`，quant_method=`compressed-tensors`）：权重 **FP8 E4M3 per-channel 静态** scale（memoryless_minmax），激活 **per-token 动态** 量化（symmetric）；**7 条 ignore 正则不量化**：lm_head / embed_tokens / `mlp.gate`（router）/ shared_expert_gate / **linear_attn 全部** / visual 全部 → linear_attn 权重 BF16，full_attn 投影 + MoE 专家 FP8；embed/lm_head BF16
- 单专家体积 ≈ gate 1.05MB + up 1.05MB + down 1.05MB（FP8）+ scale ≈ **3.15MB**；单步 decode 每层取 8 个专家 ≈ 25.2MB，40 层 ≈ **1.0GB/步（冷盘）**
- full_attn：16 q 头 / 2 kv 头 / head_dim 256，**mrope**（interleaved，section 11/11/10，partial_rotary 0.25，theta 1e7）
- linear_attn（GDN）：16 k 头×128 / 32 v 头×128，conv kernel 4，ssm dtype fp32；**递归态固定 32×128×128×4B≈2MB/层，与序列长度无关**
- KV 只发生在 10 个 full_attn 层：2 kv 头×2(KV)×256×2B = **2KB/token/层 → 20KB/token**；256K 上下文 ≈ 5.1GB → 超出 RAM 部分走磁盘层（需求 1 的核心场景）
- MTP：`mtp_num_hidden_layers=1`，权重在 `model-mtp.safetensors`（1.69GB，全 BF16；其中 256 专家以融合张量 `gate_up_proj`/`down_proj` 存放）→ MTP 专家**同样走零驻留流式**
- 多模态：含 vision tower（`qwen3_5_moe_vision`，BF16）与 mrope；文本推理为主路径，视觉按「注册点+可选实现」处理（见 §4 矩阵）
- generation_config：do_sample=true、eos=[248046,248044]、pad=248044、temp=1.0、top_k=20、top_p=0.95

### 0.2 参考实现定位（vllm/ 克隆，只读）

| 参考源 | 用途 |
|---|---|
| `vllm/v1/core/`（block_pool、kv_cache_manager、kv_offload、simple_kv_offload） | 分页 KV、前缀缓存、KV 卸载分层设计对照 |
| `vllm/v1/engine/`、`v1/.../sched` | 调度器、连续批处理、chunked prefill 语义对照 |
| `vllm/vllm/model_executor/models/qwen3_5.py / qwen3_5_mtp.py / qwen3_next.py` | 模型层数据流、MTP propose/verify 数据流对照 |
| `vllm/vllm/v1/spec_decode/`、`sample/`、`structured_output/` | 投机解码、采样参数集、受控解码语义对照 |
| `transformers/models/qwen3_5_moe/`（已装 5.15.0） | GatedDeltaNet 前向数值对照基准 |

> 定位：vLLM 是 **参考与功能基线**，不是运行时。本引擎纯 Python + numba + torch(CPU)，独立实现。

---

## 1. 需求分解

| # | 需求 | 验收口径 |
|---|---|---|
| R1 | **磁盘优先 KV Cache**（RAM 不足时） | KV 总容量可配且大于 RAM；高负载下进程自身 KV 内存占用 ≤ 配置上限，超出部分透明落盘；命中/驱逐/回写有指标 |
| R2 | **零驻留流式专家加载** | 稳态运行下，进程内「专家权重」驻留量 ≈ 0（仅当前步在算的缓冲区）；用 psutil 实测证明占用与专家总参数量无关（换 16/32/256 专家配置做对照实验） |
| R3 | **分层流水线异步加载**（计算‖预取） | 三层流水线（N 层计算 ‖ N+1 层路由+读 ‖ N+2 层投机预取）；IO 等待被计算掩盖的比率（overlap ratio）≥ 80%（预热后）并输出指标；AVX2 预取路径（§3.3）生效 |
| R4 | **vLLM 克隆版本的全部功能** | 按 §4 功能矩阵逐条实现；硬件绑定项给出 CPU 降级路径并文档化，不允许静默缺省 |
| R5 | **参数全可调**：CLI 或 toml，大小写不敏感 | §6 参数表全覆盖；`--Temperature=0.95` / `--temperature==0.95` / toml `[sampling] Temperature=0.95` 均生效；优先级 CLI > toml > 默认 |
| R6 | python/ 目录全部重写 | 旧文件已删除，全新文件树（§5）；`from ... import ...` 风格 |
| R7 | 实时进度/困难写入 docs | `docs/python/进度日志.md` 每个阶段与每次遇阻即更新 |
| R8 | **模型架构全覆盖**（v1.2：对齐 vLLM 注册表 434 条目） | 每个架构恰好归入一层且可查询（`--list-architectures` 全表+层级）：L0 通用组装器原生（声明式 spec，Ornith 主测）/ L1 transformers 兜底（功能可用，显式标注）/ L2 显式不支持清单（报错含原因）；**零静默「未支持」**；test_config 7 族全部 L0 重点验证（§3.4） |
| R9 | **内存占用最小化**（v1.1 新增） | 进程**私有 RSS ≤ 2.5GB**（不含 OS 页缓存）：全部权重 mmap 流式（embed/lm_head 逐行 gather），KV L1 默认 1GB 可缩至 512MB；psutil 实测验收（§8 R9 行） |
| R10 | **dense 模型层流式加载**（v1.3 新增） | 静态权重（非专家）总字节 > 内存预算时 auto 切换层流式：仅 `weight_ring_layers` 层权重驻留（可 sublayer 分片），计算完即弃；私有 RSS 与模型总参数解耦；数值与 page cache 模式逐 token 一致（§3.5） |
| R11 | **资源限制系统**（v1.5 新增，**默认 50%**） | CPU/内存/磁盘 IO 各限机器资源的可配比例（`resource_pct=50` 全局一钮，分资源可覆盖）：CPU 线程预算硬上限+线程优先级降档、内存私有 RSS 硬上限+看门狗五级自限（全程可逆不丢请求）、磁盘 IO 令牌桶限速；`--info` 打印预算表；Windows 合作式自治（无 cgroups，诚实标注）；验收 `test_resources.py` + P8 50% vs 100% 基准对照（§3.7） |

---

## 2. 总体架构

```
                 ┌────────────────────────────────────────────────────────────┐
                 │  入口层（可插拔前端）                                        │
                 │  CostCut-Infer.py(CLI) · api_server.py(OpenAI API) ·       │
                 │  tui_chat.py(Rich 终端) · Python SDK (CostCutInfer 类)      │
                 └──────────────┬─────────────────────────────────────────────┘
                                │ EngineOptions（统一配置，§6）
                 ┌──────────────▼─────────────────────────────────────────────┐
                 │  引擎编排层 engine.py                                        │
                 │  · 请求队列 / 连续批处理调度（参考 vLLM v1 scheduler 语义）    │
                 │  · chunked prefill + decode 混合步进                         │
                 │  · MTP 投机解码（propose→verify，接受长度统计）               │
                 │  · 指标总线（吞吐/延迟/IO 掩盖率/缓存命中，Prometheus 格式）   │
                 │  · 资源限制看门狗（CPU/内存/IO ≤50% 默认，§3.7）               │
                 └──────┬───────────────────────┬─────────────────────────────┘
        ┌───────────────────────▼──────────────┐  ┌──────────────────────────────▼─────────┐
        │ 模型执行 models/ 注册表 R8           │  │ 采样层 sampling.py                     │
        │ · ModelSpec 规范化                   │  │ · 全参数采样(temp/topk/topp/min_p/…)   │
        │ · 共享积木 GQA/MLA/GDN/MoE           │  │ · 重复/频率/存在惩罚、stop strings      │
        │ · RoPE 族/MTP/线性注意力             │  │ · 结构化输出（JSON schema 约束）        │
        │ · 7 架构组装（Ornith/DV4/Kimi/…）    │  │ · logprobs 计算                        │
        └───────────────────────┬──────────────┘  └────────────────────────────────────────┘
        ┌───────────────▼─────────────────────────────────────────────────────┐
        │ 三大核心机制（本次重写的灵魂）                                         │
        │                                                                     │
        │ ① 两级 KV Cache（需求1）        ② 零驻留专家流（需求2）               │
        │   L1: 内存分页块池（block=16tok）   专家权重直接 mmap 原 safetensors   │
        │   L2: 磁盘块文件（append+wal）    → 每层路由后只读 top-8 专家段        │
        │   前缀块哈希复用（vLLM prefix cache 语义）   → 计算完立即失效，不缓存    │
        │   L1 满→按 LRU 把冷块压缩/原样写 L2  → 共享专家常驻（每层仅 3MB）       │
        │   L2 块按需换回 L1（带 readahead）                                          │
        │                                                                     │
        │ ③ 三层流水线（需求3）                                                    │
        │   Stage C(计算):   layer N  GEMM/注意力/MoE 算子（numba AVX2）         │
        │   Stage P(预取):   layer N+1 路由→定位→mmap 读 8 专家→校验             │
        │   Stage S(投机):   layer N+2 用上一步路由结果投机读（命中率指标）        │
        │   跨层: N 层 attention/GDN 状态 ‖ N+1 专家 IO 重叠                       │
        └───────────────┬─────────────────────────────────────────────────────┘
        ┌───────────────▼─────────────────────────────────────────────────────┐
        │ IO 底座 io/                                                         │
        │ · mmap 专家读取器（带页对齐、prefetch 提示）                           │
        │ · 顺序读线程池（O_READAHEAD 语义 + numba AVX2 预取内核）               │
        │ · KV 磁盘层（块文件、索引、淘汰）                                      │
        │ · 权重清单扫描/校验（启动时建 index cache）                            │
        └─────────────────────────────────────────────────────────────────────┘
```

**数据流（decode 一步）**：
1. 调度器组 batch（decode 步 + 可能的 chunked prefill）→ 采样出 token 进入 MTP propose（1 层 MTP 生成 1 个草稿 token，可配）
2. 主模型 forward：embed → 40 层循环。每层内：
   - full_attn 层：从 L1/L2 取 KV 块做注意力；linear_attn 层：更新 2MB 递归态（常驻 L1）
   - MoE：router（常驻 gate 权重）→ top-8 → **流水线 Stage P 此刻应已把 8 个专家的 FP8 权重段读入 ring buffer** → numba 融合 `silu(gate·x)·up·x → dequant(FP8→BF16) → down·h` → 结果立即释放 buffer
3. 逐层完成时：`prefetch(layer+2)` 发出投机预取（用上一步同位置路由的历史专家 id 做 hint，命中则省一次路由等待）
4. verify：主模型对 [实际 token, 草稿 token] 做单步并行 forward，argmax 链式接受
5. 采样 → 输出；KV 写入分页块池（L1 满则触发 L2 下沉）

**为什么零驻留可行**：safetensors 是「8 字节长度 + JSON 头 + 连续 tensor 段」的纯内存映射格式，`mmap` 后每个专家就是文件内一段 `(offset, len)`。读它不产生"驻留"——OS 页缓存是系统级共享的（所有进程可用、自动淘汰），进程私有内存里只有正在算的那几 MB ring buffer。**专家驻留量 = 0 是结构性保证**，不靠手工管理。

### 2.1 内存模型（R9：进程私有 RSS ≤ 2.5GB）

v1.0 曾把 embed/lm_head（合计 ≈2GB）计为常驻——**v1.1 推翻**：它们不需要驻留。

**关键观察**：embed 与 lm_head 都是 `[vocab=248320, hidden=2048]` BF16 的"行表"，每行仅 **4KB**。embed 前向 = 按 token id 逐行 gather，lm_head 前向 = 按 batch 内每个请求的当前 hidden 做"行点积"（每请求只读 1 行）。即：每步对这两个 2GB 大表的实际访问量 = `seq_len × 4KB`，**全量 mmap 后逐行流式读，私有占用 ≈ 0**（热行由页缓存承接）。

**私有 RSS 预算表**（11.7GB RAM 机器，`kv_policy=hybrid`、`kv_l1_bytes=1GB` 默认）：

| 项 | 大小 | 说明 |
|---|---|---|
| torch/numpy/Python 基线 | ≈ 0.5GB | 解释器 + 库初始化，实测为准 |
| L1 KV 块池 | **1GB**（默认，可 512MB~4GB） | 唯一大块私有内存；`disk_first` 模式 = 512MB 热窗 |
| 激活 + 专家 ring buffer | ≈ 0.3GB | 40 层 batch 激活 + 2 活跃层 ring |
| 递归态（GDN 30×2MB 等） | ≈ 0.1GB | 与序列长无关 |
| 共享专家 + router gate（40 层） | ≈ 0.13GB | 每步必用，mmap 热页（OS 自然缓存） |
| **合计** | **≈ 2.0GB** | 留 0.5GB 余量至 2.5GB 红线 |

**不进私有预算的**（全走 OS 页缓存，可回收、与专家 IO 共享缓存）：全部静态投影权重（Ornith ≈2.3GB，热页会被 OS 留住）、embed/lm_head 行、L2 KV 块文件。
**效果**：KV 总量可配 64GB（远超物理内存）而不 OOM；权重总规模（35B→更大模型）**不改变私有 RSS**——私有内存与模型总参数完全解耦，这是 R2（专家）在 R9 下的自然推广。
**兜底**：`--mem-budget-gb` 参数让引擎按机器实测内存反推 L1/激活规模并在启动报告打印预算表；若 `MemAvailable < 预算×1.3` 启动即告警并自动收缩 L1。
**R10 补充（v1.3）**：上表默认所有静态权重走 OS 页缓存（共享、可淘汰）。当静态权重总字节超出内存预算（dense 大模型场景，如 70B 级）时，引擎 auto 切换**层流式模式**（§3.5）：私有内存主项变为 WeightRing（`weight_ring_layers × 单层字节`，可 sublayer 分片），仍与模型总参数解耦；该模式下私有 RSS 目标 = max(2.5GB, ring 预算 + 其余项)，由 `mem_budget_gb` 统一管控。

---

## 3. 三大核心机制详设

### 3.1 需求 1：磁盘优先 KV Cache（两级分页）

**块设计**（对齐 vLLM v1 block pool 语义）：
- 块大小 `block_size=16 token`（可配）。每 full_attn 层每块 = 16 × 2 kv 头 × 256 dim × 2 × 2B = 512KB；10 层合计 5MB/块。
- **只有 10 个 full_attn 层有 KV**；GDN 层状态是固定 2MB 递归态，天然「O(1) 常驻」，不占 KV 预算（这是 hybrid 架构的红利，需在指标里体现）。

**L1（内存块池）**：
- 预分配 `kv_l1_bytes`（默认 2GB）arena，块表 = (块号 → 层偏移映射)。
- 每块带**内容哈希**（token id 链式哈希，同 vLLM prefix cache）：相同前缀的请求共享块 → **前缀缓存**自动获得（vLLM 功能，§4）。
- GDN 递归态随块快照：块哈希相同的请求，GDN 态也可从块快照恢复（长上下文前缀复用的关键，vLLM 对 hybrid 模型的 KV 快照同理）。

**L2（磁盘块文件）**：
- 每请求会话一个 `.kvdb` 文件（`kv_cache_dir`，默认 `python/.kv_cache/`），固定 5MB 块槽 + 8B 头（哈希+层号+状态）。
- **下沉策略**：L1 水位 > `evict_high_water`（80%）时，按 LRU 把「最近 `kv_hot_window` 步未访问」的块原样（FP16）写入 L2；可选 `kv_l2_compression=none|lz4`（lz4 需纯 Python 轮子，默认 none，保证零依赖）。
- **换入策略**：块被访问但不在 L1 → 从 L2 读回（5MB 顺序读，NVMe 约 1ms / SSD 数 ms），**读回前先触发对后续块的 readahead 预取**（§3.3 的 AVX2 预取同样作用于 KV）。
- **磁盘优先模式** `kv_policy=disk_first`：L1 仅作「热窗」，块一旦写入 L2 就不再占 L1 配额——用于「RAM 极度不足」场景，L1 可缩到 512MB。
- 块淘汰：LRU + 哈希引用计数（前缀共享块 ref>0 不淘汰）。淘汰指标：`kv_l2_hits / evictions / demote_bytes / promote_bytes`。

**崩溃安全**：L2 文件带 64B 文件头（magic + 层数 + 块表偏移），进程崩溃后按头部截断到最后一个完整块即可复用；`kv_cache_dir` 可配置 `ttl` 自动清理。

### 3.2 需求 2：零驻留流式专家加载

**权重布局利用**（不做任何重打包，零预处理时间）：
- 启动时扫描 16 个主 shard 的 safetensors 头（只读 8B+JSON，≈秒级），建立 `专家 (layer, expert_id) → (shard 文件, offset, len, scale_offset)` 清单，落盘到 `python/.kv_cache/expert_index.json` 缓存。
- 每层 256 专家 × (3 个 FP8 权重段 + 1 个 BF16 scale 段) 共 4 个 mmap 句柄按 shard 复用（16 个文件各 1 个 `mmap` 句柄常驻，**句柄 ≠ 驻留数据**，OS 页缓存按页管理）。

**ring buffer 协议**：
- 每层 1 个 `ExpertRing`：容量 = `top_k × 单专家字节 × prefetch_slots`（默认 8×3.15MB×2 ≈ 50MB/层 × 2 活跃层 ≈ 100MB，`expert_ring_slots` 可配）。
- 计算 kernel 只从 ring buffer 读；读入由 Stage P/S 完成（§3.3）。计算完 ring buffer 槽位标记 free，**不写回磁盘、不进页缓存策略干预**（页缓存是 OS 的事，命中反而是好事）。
- 共享专家 + router gate：每层约 3.2MB，**常驻**（40 层合计 0.13GB）——这不是专家，是结构权重。
- MTP 专家：同样零驻留（`model-mtp.safetensors` 里的融合 `gate_up_proj`/`down_proj` 大张量按行段切读）。

**验证手段**（R2 验收）：
- `tests/test_zero_residency.py`：跑 200 步 decode，每步 `psutil.Process().memory_info().rss` 采样，断言 RSS 增量与「专家总参数量」解耦——用 monkeypatch 把 `num_experts` 从 256 改到 16 跑同样脚本，RSS 曲线应几乎重合（差 < 5%）。
- 引擎 `--profile-memory` 输出逐层「驻留/流式」权重清单到 `metrics.json`。

### 3.3 需求 3：三层流水线异步加载 + AVX2 预取

**流水线形态**（每步 decode，40 层推进）：

```
时间轴 ──────────────────────────────────────────────────────▶
计算核(1-2 线程)   [L1 计算]  [L2 计算]  [L3 计算]  [L4 计算] …
预取核(1 线程)     [L2 读8专家][L3 读8专家][L4 读8专家][L5 …]
投机核(1 线程)     [L3 投机读(用上一步路由hint)][L4 投机读]…
路由核(复用预取核)  L2 router 完成后立即发 L2 真实读请求
```

- **Stage C（计算）**：主线程跑 layer N 的 GEMM（numba `@njit(cache=True, fastmath=True)`，`prange` 并行，torch 侧仅做张量搬运；GEMM 内部 `torch.matmul` 已走 MKL/oneDNN AVX512，numba 负责融合 dequant+silu+scale 这类 torch 表达不了的细粒度算子）。
- **Stage P（预取，线程池 1 worker + `queue.Queue`）**：layer N+1 的 router 权重常驻 → 先算 router（便宜，2048→256 点积）→ 得 top-8 → 向 `ExpertReader` 发读请求 → `mmap[offset:offset+size]` 拷入 ring buffer（numba 融合拷贝+反量化成 BF16 的「计算型预取」，读进来顺手 dequant，计算阶段零开销）。
- **Stage S（投机预取，同线程低优先级队列）**：layer N+2 用上一步 decode 的同层路由历史（top-8 expert id 的滑动窗口，`speculative_route_history=4` 步并集）投机读。命中率指标 `spec_route_hit_rate`；未命中部分由 Stage P 补读（重叠读，mmap 天然幂等）。
- **AVX2 预取（需求点名项）**：
  1. 读请求到达时，对 50MB ring buffer 用 numba 手写 `prefetch` 内联汇编（`prefetcht0` 指令序列，步长 64B 行 → 256B 步）提前把即将写入的页拉进 L2/L3——掩盖 mmap 首次缺页开销；
  2. `readahead` 语义：Windows 无 POSIX `posix_fadvise`，用 `ctypes` 调 `SetFilePointerEx` + 预读后续块实现顺序读提示（NTFS 对顺序 IO 有自身 readahead，我们保证请求顺序性）；
  3. 读拷贝 kernel 本身 `numba` `prange` 分片 + 64B 对齐 `memcpy` 等价实现，8 专家并行拷贝。
  4. 所有 `prefetch` 指令走 **try/except 能力探测**（本核 AVX512 可用则同时用 `zmm` 批量拷贝路径；无 AVX2 的老核自动退化为 `memcpy` 纯 Python 路径——保持 §6 `prefetch_mode=auto|avx2|off`）。
- **掩盖率指标**（R3 验收）：每步记录 `io_wait_ms`（Stage P 未按时完成导致 C 阻塞的时间）与 `compute_ms`，输出 `overlap_ratio = 1 - io_wait/compute`，目标预热后 ≥ 80%（NVMe 冷读 1GB/步 ≈ 若盘速 3GB/s 则 330ms，远超单步计算 ~200ms → **纯 IO 掩盖不够，必须靠 Stage S 投机 + OS 页缓存温热化**；进度日志会持续跟踪真实数字并调参：`prefetch_layers_ahead` 默认 2、`expert_readahead_mb` 默认 128）。

> 诚实预期：i7-1065G7 平台磁盘吞吐是变量（SSD/NVMe/页面冷热）。若实测 overlap < 目标，计划中的回退顺序：① 增大 Stage S 提前层数（3→4）② 增大页缓存预热窗口 ③ 文档化「该盘型下推荐 `expert_cache_tier=page_cache_only`（依赖 OS 页缓存，放弃显式 ring buffer 预取）」模式。此权衡会实时写入进度日志。

### 3.4 需求 R8：模型架构全覆盖（v1.2：注册表级对齐 vLLM，三层覆盖）

**用户要求**：vLLM 对所有模型都有适配，本引擎同样——registry 里每个架构都必须有明确执行路径，**零静默「未支持」**。

**覆盖账本（本地 vLLM 0.27.1 registry 实测解析，2026-08-27）**：
- `_VLLM_MODELS` 共 **434** 条目：文本生成 121 / 多模态 115 / 投机解码 58 / embedding 33 / late-interaction 11 / 序列分类 10 / 词元分类 6 / reward 3 / transformers-supported 13 / transformers-backend 10
- 279 个模型文件中 **62 个是标准 decoder 模式**（Llama 24 / Qwen2 21 / Gemma 12 / Mistral 5 家族子类）——绝大多数架构 = 标准 GQA/MQA decoder + 变体开关（MoE / sliding-window / rope / 量化 / 多模态 wrapper）；特殊注意力（GDN / mamba / Kimi 线性等）仅 9 个文件
- ⇒ 全覆盖的正确做法不是手写 434 个模型，而是 **1 个通用组装器 + 声明式 spec 数据 + 少量特殊积木 + transformers 兜底**（vLLM 自身结构同理——它就有 `_TRANSFORMERS_BACKEND_MODELS` 长尾兜底层）

**三层覆盖（每个架构恰好归入一层，`--list-architectures` 输出全表+层级，随时可验证）**：

- **L0 原生快速路径**：`ModelSpec v2` → `models/generic.py` 通用组装器从 spec 构建。spec 是**声明式数据**（族模板 + 架构覆盖，非手写代码）：注意力类型（GQA/MQA/MLA/GDN/KimiLinear/sliding-window/hybrid layer_types）+ MoE（num/topk/shared/gate）+ rope 族 + norm 族 + **张量名映射模板**（config → safetensors key 名，架构间主要差异所在）+ 特性 flags（attn_output_gate / qk_norm / tie_word_embeddings / MTP / sliding interval）+ 任务类型。共享积木：norm / rope / gqa / mla / gdn / kimi_linear / moe / mtp / **heads（任务头）**。**Ornith 主路径与通用组装器同一代码路径**（Ornith 也是一份 spec），主测路径不享受特权代码。
- **L1 transformers 兜底层**：spec 不可表达 / 长尾 vision tower / 特殊 head 的架构 → 经 transformers 5.15 `AutoModel` 运行（对齐 vLLM `_TRANSFORMERS_BACKEND_MODELS` 设计）：功能可用但慢；**三大机制（两级 KV / 零驻留专家 / 流水线）不适用于本层**（用 transformers 自身缓存），启动报告与指标中显式标注 tier。⚠ 本层将 transformers 从「仅参考」放宽为「兜底运行时」（决策 D3，范围严格限于 L1 层架构）。
- **L2 显式不支持清单**：vLLM 自身已移除的架构（`_PREVIOUSLY_SUPPORTED_MODELS` 50 条）及 spec 不可表达且 transformers 无 modeling 实现者 → 显式报错，列出该架构的 vLLM 移除版本 / 最接近的可支持架构。

**落地结构（`ccut/models/`，对齐 vLLM registry + 共享 layers 的设计）**：

1. **registry 同步工具（覆盖正确性的机械保证）**：`tools/sync_vllm_registry.py` 解析本地 `vllm/model_executor/models/registry.py`（2026-08-27 实测可解析：10 个子字典、434 条目、无平台条件分支）→ 生成 `ccut/models/registry_table.json`（架构名 → tier(L0/L1/L2) + 族模板 + 备注），并纳入 git。**`test_registry_coverage.py`**：断言「本表条目数 = vLLM registry 解析数、无架构落在任何层之外、L2 条目均有理由字段」——vLLM 升级后跑一遍同步+测试即可增量对齐。
2. **`ModelSpec v2`（规范化描述符）**：`parse_config(模型目录) → ModelSpec`。吸收 config 形态差异（顶层 vs `text_config` 嵌套、`architectures` vs `model_type`、rope 各种写法、quant 各格式）→ 引擎唯一内部表示：层列表（每层 attention 类型 + MoE 参数）+ rope spec + norm spec + quant spec + mtp spec + embedding/head 规格 + 任务类型。**config 解析与架构实现解耦**。
3. **族模板 + 架构覆盖（spec 数据来源）**：`models/families/*.json` 声明式模板（llama / qwen2 / gemma / mistral / deepseek-mla / gdn-hybrid / kimi / minimax / hy…，覆盖 62 个标准 decoder 文件所属家族）+ 每架构的 override（张量名映射模板 = 架构间主要差异、特性 flags、异常项）。新增一个「标准家族」架构 = 加几行 JSON，**不改代码**。
4. **共享积木库（各积木独立单测）**：`attn_gqa`（含 mrope/partial-rotary/sliding-window/hybrid layer_types）、`attn_mla`（压缩 KV 布局 `(kv_lora_rank+qk_rope_head_dim)`/token/层 + 滑动窗口变体）、`attn_gdn`（GatedDeltaNet）、`attn_kimi_linear`（Kimi-K3 线性注意力，参考 transformers `kimi_k25`）、`moe`（top-k router + 共享专家 + 门控 + 零驻留 reader，§3.2 复用；896 专家/层的 Kimi-K3 同样零驻留）、`rope`（default/mrope/yarn/deepseek_yarn）、`norm`（RMSNorm 族含 q/k norm）、`mtp`（通用 N 层 proposer）、`heads`（任务头：causal LM / embedding pooling / 分类 / reward 标量——对齐 vLLM 的 10 类任务字典）。
5. **`models/generic.py` 通用组装器**：spec → 层装配 + 权重映射 + KV 预算声明。**Ornith、DeepSeek-V4、Kimi-K3、Llama、Qwen2.5… 全走这一条路径**（Ornith 主测路径 = 族模板 gdn-hybrid + override，不享受特权代码）。
6. **L1 兜底后端**：`models/backend_transformers.py`——L1 层架构经 transformers 5.15 `AutoModel` 运行；启动报告打印「该请求走 L1 层：功能可用，三大机制不生效」；`--arch-tier=strict` 可强制只允许 L0（服务高吞吐场景拒绝降级）。
7. **KV 预算泛化**：§3.1 块池按「每层 KV 字节/token」参数化——MLA 压缩布局（DeepSeek-V4-Flash ≈ 43 层 × (512+64) × 2B ≈ **~49KB/token** vs Ornith GQA 20KB/token），GDN/线性层 O(1) 递归态。两级 KV、前缀缓存、抢占 swap 架构无关，直接复用。

**验证策略（分级，成本与层级匹配）**：
- **全覆盖机械验证（434 条，CI 必跑）**：`test_registry_coverage.py`（账本对拍）+ `test_model_spec_fuzz.py`：对 L0 架构批量「合成 config（按族模板随机化参数）→ parse → 构建小尺寸随机权重 → 2 步前向无 NaN/形状正确/KV 预算=公式」；无需真实权重。
- **L0 重点验证（test_config 7 族 + 标准家族代表 llama/qwen2/gemma/mistral，共 ~11 个）**：config 黄金断言 + 随机权重冒烟 + `--verify-against=transformers` 真权重对拍钩子（transformers 5.15 已装 deepseek_v4 / glm_moe_dsa / kimi_k25 / longcat_flash / minimax_m3_vl / hy_v3 / qwen3_5_moe / llama / qwen2 等官方实现，放真权重即对拍，容差与 P1 相同）。
- **L1 层**：抽样 3~5 个架构 transformers 后端冒烟（能加载、能生成、tier 标注正确）；全量 434 个不逐一实测（无权重，诚实标注）。
- **L2 层**：报错文案单测（含 vLLM 移除版本/替代建议）。

**范围声明（v1.2 取代 v1.1 的 7 族限定）**：对齐目标 = **本地 vLLM 0.27.1 registry 的 434 条**（快照固化在 registry_table.json）；vLLM 之后新增架构 = 跑同步工具增量对齐（文档化流程）；外部注册点 `register_architecture()` 保留（可注册 registry 外架构，tier 标 L0-custom）。

### 3.5 需求 R10：dense 模型层流式加载（WeightRing，v1.3）

**触发场景**：某些架构的静态权重（非专家部分）大到放不下内存预算——如 70B 级 dense 模型（BF16 ≈ 140GB / FP8 ≈ 70GB）、大 MLA 投影模型。v1.2 默认靠 OS 页缓存承接静态权重（共享、可淘汰；内存吃紧时 thrash）。本特性提供**显式有界流式模式**，与 R2（专家零驻留）构成完整对偶。

**auto 检测（`weight_streaming=auto`）**：启动时从权重清单算 `static_weight_bytes`（非专家权重段总字节），与内存预算（可用内存 − KV L1 − 激活 − 专家 ring − 基线，即 `mem_budget_gb` 反推后的余量）比较：
- `≤ 50%` 余量 → **page cache 模式**（v1.2 默认，mmap 按需读、靠页缓存）；
- `> 50%` 余量 → **层流式模式**（本节），启动报告打印模式判定 + 预计带宽影响（决策 D4：阈值取固定 50%，简单可预期）；
- 手动覆盖：`weight_streaming=on|off|auto`（off 在内存不足场景启动即告警）。

**机制（复用 §3.3 三层流水线，不新增线程）**：
- `WeightRing` arena：`weight_ring_layers`（默认 2）槽，每槽容量 = 该模型最大单层静态权重大小（MoE 模型层静态权重 = attn 投影 + 共享专家 + norms + router；dense = attn + FFN）。槽位状态机：`free → prefetching → ready → computing → free`。
- 流水线映射：Stage C（layer N 计算）从 ring 读静态权重；**Stage P**（layer N+1）并行两件事：top-k 专家读（R2）+ 发起 layer N+1 静态权重读入槽 (a)；**Stage S**（layer N+2）并行：专家投机读 + layer N+2 静态权重读入槽 (b)。layer N 算完 → 槽 (a) 释放 → 供 N+3 用。「层权重预取」与「专家预取」同阶段并行，AVX2 `prefetcht0` / readahead / `prange` 拷贝核原样复用。
- **sublayer 分片**（`weight_stream_chunk`）：单层静态权重大小 > 槽位 50% 时（如 dense 单层 FFN ≈ 1~2GB），按子模块（attn 块 / FFN gate-up / down）分槽、分块计算即释放——auto 默认启用，可强制 `layer` / `sublayer`。
- embed/lm_head 不受影响（已逐行流式，§2.1，不占 ring）；KV / GDN 态 / 共享专家不受影响。

**内存账**：私有 RSS（流式模式）= 基线 + KV L1 + 激活 + **WeightRing** + 专家 ring。ring 大小只取决于 `ring_layers × 单层（或 sublayer）规模`，**与模型总参数量无关**——R2（专家）+ R10（静态权重）共同构成「私有内存与模型规模完全解耦」：两者叠加，进程可承载任意规模模型（受限于磁盘而非内存）。

**诚实吞吐预期**：decode 一步需读全模型静态权重——dense 70B BF16 ≈ 140GB/步，3GB/s 盘速 ≈ 47s/token，1GB/s ≈ 140s/token。**本模式目标是「内存可行性」（能跑），不是「快」**；启动报告打印实测盘速 → 预计 token/s，运行期输出每步带宽指标（`weight_stream_bandwidth_log`）。Ornith 主测静态部分仅 ~2.3GB，该模式在通常不触发（其专家已流式）——本特性主要服务 R8 覆盖路径下的 dense 族（Llama-70B / Qwen-72B / GLM dense 等，对应 vLLM 的 `cpu_offload_gb` 分层加载概念）。

**验收（R10）**：`tests/test_weight_streaming.py`——合成 dense 模型（磁盘权重总量 > 可用内存，如 4GB）：① auto 正确切换流式模式（阈值两侧检测单测）；② 10 步 forward 私有 RSS ≤ ring 预算 + 10% 余量；③ 数值输出与 page cache 模式（同步读）**逐 token 一致**；④ sublayer 分片路径单测；⑤ `off` 强制 page cache 模式且内存不足时启动报告明确告警。

### 3.6 量化子系统（v1.4，参考 vLLM `layers/quantization/` 体系）

**参考对象（vLLM 0.27.1 实际结构，已核查）**：
- 三层分发：`QUANTIZATION_METHODS` 注册表（`config/quantization.py`）→ 每格式一个文件里的 `XxxConfig(QuantizationConfig)`（`from_config` 解析 checkpoint / `override_quantization_method` 处理用户 CLI 覆盖 / `get_quant_method` 按层 kind 返回 `QuantizeMethodBase` 子类）→ `method.create_weights / apply / process_weights_after_loading` 被模型层统一调用
- 声明式规格：`QuantKey(dtype, scale, scale2, symmetric, GroupShape)`（`utils/quant_utils.py`，30+ 预定义键：kFp8DynamicTokenSym / kInt8StaticChannelSym / kMxfp8Dynamic / kNvfp4…）——量化格式 = 数据声明，kernel 按 key 派发
- 用户面：`--quantization` 接受格式名或在线简写（`fp8_per_token` / `int8_per_channel_weight_only` / `mxfp8` / `nvfp4_per_token`…，`_ONLINE_SHORTHANDS` 糖化成 `QuantizationConfigArgs{linear, moe, ignore}`）
- 每格式一文件：fp8 / compressed_tensors / gptq / awq / bitsandbytes / mxfp4 / mxfp8 / nvfp4(online) / torchao / kv_cache（`BaseKVCacheMethod`）…
- 通用工具：`scaled_quantize / scaled_dequantize / group_broadcast / pack_rows`（numba 等价重写）

**本引擎落地（`ccut/quant/`，结构一一对应，去 CUDA 化）**：

1. **checkpoint 自动检测（主路径，P1）**：`registry.py` 读 `config.json.quantization_config` → `quant_method` 查注册表 → 实例化 `XxxConfig.from_config()` → 引擎按层取 method。未注册 quant_method → 显式报错（列出已支持清单 + 建议 L1 后端）。
   - **compressed-tensors（Ornith 实际格式，主测路径）**：`config_groups[*].{weights, input_activations}.strategy ∈ {channel|token|tensor|group(group_size)}` + `ignore` 正则列表 → 归一为每层 `QuantSpec(weight=QuantKey, activation=QuantKey)`；7 条 ignore 正则（lm_head/embed/router/linear_attn/visual）按 vLLM `is_layer_skipped` 语义匹配 → 这些层走 BF16 无量化路径（与 §0.1 事实一致）。
2. **CPU 量化内核矩阵（`kernels.py`，numba SIMD）**——**诚实标注 CPU 与 GPU 的收益差异**：

   | 计算路径 | CPU 实际收益 | 定位 |
   |---|---|---|
   | **W8A16**（FP8 权重 → dequant → BF16 matmul） | **精确等价 BF16，磁盘/带宽减半**——本引擎默认；FP8 在 CPU 上无原生 dot，dequant 是 SIMD 查表+mul，开销可忽略 | 主路径（P1，Ornith） |
   | **W8A8**（激活 per-token 动态量化 → FP8×FP8 → 反量化累加） | 磁盘减半同 W8A16；计算端 CPU 无 FP8 dot，需软件模拟（dequant 后 BF16 dot 或 FMA 近似）——**收益 ≈ W8A16，默认关闭**，`fp8_compute_mode=w8a16|w8a8` 可开做对照 | 可选项（P1 骨架，P8 对照基准） |
   | **INT8 W8A8 VNNI**（`vpdpbusd`，1065G7 AVX512-VNNI 原生） | **CPU 上唯一有真 dot 加速的量化**：int8 dot 吞吐 = BF16 的 2×（8 字节/32B 向量 vs 2 字节），且权重磁盘减半 | 目标路径（P8，`quantize_weights` 在线生成 INT8 权重） |
   | **MX 格式**（mxfp8/mxfp4/nvfp4，32 组 E8M0 指数 scale） | dequant 内核 = E8M0 指数展开 + 128-block 广播（`prep_scale_for_group_broadcast` 移植）；nvfp4 需 FP8 E4M3 二次 scale | checkpoint 兼容 + 在线（P8，🟡） |
   | **W4 weight-only**（GPTQ/AWQ/bnbf-NF4） | int4/nf4→BF16 流式 dequant + matmul；**对 R10 层流式收益最大**（磁盘再减半，流式带宽需求减半） | 兼容（P8，🟡） |

3. **在线量化（`online.py`，P8）**：`--quantization fp8_per_token|int8_per_channel_weight_only|mxfp8|nvfp4_per_token`（简写表对齐 vLLM `_ONLINE_SHORTHANDS`，子集+可扩）：BF16/FP16 checkpoint 走**加载期量化**——权重流经 ExpertReader/WeightRing 时用 numba `scaled_quantize`（per-channel/per-token/per-128-group，memoryless_minmax observer 在线算 amax）就地量化进 ring，磁盘仍存 BF16、运行时按量化路径走；与 checkpoint 自带量化**互斥**（两者同设 → 报错）。`[quant] quantization` 参数走 casefold 体系，`--list-params` 列出全部简写。
4. **KV cache 量化（`kv.py`，P8）**：`kv_cache_dtype=auto|bf16|fp8`（对齐 vLLM `kv_cache_dtype` + `BaseKVCacheMethod` 语义）：auto 从 checkpoint `kv_cache_scheme` 读（Ornith=null→BF16）；fp8 = per-token 对称量化 + 每 token 1B scale，**KV 块 L1/L2 字节减半**（R1 协同：同样 1GB L1 装 2× token；256K 上下文从 5.1GB→2.6GB）；量化/反量化发生在块写入/读回的 numba 路径里，注意力计算仍 BF16。
5. **与三大机制的集成点**：量化是「**数据格式层**」，机制是「**搬运层**」——正交组合：ExpertReader 读 FP8 段 → ring buffer 按计算路径 dequant（W8A16）或保留 FP8（W8A8）；WeightRing（R10）流式的权重同样按 spec 量化路径走；KV 块池（R1）块大小公式已按 `kv_cache_dtype` 参数化。**验收统一口径**：任一量化路径下，与 BF16 参考（transformers 官方实现）的 token 级一致率 ≥ 99%（量化误差是模型属性，如实记录分布）。
6. **测试**：`test_quant_kernels.py`（各 QuantKey 的 quantize/dequantize 往返误差 < 1ULP/2ULP 分组、group_broadcast 布局、与 torch 参考值对拍）、`test_quant_registry.py`（compressed-tensors config 解析 → 40 层逐层 QuantSpec 黄金断言，含 7 条 ignore 命中层正确走 BF16；未注册格式报错文案）、`test_quant_e2e.py`（Ornith FP8 W8A16 主路径 = P1 数值基准；在线 INT8 端到端 32 token 对齐 ≥ 99%；KV-fp8 前后 KV 字节减半断言）。

### 3.7 资源限制系统（R11，v1.5，**默认 50%**）

**目标**：引擎是共享机器的「好公民」——CPU / 内存 / 磁盘 IO 三类资源各限制在机器资源的可配比例内，**默认 50%**（`resource_pct=50` 一个旋钮，三类同时生效；`resource_{cpu,mem,io}_pct` 可分资源覆盖）。保证机器上其他进程（用户工作、其他服务）不被推理占死。

**Windows 平台事实（诚实前提）**：无 cgroups/硬隔离 → 限制全部为**合作式自治**（启动期预算 + 运行期限速 + 看门狗自限），不是内核强制。引擎保证「不故意超限」，极端情况（页错误风暴、内核缓冲）不绝对硬隔离——文档明示（K13）。

**三类资源实现（`ccut/resources/`）**：

| 资源 | 限值（50% 默认） | 机制 | 强度 |
|---|---|---|---|
| **CPU** | 引擎线程预算 = 50% × 8 逻辑核 = **4 线程** | `budget.py` 分配：scheduler 1 + prefetch 1 + speculative 1 + 计算（numba `prange` / torch matmul）2；`torch.set_num_threads(2)` + numba OMP 线程限 + 全引擎线程 ctypes `SetThreadPriority(BELOW_NORMAL)` 降档（软让出前台）；优先于 [engine] 线程参数生效 | 强（线程数硬上限 + 优先级） |
| **内存** | 私有 RSS 硬上限 = 50% × 11.7GB ≈ **5.85GB** | 启动预算表：L1/ring/激活取 min(`mem_budget_gb` 目标 2.5GB，硬上限 5.85GB) 推导——**R9 的 2.5GB 是「打算多省」的目标，50% 上限是「绝不越过」的顶**，正常目标远在上限内；`watchdog.py` 每 1s 采样 psutil，连续 3 次超限触发分级自限（下表） | 强（自限，分级可逆） |
| **磁盘 IO** | 读写带宽 ≤ 50% × 实测盘速（P0 探测，如 3GB/s → **1.5GB/s**） | `limiter.py` 令牌桶（100ms 补满）门控 ExpertReader / WeightRing / KV L2 读写 + 并发读 ≤ 2~4；桶空 IO 线程休眠（**休眠即让出**） | 中（控主动读；OS 页错误首触读部分不可控，如实标注） |

**看门狗自限升级链**（内存/CPU 超限，连续 3 采样 ≈3s）：
① 暂停 Stage S 投机预取 → ② 收缩专家/权重 ring 槽位 → ③ KV L1 冷块驱逐（下沉 L2，复用 R1 机制）→ ④ 背压：调度器暂停接收新请求（在跑的跑完）→ ⑤ 仍超限则日志告警 + `resource_throttle_events` 指标。**迟滞恢复**：回落至限值 ×0.8 以下连续 10s 后按 ④→① 逆序回退。全程**可逆、不丢弃请求**（区别于 OOM 崩溃）。

**与既有机制的关系（正交，无冲突）**：
- 与 R9：目标（2.5GB）在上限（5.85GB）内共存，上限只在目标被突破时（并发膨胀/KV 扩张）兜底；
- 与 R10：WeightRing auto 阈值的「内存余量」分母 = 资源上限反推，同一 `budget.py` 出数；
- 与 R3 流水线：Stage P/S 的读全过 IO 令牌桶——限速不破坏掩盖语义，桶空时优雅退化为「计算等 IO」，overlap 指标如实反映下降（不造假）；
- 与 R1 KV：看门狗 ③ 级驱逐直接调用 L1→L2 下沉路径。

**指标（P6 入 Prometheus）**：`ccut_resource_{cpu_pct, mem_bytes, io_mbps}`（当前用量）+ 同名 `_limit`（限值）+ `ccut_resource_throttle_total{stage=…}` + `ccut_resource_throttle_active`（当前 0~5 级）。`--info` 启动即打印完整预算表（线程分配 / 内存上限 / IO 速率 / 推导过程）。

**验收（R11）**：`tests/test_resources.py`——① 线程预算：实测引擎全部线程（numba/OMP/torch/流水线）≤ 4 @50% 且分配符合预算表；② 令牌桶：读 100MB 实测速率落在 [0.45, 0.55]×盘速（±15%）；③ 看门狗：注入假 RSS 采样器（超限）→ 断言 ①→④ 升级有序发生、指标计数、在途请求正常完成；④ 迟滞：采样器回落 → 逆序回退；⑤ `--resource-pct` 覆盖优先级（全局 vs 分资源）单测。P8 基准：50% vs 100% 吞吐/延迟对照 + **并行后台压测**（CPU 燃烧 + 盘读）实测「系统剩余余量」证明 50% 真的留出了 50%。

---

## 4. 功能矩阵（R4：vLLM 克隆版本的全部功能）

范围界定：「全部功能」= vLLM 0.27.1 v1 引擎中**与 LLM 推理服务相关**的功能面。纯硬件后端（CUDA/ROCm/XPU kernel、FlashAttention/MLA 专用 kernel、TP/PP 多卡、LoRA 在线训练、profiler 的 NCU 集成、Ray 集群、disaggregated prefill 的 RDMA 等）在本机无对应硬件，**按「CPU 降级路径 + 文档标注」实现等价语义**，不允许静默缺省。逐条如下（✅=等价实现 / 🟡=降级实现 / ⬜=不适用于 CPU 单机，文档说明）：

| 模块（vLLM 参考路径） | 功能 | 本引擎落地 | 状态 |
|---|---|---|---|
| 引擎核心 `v1/engine/` | LLMEngine 生命周期、offload 到前台/后台线程 | `engine.py`：主线程 + 工作线程（流水线核） | ✅ P3 |
| 调度 `v1/core/sched/` | continuous batching、chunked prefill、请求抢占/抢占恢复（recompute vs swap） | 同一调度器：batch 内混排 prefill chunk + decode；KV 不足时抢占 → 状态 swap 到 L2（GDN 态快照 + KV 块下沉），恢复时回载 | ✅ P3 |
| KV 管理 `v1/core/`（block_pool/kv_cache_manager） | 分页 KV、block 复用（prefix caching）、hash 链 | §3.1 L1 块池 + 前缀哈希 | ✅ P2 |
| KV 卸载 `v1/kv_offload` `v1/simple_kv_offload` | KV 内存→磁盘分层、watermark 触发 | §3.1 L2 磁盘层 | ✅ P2（本引擎原生强化） |
| 权重 offload | `cpu_offload_gb`（权重卸载）/ 分层加载 | **R10 层流式加载**（§3.5：auto 检测 + WeightRing + sublayer 分片；CPU 单机场景的等价物） | ✅ P4 |
| 请求 `v1/request.py` | Request 状态机（waiting/running/finished/aborted）、优先级 | 同语义 + `request_priority` | ✅ P3 |
| 采样 `v1/sample/` | temperature/top_k/top_p/min_p/typical/rep_penal/length_penal/frequency/presence、early stop、seeds、best_of/n、logprobs、prompt_logprobs、allowed/bad words、detokenize 增量 | `sampling.py` 全参数实现（§6 参数表全覆盖） | ✅ P4 |
| 受控输出 `v1/structured_output/` | JSON schema / regex / grammar 约束解码（vLLM 用 outlines/xgrammar） | 用已装 `outlines_core` 实现 FSM 掩码解码（grammar → logits mask 每步应用） | ✅ P6 |
| 投机解码 `v1/spec_decode/` | ngram / EAGLE / MTP proposer | **MTP 1 层 proposer**（模型自带 MTP 头，`mtp_num_hidden_layers=1`）：propose 1 token（可配 0 关闭）+ verify 接受统计 `accept_len`；ngram proposer（prompt 内复用，`ngram_window` 可配）作为备选 | ✅ P5 |
| 模型执行 | 权重加载（safetensors 分片、quantization config 解析） | 自研 mmap 加载器（§3.2）+ `quant/` 注册表解析 checkpoint 量化配置（§3.6） | ✅ P1 |
| 量化核心 `layers/quantization/` | `QUANTIZATION_METHODS` 注册表 + 每格式 `QuantizationConfig` + `get_quant_method` 按层分发 + `QuantKey` 声明式规格 | `ccut/quant/`：registry + spec（QuantKey 移植）+ method 分发 + 格式文件（§3.6） | ✅ P1 |
| 量化 FP8 | W8A8（dynamic act / static weight）、W8A16、per-tensor、128-block | numba 内核：FP8 E4M3↔BF16 SIMD 转换 + 融合 GEMM；**CPU 无原生 FP8 dot → 默认 W8A16 计算（精确反量化），W8A8 可切换**（§3.6/D5） | ✅ P1（主测 W8A8 存储） |
| 量化 compressed-tensors | Ornith 实际格式（config_groups + ignore 正则） | `quant/compressed_tensors.py`：config_groups→QuantSpec、7 条 ignore 正则、scale 布局解析 | ✅ P1 |
| 量化 INT8 | W8A8（VNNI）/ W8A16 weight-only | `quant/int8.py`：**AVX512-VNNI（vpdpbusd）原生加速**（CPU 上唯一有真 dot 收益的量化） | ✅ P8 |
| 量化 MX 格式 | mxfp8/mxfp4/nvfp4（32 组 E8M0 指数 scale） | `quant/mx.py`：dequant 内核 + 在线量化路径（checkpoint 格式解析） | 🟡 P8 |
| 量化 GPTQ/AWQ（W4A16） | marlin kernel（GPU 专用） | CPU 等价：**dequant weight-only**（int4→BF16 流式 + matmul，磁盘体积减半对 R10 流式收益大） | 🟡 P8 |
| 量化 bitsandbytes | NF4/INT8 | dequant weight-only 等价路径 | 🟡 P8 |
| 量化 torchao/humming/inc/quark/modelopt | GPU 生态专用 | ⬜ 显式报错 → L1 transformers 后端兜底 | ⬜/🟡 |
| 在线量化 | `--quantization fp8_per_token / int8_per_channel_weight_only / mxfp8 / nvfp4_per_token …` 简写（`QuantizationConfigArgs`：linear/moe spec + ignore） | `quant/online.py`：BF16/FP16 checkpoint **加载期量化**进权重流（numba quantize 内核），在线 FP8/INT8/MXFP8；与 checkpoint 量化互斥校验 | ✅ P8 |
| KV cache 量化 | `kv_cache_dtype=fp8`（`Fp8KVCacheMethod`，per-token scale） | `quant/kv.py`：KV 块写入/读回时 numba 量化/反量化，**L1/L2 容量减半**（R1 协同）；默认 auto（Ornith `kv_cache_scheme=null`→BF16） | ✅ P8 |
| 多模态 `multimodal/` | 图像输入（processor 集成、token 替换、encoder cache） | 注册点架构（沿用旧 docs/python/多模态适配方案.md 设计）；Ornith vision tower（27 层 ViT）实现为可选路径 `--enable-vision`，默认关闭（CPU 上 ViT 慢，文档标注） | 🟡 P7 |
| 入口 `entrypoints/` | OpenAI 兼容 API（/v1/chat/completions、/v1/completions、/v1/models、流式 SSE、tools/function-calling 解析、logprobs 返回、inference parameters 全量） | `api_server.py`：fastapi+uvicorn，全部端点 + SSE 流式 + tool-call 文本解析（Qwen 工具格式） | ✅ P6 |
| 入口 `entrypoints/llm.py` | 离线 batch LLM 类（`llm.generate([prompts], SamplingParams)`） | `sdk.py`：`CostCutInfer.from_pretrained()` + `.generate()` + `.chat()` | ✅ P4 |
| 入口 `entrypoints/openai` 其余 | embeddings（pooling 模型） | 本模型非 embedding 模型：实现端点但返回 400「模型不支持」+ 文档说明（vLLM 语义一致） | 🟡 P6 |
| Tokenizer/renderers | chat template、tool parser、增量 detok | `tokenizers` + 仓库内 `chat_template.jinja`；增量 detokenize 带 `skip_special_tokens` | ✅ P3 |
| 指标 `v1/metrics/` | Prometheus `/metrics`（请求数、排队/推理时间、token 吞吐、KV 用量、preempt 次数、接受率） | 指标总线 → `/metrics` 端点 + 文件 dump（§3 各机制指标全含） | ✅ P6 |
| 日志/可观测 | 分级日志、请求 trace id | `logging` 分级 + `--log-level`；每请求 `request_id` 贯穿 | ✅ P3 |
| 资源限制 | vLLM 无独立资源限制模块（依赖部署层 cgroup/k8s limits） | **R11 资源限制系统**（§3.7）：CPU 线程预算+优先级 / 内存 RSS 看门狗五级自限 / IO 令牌桶限速，默认 50%——单机部署场景的内置等价物 | ✅ P0 预算 + P3 限速/看门狗 |
| 配置 `config/` | 环境变量 + config 全参数体系 | §6 统一配置层（CLI/toml/env 三源，大小写不敏感） | ✅ P0 |
| 并发 `v1/executor/` | TP/PP/EP 多卡、LoRA、量化内核切换 | ⬜ CPU 单机：`tensor_parallel_size` 等参数保留接口，值 ≠1 时显式报错「本引擎 CPU 单机版不支持多卡」 | ⬜ |
| 分布式 `distributed/` | 多机 RPC、KV transfer、disagg | ⬜ 同上，参数保留并显式报错 | ⬜ |
| 平台 `platforms/` | CUDA/ROCm/XPU 检测 | `platforms.py`：CPU 能力探测（AVX2/AVX512/内存/盘速基准），启动报告 | ✅ P0 |
| 工具 `scripts/`、benchmarks | 基准脚本 | `benchmarks/`：吞吐/延迟/TTFT/TPOT、IO 掩盖率、零驻留验证 | ✅ P8 |

> 矩阵在开发中随实测动态修订（进度日志同步）。「🟡 降级」项若用户要求必须 1:1 全硬件行为，需单列评估。

---

## 5. 新文件树（python/ 全部重写）

```
python/
├── CostCut-Infer.py          # 唯一 CLI 入口（需求 3 命名），argparse 宽松解析（§6）
├── api_server.py             # OpenAI 兼容 API 服务（uvicorn 拉起）
├── tui_chat.py               # Rich 终端聊天（流式、/help /metrics /reload）
├── sdk.py                    # Python SDK：CostCutInfer 类（离线 generate/chat）
├── engine.toml               # 默认配置（toml，分节：engine/kv_cache/experts/pipeline/sampling/model/api）
├── requirements.txt          # 仅运行时依赖（torch/numpy/numba/safetensors/tokenizers/fastapi/uvicorn/orjson/rich/psutil/outlines_core）
├── ccut/                     # 引擎包
│   ├── __init__.py
│   ├── config.py             # 统一配置：schema+默认值+CLI/toml/env 合并+大小写折叠+校验
│   ├── platforms.py          # CPU/AVX/内存/盘速探测，启动报告
│   ├── engine.py             # 编排：调度、批处理、MTP verify、指标总线
│   ├── models/               # R8 架构层（注册表级全覆盖，§3.4）
│   │   ├── __init__.py
│   │   ├── spec.py           # ModelSpec v2 规范化描述符 + config 解析器
│   │   ├── registry.py       # registry_table 加载 / tier 路由 / register_architecture / L2 报错
│   │   ├── registry_table.json  # 434 架构账本（tools 从 vLLM registry 生成，入 git）
│   │   ├── generic.py        # 通用组装器：spec → 层装配 + 权重映射 + KV 预算（L0 唯一代码路径）
│   │   ├── families/         # 族模板 + 架构 override（声明式 JSON，新增架构不改代码）
│   │   │   └── *.json        # llama/qwen2/gemma/mistral/deepseek-mla/gdn-hybrid/kimi/minimax/hy/…
│   │   ├── backend_transformers.py  # L1 兜底：transformers AutoModel 后端（显式 tier 标注）
│   │   ├── blocks/           # 共享积木（各积木独立单测）
│   │   │   ├── __init__.py
│   │   │   ├── attn_gqa.py   # GQA（mrope/partial-rotary/sliding-window/hybrid layer_types）
│   │   │   ├── attn_mla.py   # MLA（压缩 KV 布局，滑动窗口变体）
│   │   │   ├── attn_gdn.py   # Gated DeltaNet（Ornith）
│   │   │   ├── attn_kimi_linear.py  # Kimi-K3 线性注意力
│   │   │   ├── moe.py        # router + 专家调度 + 融合专家计算（numba）
│   │   │   ├── rope.py       # default/mrope/yarn/deepseek_yarn
│   │   │   ├── norm.py       # RMSNorm 族（含 q/k norm）
│   │   │   ├── mtp.py        # 通用 N 层 MTP proposer（专家零驻留）
│   │   │   └── heads.py      # 任务头：LM/embedding pooling/分类/reward
│   ├── tools/
│   │   └── sync_vllm_registry.py  # 解析本地 vllm/ registry → registry_table.json + tier 建议
│   ├── sampling.py           # 全参数采样 + 惩罚 + stop + 结构化输出掩码
│   ├── kv/
│   │   ├── __init__.py
│   │   ├── blocks.py         # L1 内存块池、块哈希、LRU、refcount
│   │   ├── disk.py           # L2 磁盘块文件、下沉/换入、readahead
│   │   └── coordinator.py    # 两级协调、watermark、抢占 swap
│   ├── experts/
│   │   ├── __init__.py
│   │   ├── index.py          # 专家清单扫描（shard 头解析）、index 缓存
│   │   ├── reader.py         # mmap ExpertReader、ring buffer、AVX2 预取核（numba 内联 asm）
│   │   └── pipeline.py       # 三层流水线状态机（C/P/S 阶段、队列、时序指标）
│   ├── weights/
│   │   ├── __init__.py
│   │   ├── manager.py        # 静态权重 mmap 视图、逐层张量表、page cache/流式模式判定（R10 auto 检测）
│   │   └── stream.py         # WeightRing（R10）：槽位状态机、sublayer 分片、带宽指标
│   ├── io_/                  # （包名 io 与 stdlib 冲突，沿用旧项目 io_ 约定）
│   │   ├── __init__.py
│   │   ├── safetensors_io.py # 轻量 safetensors 读取（mmap、按段）
│   │   └── windows_io.py     # ctypes Windows API：readahead 提示、大页/文件对齐
│   ├── resources/            # §3.7 资源限制系统（R11，v1.5，默认 50%）
│   │   ├── __init__.py
│   │   ├── budget.py         # 三资源预算推导（resource_pct→线程数/内存上限/IO 速率；优先于 [engine] 线程参数）
│   │   ├── limiter.py        # 令牌桶 IO 限速 + 线程预算落地（torch/numba 线程数、SetThreadPriority）
│   │   └── watchdog.py       # psutil 看门狗：采样/连续 3 次触发/五级自限/迟滞恢复/不丢请求
│   ├── tokenization.py       # tokenizer 封装、chat template、增量 detok
│   ├── metrics.py            # 指标总线（prometheus 文本格式）
│   ├── quant/                # §3.6 量化子系统（对齐 vLLM layers/quantization 结构）
│   │   ├── __init__.py
│   │   ├── spec.py           # QuantKey/ScaleDesc/GroupShape（声明式规格，移植 vLLM QuantKey）
│   │   ├── method.py         # QuantizeMethodBase/QuantizationConfig/get_quant_method 层分发
│   │   ├── registry.py       # QUANTIZATION_METHODS 注册表 + checkpoint quant_method 解析
│   │   ├── kernels.py        # numba 内核：FP8/INT8/MX 转换、group broadcast、融合 GEMM、VNNI
│   │   ├── fp8.py            # Fp8Method（W8A8/W8A16、per-tensor/channel/token/128-block）
│   │   ├── int8.py           # Int8Method（W8A8 VNNI / W8A16 weight-only）
│   │   ├── mx.py             # MxMethod（mxfp8/mxfp4/nvfp4：32 组 E8M0 scale）
│   │   ├── compressed_tensors.py  # Ornith checkpoint 格式：config_groups→spec + ignore 正则
│   │   ├── weight_only.py    # gptq/awq/bnbf dequant weight-only（int4/nf4→BF16 流式）
│   │   ├── online.py         # 在线量化：QuantizationConfigArgs（linear/moe/ignore）+ 加载期量化
│   │   └── kv.py             # KV cache 量化（fp8 per-token，L1/L2 容量减半）
│   └── vision/               # P7 可选：Ornith ViT + merger（注册点模式）
│       ├── __init__.py
│       └── ornith_vision.py
└── tests/
    ├── test_config.py        # 大小写不敏感、优先级、非法值
    ├── test_safetensors_io.py
    ├── test_expert_index.py
    ├── test_reader.py        # ring buffer、AVX2 路径开关、数值
    ├── test_pipeline.py      # 三层时序、overlap 指标
    ├── test_kv_blocks.py     # 块池、前缀哈希、LRU
    ├── test_kv_disk.py       # 下沉/换入/崩溃恢复
    ├── test_model_spec.py    # R8：test_config 7 族 + 4 标准家族 config→ModelSpec 黄金断言
    ├── test_registry_coverage.py  # R8：434 条账本对拍（层级归属完整/L2 有理由/条目数=vLLM 解析数）
    ├── test_model_spec_fuzz.py    # R8：L0 架构合成 config 批量冒烟（parse+构建+2 步前向无 NaN）
    ├── test_arch_smoke.py    # R8：重点架构小尺寸随机权重前向（KV 预算符合公式）
    ├── test_backend_l1.py    # R8：transformers 兜底层抽样冒烟（加载/生成/tier 标注）
    ├── test_blocks_gdn.py    # GDN 积木与 transformers 官方实现数值对齐（短序列）
    ├── test_blocks_attn.py   # GQA/MLA 积木与 transformers/vLLM 参考对齐
    ├── test_blocks_moe.py    # router top-k、专家输出对齐
    ├── test_quant_kernels.py # 量化：各 QuantKey 往返误差 / group broadcast / 与 torch 对拍 / VNNI 路径
    ├── test_quant_registry.py# 量化：compressed-tensors config→逐层 QuantSpec 黄金断言（7 条 ignore 命中层走 BF16）
    ├── test_quant_e2e.py     # 量化：Ornith FP8 主路径 / 在线 INT8 对齐≥99% / KV-fp8 容量减半
    ├── test_resources.py     # R11：线程预算实测 / 令牌桶速率 / 看门狗五级升级 / 迟滞恢复 / 覆盖优先级
    ├── test_mtp.py           # propose/verify 接受逻辑
    ├── test_sampling.py      # 各参数分布正确性（含 seed 可复现）
    ├── test_zero_residency.py# R2 验收：RSS 与专家参数量解耦
    ├── test_weight_streaming.py # R10 验收：dense>内存 auto 切换 / RSS≤ring 预算 / 数值逐 token 一致
    ├── test_overlap.py       # R3 验收：overlap_ratio 指标
    └── test_e2e.py           # 端到端：短 prompt 生成 + API 冒烟
```

约定：
- 所有第三方导入一律 `from x import y`（R6）；numba 内联汇编仅出现在 `experts/reader.py` 与 `quant.py`，集中管理并带能力探测回退。
- `ccut` = CostCut 缩写包名；不引入需要编译的依赖（无 C 扩展、无 CUDA）。
- 旧 `python/` 文件已全部删除（git status 可见），本树为全新实现。

---

## 6. 参数体系（R5：全可调、CLI/toml、大小写不敏感）

**三源合并，优先级 CLI > toml > 环境变量（`CCUT_` 前缀）> 内置默认**；所有键在合并前统一 `casefold()`，即 `--Temperature=0.95`、`--temperature==0.95`（宽松解析把第二个 `=` 视为值的一部分剥掉）、toml `[sampling] Temperature = 0.95` 等价。CLI 支持 `--key=value` 与 `--key value` 两种形式（用 `parse_known_args` + 自定义归一化，不用 argparse 的类型强转，全部走 schema 声明的类型与取值域校验，非法值启动即报错并列出全部合法键——`--list-params` 输出完整参数表）。

分节参数表（默认值；`★`=三大核心机制直接相关）：

**[model]**
| 参数 | 默认 | 说明 |
|---|---|---|
| model_path | ./models/Ornith-1.5-35B-A3B-MTP-FP8 | 权重目录 |
| dtype | auto | auto 从 config 探测 |
| max_model_len | 32768 | 最大上下文（上限 262144） |
| arch_tier | auto | R8：auto=L0 优先、缺则 L1 兜底；strict=仅 L0（L1 架构直接报错） |
| enable_vision | false | Ornith ViT 路径（P7，CPU 慢） |
| trust_remote_code | false | — |

**[engine]**
| 参数 | 默认 | 说明 |
|---|---|---|
| max_num_seqs | 8 | 并发请求上限 |
| max_num_batched_tokens | 8192 | 每步 token 预算（prefill+decode 共享） |
| chunked_prefill_size | 4096 | prefill 分块 |
| enable_prefix_caching | true | 前缀块复用 |
| scheduler_delay_factor | 0.0 | 参考 vLLM 语义 |
| num_scheduler_threads / num_worker_threads | 1 / 4 | 调度与算子线程 |
| tensor_parallel_size 等 | 1 | CPU 单机版：≠1 显式报错（§4 ⬜ 项） |
| log_level | INFO | DEBUG/INFO/WARNING/ERROR |
| seed | null | 全局随机种子 |
| mem_budget_gb | auto(2.5) | R9：进程私有内存预算；auto=2.5GB，启动时反推 L1/激活规模并打印预算表；MemAvailable < 1.3×预算 自动收缩 |

**[kv_cache]（需求 1）**
| 参数 | 默认 | 说明 |
|---|---|---|
| kv_l1_bytes | 1073741824 (1GB) | L1 内存块池大小（R9 默认 1GB；范围 512MB~4GB；disk_first 模式 = 512MB 热窗） |
| kv_l2_dir | ./.kv_cache（相对 python/） | L2 磁盘目录 |
| kv_l2_max_bytes | 68719476736 (64GB) | L2 上限（可远超 RAM） |
| kv_policy | hybrid | hybrid / **disk_first**（RAM 极省：L1 仅热窗）/ memory_first |
| kv_block_size | 16 | token/块 |
| kv_evict_high_water / low_water | 0.8 / 0.6 | L1 水位触发下沉与回收 |
| kv_hot_window_steps | 16 | 冷判定窗口（步） |
| kv_l2_compression | none | none / lz4（若装） |
| kv_cache_ttl_seconds | 0 | 0=不过期；会话级 .kvdb 清理 |

**[experts]（需求 2）**
| 参数 | 默认 | 说明 |
|---|---|---|
| expert_residency | **zero** | zero（零驻留，本引擎默认）/ page_cache_only（纯靠 OS 页缓存，无显式预取） |
| expert_ring_slots | 2 | 每层 ring buffer 槽位数（×top_k×单专家字节） |
| expert_index_cache | ./.kv_cache/expert_index.json | 专家清单缓存 |
| expert_verify_crc | false | 读后抽样 CRC 校验（调试用） |

**[weights]（R10 dense 层流式，v1.3）**
| 参数 | 默认 | 说明 |
|---|---|---|
| weight_streaming | auto | auto（§3.5 阈值 auto 切换）/ on（强制流式）/ off（强制 page cache 模式，内存不足时启动告警） |
| weight_ring_layers | 2 | WeightRing 槽位数（流式模式私有内存主项；1 更省内存） |
| weight_stream_chunk | auto | auto=单层 > 槽位 50% 自动 sublayer；layer / sublayer 强制 |
| weight_stream_bandwidth_log | true | 每步打印静态权重读取带宽 / 预计 token/s |

**[quant]（量化子系统，§3.6，v1.4）**
| 参数 | 默认 | 说明 |
|---|---|---|
| quantization | auto | auto=按 checkpoint `quantization_config` 自动；可指定在线简写（`fp8_per_token` / `int8_per_channel_weight_only` / `mxfp8` / `nvfp4_per_token`，对齐 vLLM `_ONLINE_SHORTHANDS` 子集，全大小写不敏感）；与 checkpoint 自带量化互斥（同设报错） |
| quant_ignore | [] | 在线量化时的 ignore 层名正则列表（对齐 vLLM `QuantizationConfigArgs.ignore`） |
| fp8_compute_mode | w8a16 | w8a16（默认，dequant 后 BF16 matmul，精确）/ w8a8（激活动态量化路径，CPU 对照基准） |
| kv_cache_dtype | auto | auto（checkpoint `kv_cache_scheme`，Ornith→BF16）/ bf16 / fp8（KV 容量减半，§3.6-4） |

**[resources]（资源限制系统，§3.7，v1.5，默认 50%）**
| 参数 | 默认 | 说明 |
|---|---|---|
| resource_pct | **50** | 全局资源限制比例（%）：CPU/内存/IO 同时生效；被下面三个分资源项覆盖（分资源默认 auto=全局值） |
| resource_cpu_pct | auto | auto=resource_pct；CPU 线程预算 = pct × 逻辑核（scheduler/prefetch/speculative/计算 分配，优先于 [engine] 线程参数）+ 线程优先级 BELOW_NORMAL 降档 |
| resource_mem_pct | auto | auto=resource_pct；内存私有 RSS 硬上限 = pct × 物理内存（看门狗超限五级自限） |
| resource_io_pct | auto | auto=resource_pct；磁盘 IO 带宽限 = pct × 实测盘速（令牌桶，门控 ExpertReader/WeightRing/KV-L2 读写） |
| resource_monitor_interval | 1.0 | 看门狗采样间隔（秒） |
| resource_throttle | auto | auto（超限自限）/ warn（仅告警不自限）/ off（仅打印预算表） |

**[pipeline]（需求 3）**
| 参数 | 默认 | 说明 |
|---|---|---|
| pipeline_depth | 3 | 三层流水线：计算 N ‖ 预取 N+1 ‖ 投机 N+2 |
| prefetch_layers_ahead | 2 | 投机预取提前层数 |
| prefetch_mode | auto | auto / avx2 / off（AVX2 预取指令开关，能力探测兜底） |
| expert_readahead_mb | 128 | 顺序读 readahead 提示量 |
| speculative_route_history | 4 | 投机路由历史窗口（步） |
| pipeline_metrics | true | 输出 overlap_ratio 等时序指标 |

**[sampling]（对齐 vLLM SamplingParams 全集，默认取模型 generation_config）**
| 参数 | 默认 | 参数 | 默认 |
|---|---|---|---|
| temperature | 1.0 | min_p | 0.0 |
| top_p | 0.95 | typical_p | 1.0 |
| top_k | 20 | repetition_penalty | 1.0 |
| presence_penalty | 0.0 | frequency_penalty | 0.0 |
| length_penalty | 1.0 | early_stopping | false |
| n | 1 | best_of | 1 |
| max_tokens | null(=max_model_len-输入) | max_completion_tokens | null |
| stop | [] | stop_token_ids | [] |
| seed | null | ignore_eos | false |
| logprobs | null | prompt_logprobs | null |
| guided_json / guided_regex / guided_grammar | null | detokenize | true |

**[spec_decode]（MTP）**
| 参数 | 默认 | 说明 |
|---|---|---|
| enable_mtp | true | 用模型自带 1 层 MTP |
| mtp_draft_tokens | 1 | 每步草稿 token 数 |
| enable_ngram | false | ngram 备选 proposer |
| ngram_window | 8 | ngram 窗口 |

**[api]**（api_server.py 用）
| 参数 | 默认 | 说明 |
|---|---|---|
| host / port | 0.0.0.0 / 8000 | 监听 |
| api_key | null | 可选鉴权 |
| sse_heartbeat_seconds | 15 | 流式心跳 |
| cors | true | — |

> `--list-params` 输出此全表（运行时动态生成，单一事实来源 = `ccut/config.py` 的 schema，避免文档漂移）。

---

## 7. 阶段计划（P0–P8，含 P1.5，详细到可执行步骤）

> 每阶段定义「出口准则」，达成才进入下一阶段。进度实时写入 `docs/python/进度日志.md`（模板见 §10）。

### P0 地基（配置/平台/IO 底座/架构账本）— 预计 0.5 天
1. `ccut/config.py`：schema 声明（§6 全参数）+ 三源合并 + casefold + 校验 + `--list-params`。
2. `ccut/platforms.py`：AVX2/AVX512/VNNI 探测（cpuid）、内存/盘速基准（4KB 随机 + 1MB 顺序读 64MB 样本）、启动报告打印。
3. `ccut/io_/safetensors_io.py`：mmap 读 safetensors（只解析头、按段读、零拷贝视图）。
4. `ccut/experts/index.py`：专家清单扫描（17 个 shard 头）→ `(layer, expert) → (shard, offset, len)` + 落盘缓存；校验 62565 张量计数与 manifest 一致。
5. **`ccut/tools/sync_vllm_registry.py`**：解析本地 `vllm/model_executor/models/registry.py` → 生成 `ccut/models/registry_table.json`（434 条：架构名 → tier 建议(L0/L1/L2) + 族猜测 + 理由）；`CostCut-Infer.py --list-architectures` 输出全表。`test_registry_coverage.py` 首版（条目数对拍 + 层级完整性）。
6. **`ccut/resources/budget.py` 首版（R11）**：解析 `resource_pct`（默认 50）→ 三资源预算表（线程数 = pct×逻辑核 / 内存上限 = pct×物理内存 / IO 速率 = pct×盘速），`--info` 打印完整预算表与推导过程；`test_resources.py` 首版（预算推导 + 覆盖优先级 + 大小写）。
7. `CostCut-Infer.py` 骨架：`--list-params`、`--list-architectures`、`--info`、`--version` 四个命令可用。
8. 测试：`test_config.py`（大小写/优先级/非法值）、`test_safetensors_io.py`（与 safetensors 库读数一致）、`test_expert_index.py`。
- **出口准则**：四条命令跑通；`test_*` 全绿；专家清单覆盖 40×256 层×专家无遗漏；registry_table.json 434 条与 vLLM 解析数一致；`--info` 预算表正确（8 核 @50% → 4 线程）。

### P1 模型数值正确（Ornith 单请求、无流水线、专家同步读）— 预计 2 天
1. `ccut/quant/` 量化子系统（§3.6）：`spec.py`（QuantKey/ScaleDesc/GroupShape）+ `method.py`（QuantizeMethodBase/QuantizationConfig/`get_quant_method` 层分发）+ `registry.py`（QUANTIZATION_METHODS + checkpoint quant_method 解析）+ `kernels.py`（FP8 E4M3↔BF16 SIMD 转换 + 融合 GEMM，`numba` `prange`；CPU 无原生 FP8 dot → **默认 W8A16 精确反量化**）+ `compressed_tensors.py`（Ornith 格式：config_groups→QuantSpec + 7 条 ignore 正则）；`test_quant_registry.py` 首版（40 层逐层 QuantSpec 黄金断言）+ `test_quant_kernels.py`（与 torch matmul 参考值对拍）。
2. `ccut/models/spec.py` + `registry.py` 骨架 + `blocks/`：`norm.py`、`rope.py`（default/mrope）、`attn_gqa.py`（16q/2kv、head_dim 256、**mrope** partial_rotary 0.25、q/k norm、attn_output_gate；KV 直接写内存数组，L1 块池接口预留）、`attn_gdn.py`（GatedDeltaNet：in_proj 拆分 q/k/v/z/b/a、causal conv1d(k=4)、gated delta rule 递归更新、A_log/dt_bias、out proj、norm）——**关键难点**：与 `transformers/models/qwen3_5_moe` 官方实现短序列（seq≤64）逐层对齐，容差 1e-2（BF16）、`moe.py`（256 专家 softmax top-8 + 共享专家 + `shared_expert_gate` 标量门控 + 专家融合计算 `silu(g)·u → dequant → down`；此阶段专家**同步读**，验证数值用）。
3. `models/generic.py` 通用组装器 + `families/gdn-hybrid.json`（Ornith 族模板）：spec → 40 层堆栈 + embed/lm_head **mmap 逐行流式**（§2.1）+ mrope 位置；**Ornith 即走通用路径**（主测不享特权代码）；`ccut/tokenization.py` 接入 chat_template.jinja。
4. 数值基准：固定 prompt + temperature=0，与 transformers 官方 `Qwen3_5MoeTextModel`（BF16 反量化后参考）输出 logits 对齐（top-1 token 一致序列 ≥ 前 32 token）；记录每层余弦相似度表。
- **出口准则**：单 prompt 生成 128 token，与参考实现 token 级一致（或日志记录逐层相似度 ≥ 0.99 并解释残差来源）；**psutil 私有 RSS ≤ 2.5GB**（R9，embed/lm_head 流式生效后）。

### P1.5 架构全覆盖（R8：434 条逐层落实）— 预计 2.5 天
1. `models/spec.py` 完成 config 归一全变体（`text_config` 嵌套 / rope 变体 / quant 变体 / hybrid layer_types / 多模态 config 剥离）；`test_model_spec.py`：test_config 7 族 + llama/qwen2/gemma/mistral 代表共 ~11 个黄金断言。
2. `models/families/*.json`：llama / qwen2 / gemma / mistral / deepseek-mla / gdn-hybrid / kimi / minimax / hy 等族模板（覆盖 62 个标准 decoder 文件所属家族）+ 张量名映射模板；**registry_table 中 L0 条目逐一挂到某族或标记升级理由**。
3. `blocks/attn_mla.py`：MLA 积木（q_lora 分解、压缩 KV `(kv_lora_rank+qk_rope_head_dim)` 布局、decode/prefill 两路、滑动窗口变体）——参考 vLLM `deepseek_v2.py`/`deepseek_v3.py` 数据流 + transformers `deepseek_v4` 数值；`blocks/attn_kimi_linear.py`：Kimi-K3 线性注意力（参考 transformers `kimi_k25`）；rope 补 yarn/deepseek_yarn；`blocks/heads.py` 任务头。
4. `test_model_spec_fuzz.py`：L0 架构批量「合成 config 随机化参数 → parse → 小尺寸随机权重构建 → 2 步前向无 NaN/形状/KV 预算=公式」；`test_arch_smoke.py`：7 族重点架构专项（KV 字节/token 实测=公式）。
5. `models/backend_transformers.py` L1 兜底层 + `--arch-tier=strict`；`test_backend_l1.py` 抽样 3~5 架构冒烟；`--list-architectures` 全表含 tier 与理由。
6. `test_registry_coverage.py` 完整版：**434 条每条恰归一层、L2 均有理由、条目数=vLLM 解析数**；未注册架构（registry 外）走 L1 探测或显式报错。
- **出口准则**：`test_registry_coverage` + fuzz + smoke 全绿；MLA/Kimi 积木与 transformers 对应实现短序列 logits 对齐（同 P1 容差）；`--list-architectures` 434 条可查询、零静默未处理；MLA KV 预算实测=公式值（进 `--info`）。

### P2 两级 KV Cache（需求 1）— 预计 1.5 天
1. `ccut/kv/blocks.py`：L1 arena 块池、块哈希链（前缀复用）、LRU、refcount。
2. `ccut/kv/disk.py`：L2 块文件（5MB 槽 + 头）、下沉/换入、文件头崩溃截断恢复、readahead 预取。
3. `ccut/kv/coordinator.py`：watermark 调度、`kv_policy` 三模式、抢占 swap 接口（GDN 态快照随块）。
4. 接入 attention：KV 读写全部走块池。
5. 测试：`test_kv_blocks.py`、`test_kv_disk.py`（含 kill -9 模拟崩溃后恢复）；`test_zero_residency.py` 骨架先立。
- **出口准则**：`max_model_len=131072` + `kv_l1_bytes=1GB` + `kv_l2_max_bytes=8GB` 连续生成 4096 token 不 OOM、无正确性回退；`kv_policy=disk_first` 下 L1 实际占用 ≤ 512MB（psutil 验证）；前缀缓存命中时 TTFT 下降 ≥ 50%（bench 脚本输出）。

### P3 引擎编排与调度 — 预计 1.5 天
1. `ccut/engine.py`：请求状态机（waiting/running/finished/aborted）、continuous batching、chunked prefill + decode 混排、抢占/恢复。
2. `ccut/sampling.py` 全参数实现 + seed 可复现 + logprobs + stop。
3. `ccut/metrics.py`：指标总线（prometheus 文本）+ 请求级 request_id + 分级日志。
4. `sdk.py`：`CostCutInfer.from_pretrained/generate/chat`（对齐 vLLM LLM 类语义，含 batch 输入）。
5. `tui_chat.py`：Rich 流式聊天 + `/help /metrics /stop /reload`。
6. **R11 运行期组件**：`resources/limiter.py`（IO 令牌桶门控 ExpertReader/WeightRing/KV-L2 读写 + torch/numba 线程数落地 + `SetThreadPriority` 降档）、`resources/watchdog.py`（psutil 采样 + 连续 3 次超限五级自限 + 迟滞恢复 + 背压接调度器）；`test_resources.py` 完整版（线程预算实测 ≤ 4 @50% / 令牌桶速率 ±15% / 看门狗升级有序 / 迟滞回退 / 覆盖优先级）。
7. 测试：`test_sampling.py`（分布正确性 + 可复现 + 惩罚单调性）、`test_e2e.py`（CLI 短生成）。
- **出口准则**：`CostCut-Infer.py --prompt "..." --max-tokens 64 --temperature 0.7` 出流式结果；并发 8 请求 batch 正确（各请求输出互不串扰，用不同 seed 断言）；TUI 可用；**R11 运行期验收全绿（线程 ≤4、IO 速率 ≤50% 盘速、看门狗不丢请求）**。

### P4 零驻留专家流 + dense 层流式（需求 2 + R10）— 预计 2 天
1. `ccut/experts/reader.py`：mmap ExpertReader + ring buffer 协议 + 读入即反量化（BF16 ring）+ 槽位回收。
2. MTP 专家融合张量切读（gate_up_proj 按专家行段）。
3. `expert_residency=zero/page_cache_only` 双模式 + `expert_verify_crc`。
4. 接入 moe.py：计算只读 ring buffer；`--profile-memory` 逐层驻留清单。
5. **R10 层流式**：`ccut/weights/manager.py`（静态权重清单 + auto 模式判定：static_weight_bytes vs 内存余量 50% 阈值）+ `ccut/weights/stream.py`（WeightRing 槽位状态机 + sublayer 分片 + 带宽指标）；流水线 Stage P/S 扩展为「专家 + 层权重」双预取（§3.5）；generic 组装器静态权重读取全走 manager（page cache/流式两路同接口，数值等价）。
6. 测试：`test_reader.py`（数值 = 同步读路径）、`test_zero_residency.py` **正式验收**（256 vs 16 专家 monkeypatch 对照，RSS 差 < 5%）、`test_weight_streaming.py` **R10 验收**（合成 dense 模型权重 > 可用内存：auto 切换正确 / 10 步 RSS ≤ ring 预算+10% / 数值与 page cache 模式逐 token 一致 / sublayer 路径 / off 告警）。
- **出口准则**：200 步 decode RSS 曲线稳定（不随步数增长）；零驻留验收通过并出对照图数据（写入进度日志）；性能不回退 > 10%（vs P1 同步读，差值应被页缓存掩盖）；R10 五项断言全绿；Ornith（静态 2.3GB）保持 page cache 模式且行为不变。

### P5 MTP 投机解码 — 预计 1 天
1. `ccut/mtp.py`：1 层 MTP（pre_fc_norm ×2 → fc(2D→D) → MTP decoder 层（full attn，共享主模型 KV 语义）→ 1 token）；verify：主模型对 [actual, draft] 双 token 单步 forward + 链式接受。
2. ngram proposer（备选，`enable_ngram`）。
3. 接受率指标 `spec_accept_rate`、`avg_accept_len` 进指标总线。
4. 测试：`test_mtp.py`（draft 正确时必接受 / 错误时拒绝 / 关闭时行为等同 P4）。
- **出口准则**：`enable_mtp=true` 时 TPOT 下降 ≥ 25%（接受率 > 0.3 前提下；若接受率低则记录原因——MTP 头为 BF16 独立训练，接受率是模型属性，如实记录不硬凑）；数值正确性不回退。

### P6 API 服务与受控输出 — 预计 1.5 天
1. `api_server.py`：/v1/chat/completions、/v1/completions、/v1/models、/metrics、SSE 流式、全量 inference parameters（对齐 vLLM 请求 schema）、tool-call 文本解析（Qwen 格式 `{"name":...,"arguments":...}`）。
2. 受控输出：`outlines_core` grammar → logits mask（每步应用），guided_json/regex/grammar 三入口。
3. embeddings 端点（400 不支持，语义对齐）。
4. 测试：`test_e2e.py` 扩展（httpx 打全端点、流式分片完整性、schema 违规为 0）。
- **出口准则**：`curl` 复现 vLLM 标准调用姿势（chat 流式 / completions / logprobs / json mode）全部 200 且输出合法；json mode 下 100 次采样 100% 合法 JSON。

### P7 多模态（可选降级项）— 预计 1 天
1. `ccut/vision/ornith_vision.py`：27 层 ViT + patch_embed + merger（BF16，numba 加速 matmul）；图像预处理（transformers processor 配置驱动）。
2. 图像 token 替换 + mrope 三维位置（section 11/11/10 的图像块展开）。
3. `--enable-vision` 开关 + chat 接口 image_url（base64/file）。
- **出口准则**：一张 512×512 图能进模型并生成描述（CPU 慢属预期，记录耗时）；`enable_vision=false` 时零开销。若时间不足，降级为「注册点 + 文档 + 单测桩」，进度日志如实标注。

### P8 基准、量化扩展、资源基准、文档、收尾 — 预计 2 天
1. `benchmarks/`：吞吐/TTFT/TPOT 曲线（序列长 1K/4K/16K/32K）、IO 掩盖率专项、零驻留专项、前缀缓存收益、**资源限制专项**（R11）：`--resource-pct 50` vs `100` 吞吐/延迟对照 + 并行后台压测（CPU 燃烧 + 盘读）实测「系统剩余余量」，验证 50% 确实留出 50%。
2. **量化扩展（§3.6 第 2~4 项）**：`int8.py` INT8 W8A8 VNNI（`vpdpbusd`，本机 AVX512-VNNI 真加速）+ `mx.py`（mxfp8/mxfp4/nvfp4 dequant 内核）+ `weight_only.py`（GPTQ/AWQ/bnbf int4/nf4 流式 dequant）+ `online.py` 在线量化（`--quantization fp8_per_token / int8_per_channel_weight_only / mxfp8`，BF16 checkpoint 加载期量化，与 checkpoint 量化互斥校验）+ `kv.py` KV-fp8（L1/L2 容量减半，`kv_cache_dtype` 接入 §3.1 块大小公式）；`test_quant_e2e.py` 全绿（在线 INT8 32 token 对齐 ≥ 99%、KV-fp8 字节减半断言）。
3. `README.md`（python/ 重写说明 + 三大机制 + 量化设计 + 资源限制设计 + 参数全表 + 基准数字）；更新根 README 与 `docs/python/进度日志.md` 收尾总结。
4. 全量测试回归 + `code_review` 工具走一遍工作区 diff。
5. git 提交（用户确认时机）。
- **出口准则**：全部测试绿；基准数字入档（含 W8A16 vs W8A8 vs INT8-VNNI 三条量化路径的速度/精度对照表 + 50% vs 100% 资源限制对照表与「留出余量」实测）；文档与代码参数表一致（`--list-params` 对拍）。

**总计约 15.5 个工作日（单人：P0 0.5 + P1 2 + P1.5 2.5 + P2 1.5 + P3 1.5 + P4 2 + P5 1 + P6 1.5 + P7 1 + P8 2）。** 阶段间允许 0.5 天弹性；任何阶段阻塞 > 2 轮排查即在进度日志「困难」节记录并给出 A/B 方案。

---

## 8. 验收清单（对应需求编号）

| 需求 | 验证命令/脚本 | 通过标准 |
|---|---|---|
| R1 磁盘优先 KV | `python CostCut-Infer.py --kv-policy=disk_first --kv-l1-bytes=536870912 --max-model-len=131072 ...` + `benchmarks/bench_kv.py` | L1 psutil 实测 ≤ 配置；L2 命中/下沉指标正常；崩溃恢复单测绿 |
| R2 零驻留专家 | `python -m tests.test_zero_residency` | 256 vs 16 专家 RSS 差 < 5%；`--profile-memory` 显示专家驻留 = ring buffer 量 |
| R3 三层流水线 | `benchmarks/bench_overlap.py` | 预热 50 步后 `overlap_ratio` ≥ 0.8（盘速允许时）；`prefetch_mode=avx2` 与 `off` 对比有可测差异 |
| R4 全功能 | §4 矩阵逐项 + `test_e2e.py` | ✅ 项全绿；🟡 项降级行为有单测；⬜ 项参数报错文案有单测 |
| R5 参数 | `test_config.py` + `--list-params` 对拍 | 大小写变体、`==` 宽松写法、三源优先级全通过 |
| R6 全重写 | `git status` | python/ 下全部新文件；`grep -rE "^import (torch\|numpy\|...)$" python/` 为 0 命中（第三方均 from-import） |
| R7 实时文档 | `docs/python/进度日志.md` | 每阶段 ≥ 1 条进度、每困难 ≥ 1 条记录（含时间戳与处置） |
| R8 架构全覆盖 | `test_registry_coverage.py` + `test_model_spec_fuzz.py` + `test_arch_smoke.py` + `test_backend_l1.py` + `--list-architectures` | vLLM registry **434 条**每条恰归一层（L0/L1/L2）且可查询；L0 重点 ~11 架构黄金断言+冒烟全绿；MLA/GDN/Kimi 积木对拍通过；L1 抽样冒烟通过；L2 报错含原因；**零静默未处理** |
| R9 内存最小化 | `--profile-memory` + `tests/test_zero_residency.py`（内存节） | 200 步 decode 稳态**私有 RSS ≤ 2.5GB**（psutil 实测，含预算表打印）；`mem_budget_gb` 收缩行为单测通过 |
| R10 dense 层流式 | `tests/test_weight_streaming.py` | 合成 dense（权重 > 可用内存）auto 切换正确；10 步 RSS ≤ ring 预算+10%；**数值与 page cache 模式逐 token 一致**；sublayer 分片路径通过；`weight_streaming=off` 内存不足告警可见；带宽指标输出 |
| R11 资源限制 | `tests/test_resources.py` + `benchmarks/bench_resources.py` | 默认 50% 下：线程 ≤ 4（@8 核）；IO 实测速率 ≤ 0.55×盘速；看门狗五级升级有序 + 迟滞回退 + **在途请求不丢**；50% vs 100% 基准对照 + 后台压测「留出余量」实测 |

---

## 9. 风险登记（开发中动态更新）

| # | 风险 | 概率 | 影响 | 缓解 |
|---|---|---|---|---|
| K1 | GDN（GatedDeltaNet）数值对齐困难（官方实现用 fla kernel，CPU 无对应） | 中 | 高（P1 阻塞） | 以 transformers 纯 PyTorch 参考路径为准（qwen3_next 有 fallback）；chunk 前向改递归前向（decode 场景天然递归）；prefill 长序列性能另调 |
| K2 | 盘速不足导致 overlap < 80%（i7 平台可能机械盘/低速 SSD） | 中 | 中 | §3.3 回退阶梯；page_cache_only 模式；如实记录基准数字 |
| K3 | RAM 11.7GB 紧张（页缓存与私有预算争抢） | 中 | 中 | v1.1 已重构：embed/lm_head 不再常驻（mmap 逐行流式，§2.1），私有预算 ≈2.0GB（L1 默认 1GB 可缩 512MB）；`mem_budget_gb` 启动自检收缩；OOM 预案：激活分块 + L1 降到 512MB |
| K4 | numba 内联汇编 AVX2 在 3.14/LLVM 下的兼容性 | 低 | 中 | 能力探测 + `prefetch_mode=off` 纯 numpy 路径保底；asm 代码独立函数隔离 |
| K5 | MTP 接受率低导致投机解码负收益 | 中 | 低 | 接受率 < 0.2 时引擎自动建议关闭（日志提示）；`enable_mtp=false` 一键回退 |
| K6 | 35B 模型 CPU 生成速度过慢被误判为 bug | 高 | 低 | P1 即记录实测 token/s 基线入档；进度日志明示「CPU 推理 35B 预期个位数 token/s」，验收看正确性+机制指标，不看绝对速度 |
| K7 | vLLM「全部功能」边界争议（多卡/LoRA 等） | 低 | 中 | §4 矩阵已显式分级；⬜ 项保留参数接口+明确报错；用户可在 P3 前调整矩阵 |
| K8 | MLA/线性注意力积木数值对齐困难（R8 风险） | 中 | 中（P1.5 阻塞） | MLA 以 transformers `deepseek_v4` + vLLM `deepseek_v2/v3` 双参考对拍；Kimi 线性注意力以 `kimi_k25` 为基准；每积木独立单测，失败不拖累 Ornith 主路径（积木级隔离） |
| K9 | 434 架构中族模板覆盖不足（个别架构 config 变体超出 spec v2 表达力） | 中 | 中 | 这类架构自动落 L1（transformers 兜底，功能可用）而非阻塞——tier 判定在 P0 同步工具中机械化完成，P1.5 只处理「可升 L0」的；L2 仅留给 vLLM 已移除 + transformers 无实现者（有理由字段） |
| K10 | L1 兜底层依赖 transformers 5.15 对长尾架构的建模覆盖（v1.2 引入） | 中 | 低 | L1 加载失败 → 显式报错 + 指引（升级 transformers / 联系提供 spec）；L1 架构不承诺三大机制，文档与启动报告均标注 |
| K11 | R10 层流式在极低带宽盘上完全不可用（dense 大模型 decode 每步读全盘权重） | 中 | 中（仅 dense 大模型场景） | 诚实定位「内存可行性优先」：启动报告打印实测盘速 → 预计 token/s，低于阈值（<0.05 tok/s 估）时强烈建议 off/换盘/量化更小权重；不影响 Ornith 主测（静态仅 2.3GB，通常不触发流式） |
| K12 | R11 资源限制在 Windows 上非内核级强制（合作式自治，极端情况可能短暂超限） | 中 | 中 | 诚实标注：引擎保证「不故意超限」（线程硬上限 + 令牌桶 + 看门狗自限），页错误风暴/内核缓冲部分不可控；文档与 `--info` 均明示限制强度分级（强/中）；若用户需要硬隔离，指引 Docker Desktop WSL2 cgroup / 任务管理器作业对象（Job Object）外部包裹 |
| K13 | 资源限速与 R3 流水线掩盖率目标冲突（IO 被限到 50% 盘速，冷读 overlap 可能不达标） | 中 | 中 | 两者是「快 vs 省」的显式权衡，不互相破坏：overlap 目标在 50% 限制下按 0.5×盘速重新核算（§3.3 回退阶梯仍适用）；`--info` 启动报告同时打印两者（限制后盘速 → 预计 token/s），用户可用 `resource_io_pct=100` 换取速度 |

---

## 10. 进度日志规范（`docs/python/进度日志.md`）

- 结构：`## 当前状态`（一句话+阶段+完成度）→ `## 阶段进度`（P0–P8 表：状态/完成度/关键产出）→ `## 实时日志`（时间戳倒序：做了什么/结果/下一步）→ `## 困难与决策`（编号 D1…：现象/根因/方案A/B/选定/结果）。
- 更新时机：每完成一个文件模块、每跑通一个测试、每遇阻塞、每做设计变更 → **即时**追加（不等阶段结束）。
- 诚实原则：数字（速度/命中率/内存）必须实测后写入，估计值标注 `(估)`。

---
