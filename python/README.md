# CostCut Infer（Python 版）

> MoE 大模型推理引擎的 Python 实现（技术探索 / 新功能首发版本——发布版为 Rust）。
> 支持 **CPU / GPU（CUDA）/ NPU（昇腾）/ APU（AMD ROCm）**（`engine.toml [device].kind`——`""` 自动检测）。

## 统一入口（main.py）

所有功能通过 `python main.py <子命令>` 进入：

| 子命令 | 功能 | 等价入口 |
|---|---|---|
| `chat` | CLI 对话 | cli_chat.py |
| `tui` | TUI 对话（textual） | tui_chat.py |
| `api` | OpenAI 兼容 API + Web 管理页 | api_server.py |

```bash
python main.py chat                # CLI 对话（首次输入触发模型构建）
python main.py tui                 # TUI 对话
python main.py api --port 8000     # OpenAI 兼容 API（http://127.0.0.1:8000——/v1 + /docs + / 管理页）
```

## 架构（liteengine 分层——镜像 Rust 结构）

```
python/
  main.py                 # 统一入口（chat / tui / api）
  cli_chat.py             # CLI 对话（ChatSession）
  tui_chat.py             # TUI 对话（textual）
  api_server.py           # OpenAI 兼容 API 服务（参考 vLLM——/v1/models + /v1/chat/completions）
  engine.toml             # 配置（模型 / 设备 / 采样 / 优化开关）
  liteengine/
    core/                 # 核心原语：norm（RMSNorm）、rope（RoPE/YaRN）
    engine/               # 推理引擎：model、layer、attention、moe、mtp、speculator、
                          #   cache（KV 缓存）、sampling、registry（插件注册）、engine（外壳）
    io/                   # 加载：loader（safetensors 惰性读取）、gguf
    quant/                # 量化：awq、gptq、fp8、nvfp4、dequantize、unpack
    config.py             # 配置解析（EngineConfig / DeviceConfig——设备自动检测）
    model_config.py       # 模型配置归一化（架构探测）
    lazy_loader.py        # 惰性导入
  tests/                  # 回归测试（test_config / test_moe_arch / test_quant 等）
```

- **分层原则**：`core`（原语）→ `engine`（推理）→ `io`（加载）→ `quant`（量化）——与 Rust 的 `core/engine/io/quant` 一一对应
- **注册表**（`engine/registry.py`）：注意力 / MoE 格式 / 量化方法 / 架构归一化四注册点即插即用

## 设备支持

`engine.toml [device]`：

```toml
[device]
kind = ""        # "" 自动检测（CUDA→NPU→ROCm→CPU）；或显式 cpu/gpu/npu/apu
threads = 0      # 0 = 自动
fp16 = false     # GPU/NPU/APU 推荐 true
```

## 功能

- **OpenAI 兼容 API**（参考 vLLM）：`/v1/models` + `/v1/chat/completions` + `/v1/health` + `/docs`（Swagger）+ `/`（Web 管理页）——OpenAI SDK / curl / LangChain 兼容（`base_url` 指向即可）
- **投机解码**（dspark——KV 缓存续接）、**GGUF**、**多量化**（AWQ/GPTQ/FP8/NVFP4）、**完整 YaRN**（NTK-by-parts + 温度缩放）

## 依赖

`torch`（必需）+ `fastapi` / `uvicorn`（API）/ `textual`（TUI）——`pip install fastapi uvicorn textual`。

详细指南：`../docs/total/OpenAI-API与TUI指南.md`（API/TUI 用法）、`../docs/total/性能测试汇总.md`（性能数据）。
