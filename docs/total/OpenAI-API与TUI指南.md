# OpenAI 兼容 API 与 TUI 指南（Python 版）

> 日期：2026-08-22 ｜ 参考：vLLM 的 OpenAI 兼容接口 ｜ 库：fastapi + uvicorn + textual（pip 已装）

## 1. OpenAI 兼容 API 服务

参考 vLLM 的 `/v1` 接口（OpenAI SDK 兼容），Python 版新增 `python/api_server.py`：

```bash
python api_server.py [--host 0.0.0.0] [--port 8000] [--model <name>]
```

### 接口清单

| 接口 | 说明 | 返回格式（参考 vLLM/OpenAI） |
|---|---|---|
| `GET /v1/models` | 模型列表 | `{object: list, data: [{id, object, created, owned_by}]}` |
| `POST /v1/chat/completions` | 对话补全 | `{id, object: chat.completion, choices: [{message: {role, content}}], usage}` |
| `GET /v1/health` | 健康检查 | `{status, model}` |
| `GET /` | **Web 管理页面**（模型信息 + 接口清单 + Swagger 链接） | HTML |
| `GET /docs` | API 文档 | Swagger UI（fastapi 自动） |

### 请求体（chat/completions）

```json
{
  "model": "default",
  "messages": [{"role": "user", "content": "你是谁？"}],
  "temperature": 0.7, "top_p": 0.9, "max_tokens": 512
}
```

### 兼容客户端

- **OpenAI SDK**：`base_url="http://127.0.0.1:8000/v1"`（模型 id 用 `models` 列表中的 id）
- **curl**：`curl http://127.0.0.1:8000/v1/chat/completions -d @req.json`
- **LangChain / 其他 OpenAI 兼容工具**：base_url 指向本服务即可

### 说明

- 模型惰性构建（首次请求触发——约 1-2 分钟）；复用 `cli_chat.ChatSession` 的生成链路（历史 + 采样参数）
- 生成走 liteengine 纯 Python 推理（torch BLAS——lm_head 约 95.9ms/token）

## 2. TUI 对话界面

原 CLI（`cli_chat.py`）保留，新增 `python/tui_chat.py`（textual——终端交互界面）：

```bash
python tui_chat.py [--model <name>]
```

### 功能

- 历史区（上下滚动）+ 输入框 + 发送按钮（Enter 或点击发送）
- 首次输入触发模型构建（复用 ChatSession——历史 + 采样参数）
- 用户/助手消息分色显示

### 依赖

`textual`（pip 已装 8.2.8）——需支持 ANSI 的终端（Windows Terminal / 现代终端）。

## 3. 结构

```
python/
  cli_chat.py       # 原 CLI（保留——脚本对话）
  tui_chat.py       # 新 TUI（textual——终端交互）
  api_server.py     # 新 OpenAI 兼容 API 服务（fastapi + Web 管理页）
```

## 4. 使用示例

```bash
# 1) API 服务
python api_server.py --port 8000

# 2) 另一个终端——OpenAI 兼容调用
curl http://127.0.0.1:8000/v1/models
curl http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"default","messages":[{"role":"user","content":"你好"}]}'

# 3) TUI 对话
python tui_chat.py
```

详细配置见 `推荐配置指南.md`；性能数据见 `性能测试汇总.md`。
