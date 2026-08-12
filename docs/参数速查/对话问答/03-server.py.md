# 03 · server.py 参数

> 文件：`src/api/server.py`（既有），模块级 FastAPI 应用
> 功能：**HTTP 入口**——`POST /chat/stream` 把 `stream_chat` 生成器包成 SSE 流式响应推给前端。

## 构造参数

| 参数 | 默认值 | 大白话 | 技术性 |
|------|--------|--------|--------|
| `FastAPI(title, version, description)` | `"AI 知识库助手 API"` / `"1.0.0"` | 给 Swagger 文档看的元信息 | FastAPI 应用构造元数据 |
| `CORSMiddleware(allow_origins=["*"], ...)` | 全放行 | 允许任何网页跨域调接口 | CORS 跨域中间件，开发期全放行 |

> 启动方式：`uvicorn run_api:app --reload --port 8000`（见 `run_api.py`）。

## 读取的 config（只列本功能链路消费的）

| config 参数 | 默认值 | 大白话 | 技术性 | 谁消费 |
|------------|--------|--------|--------|--------|
| 无 | — | server 层不读配置参数 | 全部逻辑委托给全局单例 `chat_service` | — |

## 核心方法

### `chat_stream(request: ChatRequest)`（`server.py:60`）

```
ChatRequest 校验通过
→ return StreamingResponse(
      chat_service.stream_chat(message=request.message,
                               session_id=request.session_id),   # 异步生成器：逐条 yield SSE 字符串
      media_type="text/event-stream",                            # MIME 告诉浏览器这是流
      headers={
        "Cache-Control": "no-cache",         # 不许缓存
        "Connection": "keep-alive",          # 长连接
        "X-Accel-Buffering": "no",           # 关掉 Nginx 缓冲，保证逐字到前端
      })
```

> **设计要点**
> - `StreamingResponse` 接受一个异步生成器，生成器每 `yield` 一条 SSE 字符串就立刻推给前端——**这就是打字机效果的传输层**。
> - `X-Accel-Buffering: no` 是为了**绕过反向代理/网关的缓冲**：默认 Nginx 会攒满一段才下发，导致打字机变"一顿一顿"。【推断】理由：该 header 是标准反缓冲手段，配合 `text/event-stream` 使用。
> - 错误不靠 HTTP 状态码表达，而是 `chat_service` 内部捕获 LLM 异常后 `yield` 一个 `error` 事件——前端统一按事件协议解析，不用区分 4xx/5xx。

## 该文件在链路中的位置

```
ui/app.py → POST /chat/stream → server.chat_stream → chat_service.stream_chat → StreamingResponse(SSE) → 前端
```
