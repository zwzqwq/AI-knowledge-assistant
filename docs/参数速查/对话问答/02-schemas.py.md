# 02 · schemas.py 参数

> 文件：`src/api/schemas.py`（既有），模块级 Pydantic 模型
> 功能：**HTTP 数据合同**——对话请求进来先过校验，不合法直接 422，不碰业务逻辑。

## 构造参数

| 参数 | 默认值 | 大白话 | 技术性 |
|------|--------|--------|--------|
| `ChatRequest.message` | 必传（`...`） | 用户问题，空/超长拒绝 | `Field(..., min_length=1, max_length=2000)` |
| `ChatRequest.session_id` | `"default"` | 不传就用默认会话 | `Field(default="default")` |

## 读取的 config（只列本功能链路消费的）

| config 参数 | 默认值 | 大白话 | 技术性 | 谁消费 |
|------------|--------|--------|--------|--------|
| 无 | — | 校验规则硬编码在字段定义里 | 不读 config，`server.py` 依赖注入实例 | — |

## 核心方法

Pydantic 模型无方法。`ChatRequest` 被 `server.py` 的 `chat_stream` 作为参数类型注解——FastAPI 自动解析请求体 JSON 并校验，非法请求在进入 service 前返回 422。

> **设计要点**
> - 长度上限 2000 是**防滥用护栏**：超长问题直接拒绝，避免把超大文本喂进 Agent 循环烧 Token。
> - `ChatResponse` 是本项目 Phase 2 遗留（非流式响应模型），当前对话链路走 SSE 流式，**不消费它**——按参数裁剪规则不展开。

## 该文件在链路中的位置

```
ui/app.py → POST /chat/stream → schemas.ChatRequest（校验）→ server.chat_stream → chat_service
```
