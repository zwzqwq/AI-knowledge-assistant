# 01 · app.py 参数

> 文件：`src/ui/app.py`（既有），模块级脚本（Streamlit 入口）
> 功能：**对话链路的用户入口**——收集问题 → 走 HTTP 调后端 → 流式渲染 SSE 回答。

## 构造参数

Streamlit 脚本，无类、无 `__init__`。关键**模块级常量**：

| 参数 | 默认值 | 大白话 | 技术性 |
|------|--------|--------|--------|
| `API_BASE` | `"http://127.0.0.1:8000"` | 后端地址，Streamlit 和 FastAPI 都在本机 | 前端唯一知道的"后端在哪"的地址 |
| `API_BASE` 相关 timeout | 5s（管理接口）/ 120s（对话流） | 管理操作快、对话慢，超时时间分开 | 管理接口 `httpx.timeout=5`，对话流 `httpx.stream(timeout=120)` |

## 读取的 config（只列本功能链路消费的）

| config 参数 | 默认值 | 大白话 | 技术性 | 谁消费 |
|------------|--------|--------|--------|--------|
| 无 | — | 前端是纯 HTTP 客户端，不直接读 config | 所有配置都在后端，前端只发请求 | — |

## 核心方法

### 对话发送 + SSE 流式渲染（`app.py:220-254`）

```
prompt = st.chat_input("输入你的问题...")            # 用户输入
  → 本地追加 user 消息、渲染
  → httpx.stream("POST", /chat/stream,
                 json={message: prompt, session_id}, timeout=120)
     for line in response.iter_lines():             # 逐行读 SSE
       if line.startswith("data: "):
         data = json.loads(line[6:])                # 去掉 "data: " 前缀
         - "sources" in data → 显示来源标签（知识库/图谱/联网/AI自身）
         - "content" in data → full_answer += content, 渲染 full_answer + "▌"  ← 打字机
         - "error" in data   → st.error
         - "done" in data    → 结束
  → 把完整答案（含 sources）存入本地 messages
```

> **设计要点**
> - 前端**只管三个事件**：`source`（来源）、`token`（内容逐字）、`error/done`（收尾）。后端事件协议变化才需要改前端——前后端以事件协议解耦。
> - `st.session_state.messages` 是**前端自己的渲染缓存**，不是权威历史；权威历史在后端 `ConversationHistory`，切换会话时从 `/session/{id}/history` 拉取刷新（`app.py:142`）。
> - 会话切换失败时的降级：`/sessions` 后端不可用时本地 `uuid` 生成 session_id（`app.py:107-119`）。【推断】理由：兜底不阻塞用户提问。证据：`except Exception` 分支内 fallback 本地生成。

## 该文件在链路中的位置

```
用户输入 → src/ui/app.py → httpx POST /chat/stream → (后端 SSE 流回) → 逐字渲染
```
