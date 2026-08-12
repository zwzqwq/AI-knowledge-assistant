# 05 · history.py 参数

> 文件：`src/memory/history.py`（既有），类 `ConversationHistory`
> 功能：**会话历史容器**——每轮 user/assistant 原文存内存，供追问上下文注入使用。

## 构造参数

| 参数 | 默认值 | 大白话 | 技术性 |
|------|--------|--------|--------|
| `max_turns` | `None → config.HISTORY_MAX_TURNS` | 最多记几轮对话 | 不传就取全局默认 |

## 读取的 config（只列本功能链路消费的）

| config 参数 | 默认值 | 大白话 | 技术性 | 谁消费 |
|------------|--------|--------|--------|--------|
| `HISTORY_MAX_TURNS` | 6 | 会话最多保留 6 轮 | 决定 `messages` 数组保留多少轮 | `__init__` 默认值 |

## 核心方法

### `add_user(content)` / `add_assistant(content)`（`history.py:14,18`）

```
messages.append({"role": "user"/"assistant", "content": content})
```

### `format()`（`history.py:20`）

```
recent = messages[-max_turns*2:]   # ×2 因为一轮 = user + assistant
→ 拼成 "用户: ...\n助手: ...\n" 文本
```

> **设计要点**
> - **只存原文，不做图内窗口管理**。真正喂给 LLM 的消息窗口（摘要压缩、截断）是 LangGraph `state["messages"]` + `conversation_summary` 的事（`nodes.py`）——**这里只是"跨请求的会话记忆"**。
> - 本链路里 `stream_chat` 用的主要是 `add_user`/`add_assistant`/`messages` 属性 + 倒序遍历找上轮回答；`format()` 是早期 Phase 拼接 prompt 的遗留，当前对话链路**不消费**——按参数裁剪规则不展开。

## 该文件在链路中的位置

```
chat_service.stream_chat → history.add_user/add_assistant →（追问上下文读取 messages）→ 图执行 → 回写
```
