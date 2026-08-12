# 04 · chat_service.py 参数

> 文件：`src/services/chat_service.py`（既有），类 `ChatService`
> 功能：**对话链路的编排中枢（导演）**——管会话历史、组装检索器与图、跑图、把结果转成 SSE 事件。

## 构造参数

| 参数 | 默认值 | 大白话 | 技术性 |
|------|--------|--------|--------|
| `__init__()` | 无参 | 一启动就把加载器、检索器管理、会话表备好 | `DocumentLoader()` + `Retriever()` + `self._sessions: dict[str, ConversationHistory]` |

> 全局单例 `chat_service = ChatService()`（`chat_service.py:311`）——所有请求共享同一实例、同一会话表。**单进程内存态**：重启后会话/向量库引用丢失（向量库本身在磁盘，重启可重新加载）。

## 读取的 config（只列本功能链路消费的）

| config 参数 | 默认值 | 大白话 | 技术性 | 谁消费 |
|------------|--------|--------|--------|--------|
| `LLM_*`（MODEL/API_KEY/BASE_URL/TEMPERATURE/MAX_TOKENS/TIMEOUT/MAX_RETRIES） | 见 15-config | 决定对话用哪个模型、怎么连 | 传给 `_make_llm` / `_make_router_llm` | `nodes.py`（经 graph 间接） |

## 核心方法

### `stream_chat(message, session_id="default")`（`chat_service.py:163`）★

```
history = _get_or_create_history(session_id)   # ① 取/建会话历史
history.add_user(message)                      # ② 记用户消息
retriever = get_hybrid_retriever()             # ③ 现组装混合检索器（向量+BM25+Rerank）
agent = build_agent_graph(retriever)           # ④ 编译 LangGraph
# ⑤ 追问上下文注入：
#    若历史里有上轮回答(>20字) → 拼"对话背景——上轮问题/上轮回答/当前追问"
initial_state = {messages:[HumanMessage(context_message)],
                 iteration:0, final_answer:"", conversation_summary:""}
final_answer = ""
async for chunk in agent.astream(initial_state):   # ⑥ 逐节点执行图
    for node_name, node_output in chunk.items():
        收集 messages；若 generate/router 有 final_answer → 记录
# ⑦ 来源判定：扫描所有 ToolMessage
#    retrieve 有内容→knowledge_base；graph_query→knowledge_graph；
#    web_search→web_search；全空→llm
yield _sse_event("source", {sources})            # ⑦ 先发来源
for char in final_answer:                         # ⑧ 逐字输出（打字机）
    yield _sse_event("token", {"content": char})
history.add_assistant(final_answer)               # ⑨ 记回答
yield _sse_event("done", {})
# 异常兜底：AuthenticationError/RateLimit/Timeout/Connection/APIError/Exception
#   → yield _sse_event("error", ...) 给前端可读信息
```

> **设计要点**
> - **编排与图分离**：`stream_chat` 不关心"怎么决策"，只负责"把问题放进去、把答案拿出来、转成 SSE"。图逻辑全在 `graph.py`/`nodes.py`。
> - **一次提问 = 多次 LLM 调用**：`astream` 是节点级流（每个节点跑完 yield 一次），不是 token 级流——`generate_node` 内部一次性生成完整回答，真正的"逐字"发生在**这层的回放**（`for char in final_answer`）。
> - **追问上下文注入**：图内 router 只看当前一轮，跨轮信息靠这里拼成 context_message 塞进 initial_state（`chat_service.py:183-202`）。
> - **来源判定靠"内容长度 + 占位符"启发式**：`len(m.content) > 100` 且不含"（知识库中未找到"等占位 → 认为该工具真出了结果。这是轻量方案；更严谨做法是让工具返回结构化成功标志。【推断】理由：代码用 `>100` 字符+排除占位串判断"有无结果"，无显式状态字段。证据：`chat_service.py:238-250`。
> - 节点执行可观测性：`agent.astream` 每个节点输出不直接打印，靠 `nodes.py` 各节点的 `logger.info` 追踪（Router 决策、Retrieve/WebSearch/GraphQuery 查询），日志落盘到 `logs/`（见 15-config）。

## 该文件在链路中的位置

```
ui/app.py → server.py → chat_service.stream_chat → history/retriever/graph → agent.astream → SSE 事件 → server.py → 前端
```
