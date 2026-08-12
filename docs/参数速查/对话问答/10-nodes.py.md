# 10 · nodes.py 参数

> 文件：`src/agent/nodes.py`（既有），模块级函数集
> 功能：**对话链路的大脑**——6 类节点函数 + LLM 工厂 + 消息窗口管理，一次提问的"决策→执行→生成"全在这里落地。
> Function Calling / 工具循环概念见 [Notes/AI-Agent/AI-Agent核心概念-小林.md](../../../../Notes/AI-Agent/AI-Agent核心概念-小林.md)。

## 构造参数

模块级函数，无类。关键**函数签名**：

| 函数 | 参数 | 大白话 | 技术性 |
|------|------|--------|--------|
| `_make_llm()` / `_make_router_llm()` | 无参 | 建生成/决策 LLM | 读 config 建 `ChatOpenAI` |
| `retrieve_node(state, retriever)` | 必传 | 执行知识库检索 | 从 state 找 tool_call → `retriever.invoke(query)` |
| `generate_node(state, llm)` | 必传 | 生成最终回答 | 收集 ToolMessage → prompt → LLM |
| `_summarize_old_messages(state, llm)` | 必传 | 压缩旧对话 | 增量摘要 + RemoveMessage 删旧 |
| `router_node(state)` | 必传 | 决策 | 内部自建 router LLM |

## 读取的 config（只列本功能链路消费的）

| config 参数 | 默认值 | 大白话 | 技术性 | 谁消费 |
|------------|--------|--------|--------|--------|
| `LLM_MODEL` | `deepseek-chat` | 用哪个模型 | `_make_llm`/`_make_router_llm` |
| `LLM_API_KEY` / `LLM_BASE_URL` | 见 15-config | 连哪个 API | 同上 |
| `LLM_TEMPERATURE` | 0.7 | 生成回答的随机度 | `_make_llm` |
| `LLM_MAX_TOKENS` | 2048 | 回答最长多少 token | `_make_llm` |
| `LLM_TIMEOUT` / `LLM_MAX_RETRIES` | 60 / 2 | 超时/重试 | 两个工厂 |
| `ROUTER_MAX_RECENT_MESSAGES` | 8 | 决策时保留最近几条纯对话 | `_build_router_messages` |
| `ROUTER_TOOL_RESULT_MAX_CHARS` | 200 | 决策时 ToolMessage 截断长度 | `_build_router_messages` |
| `ROUTER_SUMMARY_MAX_CHARS` | 800 | 摘要上限 | `_summarize_old_messages` |

> router LLM 的 `temperature=0`、`max_tokens=512` 是**硬编码**（`nodes.py:40-41`），不走 config——决策要确定性，输出只要 tool_calls。

## 核心方法

### `_make_router_llm()`（`nodes.py:35`）—— 决策 LLM

```
ChatOpenAI(model=LLM_MODEL, temperature=0, max_tokens=512, ...)
```

### `router_node(state)`（`nodes.py:216`）—— 决策节点

```
llm = _make_router_llm().bind_tools(TOOLS)   # 让 LLM 能输出 tool_calls
messages_to_send = _build_router_messages(state)
response = llm.invoke(messages_to_send)
return {"messages": [response], "iteration": iteration+1}
```

### `_build_router_messages(state)`（`nodes.py:62`）—— 窗口精简（关键设计）

```
① 当前轮（最后一条 HumanMessage 之后）→ 全保留
     ToolMessage 截断到 ROUTER_TOOL_RESULT_MAX_CHARS（Router 只要"有无结果/是否相关"的粗信号）
② 历史纯对话 → 只留最近 ROUTER_MAX_RECENT_MESSAGES 条
③ 旧轮工具消息 → 全丢（跨轮无用，极占 Token）
④ conversation_summary → 以 SystemMessage 注入
```

### `retrieve_node(state, retriever)`（`nodes.py:242`）

```
tc = _find_pending_tool_call(state["messages"], "retrieve")   # 找未执行的 retrieve 调用
if tc is None: return {}
docs = retriever.invoke(tc["args"]["query"])                  # 混合检索
context = 有结果 → "[片段 1] (来源: xxx)\n内容..." ; 无 → "（知识库中未找到相关内容）"
return {"messages": [ToolMessage(content=context, tool_call_id=tc["id"], name="retrieve")]}
```

### `generate_node(state, llm)`（`nodes.py:346`）—— 生成节点

```
user_question = 最后一条 HumanMessage.content
round_start = _find_last_human_index(messages)                # 界定当前轮
current_tool_messages = 本轮所有 ToolMessage
# 特例：router 已直接给文本且本轮无工具交互 → 透传，不重复生成
if 上一条 AIMessage 有 content 且无 tool_calls 且无工具消息:
    return {"final_answer": last_msg.content}
# 正常：同工具取最后一条结果 → 拼 tool_results_text
system_prompt = build_generate_prompt(user_question, tool_results_text)   # 按问题类型选 prompt
response = llm.invoke([SystemMessage(system_prompt), HumanMessage(user_question)])
return {"final_answer": response.content, "messages": [response]}
```

### `_find_pending_tool_call(messages, tool_name)`（`nodes.py:311`）—— 工具循环的"记账本"

```
start_idx = _find_last_human_index(messages)       # 只扫当前轮
倒序遍历本轮消息，找到带 tool_calls 的 AIMessage
  对每个 tool_call：tool_call_id 是否已有对应 ToolMessage？
    已有 → 已执行，跳过；没有 → 返回这个未执行的调用
找不到 → return None（该工具本轮没被调用或已执行完）
```

> **设计要点**
> - **工具声明与执行分离**：`tools.py` 只出 JSON Schema 给 LLM 看，真正执行在本节点（用 `_find_pending_tool_call` 手动找未执行的 tool_call）——不用 LangChain `ToolNode`，对"执行哪个/几次/结果怎么放回"完全可控。
> - **`tool_call_id` 是关联键**：ToolMessage 必须带 `tool_call_id` 与 AIMessage 的 tool_call 对应（LangChain 协议），`_find_pending_tool_call` 靠它判断"是否已执行"（`nodes.py:331`）。
> - **只扫当前轮**（`_find_last_human_index` 之后）——注释明确"不再全局倒序，避免跨轮 tool_call 污染"（`nodes.py:314`）。
> - **generate 透传特例**：router 直接给文本且无工具交互时不重复调 LLM（省一次调用）。
> - **同工具取最后一条**：`latest_per_tool[tc["name"]]`——同一工具被执行多次（如重试）时，只留最后一次结果喂给 generate（`nodes.py:371`）。

## 该文件在链路中的位置

```
graph.py 注册 → astream 执行时：
  summarize → _summarize_old_messages
  router → router_node → _build_router_messages → LLM
  retrieve/web_search/graph_query → 对应节点（_find_pending_tool_call 驱动）
  generate → generate_node → final_answer → chat_service
```
