# 08 · graph.py 参数

> 文件：`src/agent/graph.py`（既有），模块级函数 `build_agent_graph`
> 功能：**LangGraph 图组装**——把节点和条件边注册成状态机并编译，是"决策循环"的拓扑定义。
> LangGraph 状态机概念见 [Notes/AI-Agent/AI-Agent核心概念-小林.md](../../../../Notes/AI-Agent/AI-Agent核心概念-小林.md)。

## 构造参数

`build_agent_graph(retriever, llm=None)`：

| 参数 | 默认值 | 大白话 | 技术性 |
|------|--------|--------|--------|
| `retriever` | 必传 | 知识库检索器，喂给 retrieve_node | `chat_service` 传 `HybridRetriever` 实例 |
| `llm` | `None` | 不传就自建生成 LLM | `llm = _make_llm()` |

## 读取的 config（只列本功能链路消费的）

| config 参数 | 默认值 | 大白话 | 技术性 | 谁消费 |
|------------|--------|--------|--------|--------|
| 无 | — | 图拓扑是硬编码连好的 | 不读 config；LLM 参数在 `nodes._make_llm` | — |

## 核心方法

### `build_agent_graph(retriever, llm=None)`（`graph.py:19`）

```
workflow = StateGraph(AgentState)
# 注册 6 个真节点 + 1 个虚拟路由
add_node("summarize", ...)  add_node("router", router_node)
add_node("retrieve", lambda s: retrieve_node(s, retriever))
add_node("web_search", web_search_node)   add_node("graph_query", graph_query_node)
add_node("generate", lambda s: generate_node(s, llm))
add_node("pick_next_tool", lambda s: {})  # pass-through，无操作
set_entry_point("summarize")              # 入口：先压缩旧对话
add_edge("summarize", "router")
# Router → 有 tool_calls? pick_next_tool : generate
add_conditional_edges("router", route_after_router, {...})
# pick_next_tool → 分发到具体工具
add_conditional_edges("pick_next_tool", pick_next_tool, {...})
# 每个工具执行完 → should_continue（还有未执行→pick_next_tool；全完→router；迭代上限→generate）
for tool in ["retrieve","web_search","graph_query"]:
    add_conditional_edges(tool, should_continue, {...})
add_edge("generate", END)
return workflow.compile()   # ← 返回编译后的可调用对象
```

> **设计要点**
> - **`pick_next_tool` 是虚拟节点**：无操作（返回 `{}`），只作为条件分发的"岔路口"，让图结构清晰（Router → 岔路口 → 具体工具）。
> - **工具节点用闭包绑注入依赖**：`lambda s: retrieve_node(s, retriever)`、`lambda s: generate_node(s, llm)`——检索器和 LLM 从外部注入，图本身不感知具体实现，利于测试（可注入 mock）。
> - **循环结构**：工具 → `should_continue` → 回 `pick_next_tool`/`router`，这是 LangGraph 用"条件边回指"实现的循环，没有 `add_loop` 之类显式 API。
> - 【推断】为什么 `summarize` 是入口而非直接 router：先压缩/清理旧消息，保证 router 看到的窗口精简。证据：`graph.py:82-83` + `nodes._summarize_old_messages` 注释。

## 该文件在链路中的位置

```
chat_service.stream_chat ④ → build_agent_graph(retriever) → (stream_chat ⑥) agent.astream → 逐节点执行
```
