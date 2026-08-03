"""LangGraph Agent 图组装"""

from langgraph.graph import StateGraph, END

from src.config import logger
from src.agent.state import AgentState
from src.agent.nodes import (
    _make_llm,
    _summarize_old_messages,
    router_node,
    retrieve_node,
    web_search_node,
    graph_query_node,
    generate_node,
)
from src.agent.edges import route_after_router, pick_next_tool, should_continue


def build_agent_graph(retriever, llm=None):
    """构建并编译 LangGraph Agent

    Graph 结构:

                  ┌─────────────────────────────┐
                  │         router_node          │
                  │  state["messages"] → LLM     │
                  │  返回 AIMessage              │
                  └─────────────┬───────────────┘
                                │ route_after_router
                  有 tool_calls │         无 tool_calls
                                │               │
                    ┌───────────┘               │
                    ▼                           │
          ┌──────────────────┐                  │
          │  pick_next_tool  │ (pass-through)   │
          │  扫描 messages   │                  │
          │  找未执行的       │                  │
          │  tool_call        │                  │
          └────────┬─────────┘                  │
                   │ pick_next_tool             │
        ┌──────────┼──────────┐                 │
        ▼          ▼          ▼                 │
    ┌────────┐┌──────────┐┌──────────┐         │
    │retrieve││web_search││graph_    │         │
    │_node   ││_node     ││query_node│         │
    │ToolMsg ││ToolMsg   ││ToolMsg   │         │
    └───┬────┘└────┬─────┘└────┬─────┘         │
        │          │           │                │
        └──────────┼───────────┘                │
                   │ should_continue            │
           还有未执行? → pick_next_tool          │
           全部执行完? → router                  │
                   │                            │
                   └────────────┬───────────────┘
                                │
                                ▼
                     ┌──────────────────┐
                     │  generate_node   │
                     │  收集 ToolMessage│
                     │  → 最终回答      │
                     └────────┬─────────┘
                              │
                              ▼
                             END
    """
    if llm is None:
        llm = _make_llm()

    workflow = StateGraph(AgentState)

    # 节点注册
    workflow.add_node("summarize", lambda s: _summarize_old_messages(s, llm))
    workflow.add_node("router", router_node)
    workflow.add_node("retrieve", lambda s: retrieve_node(s, retriever))
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("graph_query", graph_query_node)
    workflow.add_node("generate", lambda s: generate_node(s, llm))

    # pick_next_tool 是虚拟路由节点，无操作的 pass-through
    workflow.add_node("pick_next_tool", lambda s: {})

    workflow.set_entry_point("summarize")
    workflow.add_edge("summarize", "router")

    # Router → 根据是否有 tool_calls 分发
    workflow.add_conditional_edges(
        "router",
        route_after_router,
        {
            "pick_next_tool": "pick_next_tool",
            "generate": "generate",
        },
    )

    # pick_next_tool → 分发到具体工具
    workflow.add_conditional_edges(
        "pick_next_tool",
        pick_next_tool,
        {
            "retrieve": "retrieve",
            "web_search": "web_search",
            "graph_query": "graph_query",
            "generate": "generate",
        },
    )

    # 每个工具 → should_continue
    for tool_node in ["retrieve", "web_search", "graph_query"]:
        workflow.add_conditional_edges(
            tool_node,
            should_continue,
            {
                "pick_next_tool": "pick_next_tool",
                "router": "router",
                "generate": "generate",
            },
        )

    workflow.add_edge("generate", END)

    app = workflow.compile()
    logger.info("LangGraph Agent 编译完成")
    return app
