"""
LangGraph Agent —— 智能决策循环

这是 Phase 2 的核心模块，替代 agent_chain.py 的硬编码路由。

┌──────────────────────────────────────────────────────────┐
│                    Agent 决策循环                          │
│                                                          │
│   router ──→ retrieve ──→ router ──→ web_search ──→ ...  │
│      │          ↑              ↑                         │
│      │          └──── 工具结果返回，LLM 再看 ────┘         │
│      │                                                   │
│      └──→ generate ──→ END   (LLM 觉得够了就直接回答)      │
└──────────────────────────────────────────────────────────┘

每个节点的职责：
  router:       LLM 决定"要调工具"还是"直接回答"
  retrieve:     查 ChromaDB，结果写回 state
  web_search:   查 Bing，结果写回 state
  generate:     汇总上下文生成最终回答
"""
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage

from src.config import config, logger
from src.agent.state import AgentState, ToolCall
from src.agent.tools import retrieve as retrieve_tool, web_search as web_search_tool, graph_query as graph_query_tool
from src.agent.web_search import search_bing, format_search_results
from src.kg.graph_store import GraphStore

# 所有可用工具
TOOLS = [retrieve_tool, web_search_tool, graph_query_tool]

# 防止死循环
MAX_ITERATIONS = 5  # Phase 3: 多了一个工具，多给点循环次数


# ═══════════════════════════════════════════════════════════
# System Prompt
# ═══════════════════════════════════════════════════════════

ROUTER_SYSTEM_PROMPT = """你是知识库助手的路由器。你唯一的职责是：根据检索结果判断下一步该做什么。你不是 chatbot，不能直接回答用户问题。

可用工具：
- retrieve: 从本地知识库检索文档片段
- graph_query: 在知识图谱中查询实体关系
- web_search: 在互联网上搜索补充信息

决策规则（按优先级，严格遵守）：
1. 首次收到用户消息 → 必须调 retrieve + graph_query，不输出任何文字
2. 收到检索结果后 → 判断结果是否与用户问题相关
   - 相关 → 所有工具都执行完了就调用 generate，否则继续执行未完成的工具
   - 不相关 → 必须调 web_search，绝不反问用户
3. 收到 web_search 结果后 → 直接调用 generate，不输出任何文字
4. 禁止行为：不反问、不引导、不打招呼、不输出解释性文字。你的唯一表达方式是 tool_calls"""



# ═══════════════════════════════════════════════════════════
# LLM 工厂（复用实例，减少重复创建）
# ═══════════════════════════════════════════════════════════

def _make_llm():
    return ChatOpenAI(
        model=config.LLM_MODEL,
        api_key=config.LLM_API_KEY,
        base_url=config.LLM_BASE_URL,
        temperature=config.LLM_TEMPERATURE,
        max_tokens=config.LLM_MAX_TOKENS,
        timeout=config.LLM_TIMEOUT,
        max_retries=config.LLM_MAX_RETRIES,
    )


def _make_router_llm():
    """Router 专用 — temperature=0 保证决策确定性"""
    return ChatOpenAI(
        model=config.LLM_MODEL,
        api_key=config.LLM_API_KEY,
        base_url=config.LLM_BASE_URL,
        temperature=0,
        max_tokens=512,
        timeout=config.LLM_TIMEOUT,
        max_retries=config.LLM_MAX_RETRIES,
    )


# ═══════════════════════════════════════════════════════════
# 节点函数
# ═══════════════════════════════════════════════════════════

def router_node(state: AgentState) -> dict:
    """
    路由器节点 —— LLM 决定"下一步做什么"。

    LLM 看到当前对话 + 之前工具的结果后，自己决定：
      A. 调用某个工具（retrieve / web_search）
      B. 直接回答（觉得信息够了）

    这个决策不是硬编码的，而是 LLM 根据上下文自由判断。
    """
    llm = _make_router_llm()
    llm_with_tools = llm.bind_tools(TOOLS)

    messages = [SystemMessage(content=ROUTER_SYSTEM_PROMPT)]

    # 取原始用户问题
    user_content = ""
    for msg in state.get("messages", []):
        if isinstance(msg, HumanMessage):
            user_content = msg.content
            break
        elif isinstance(msg, dict) and msg.get("role", "") in ("human", "user"):
            user_content = msg.get("content", "")
            break

    messages.append(HumanMessage(content=user_content))

    # 如果之前有工具结果，用 ToolMessage 把结果传给 LLM
    #  注意：区分两种信息流：
    #  - tool_calls 里的 result → 上轮工具执行结果（保留，用于连续工具调用场景）
    #  - final_answer → 上轮终答（router_node 无法获取，由 ChatService 通过 context_message 传入）
    has_tool_results = False
    for tc in state.get("tool_calls", []):
        if tc["result"]:
            has_tool_results = True
            # DeepSeek API 要求：ToolMessage 必须紧跟在带 tool_calls 的 AIMessage 之后
            # - AIMessage: 告诉 LLM "我调用了哪个工具"（name + args + id）
            # - ToolMessage: 告诉 LLM "工具返回了什么结果"（content）
            # - 两者通过 id 关联，LLM 才能正确理解哪个结果对应哪个调用
            messages.append(AIMessage(
                content="",
                tool_calls=[{"name": tc["name"], "args": tc["args"], "id": tc["id"]}]
            ))
            messages.append(ToolMessage(
                content=tc["result"],
                tool_call_id=tc["id"],
            ))

    # 如果没有任何工具结果（首次请求或追问场景），把 system prompt 的指示加得更清楚
    if not has_tool_results:
        if "对话背景" in user_content:
            messages[0] = SystemMessage(content=ROUTER_SYSTEM_PROMPT + '\n\n当前消息包含对话背景和用户追问。请定位到背景中上轮助手回答的具体内容，直接基于背景展开讲解。如果涉及知识库中的概念，先检索知识库补充信息。')
        else:
            messages[0] = SystemMessage(content=ROUTER_SYSTEM_PROMPT)

    response = llm_with_tools.invoke(messages)

    # 判断 LLM 是否想调工具
    if response.tool_calls:
        tool_calls = []
        for tc in response.tool_calls:
            tool_calls.append(ToolCall(
                name=tc["name"],
                args=tc["args"],
                result="",
                id=tc.get("id", ""),
            ))

            # 如果图中有多条边但没被全部搜索，聚合一下
            # DeepSeek 可能会把两个 graph_query 拆成两次——我们合并它们的结果
            if tc["name"] == "graph_query":
                entity = tc["args"].get("entity", "")
                # 预热查询（让 graph_query_node 拿到结果）
                pass

        logger.info(f"Router: LLM 决定调用工具 → {[t['name'] for t in tool_calls]}")
        # 合并新旧 tool_calls，避免覆盖已执行的工具结果
        merged_calls = state.get("tool_calls", []).copy()
        for new_tc in tool_calls:
            already_done = any(
                old["name"] == new_tc["name"] and old["args"] == new_tc["args"] and old["result"]
                for old in merged_calls
            )
            if not already_done:
                merged_calls.append(new_tc)
        return {
            "tool_calls": merged_calls,
            "messages": [response],
            "iteration": state.get("iteration", 0) + 1,
        }
    else:
        # 无 tool_calls — 聊天/闲聊直接透传
        logger.info("Router: LLM 直接回答（无工具调用）")
        iteration = state.get("iteration", 0) + 1
        return {
            "final_answer": response.content,
            "tool_calls": [],
            "messages": [response],
            "iteration": iteration,
        }

    # 死代码防护：到这一定是有 tool_calls 的
    return {
        "tool_calls": state.get("tool_calls", []),
        "final_answer": state.get("final_answer", ""),
        "iteration": state.get("iteration", 0),
    }


def retrieve_node(state: AgentState, retriever) -> dict:
    """
    RAG 检索节点 —— 从 ChromaDB 中搜索相关文档片段。

    取 state["tool_calls"] 中第一条 retrieve 类型的调用，
    执行检索，结果以 ToolMessage 回填到 messages 中。
    """
    retrieve_call = None
    for tc in state["tool_calls"]:
        if tc["name"] == "retrieve" and not tc["result"]:
            retrieve_call = tc
            break

    if retrieve_call is None:
        logger.warning("retrieve_node: 没找到待执行的 retrieve 调用")
        return {}

    query = retrieve_call["args"].get("query", "")
    logger.info(f"Retrieve: 搜索 → '{query}'")

    docs = retriever.invoke(query)

    if docs:
        parts = []
        for i, doc in enumerate(docs, 1):
            src = doc.metadata.get('source', '')
            parts.append(f"[片段 {i}] (来源: {src})\n{doc.page_content}")
        context = "\n\n".join(parts)
    else:
        context = "（知识库中未找到相关内容）"

    retrieve_call["result"] = context

    # 构建 ToolMessage —— 关键：tool_call_id 必须匹配
    tool_msg = ToolMessage(
        content=context,
        tool_call_id=retrieve_call["id"],
    )

    logger.info(f"Retrieve: 找到 {len(docs)} 个片段")

    return {
        "tool_calls": state["tool_calls"],
        "messages": [tool_msg],
        "iteration": state.get("iteration", 0) + 1,
    }


def web_search_node(state: AgentState) -> dict:
    """
    联网搜索节点 —— 在 Bing 上搜索。

    与 retrieve_node 同样的结构：找到调用 → 执行 → ToolMessage 回填。
    """
    search_call = None
    for tc in state["tool_calls"]:
        if tc["name"] == "web_search" and not tc["result"]:
            search_call = tc
            break

    if search_call is None:
        logger.warning("web_search_node: 没找到待执行的 web_search 调用")
        return {}

    query = search_call["args"].get("query", "")
    logger.info(f"WebSearch: 搜索 → '{query}'")

    results = search_bing(query, max_results=3)
    context = format_search_results(results)

    search_call["result"] = context

    tool_msg = ToolMessage(
        content=context,
        tool_call_id=search_call["id"],
    )

    return {
        "tool_calls": state["tool_calls"],
        "messages": [tool_msg],
        "iteration": state.get("iteration", 0) + 1,
    }


def graph_query_node(state: AgentState) -> dict:
    """
    知识图谱查询节点 —— 在 NetworkX 中查找实体关系。

    结构与 retrieve_node / web_search_node 完全一致：
    找到 graph_query 调用 → 查图谱 → ToolMessage 回填。
    """
    gq_call = None
    for tc in state["tool_calls"]:
        if tc["name"] == "graph_query" and not tc["result"]:
            gq_call = tc
            break

    if gq_call is None:
        logger.warning("graph_query_node: 没找到待执行的 graph_query 调用")
        return {}

    entity = gq_call["args"].get("entity", "")
    logger.info(f"GraphQuery: 查询实体 → '{entity}'")

    store = GraphStore()
    context = store.query_to_text(entity)

    gq_call["result"] = context

    tool_msg = ToolMessage(
        content=context,
        tool_call_id=gq_call["id"],
    )

    return {
        "tool_calls": state["tool_calls"],
        "messages": [tool_msg],
        "iteration": state.get("iteration", 0) + 1,
    }


def generate_node(state: AgentState, llm) -> dict:
    """
    生成最终回答 —— 汇总所有工具结果，流式输出给用户。

    关键设计：
      - 用 llm.astream() 替代 llm.invoke()，让 astream_events 能捕获 token 流
      - LangGraph 的 astream_events 是事件驱动架构：LLM 每产一个 token
        就触发 on_chat_model_stream → 外层 ChatService 推 SSE
    """
    # 1. 找到原始用户问题
    user_question = ""
    for msg in state.get("messages", []):
        if isinstance(msg, HumanMessage):
            user_question = msg.content
            break
        elif isinstance(msg, dict) and msg.get("role", "") in ("human", "user"):
            user_question = msg.get("content", "")
            break

    # 如果 router 已经给了 final_answer（追问直接回答），直接透传
    if state.get("final_answer"):
        logger.info("Generate: final_answer 已在 router 中生成，透传输出")
        return {
            "final_answer": state["final_answer"],
        }

    # 2. 收集所有工具的搜索结果
    tool_results_text = ""
    for tc in state.get("tool_calls", []):
        if tc["result"]:
            tool_results_text += f"\n### {tc['name']} 搜索结果（查询: {tc['args'].get('query', '')}）\n{tc['result']}\n"

    # 3. 构建全新消息列表
    if tool_results_text:
        system_prompt = f"""你是一个知识库助手。以下是通过工具收集到的信息片段：

{tool_results_text}

回答示例：

<example_input>什么是MySQL？[检索到引擎对比]</example_input>
<example_output>MySQL 是一个开源的关系型数据库管理系统（RDBMS），由 Oracle 公司开发和维护。它是全球最流行的开源数据库之一，核心特性包括：支持 ACID 事务、提供多种存储引擎（InnoDB、MyISAM 等）、高并发下的 MVCC 机制、标准化 SQL 查询语言、丰富的索引类型。MySQL 广泛应用于 Web 应用后端，是 LAMP（Linux+Apache+MySQL+PHP）技术栈的核心组件，Facebook、Uber 等大型互联网公司都在使用它。它具备社区版（免费）和企业版（付费）两种版本。

补充一点来自你的知识库的信息：MySQL 的存储引擎中，InnoDB 是 5.5 版本后的默认引擎，支持事务和行级锁，适合高并发场景；MyISAM 不支持事务但读取速度快，适合只读/分析场景。</example_output>

回答规则：
1. 严格禁止反问——不要问用户任何问题，不要列选项让用户选，不要以"如果你有具体场景..."结尾。你只有一个任务：回答问题本身
2. 先用你自己的知识体系给出完整的定义、核心特性、典型应用场景。写一个大段落，像一个技术百科的作者在写条目
3. 然后检查检索结果——有相关的内容就用"补充一点来自你的知识库的信息："拼接进去。检索结果是佐料，你的知识才是主菜
4. 最终回答的长度应该至少 300 字，结构上只有一个大段落（定义）加一个补充段落（检索到的具体细节）"""

    else:
        system_prompt = f"""你是一个知识库助手。以下是通过工具收集到的信息：
（未使用搜索工具）

用户的问题是：「{user_question}」

请根据你的自身知识回答用户的问题。如果不知道就说不知道，不要编造。"""

    clean_messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_question),
    ]

    response = llm.invoke(clean_messages)

    logger.info("Generate: 最终回答生成完成")
    return {
        "final_answer": response.content,
        "messages": [response],
    }


# ═══════════════════════════════════════════════════════════
# 条件边函数
# ═══════════════════════════════════════════════════════════

def route_after_router(state: AgentState) -> str:
    """
    router 执行完后，根据 LLM 的决策走向下一个节点。

    返回: "retrieve" | "web_search" | "generate"
    """
    if state.get("final_answer"):
        return "generate"

    calls = state.get("tool_calls", [])
    if not calls:
        return "generate"

    for tc in calls:
        if not tc["result"]:
            return tc["name"]

    return "generate"


def should_loop(state: AgentState) -> str:
    """
    工具执行完后判断：回到 router 还是直接生成？

    终止条件:
      1. final_answer 已填充
      2. 循环次数到 MAX_ITERATIONS
      3. 所有 tool_call 都执行完了
    """
    if state.get("final_answer"):
        return "generate"

    iteration = state.get("iteration", 0)
    if iteration >= MAX_ITERATIONS:
        logger.info(f"达到最大循环次数 {MAX_ITERATIONS}，强制生成回答")
        return "generate"

    # 还有未执行完的工具 → 回 router
    for tc in state.get("tool_calls", []):
        if not tc["result"]:
            return "router"

    # 所有工具都执行完了，直接生成
    return "generate"


# ═══════════════════════════════════════════════════════════
# Graph 组装
# ═══════════════════════════════════════════════════════════

def build_agent_graph(retriever, llm=None):
    """
    构建并编译 LangGraph Agent。

    Graph 结构:

              ┌──────────────────────────┐
              │         router            │
              │  LLM 决定调工具 / 直接答   │
              └─────┬──────┬──────┬──────┘
                    │      │      │
          "retrieve"│  "web_search"  "generate"
                    ▼      ▼      │
              ┌─────────┐┌──────┐│
              │retrieve ││web_  ││
              │ (RAG)   ││search││
              └────┬────┘└──┬───┘│
                   │        │    │
                   ▼        ▼    │
                  should_loop     │
                  /        \     │
            "router"   "generate" │
                 │         │     │
                 └────┬────┘     │
                      ▼          │
                    END ←────────┘

    参数:
      retriever: LangChain Retriever 对象
      llm: 可选的共享 LLM 实例

    返回:
      编译后的 CompiledGraph
    """
    if llm is None:
        llm = _make_llm()

    workflow = StateGraph(AgentState)

    workflow.add_node("router", router_node)
    workflow.add_node("retrieve", lambda s: retrieve_node(s, retriever))
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("graph_query", graph_query_node)
    workflow.add_node("generate", lambda s: generate_node(s, llm))

    workflow.set_entry_point("router")

    workflow.add_conditional_edges(
        "router",
        route_after_router,
        {
            "retrieve": "retrieve",
            "web_search": "web_search",
            "graph_query": "graph_query",
            "generate": "generate",
        }
    )

    workflow.add_conditional_edges(
        "retrieve",
        should_loop,
        {
            "router": "router",
            "generate": "generate",
        }
    )
    workflow.add_conditional_edges(
        "web_search",
        should_loop,
        {
            "router": "router",
            "generate": "generate",
        }
    )
    workflow.add_conditional_edges(
        "graph_query",
        should_loop,
        {
            "router": "router",
            "generate": "generate",
        }
    )

    workflow.add_edge("generate", END)

    app = workflow.compile()
    logger.info("LangGraph Agent 编译完成")
    return app
