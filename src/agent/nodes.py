"""
LangGraph Agent 节点函数

每个节点返回 {"messages": [新消息]}，由 add_messages reducer 自动追加。
节点不手动创建 state["tool_calls"] 或修改 state["context"]。
"""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage, RemoveMessage
from langchain_openai import ChatOpenAI

from src.config import config, logger
from src.agent.state import AgentState
from src.agent.tools import TOOLS
from src.agent.prompts import ROUTER_SYSTEM_PROMPT, build_generate_prompt
from src.agent.web_search import search_bing, format_search_results
from src.kg.graph_store import GraphStore


# ═══════════════════════════════════════════════════════════
# LLM 工厂
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
# Router 消息窗口管理
# ═══════════════════════════════════════════════════════════

def _is_plain_message(msg) -> bool:
    """判断是否为纯对话消息（非工具相关的 System/Tool/带 tool_calls 的 AIMessage）"""
    if isinstance(msg, SystemMessage):
        return False
    if isinstance(msg, ToolMessage):
        return False
    if isinstance(msg, AIMessage) and (msg.tool_calls or getattr(msg, "name", None)):
        return False
    return isinstance(msg, (HumanMessage, AIMessage))


def _build_router_messages(state: AgentState) -> list:
    """构建传给 Router LLM 的精简消息列表

    策略：
      1. 当前轮的 HumanMessage + AIMessage(tool_calls) + ToolMessage → 全部保留
      2. 历史纯对话 → 保留最近 N 条，多余的不传（由 summarize 节点异步压缩）
      3. 旧轮的工具消息 → 全部丢弃（跨轮无用，极占 Token）
      4. conversation_summary → 以 SystemMessage 形式注入
    """
    all_messages = state.get("messages", [])
    max_recent = config.ROUTER_MAX_RECENT_MESSAGES

    # 找到当前轮的起始位置
    current_round_start = len(all_messages)
    for i in range(len(all_messages) - 1, -1, -1):
        if isinstance(all_messages[i], HumanMessage):
            current_round_start = i
            break

    history_messages = all_messages[:current_round_start]
    current_round_messages = all_messages[current_round_start:]

    # 对当前轮的 ToolMessage 做截断拷贝 —— Router 只需要粗粒度信号（有没有结果、是否相关），
    # 不需要完整内容（那是 Generate 的工作）。不改 state 里的原始消息。
    max_tool_chars = config.ROUTER_TOOL_RESULT_MAX_CHARS
    truncated_current: list = []
    for msg in current_round_messages:
        if isinstance(msg, ToolMessage) and len(msg.content) > max_tool_chars:
            truncated = msg.model_copy()
            truncated.content = msg.content[:max_tool_chars] + "..."
            truncated_current.append(truncated)
        else:
            truncated_current.append(msg)

    # 历史中只保留最近 N 条纯对话
    plain_history = [m for m in history_messages if _is_plain_message(m)]
    recent_plain = plain_history[-max_recent:] if len(plain_history) > max_recent else plain_history
    old_plain_count = max(0, len(plain_history) - max_recent)

    # 组装
    messages_to_send = [SystemMessage(content=ROUTER_SYSTEM_PROMPT)]

    existing_summary = state.get("conversation_summary", "").strip()
    if existing_summary:
        messages_to_send.append(
            SystemMessage(content=f"[对话历史摘要]\n{existing_summary}")
        )

    messages_to_send.extend(recent_plain)
    messages_to_send.extend(truncated_current)

    logger.info(
        f"Router messages: {len(all_messages)} total → {len(messages_to_send)} to LLM "
        f"(kept {len(recent_plain)} recent plain + {len(current_round_messages)} current, "
        f"dropped {old_plain_count} old plain + all old tool messages)"
    )

    return messages_to_send


def _summarize_old_messages(state: AgentState, llm) -> dict:
    """压缩旧对话并清理历史消息

    删除逻辑：删除所有历史消息，只保留最近 N 条纯对话 + 当前轮全部消息。
    被删除的历史消息分两类：
      1. 旧轮工具消息（AIMessage+tool_calls, ToolMessage）→ 直接删，跨轮无用
      2. 超出 max_recent 的旧纯对话 → 增量合并进 conversation_summary 后删除

    增量摘要：只对本次新超出的旧纯对话（old_plain）调用 LLM 做摘要，
    输入 = 已有摘要 + 新超出内容，输出 = 合并后的摘要。
    已被总结过的历史消息不会再次进入 LLM，避免重复计算。
    """
    all_messages = state.get("messages", [])
    max_recent = config.ROUTER_MAX_RECENT_MESSAGES

    # 找到当前轮的起始位置
    current_round_start = len(all_messages)
    for i in range(len(all_messages) - 1, -1, -1):
        if isinstance(all_messages[i], HumanMessage):
            current_round_start = i
            break

    history_messages = all_messages[:current_round_start]

    # 历史中的纯对话，只保留最近 max_recent 条
    plain_history = [m for m in history_messages if _is_plain_message(m)]
    recent_plain = plain_history[-max_recent:]
    old_plain = plain_history[:-max_recent]

    # 删除所有历史消息，只保留最近 N 条纯对话。
    # 用 id 判断保留（LangGraph 标准实践）—— 兼容序列化/反序列化场景，
    # 避免因对象引用变化或内容重复导致的误判。
    keep_ids = {m.id for m in recent_plain if m.id}
    removals = []
    for m in history_messages:
        if m.id and m.id in keep_ids:
            continue  # 保留
        if getattr(m, "id", None):
            removals.append(RemoveMessage(id=m.id))
        else:
            logger.warning(
                f"历史消息缺少 id，无法删除: {type(m).__name__} "
                f"(content前50字: {str(getattr(m, 'content', ''))[:50]})"
            )

    updates: dict = {}
    if removals:
        updates["messages"] = removals

    # 增量摘要：只总结本次新超出的旧纯对话，与已有摘要合并
    if old_plain:
        existing_summary = state.get("conversation_summary", "").strip()

        old_text_lines = []
        for msg in old_plain:
            role = "用户" if isinstance(msg, HumanMessage) else "助手"
            old_text_lines.append(f"{role}: {msg.content}")
        old_text = "\n".join(old_text_lines)

        # 真正增量：只把本次新增的旧对话压缩成一个独立片段，拼接在已有摘要后面。
        # 已有摘要内容不进入本次 LLM 调用 —— 早期事实永远不会被反复重压缩而丢失。
        summary_prompt = f"""你是一个对话摘要管理器。请将下面的旧对话压缩为一段独立的简短摘要片段。

本次新增的旧对话：
{old_text}

要求：
- 压缩为 30 字以内的摘要片段
- 只保留可能在未来追问中用到的信息：用户关注的主题、重要的事实和结论、未解决的问题
- 丢弃问候语、无关闲聊
- 直接返回摘要片段内容，不要加"摘要："等前缀"""

        response = llm.invoke([
            SystemMessage(content="你是一个对话摘要管理器，负责把旧对话压缩成简短片段。"),
            HumanMessage(content=summary_prompt),
        ])

        new_fragment = response.content.strip()

        # 拼接：已有摘要 + 新片段（用 | 分隔），并截断到合理长度防止无限膨胀
        existing_summary = state.get("conversation_summary", "").strip()
        merged = f"{existing_summary} | {new_fragment}".strip(" |")
        updates["conversation_summary"] = merged[:config.ROUTER_SUMMARY_MAX_CHARS]

    if not updates:
        return {}

    return updates


# ═══════════════════════════════════════════════════════════
# Router 节点
# ═══════════════════════════════════════════════════════════

def router_node(state: AgentState) -> dict:
    """Router：构建精简消息窗口 → LLM 决策"""

    iteration = state.get("iteration", 0)

    llm = _make_router_llm().bind_tools(TOOLS)

    messages_to_send = _build_router_messages(state)

    response = llm.invoke(messages_to_send)

    logger.info(
        f"Router: iteration={iteration+1}, "
        f"tool_calls={[tc['name'] for tc in (response.tool_calls or [])]}"
    )

    return {
        "messages": [response],
        "iteration": iteration + 1,
    }


# ═══════════════════════════════════════════════════════════
# 工具执行节点
# ═══════════════════════════════════════════════════════════

def retrieve_node(state: AgentState, retriever) -> dict:
    """从 messages 中找到自己未执行的 tool_call，执行检索，返回 ToolMessage"""

    tc_to_run = _find_pending_tool_call(state["messages"], "retrieve")
    if tc_to_run is None:
        return {}

    query = tc_to_run["args"].get("query", "")
    logger.info(f"Retrieve: '{query}'")

    docs = retriever.invoke(query)
    if docs:
        parts = []
        for i, doc in enumerate(docs, 1):
            src = doc.metadata.get("source", "")
            parts.append(f"[片段 {i}] (来源: {src})\n{doc.page_content}")
        context = "\n\n".join(parts)
    else:
        context = "（知识库中未找到相关内容）"

    return {
        "messages": [ToolMessage(content=context, tool_call_id=tc_to_run["id"], name="retrieve")]
    }


def web_search_node(state: AgentState) -> dict:
    """从 messages 中找到自己未执行的 tool_call，执行搜索，返回 ToolMessage"""

    tc_to_run = _find_pending_tool_call(state["messages"], "web_search")
    if tc_to_run is None:
        return {}

    query = tc_to_run["args"].get("query", "")
    logger.info(f"WebSearch: '{query}'")

    results = search_bing(query, max_results=3)
    context = format_search_results(results)

    return {
        "messages": [ToolMessage(content=context, tool_call_id=tc_to_run["id"], name="web_search")]
    }


def graph_query_node(state: AgentState) -> dict:
    """从 messages 中找到自己未执行的 tool_call，查图谱，返回 ToolMessage"""

    tc_to_run = _find_pending_tool_call(state["messages"], "graph_query")
    if tc_to_run is None:
        return {}

    entity = tc_to_run["args"].get("entity", "")
    logger.info(f"GraphQuery: '{entity}'")

    store = GraphStore()
    context = store.query_to_text(entity)

    return {
        "messages": [ToolMessage(content=context, tool_call_id=tc_to_run["id"], name="graph_query")]
    }


def _find_last_human_index(messages: list) -> int:
    """返回最后一条 HumanMessage 的索引，用于界定"当前轮"的范围"""
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            return i
    return 0


def _find_pending_tool_call(messages: list, tool_name: str) -> dict | None:
    """在当前轮（最后一条 HumanMessage 之后的 AIMessage）中寻找未执行的 tool_call

    只扫描当前轮 —— 不再是全局倒序，避免跨轮 tool_call 污染。
    """
    start_idx = _find_last_human_index(messages)

    # len(messages) - 1消息列表的最后一条，从后往前翻到本轮消息的开始索引处，-1代表倒叙，所以就是倒叙遍历本轮对话的消息
    for i in range(len(messages) - 1, start_idx - 1, -1):
        msg = messages[i]
        # 找到有tool_calls的AIMessage
        if isinstance(msg, AIMessage) and msg.tool_calls:
            # 遍历AIMessage的tool_calls，看是否有当前工具的调用请求
            for tc in msg.tool_calls:
                if tc.get("name") != tool_name:
                    continue
                # 检查是否已经执行过该工具调用，因为langchain要求toolmessage必须紧跟在aimessage后面，
                # 并且两者根据tool_call_id关联，toolmessage是在工具结点执行完之后添加的，没添加则说明该工具调用未执行过
                tc_id = tc.get("id", "")
                # 为true则表示找到该工具的toolmessage了，也就是该工具被调用过
                already_done = any(
                    isinstance(m, ToolMessage) and getattr(m, "tool_call_id", "") == tc_id
                    for m in messages[start_idx:]
                )
                # already_done为true就跳过，为false则返回该工具调用
                if not already_done:
                    return tc
    # 如果没有找到未执行的 tool_call，则返回 None，表明没有这个工具调用或者该工具调用已经执行过了
    return None


# ═══════════════════════════════════════════════════════════
# Generate 节点
# ═══════════════════════════════════════════════════════════

def generate_node(state: AgentState, llm) -> dict:
    """从 messages 中收集 ToolMessage → 构建 prompt → LLM 生成回答"""

    messages = state.get("messages", [])

    # 提取用户原始问题（最后一条 HumanMessage）
    user_question = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            user_question = msg.content
            break

    # 界定当前轮范围：最后一条 HumanMessage 之后的所有 message
    round_start = _find_last_human_index(messages)
    current_round_messages = messages[round_start:]
    current_tool_messages = [m for m in current_round_messages if isinstance(m, ToolMessage)]

    # 如果 router 直接给了文本回复且当前轮没有任何工具交互
    last_msg = messages[-1] if messages else None
    if isinstance(last_msg, AIMessage) and last_msg.content and not last_msg.tool_calls:
        if not current_tool_messages:
            logger.info("Generate: 透传 router 直接回答")
            return {"final_answer": last_msg.content}

    # 收集当前轮的 ToolMessage（同工具取最后一条）
    latest_per_tool: dict[str, str] = {}
    for msg in current_tool_messages:
        if msg.content:
            tc_id = getattr(msg, "tool_call_id", "")
            for prev in reversed(current_round_messages):
                if isinstance(prev, AIMessage) and prev.tool_calls:
                    for tc in prev.tool_calls:
                        if tc.get("id") == tc_id:
                            latest_per_tool[tc["name"]] = msg.content
                            break

    tool_results_text = ""
    for name, content in latest_per_tool.items():
        tool_results_text += f"\n### {name} 结果\n{content}\n"

    system_prompt = build_generate_prompt(user_question, tool_results_text)

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_question),
    ])

    logger.info("Generate: 完成")
    return {
        "final_answer": response.content,
        "messages": [response],
    }
