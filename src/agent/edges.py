"""条件边函数 —— 纯函数，只读 state，不做任何修改"""

from langchain_core.messages import AIMessage, ToolMessage, HumanMessage
from .state import AgentState

MAX_ITERATIONS = 12


def _find_last_human_index(messages: list) -> int:
    """返回最后一条 HumanMessage 的索引"""
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            return i
    return 0


def route_after_router(state: AgentState) -> str:
    """Router 输出后：有 tool_calls → 执行工具，无 → generate"""

    if state.get("iteration", 0) >= MAX_ITERATIONS:
        return "generate"

    messages = state.get("messages", [])
    if not messages:
        return "generate"

    last_msg = messages[-1]
    if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
        return "pick_next_tool"

    return "generate"


def pick_next_tool(state: AgentState) -> str:
    """在当前轮中，返回下一个未执行的工具节点名"""

    messages = state.get("messages", [])
    start_idx = _find_last_human_index(messages)

    for i in range(len(messages) - 1, start_idx - 1, -1):
        msg = messages[i]
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                tc_id = tc.get("id", "")
                has_response = any(
                    isinstance(m, ToolMessage) and getattr(m, "tool_call_id", "") == tc_id
                    for m in messages[start_idx:]
                )
                if not has_response:
                    name = tc.get("name", "")
                    if name in ("retrieve", "web_search", "graph_query"):
                        return name

    return "generate"


def should_continue(state: AgentState) -> str:
    """工具执行后：当前轮还有未执行的 tool_call → 继续挑，否则回 router"""

    messages = state.get("messages", [])
    start_idx = _find_last_human_index(messages)

    for i in range(len(messages) - 1, start_idx - 1, -1):
        msg = messages[i]
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                tc_id = tc.get("id", "")
                has_response = any(
                    isinstance(m, ToolMessage) and getattr(m, "tool_call_id", "") == tc_id
                    for m in messages[start_idx:]
                )
                if not has_response:
                    return "pick_next_tool"

    return "router"
