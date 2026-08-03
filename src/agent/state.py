"""
AgentState — 用 LangGraph 原生 add_messages reducer

核心设计：
  state["messages"] 是唯一数据源。
  每个节点只需返回 {"messages": [新消息]}，由 add_messages 自动追加。
  不再有平行的 tool_calls 数组或 context 字段。
"""

from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    iteration: int
    final_answer: str
    conversation_summary: str  # 旧对话的压缩摘要，替代被移除的历史消息
