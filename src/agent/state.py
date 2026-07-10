"""
LangGraph Agent 状态定义

AgentState 是 LangGraph 的核心——所有节点共享这个状态对象，
每个节点可以读/写其中的字段。

状态流转示例：
  1. router 节点：读 messages，写 tool_calls
  2. retrieve 节点：读 tool_calls，写 context（ToolMessage 回填到 messages）
  3. generate 节点：读 messages + context，写 final_answer

messages 使用 LangChain BaseMessage 对象列表（非 dict），
保证 tool_call_id 等元数据不丢失。
"""
from typing import TypedDict, Optional
from langchain_core.messages import BaseMessage


class ToolCall(TypedDict):
    """一次工具调用"""
    name: str          # 工具名："retrieve" | "web_search" | "graph_query"
    args: dict         # 参数，如 {"query": "MySQL 索引优化"}
    result: str        # 工具返回的结果
    id: str            # tool_call_id，来自 LLM 响应，用于匹配 ToolMessage


class AgentState(TypedDict):
    """
    Agent 的全局状态，在 Graph 节点间流转

    每个字段的含义：
      messages      — 对话历史（LangChain BaseMessage 对象列表）
      tool_calls    — LLM 决定要调用的工具列表
      context       — 本轮检索/搜索拿到的文档片段（供 generate_node 读取）
      final_answer  — 最终回答，填了就意味着该结束了
      iteration     — 当前循环次数，防止无限循环
    """
    messages: list[BaseMessage]
    tool_calls: list[ToolCall]
    context: str
    final_answer: str
    iteration: int
