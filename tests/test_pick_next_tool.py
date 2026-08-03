from unittest.mock import patch,Mock
import pytest
from src.agent.edges import pick_next_tool
from langchain_core.messages import AIMessage, ToolMessage, BaseMessage,HumanMessage
from tests.helpers import make_tool_call, make_state

def test_pick_next_tool_generate1():
    """
    测试 pick_next_tool 节点返回generate的路径。
    工具都被调用，返回generate。
    """
    ai_message = AIMessage(content="你好", tool_calls=[make_tool_call("tc1", "retrieve")])
    tool_message = ToolMessage(content="你好", tool_call_id="tc1")
    state = make_state([ai_message,tool_message])
    assert pick_next_tool(state) == "generate"

def test_pick_next_tool_generate2():
    """
    测试 pick_next_tool 节点返回generate的路径。
    无工具调用时和只有HumanMessage时，返回generate。
    """
    state = make_state()
    assert pick_next_tool(state) == "generate"

    humenMessage=HumanMessage(content="你好")
    state= make_state([humenMessage])
    assert pick_next_tool(state) == "generate"

def test_pick_next_tool_tool_name():
    """
    测试 pick_next_tool 节点返回工具名称的路径。
    存在未被调用的工具时，返回工具名称。
    """
    ai_message = AIMessage(content="你好", tool_calls=[make_tool_call("tc1", "retrieve")])
    state=make_state([ai_message])
    assert pick_next_tool(state) == "retrieve"