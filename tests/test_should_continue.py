from src.agent.edges import should_continue
from langchain_core.messages import AIMessage,ToolMessage
from tests.helpers import make_tool_call, make_state
import pytest


def test_should_continue_has_tool_call():
    """
    测试 should_continue 节点返回 continue 的路径。
    当当前轮还有未执行的 tool_call 时，返回 continue。
    """
    ai_message = AIMessage(content="你好", tool_calls=[make_tool_call("tc1", "retrieve")])
    state=make_state([ai_message])
    assert should_continue(state) == "pick_next_tool"

def test_should_continue_no_tool_call():
    """
    测试 should_continue 节点返回 continue 的路径。
    当当前轮没有未执行的 tool_call 时，返回 continue。
    """
    state1=make_state([])
    assert should_continue(state1) == "router"

    ai_message = AIMessage(content="你好", tool_calls=[make_tool_call("tc1", "retrieve")])
    tool_message = ToolMessage(content="你好", tool_call_id="tc1")
    state2 = make_state([ai_message,tool_message])
    assert should_continue(state2) == "router"
    
    