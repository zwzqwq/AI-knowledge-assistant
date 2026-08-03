from langchain_core.messages import BaseMessage

def make_tool_call(tc_id: str, name: str, **args) -> dict:
    """生成一条 tool_call 字典"""
    return {"id": tc_id, "name": name, "args": args}
def make_state(messages: list[BaseMessage] | None = None, iteration: int = 0) -> dict:
    """把消息列表包成 AgentState 所需的 dict"""
    return {"messages": messages if messages else [], "iteration": iteration}