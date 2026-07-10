"""
Agent 工具定义 —— 把能力和参数签名告诉 LLM

LangChain 的 @tool 装饰器会自动生成 JSON Schema，
LLM 通过 Function Calling 协议看到这些 Schema，决定调用哪个。

用法：
  tools = [retrieve, web_search]
  llm_with_tools = llm.bind_tools(tools)
  response = llm_with_tools.invoke(messages)
  # response.tool_calls 包含 LLM 决定调用的工具和参数
"""
from langchain_core.tools import tool


@tool
def retrieve(query: str) -> str:
    """
    从本地知识库中检索与 query 相关的文档片段。
    适用场景：用户问的问题可能在已上传的文档中有答案。

    参数:
        query: 搜索关键词或自然语言问题
    返回:
        检索到的相关文档片段（可能为空）
    """
    # 实际实现由 graph.py 的 retrieve_node 完成
    # 这里只是声明签名给 LLM 看
    return ""


@tool
def web_search(query: str) -> str:
    """
    在互联网上搜索与 query 相关的信息。
    适用场景：本地知识库中找不到答案，需要实时信息或补充知识。

    参数:
        query: 搜索关键词
    返回:
        搜索结果的标题和摘要
    """
    return ""


# Phase 3 会新增这个
@tool
def graph_query(entity: str) -> str:
    """
    在知识图谱中查询与 entity 相关的实体和关系。
    适用场景：用户想知道某个概念/技术/实体与其他概念之间的关系。

    参数:
        entity: 要查询的实体名称，如 "InnoDB"、"事务"、"MySQL"
    返回:
        与该实体直接关联的其他实体及其关系描述
    """
    return ""
