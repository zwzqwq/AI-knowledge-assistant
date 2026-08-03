from langchain_core.messages import AIMessage,ToolMessage
from tests.helpers import make_tool_call, make_state
import pytest
from src.agent.prompts import build_generate_prompt

def test_build_generate_prompt():
    # Test case 1: Single tool call
    assert build_generate_prompt("你好","")=="""你是一个知识库助手。
（未使用搜索工具）

用户的问题是：「你好」
请根据你的自身知识回答用户的问题。如果不知道就说不知道，不要编造。"""

    assert build_generate_prompt("mysql对比redis","aaa")=="""用户的问题是：「mysql对比redis」

以下是检索到的信息（仅供参考，与问题无关的内容请忽略）：

aaa

请按以下结构回答：
1. 分别说明问题中涉及的各方——各自的核心特征是什么
2. 做对比总结——指出各方在核心维度上的本质区别
3. 检索结果中哪一方信息不足，就用你自己的知识补充，并诚实说明

注意：
- 不要写成百科条目，直接回答问题
- 对比是服务于用户问题的，不是为对比而对比"""
    
    assert build_generate_prompt("mysql详解","aaa")=="""用户的问题是：「mysql详解」

以下是工具检索到的信息（这些信息可能相关，也可能不相关）：
aaa

回答要求：
- 先充分回答用户的问题本身：给出清晰的定义，说明核心特征或原理，列举关键要点
- 回答完核心问题后，如果检索结果中有相关的延伸内容，可以用类比或联想的方式自然地过渡过去，不需要拘泥于单一答案
- 如果检索结果与问题完全无关，直接忽略它们，用你自己的知识充分回答
- 如果不知道就说不知道，不要编造"""
