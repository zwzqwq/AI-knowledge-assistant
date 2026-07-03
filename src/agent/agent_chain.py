"""
Agent Chain —— 智能路由：RAG 检索不到时降级到 LLM 自身知识

决策逻辑：
  1. 先走 RAG Chain，用知识库内容回答
  2. 如果知识库能回答 → 返回结果，标注 "knowledge_base"
  3. 如果知识库回答不了（LLM 说"文档中没有提到"）
     → 直接让 LLM 用自己的知识回答，但前置声明"知识库中未找到相关信息"
"""
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from src.config import config, logger
from src.rag.chain import build_chain


FALLBACK_SYSTEM_PROMPT = """你是一个知识助手。用户问题在本地知识库中没有找到相关信息，请你根据自己的知识回答。

<对话历史>
{history}
</对话历史>

回答规则：
1. 直接回答用户问题，不要编造任何信息
2. 不要提到"互联网搜索"、"来源"、"搜索"等词汇——你的回答基于自身知识，不是搜索
3. 不要添加 "[来源 1]" 等脚注标记，这不是联网搜索的结果
4. 如果不确定或不知道，直接说"不清楚"
5. 回答简洁"""

FALLBACK_PROMPT = ChatPromptTemplate.from_messages([
    ("system", FALLBACK_SYSTEM_PROMPT),
    ("user", "{input}"),
])


class AgentChain:
    """
    Agent 式智能路由 Chain

    使用方式:
      chain.invoke({"history": "...", "input": "用户问题"})
      → {"answer": "回答文本", "source": "knowledge_base" | "llm"}
    """

    def __init__(self, retriever, llm=None):
        self.retriever = retriever
        self.rer_chain = build_chain(retriever, llm)
        self.llm = llm or ChatOpenAI(
            model=config.LLM_MODEL,
            api_key=config.LLM_API_KEY,
            base_url=config.LLM_BASE_URL,
        )
        logger.info("AgentChain 初始化完成")

    def invoke(self, inputs: dict) -> dict:
        """
        执行智能路由

        参数:
          inputs: {"history": "...", "input": "用户问题"}

        返回:
          {"answer": str, "source": "knowledge_base" | "llm"}
        """
        query = inputs["input"]

        # ─── 第 1 步：先走 RAG，看知识库能不能回答 ────────────
        result = self.rer_chain.invoke(inputs)
        answer = result.content

        # 判断 RAG 是否真的给出了有用回答
        cant_answer = any(marker in answer for marker in [
            "文档中没有提到",
            "未检索到相关文档",
            "文档中没有相关",
            "没有提到",
        ])

        if not cant_answer:
            # RAG 能回答 → 直接用
            return {"answer": answer, "source": "knowledge_base"}

        # ─── 第 2 步：RAG 回答不了，降级到 LLM 自身知识 ────
        logger.info("RAG 无法回答，降级到 LLM 自身知识")
        response = self.llm.invoke(FALLBACK_PROMPT.format_prompt(
            history=inputs.get("history", ""),
            input=query,
        ))

        return {
            "answer": f"⚠️ 知识库中未找到相关信息，以下基于 AI 自身知识回答：\n\n{response.content}",
            "source": "llm",
        }
