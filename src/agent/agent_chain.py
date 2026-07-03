"""
Agent Chain —— 智能路由：RAG 检索不到时自动联网搜索

决策逻辑：
  1. 先做 RAG 检索，看上下文的长度和信息量
  2. 如果检索到的文档片段充分（总长度 > 100 字符）→ 用 RAG 链回答
  3. 如果检索到的文档太短或为空 → 自动切换为联网搜索，基于搜索结果回答
"""
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from src.config import config, logger
from src.rag.chain import build_chain
from src.agent.web_search import search_bing, format_search_results

# ─── 联网搜索专用 Prompt ───────────────────────────────────

WEB_SYSTEM_PROMPT = """你是一个知识助手。用户问题在本地知识库中没有找到相关信息，以下是从互联网搜索到的相关内容：

<对话历史>
{history}
</对话历史>

<网络搜索结果>
{context}
</网络搜索结果>

回答规则：
1. 基于搜索结果回答用户问题
2. 如果搜索结果也不相关，诚实地说"未能找到相关信息"
3. 回答时标注信息来源（使用 [来源 1] 等标记）
4. 回答简洁"""

WEB_PROMPT = ChatPromptTemplate.from_messages([
    ("system", WEB_SYSTEM_PROMPT),
    ("user", "{input}"),
])

# 当检索到的总内容少于这个字符数时，触发联网搜索
MIN_CONTEXT_LENGTH = 80


class AgentChain:
    """
    Agent 式智能路由 Chain

    使用方式（和普通 Chain 一样）:
      chain.invoke({"history": "...", "input": "用户问题"})
      → {"answer": "回答文本", "source": "knowledge_base" | "web_search"}
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
          {"answer": str, "source": "knowledge_base" | "web_search"}
        """
        query = inputs["input"]

        # ─── 第 1 步：先走 RAG Chain ────────────────────────
        result = self.rer_chain.invoke(inputs)
        answer = result.content

        # ─── 第 2 步：判断 RAG 是否真能回答 ──────────────────
        # 如果 LLM 说"文档中没有提到"之类的话，说明知识库不够用
        cant_answer = any(marker in answer for marker in [
            "文档中没有提到",
            "未检索到相关文档",
            "文档中没有相关",
            "没有提到",
        ])

        if not cant_answer:
            # RAG 能回答 → 直接用
            return {"answer": answer, "source": "knowledge_base"}

        # ─── 第 3 步：联网搜索回退 ─────────────────────────
        logger.info("RAG 无法回答，触发联网搜索")
        web_results = search_bing(query)
        web_context = format_search_results(web_results)

        web_prompt_input = {
            "history": inputs.get("history", ""),
            "context": web_context,
            "input": query,
        }
        response = self.llm.invoke(WEB_PROMPT.format_prompt(**web_prompt_input))

        return {"answer": response.content, "source": "web_search"}
