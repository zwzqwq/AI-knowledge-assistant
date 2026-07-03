"""
RAG Chain 组装 ★ 核心模块

这是整个项目的中枢——把检索 + Prompt + LLM 用 LCEL 管道串联起来。

LCEL 的优势：
  检索结果变化 → Chain 自动重新检索上下文
  不需要手动管理中间状态

管道流程：
  {"history": ..., "input": "用户问题"}
    → 检索: input → Retriever → format_docs → "context" 字符串
    → 组装: Prompt(历史 + 上下文 + 问题)
    → 生成: LLM 返回回答
"""

from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.config import config, logger


# ─── Prompt 模板 ───────────────────────────────────────────
# 面试常考：你为什么这样设计 Prompt？
#   1. 先给身份（"你是一个..."）—— 约束回答风格
#   2. 对话历史在前，上下文在后 —— 让 LLM 先理解对话背景
#   3. 明确 "如果上下文中没有相关信息，就说不知道" —— 减少幻觉
#   4. 上下文用分隔符包裹 —— 防止 prompt injection
SYSTEM_PROMPT = """你是一个知识助手，根据提供的文档内容回答用户问题。

<对话历史>
{history}
</对话历史>

<参考文档>
{context}
</参考文档>

回答规则：
1. 优先使用参考文档中的信息回答
2. 如果文档中没有相关信息，诚实地说"文档中没有提到这方面的内容"
3. 回答简洁，不要编造文档中没有的内容"""

PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("user", "{input}"),
])


# ─── 工具函数 ──────────────────────────────────────────────

def format_docs(docs: list) -> str:
    """把检索到的文档片段拼接成一个字符串"""
    if not docs:
        return "（未检索到相关文档）"
    parts = []
    for i, doc in enumerate(docs, 1):
        parts.append(f"[片段 {i}]\n{doc.page_content}")
    return "\n\n".join(parts)


def build_chain(retriever, llm=None):
    """
    构建 RAG Chain

    参数:
      retriever: LangChain Retriever 对象（来自 retriever.Retriever.get_retriever()）
      llm: 可选，共享的 LLM 实例；不传则自动创建

    返回:
      可调用的 LCEL Runnable 对象
      invoke({"history": "...", "input": "问题"}) → AIMessage
    """
    if llm is None:
        llm = ChatOpenAI(
            model=config.LLM_MODEL,
            api_key=config.LLM_API_KEY,
            base_url=config.LLM_BASE_URL,
        )

    # ─── LCEL 管道 ★ 理解这个就等于理解了 RAG 链 ─────────
    chain = (
        {
            # 管道并行执行三个分支：
            "history": RunnableLambda(lambda d: d["history"]),
            "context": (
                RunnableLambda(lambda d: d["input"])
                | retriever                    # "问题" → [Document, ...]
                | RunnableLambda(format_docs)  # [Document, ...] → "文本..."
            ),
            "input": RunnableLambda(lambda d: d["input"]),
        }
        | PROMPT_TEMPLATE  # dict → ChatPromptValue
        | llm              # ChatPromptValue → AIMessage
    )

    logger.info("RAG Chain 构建完成")
    return chain
