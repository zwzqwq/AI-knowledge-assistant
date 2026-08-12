"""
HybridRetriever —— 混合编排器（"指挥"）

功能：把两条召回路（向量看语义 + BM25 认死理）的结果合并去重，
      交给 Reranker 精排，返回最终 top-k 精准文档。
      自己不召回、不打分，只做"合并 + 去重 + 编排顺序"。

在整体链路中的位置（召回后 → 精排前）：
  query
    ├─ 向量检索（语义路）→ top-N
    ├─ BM25（认死理）    → top-N
    │
    ▼
   HybridRetriever（本文件）→ 合并去重 → 候选池
    │
    ▼
   Reranker.rerank → 打分重排 → 截断 top-k   ← 调用已有组件
    │
    ▼
   top-k 精准文档 → LLM 生成答案

下游契约（零改动）：
  retrieve_node 只调 retriever.invoke(query) 返回 list[Document]，
  每篇有 page_content + metadata["source"]。本类保证这三样即可。

为什么继承 BaseRetriever 而不是普通类：
  LangChain 检索器的标准基类。实现 _get_relevant_documents 即可：
    1. invoke() 由基类自动提供（下游照常调用，零改动）
    2. 天然兼容 LCEL 管道（| retriever |）——旧 chain.py 若重新启用也能接
    3. 面试谈资：自定义检索器要接入 LangChain 生态，继承 BaseRetriever 是标准做法

为什么不用 LangChain EnsembleRetriever：
  它做 score fusion（分数相加），解决不了向量分和 BM25 分量纲不可比，
  也接不了 cross-encoder 精排。我们是 two-stage（召回 + 精排），
  更接近工业 RAG 标准形态。
"""
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from src.config import logger


class HybridRetriever(BaseRetriever):
    """混合检索编排器：合并两路召回 → 去重 → Rerank 精排"""

    def __init__(self, vector_retriever, bm25_retriever, reranker):
        """
        Args:
            vector_retriever: 向量检索器（LangChain retriever，.invoke(query) -> list[Document]）
            bm25_retriever: BM25 检索器（.invoke(query) -> list[Document]）
            reranker: 精排器（.rerank(query, candidates) -> list[Document]）
        """
        super().__init__()
        self._vector = vector_retriever
        self._bm25 = bm25_retriever
        self._reranker = reranker

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager=None,
        **kwargs,
    ) -> list[Document]:
        """
        混合检索入口（BaseRetriever 抽象方法）：并行召回 → 合并去重 → Rerank 精排 → top-k。

        下游调 retriever.invoke(query) 时，BaseRetriever 会自动调用本方法并传入 run_manager。

        Args:
            query: 用户问题
            run_manager: 由 BaseRetriever.invoke 传入的回调管理器（本类不需要，忽略）

        Returns:
            list[Document]: 精排后的 top-k 精准文档（分数从高到低）
        """
        # ① 两条召回路（各自内部已取 top-N，N = config.RETRIEVER_CANDIDATES）
        vector_docs = self._vector.invoke(query) if self._vector else []
        bm25_docs = self._bm25.invoke(query) if self._bm25 else []

        # ② 合并候选池：按 page_content 去重（保序）
        #    dict 以文本为 key → 同一切片只留一次；同文件不同页文本不同 → 都保留
        merged: dict[str, Document] = {}
        for doc in vector_docs + bm25_docs:
            if doc.page_content not in merged:
                merged[doc.page_content] = doc

        candidates = list(merged.values())

        # ③ 空候选池保护：两路都没捞到 → 直接返回空
        if not candidates:
            logger.warning("HybridRetriever: 两条召回路均无结果")
            return []

        logger.info(
            f"HybridRetriever: 向量路 {len(vector_docs)} 条 + BM25 路 {len(bm25_docs)} 条"
            f" → 去重后候选 {len(candidates)} 条"
        )

        # ④ Rerank 精排：打分 → 排序 → 截断 top-k
        return self._reranker.rerank(query, candidates)
