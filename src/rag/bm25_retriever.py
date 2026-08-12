"""
BM25 检索器 —— "认死理"召回器

功能：从向量库拉全量文档 → jieba 分词 → rank_bm25 建索引 → 按 query 检索 top-N。
只认字面匹配（不看语义），负责关键词精确召回，是混合检索的第二条路。

在整体链路中的位置（召回阶段的第二条路）：
  query
    ├─ 向量检索（语义路）   → top-N
    └─ BM25Retriever（本文件）→ top-N   ← 字面匹配路
           │
           ▼
       合并候选池 → Rerank 精排 → 截断取 top-k → LLM

依赖说明（为什么这么选）：
  - rank_bm25：BM25 算法成熟实现（BM25Okapi 三行建索引）
  - jieba：中文分词。BM25 是词级算法，中文必须分词才能匹配
"""
import os
from rank_bm25 import BM25Okapi
import jieba

from langchain_core.documents import Document

from src.config import config, logger


class BM25Retriever:
    """基于 BM25 的关键词检索器"""

    def __init__(self, vectorstore, top_n: int = 10):
        """
        Args:
            vectorstore: Chroma 向量库对象（已有，用于拉取全量文档）
            top_n: 检索返回的候选数量 N（注意：N 应 > 最终精排的 k）
        """
        self._vectorstore = vectorstore
        self._top_n = top_n
        # 懒加载：首次调用才建索引
        self._bm25: BM25Okapi | None = None
        self._documents: list[str] = []
        self._metadatas: list[dict] = []

    def _build_index(self):
        """从向量库拉全量文档，jieba 分词，建立 BM25 索引"""
        if self._bm25 is not None:
            return

        if self._vectorstore is None:
            raise RuntimeError("BM25Retriever: 向量库为空，无法建立索引")

        logger.info("BM25Retriever: 从向量库拉取全量文档建立索引...")
        collection = self._vectorstore._collection
        data = collection.get()  # {"ids", "documents", "metadatas"}
        self._documents = data.get("documents", []) or []
        self._metadatas = data.get("metadatas", []) or []

        if not self._documents:
            logger.warning("BM25Retriever: 向量库中没有文档，索引为空")
            self._bm25 = BM25Okapi([])
            return

        tokenized_docs = [jieba.lcut(doc) for doc in self._documents]
        self._bm25 = BM25Okapi(tokenized_docs)
        logger.info(f"BM25Retriever: 索引建立完成，共 {len(self._documents)} 篇文档")

    def _rebuild(self):
        """强制重建索引（文档增删后调用）"""
        self._bm25 = None
        self._build_index()

    def add_documents(self, chunks: list):
        """新增文档后重建索引（简化处理：全量重建，文档量小够用）"""
        if chunks:
            self._rebuild()

    def invoke(self, query: str) -> list[Document]:
        """按 query 检索，返回 top-N 条 Document"""
        self._build_index()

        # 空 query 保护：分词为空时直接返回空
        tokens = jieba.lcut(query.strip())
        if not tokens or self._bm25 is None:
            return []

        scores = self._bm25.get_scores(tokens)

        # 取分数最高的 top_n 个下标
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[: self._top_n]

        # 只保留有实际分数的结果（BM25 无命中的分数为 0）
        results = []
        for i in top_indices:
            if scores[i] <= 0:
                continue
            results.append(
                Document(
                    page_content=self._documents[i],
                    metadata=(
                        self._metadatas[i] if i < len(self._metadatas) else {}
                    ),
                )
            )
        return results
