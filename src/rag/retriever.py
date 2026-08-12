"""
检索器模块 —— 封装 ChromaDB 向量存储和检索策略
"""
import os
from langchain_chroma import Chroma

from src.config import config, logger
from src.rag.embedder import EmbeddingManager
from src.rag.bm25_retriever import BM25Retriever
from src.rag.reranker import Reranker
from src.rag.ensemble import HybridRetriever

# ChromaDB内部数据结构
#{
#     "ids": ["id1", "id2", "id3"],           # 每个切片的唯一ID
#     "metadatas": [                           # ← 这就是你的 all_metadatas
#         {"source": "a.txt", "page": 1},      # 切片1的元数据（只有属性）
#         {"source": "b.txt", "page": 2},      # 切片2的元数据
#         {"source": "a.txt", "page": 3}       # 切片3的元数据
#     ],
#     "documents": [                           # 注意！这里是独立的！
#         "这是切片1的实际文本内容...",         # 切片1的内容
#         "这是切片2的实际文本内容...",         # 切片2的内容
#         "这是切片3的实际文本内容..."          # 切片3的内容
#     ],
#     "embeddings": [                          # 可选，向量数据（通常不取，太大）
#         [0.1, 0.2, ...],                     # 切片1的向量
#         ...
#     ]
# }

class Retriever:
    """管理向量库的创建、加载、追加，以及多种检索策略"""

    def __init__(self):
        self.embeddings = EmbeddingManager().get()
        self._vectorstore: Chroma | None = None

    @property
    def exists(self) -> bool:
        """检查本地向量库是否存在"""
        return os.path.exists(config.CHROMA_DB_DIR) and os.listdir(config.CHROMA_DB_DIR)

    def _get_or_load(self) -> Chroma:
        """懒加载：已有则加载，否则返回 None"""
        if self._vectorstore is None and self.exists:
            logger.info(f"加载已有向量库到当前retriever: {config.CHROMA_DB_DIR}")
            self._vectorstore = Chroma(
                embedding_function=self.embeddings,
                persist_directory=config.CHROMA_DB_DIR,
            )
        return self._vectorstore

    @staticmethod
    def _unique_chunks(chunks: list, existing: set | None = None) -> tuple[list, int]:
        """按 page_content 去重切片，返回 (去重后列表, 跳过的数量)

        历史教训：向量库曾因重复导入积累 96% 重复切片（422 篇只剩 51 篇唯一）。
        这里同时做两层去重：
          - existing 里的内容（库中已有）→ 跳过
          - 本批 chunks 内部重复 → 跳过
        返回值：
          - 去重后的列表 ： unique
          - 跳过的数量：len(chunks) - len(unique)
        """
        seen = set(existing or [])
        unique = []
        for c in chunks:
            if c.page_content not in seen:
                unique.append(c)
                seen.add(c.page_content)
        return unique, len(chunks) - len(unique)

    def create(self, chunks: list) -> Chroma:
        """新建向量库（会覆盖已有）"""
        # 入参自身可能带重复切片（同一文档切了多份一样的），先按内容去重
        unique_chunks, skipped = self._unique_chunks(chunks)
        if skipped:
            logger.info(f"创建前去重：剔除 {skipped} 个重复切片，剩 {len(unique_chunks)} 个")
        logger.info(f"创建新向量库，{len(unique_chunks)} 个切片")
        self._vectorstore = Chroma.from_documents(
            documents=unique_chunks,
            embedding=self.embeddings,
            persist_directory=config.CHROMA_DB_DIR,
        )
        return self._vectorstore

    def add(self, chunks: list):
        """向已有向量库追加文档（自动去重，防重复导入污染）"""
        vs = self._get_or_load()
        if vs is None:
            logger.info("向量库不存在，自动创建")
            return self.create(chunks)

        # 入库去重：跳过库中内容已存在的切片（防止上次的 96% 重复事故重演）
        existing = set(vs._collection.get()["documents"])
        new_chunks, skipped = self._unique_chunks(chunks, existing)
        if skipped:
            logger.info(f"入库去重：跳过 {skipped} 个内容已存在的切片")
        if not new_chunks:
            logger.info("全部切片已存在，无需追加")
            return vs

        logger.info(f"追加前向量库数量: {vs._collection.count()}")
        vs.add_documents(new_chunks)
        logger.info(f"追加 {len(new_chunks)} 个切片到向量库，追加后数量: {vs._collection.count()}")
        return vs

    def delete_by_source(self, source_name: str) -> int:
        """按文档名删除向量库中该文档的所有切片，返回删除数量

        ChromaDB 支持 where 过滤删除：collection.delete(where={"source": 文件名})
        切片入库时 source 字段就是文档名（见 loader.py 的 source_name），
        所以这里能精确删掉某个文档的所有切片。
        """
        vs = self._get_or_load()
        if vs is None:
            logger.info(f"删除文档 '{source_name}': 向量库不存在，无需删除")
            return 0
        collection = vs._collection
        before = collection.count()
        collection.delete(where={"source": source_name})
        after = collection.count()
        deleted = before - after
        logger.info(f"删除文档 '{source_name}': 移除 {deleted} 个切片（剩余 {after}）")
        return deleted

    def get_stats(self) -> dict:
        """返回向量库统计：切片总数、文档数"""
        vs = self._get_or_load()
        if vs is None:
            return {"total_chunks": 0, "total_documents": 0}
        collection = vs._collection
        total_chunks = collection.count()
        # ChromaDB 的 metadata 里 source 字段就是文档名，去重得到文档数
        all_metadatas = collection.get()["metadatas"]
        sources = set()
        for meta in all_metadatas:
            if meta and "source" in meta:#判断元数据meta是否为空，不为空则判断文件名source是否为空
                sources.add(meta["source"])
        return {
            "total_chunks": total_chunks,
            "total_documents": len(sources),
        }

    def get_retriever(self, search_type: str | None = None):
        """
        返回 LangChain Retriever 对象（带检索配置）

        参数:
          search_type: "similarity" 或 "mmr"，None 则使用 config 默认值
        """
        vs = self._get_or_load()
        if vs is None:
            raise RuntimeError("向量库未初始化，请先创建或上传文档")

        #设置检索策略
        effective_type = search_type or config.RETRIEVER_SEARCH_TYPE

        # 检索配置，根据检索策略设置不同参数
        #定义一个字典，对应as_retriever的search_kwargs的参数，对应键值对的键值也对应search_kwargs的键值
        search_kwargs = {"k": config.RETRIEVER_K}
        if effective_type == "mmr":
            search_kwargs["fetch_k"] = config.RETRIEVER_K * 4  # 候选文档数
            search_kwargs["lambda_mult"] = 0.7  # 相似度和多样性的平衡

        return vs.as_retriever(
            search_type=effective_type,
            search_kwargs=search_kwargs,
        )

    def get_hybrid_retriever(self) -> HybridRetriever:
        """
        返回混合检索器（RAG 进阶后的主检索入口）

        组装三件套：
          ① 向量检索器（语义路）→ 召回 N 条
          ② BM25Retriever（认死理路）→ 召回 N 条
          ③ Reranker（精排）→ 对合并去重后的候选池打分，截断 top-k

        召回阶段要"广"：N = config.RETRIEVER_CANDIDATES（默认 10），
        精排阶段要"准"：k = config.RETRIEVER_RERANK_TOP_K（默认 3），
        N > k 才能给 Rerank 足够的挑选空间。
        """
        vs = self._get_or_load()
        if vs is None:
            raise RuntimeError("向量库未初始化，请先创建或上传文档")

        # ① 向量路：纯相似度召回，N 大一点捞得全（mmr 的多样性权衡已由 Rerank 接管）
        vector_retriever = vs.as_retriever(
            search_type=config.RETRIEVER_SEARCH_TYPE,
            search_kwargs={"k": config.RETRIEVER_CANDIDATES},
        )

        # ② BM25 路：复用同一个向量库拉全量文档建词级索引（懒加载，首次查询才建）
        bm25_retriever = BM25Retriever(
            vectorstore=vs,
            top_n=config.RETRIEVER_CANDIDATES,
        )

        # ③ 精排器：top_k 默认取 config.RETRIEVER_RERANK_TOP_K
        reranker = Reranker()

        # ④ 组装：合并去重 + 精排都在 HybridRetriever 内部完成
        return HybridRetriever(vector_retriever, bm25_retriever, reranker)
