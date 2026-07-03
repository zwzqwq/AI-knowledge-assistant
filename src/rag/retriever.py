"""
检索器模块 —— 封装 ChromaDB 向量存储和检索策略
"""
import os
from langchain_chroma import Chroma

from src.config import config, logger
from src.rag.embedder import EmbeddingManager


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
            logger.info(f"加载已有向量库: {config.CHROMA_DB_DIR}")
            self._vectorstore = Chroma(
                embedding_function=self.embeddings,
                persist_directory=config.CHROMA_DB_DIR,
            )
        return self._vectorstore

    def create(self, chunks: list) -> Chroma:
        """新建向量库（会覆盖已有）"""
        logger.info(f"创建新向量库，{len(chunks)} 个切片")
        self._vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=config.CHROMA_DB_DIR,
        )
        return self._vectorstore

    def add(self, chunks: list):
        """向已有向量库追加文档"""
        vs = self._get_or_load()
        if vs is None:
            logger.info("向量库不存在，自动创建")
            return self.create(chunks)
        logger.info(f"追加 {len(chunks)} 个切片到向量库")
        vs.add_documents(chunks)
        return vs

    def get_retriever(self, search_type: str | None = None):
        """
        返回 LangChain Retriever 对象（带检索配置）

        参数:
          search_type: "similarity" 或 "mmr"，None 则使用 config 默认值
        """
        vs = self._get_or_load()
        if vs is None:
            raise RuntimeError("向量库未初始化，请先创建或上传文档")

        effective_type = search_type or config.RETRIEVER_SEARCH_TYPE

        # 检索配置，根据检索策略设置不同参数
        search_kwargs = {"k": config.RETRIEVER_K}
        if effective_type == "mmr":
            search_kwargs["fetch_k"] = config.RETRIEVER_K * 4  # 候选文档数
            search_kwargs["lambda_mult"] = 0.7  # 相似度和多样性的平衡

        return vs.as_retriever(
            search_type=effective_type,
            search_kwargs=search_kwargs,
        )
