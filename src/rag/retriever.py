"""
检索器模块 —— 封装 ChromaDB 向量存储和检索策略
"""
import os
from langchain_chroma import Chroma

from src.config import config, logger
from src.rag.embedder import EmbeddingManager

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
        logger.info(f"追加前向量库数量: {vs._collection.count()}")
        vs.add_documents(chunks)
        logger.info(f"追加 {len(chunks)} 个切片到向量库，追加后数量: {vs._collection.count()}")
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
