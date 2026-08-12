"""
Reranker —— "海选后的小模型评委"

功能：接收候选文档池（已合并去重），用 bge-reranker cross-encoder 模型
      逐对打分（query + 每篇文档拼一起过模型）→ 按分数排序 → 截断取 top-k。
      只负责"精排"，不负责"合并"（那是 ensemble.py 的活）。

在整体链路中的位置（精排阶段）：
  query
    ├─ 向量检索（语义路）→ top-N
    ├─ BM25（认死理）    → top-N
    │
    ▼
   合并候选池（去重）       ← ensemble.py
    │
    ▼
   Reranker（本文件）→ 打分重排 → 截断 top-k   ← 你现在的位置
    │
    ▼
   top-k 精准文档 → LLM 生成答案

依赖说明（为什么这么选）：
  - sentence-transformers.CrossEncoder：加载 bge-reranker，批量打分
    （项目已装，embedding 依赖它，零额外大依赖）
  - modelscope.snapshot_download：国内下载模型（复用 embedder.py 的本地缓存模式）
  - bge-reranker-base：国产开源 cross-encoder，中文效果好，和 bge-small-zh 同家族
"""
import os
import math

from sentence_transformers import CrossEncoder
from modelscope.hub.snapshot_download import snapshot_download

from langchain_core.documents import Document

from src.config import config, logger


class Reranker:
    """基于 bge-reranker 的精排器：对候选池逐对打分 → 排序 → 截断 top-k"""

    def __init__(self, top_k: int | None = None):
        """
        Args:
            top_k: 精排后保留的文档数（截断数）。默认取 config.RETRIEVER_RERANK_TOP_K
        """
        self._top_k = top_k or config.RETRIEVER_RERANK_TOP_K
        self._model: CrossEncoder | None = None

    # ── 模型加载（懒加载 + 单例）──

    def _find_local_model(self) -> str | None:
        """在缓存目录中查找已下载的 rerank 模型路径"""
        import glob
        cache_dir = config.RERANK_CACHE_DIR
        patterns = [
            os.path.join(cache_dir, "BAAI", "bge-reranker-base"),
            os.path.join(cache_dir, "**", "bge-reranker-base"),
        ]
        for pattern in patterns:
            matches = glob.glob(pattern, recursive=True)
            if matches:
                return matches[0]
        return None

    def _get_model(self) -> CrossEncoder:
        """懒加载模型：优先本地缓存，没有再从 ModelScope 下载"""
        if self._model is None:
            logger.info(f"正在加载 Rerank 模型: {config.RERANK_MODEL}")
            model_path = self._find_local_model()
            if model_path:
                logger.info(f"使用本地缓存: {model_path}")
            else:
                logger.info("本地缓存未找到，从 ModelScope 下载...")
                model_path = snapshot_download(
                    config.RERANK_MODEL,
                    cache_dir=config.RERANK_CACHE_DIR,
                    # 只下载 PyTorch safetensors 格式 + 配置文件 + 分词器。
                    # 默认会连 pytorch_model.bin / onnx 一起拉（同一个模型 3 种格式，多 2G），
                    # allow_patterns 精确控制，省磁盘和时间。
                    allow_patterns=[
                        "*.json",
                        "*.safetensors",
                        "tokenizer*",
                        "sentencepiece*",
                        "README.md",
                    ],
                )
            self._model = CrossEncoder(model_path, max_length=config.RERANK_MAX_LENGTH)
            logger.info("Rerank 模型加载完成")
        return self._model

    # ── 核心逻辑：打分 + 排序 + 截断 ──

    def rerank(self, query: str, candidates: list[Document]) -> list[Document]:
        """
        对候选池逐对打分，按分数降序，截断取 top_k。

        Args:
            query: 用户问题
            candidates: 候选文档池（已合并去重）

        Returns:
            精排后的 top_k 条 Document（分数从高到低）
        """
        if not candidates:
            return []

        model = self._get_model()

        # 逐对构造输入：[[query, doc1], [query, doc2], ...]
        pairs = [[query, doc.page_content] for doc in candidates]

        # 批量打分（一次过模型，比逐条快得多）
        logits = model.predict(pairs, batch_size=config.RERANK_BATCH_SIZE)

        # sigmoid 归一化到 0~1（bge-reranker 官方建议）
        scores = [1 / (1 + math.exp(-x)) for x in logits]

        # 按分数降序取 top_k 个下标
        top_indices = sorted(
            range(len(scores)),#排序对象，分数的下标列表
            key=lambda i: scores[i],#排序规则，按照分数排序
            reverse=True,#降序排列
        )[: self._top_k]#最终得到的就是分数排行前列top_k在文档中的下标列表

        return [candidates[i] for i in top_indices]
