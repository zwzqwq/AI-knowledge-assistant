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

单例设计：
  Reranker 采用【线程安全的双重检查锁定】单例模式，与 GraphStore 保持一致。
  为什么要单例：CrossEncoder 模型权重 ~400MB，每次实例化都要从磁盘重新加载到内存，
  耗时 2~5 秒。单例保证整个进程只加载一次模型，后续问答复用同一个内存对象。
"""
import os
import math
import time
import threading

from sentence_transformers import CrossEncoder
from modelscope.hub.snapshot_download import snapshot_download

from langchain_core.documents import Document

from src.config import config, logger


class Reranker:
    """基于 bge-reranker 的精排器：对候选池逐对打分 → 排序 → 截断 top-k

    单例模式：整个进程共享同一个 Reranker 实例，模型只加载一次。
    """

    # ── 单例基础设施（与 GraphStore 同构）──
    _instance: "Reranker | None" = None       # 类级别的唯一实例引用（static 字段）
    _lock = threading.Lock()                  # 类级别的线程锁，保证 __new__ 原子性

    def __new__(cls, top_k: int | None = None):
        """重写构造：首次调用创建实例，后续调用直接返回已有实例

        双重检查锁定（Double-Checked Locking）：
          第1层 if：99% 的情况对象已存在，无需抢锁，直接返回，无锁开销
          进入锁：只有多线程竞争首次创建时才走到这里（锁保证只有一个线程能创建）
          第2层 if：第一个线程创建完后释放锁，第二个抢到锁的线程再检查一次，
                    避免两个线程都通过第1层检查后重复创建
        """
        if cls._instance is None:                              # 第 1 层检查（无锁）
            with cls._lock:                                    # 抢锁（临界区）
                if cls._instance is None:                      # 第 2 层检查（持锁状态下）
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False        # 标记：实例还没跑过 __init__
        return cls._instance

    def __init__(self, top_k: int | None = None):
        """初始化实例参数（只跑一次，因为单例返回同一个对象）

        注意：Python 中 __new__ 返回的实例，解释器会自动再调用 __init__。
        如果不加 _initialized 守卫，第二次 Reranker() 调用会重置 self._top_k，
        把之前的设置覆盖掉。_initialized 保证初始化逻辑只执行一次。
        """
        if getattr(self, "_initialized", False):
            return
        self._top_k = top_k or config.RETRIEVER_RERANK_TOP_K
        self._model: CrossEncoder | None = None
        self._initialized = True

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

        # ── 性能计时起点 ──────────────────────────────────────
        # 分段计时用于诊断 Rerank 慢在哪一环：
        #   model_load：模型懒加载（单例后只有第一次问答有耗时）
        #   prepare  ：候选对构造（CPU 拼接，通常 <10ms）
        #   inference：CrossEncoder 批量推理（主要瓶颈，CPU 上 50~200ms/对）
        #   sort     ：排序 + 截断（纯内存，通常 <5ms）
        t0 = time.perf_counter()

        # ① 懒加载模型（单例：仅首次问答真正加载，后续直接返回内存中的对象）
        model = self._get_model()

        t1 = time.perf_counter()

        # ② 逐对构造输入：[[query, doc1], [query, doc2], ...]
        #    CrossEncoder 要求 query 和 doc 拼成一条文本对，做双向 cross-attention
        pairs = [[query, doc.page_content] for doc in candidates]

        t2 = time.perf_counter()

        # ③ 批量打分（一次过模型，比逐条快得多）
        #    batch_size 控制每次喂给模型的文本对数量，越大吞吐越高（但更吃内存）
        logits = model.predict(
            pairs,
            batch_size=config.RERANK_BATCH_SIZE,
        )

        t3 = time.perf_counter()

        # ④ sigmoid 归一化到 0~1（bge-reranker 官方建议）
        #    原始 logits 是实数（可正可负），sigmoid 把它压到 [0,1] 区间，便于分数比较
        scores = [1 / (1 + math.exp(-x)) for x in logits]

        # ⑤ 按分数降序取 top_k 个下标
        #    sorted 返回的是下标列表，不是文档本身，方便后面用下标回查候选池
        top_indices = sorted(
            range(len(scores)),#排序对象，分数的下标列表
            key=lambda i: scores[i],#排序规则，按照分数排序
            reverse=True,#降序排列
        )[: self._top_k]#最终得到的就是分数排行前列top_k在文档中的下标列表

        t4 = time.perf_counter()

        # 打印分段耗时，用于定位性能瓶颈
        # 控制台看到 "inference=2.3s" 就知道慢在 CrossEncoder 推理，不是加载或排序
        logger.info(
            "Rerank timing | "
            f"model_load={t1-t0:.3f}s | "
            f"prepare={t2-t1:.3f}s | "
            f"inference={t3-t2:.3f}s | "
            f"sort={t4-t3:.3f}s | "
            f"total={t4-t0:.3f}s | "
            f"candidates={len(candidates)}"
        )

        # ⑥ 用排序后的下标回查候选池，返回对应的 Document 列表
        return [candidates[i] for i in top_indices]
