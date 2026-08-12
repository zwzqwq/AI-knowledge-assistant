# 06 · retriever.py 参数

> 文件：`src/rag/retriever.py`（既有），类 `Retriever`
> 功能：**对话链路的检索工厂**——`stream_chat` 每次提问现调用 `get_hybrid_retriever()` 组装混合检索器。
> 完整内部实现（Chroma 创建/加载/去重/向量检索）见 [RAG进阶-混合检索与重排序/04-retriever.py](../RAG进阶-混合检索与重排序/04-retriever.py.md)。本页只写**对话链路消费的点**。

## 构造参数

| 参数 | 默认值 | 大白话 | 技术性 |
|------|--------|--------|--------|
| `__init__()` | 无参 | 备好 embedding，向量库懒加载 | `EmbeddingManager().get()` + `self._vectorstore=None` |

## 读取的 config（只列本功能链路消费的）

| config 参数 | 默认值 | 大白话 | 技术性 | 谁消费 |
|------------|--------|--------|--------|--------|
| `RETRIEVER_SEARCH_TYPE` | `"similarity"` | 召回用纯相似度 | 向量路 `vs.as_retriever(search_type=...)` | `get_hybrid_retriever` |
| `RETRIEVER_CANDIDATES` | 10 | 召回阶段每路捞 10 条（广） | 向量路 `k` + BM25 `top_n` | `get_hybrid_retriever` |
| `RETRIEVER_RERANK_TOP_K` | 3 | 精排后只留 3 条（准） | Reranker 内部 top_k | `get_hybrid_retriever`（经 Reranker） |
| `CHROMA_DB_DIR` | `./chroma_db` | 向量库在磁盘哪 | 懒加载时 `Chroma(persist_directory=...)` | `_get_or_load` |

## 核心方法

### `get_hybrid_retriever()`（`retriever.py:172`）★ 对话链路唯一消费入口

```
vs = self._get_or_load()          # 懒加载：已有则加载；没有则 raise RuntimeError("向量库未初始化")
① vector_retriever = vs.as_retriever(search_type=similarity, k=RETRIEVER_CANDIDATES)
② bm25_retriever   = BM25Retriever(vectorstore=vs, top_n=RETRIEVER_CANDIDATES)
③ reranker         = Reranker()   # top_k 内部取 RETRIEVER_RERANK_TOP_K
④ return HybridRetriever(vector_retriever, bm25_retriever, reranker)
```

> **设计要点（对话链路角度）**
> - **每次提问都现组装**，不缓存复用——因为向量库可能在上传文档后变化，懒加载保证拿到最新数据。【推断】理由：`get_hybrid_retriever` 每次都调 `_get_or_load()`。证据：`chat_service.py:180` 每轮 `stream_chat` 内调用。
> - **N(10) > k(3)**：召回广、精排准，给 Rerank 足够挑选空间。
> - **失败语义**：向量库未初始化时 raise——但 `chat_service` 没有单独捕获，会落到兜底 `Exception → error 事件`（前端提示"请先上传文档"更友好，属可优化点）。【推断】证据：`stream_chat` 的异常链里没有 RuntimeError 专门分支。

## 该文件在链路中的位置

```
chat_service.stream_chat ③ → retriever.get_hybrid_retriever() → (retrieve_node 执行时) retriever.invoke(query) → 回答
```
