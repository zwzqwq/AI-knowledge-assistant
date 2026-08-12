# 02 · reranker.py 参数

> 文件：`src/rag/reranker.py`，类 `Reranker`
> 功能："评委"——对候选池逐对打分（cross-encoder）→ 排序 → 截断 top-k。
> 读取的 config 较多，因为模型路径、缓存、推理参数都是全局可调的。

## 构造参数

| 参数 | 默认值 | 大白话 | 技术性 | 谁传入 |
|------|--------|--------|--------|--------|
| `top_k` | `None` → 取 `config.RETRIEVER_RERANK_TOP_K` | 精排后留几篇 | 截断数 k。None 时用全局默认 3，也可显式传 | 调用方（ensemble） |

## 读取的 config

| config 参数 | 默认值 | 大白话 | 技术性 |
|------------|--------|--------|--------|
| `RETRIEVER_RERANK_TOP_K` | `3` | 评委最后留 3 篇 | 精排截断数（构造 top_k=None 时的默认） |
| `RERANK_MODEL` | `BAAI/bge-reranker-base` | 评委模型名 | BGE cross-encoder，中文好、轻量 |
| `RERANK_CACHE_DIR` | `./rerank_model` | 模型放本地哪 | 缓存目录，复用 embedder 的本地优先模式 |
| `RERANK_MAX_LENGTH` | `512` | 拼太长的文档截断 | token 上限，防超长文本拖慢推理 |
| `RERANK_BATCH_SIZE` | `32` | 一次给模型 32 篇 | 批量推理 batch_size，CPU 上平衡速度/内存 |

## 关键方法

### `rerank(query: str, candidates: list[Document]) -> list[Document]`

| 入参 | 说明 |
|------|------|
| `query` | 用户问题（和每篇候选拼一起过模型） |
| `candidates` | 候选池（已合并去重，来自 ensemble） |

| 返回 | 说明 |
|------|------|
| `list[Document]` | 按分数降序的 top_k 条，`metadata` 原样保留 |

**内部逻辑**（对应实现顺序）：
1. `_get_model()`：懒加载模型（本地缓存 → ModelScope 下载）
2. `pairs = [[query, doc1], [query, doc2], ...]` 逐对构造
3. `model.predict(pairs, batch_size=...)` 批量打分（一次过模型，比逐条快）
4. `sigmoid` 归一化到 0~1（bge 官方建议）
5. `sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]` 取高分下标
6. 返回 `[candidates[i] for i in top_indices]`
