# 01 · bm25_retriever.py 参数

> 文件：`src/rag/bm25_retriever.py`，类 `BM25Retriever`
> 功能："认死理"召回器——jieba 分词 → rank_bm25 建索引 → 按 query 检索 top-N。
> 注意：**本文件不读取 config 的检索参数**，所有参数走构造传入。因为它的 N 由调用方（ensemble）决定，不固定。

## 构造参数

| 参数 | 默认值 | 大白话 | 技术性 | 谁传入 |
|------|--------|--------|--------|--------|
| `vectorstore` | 无（必传） | Chroma 向量库对象 | 用于 `_collection.get()` 拉取全量文档建索引，本文件**复用向量库里的文本**，不自己存文档 | 调用方（Retriever 实例） |
| `top_n` | `10` | 检索返回多少篇候选 | 召回数 N（须 > 精排 k，给 Rerank 空间）。BM25 只负责"海选"，N 越大召回越全 | 调用方（ensemble 决定） |

## 读取的 config

无。文件头 `from src.config import config, logger` 只用到了 `logger`（打日志），`config` 目前未消费——**这是有意为之**：BM25 的 top_n 与向量路的 N 必须一致，由 ensemble 统一控制，不各自写死。

## 关键方法

### `invoke(query: str) -> list[Document]`

| 入参 | 说明 |
|------|------|
| `query` | 用户问题字符串，内部做 `jieba.lcut` 分词 |

| 返回 | 说明 |
|------|------|
| `list[Document]` | 分数 > 0 的 top_n 条，`page_content`=切片文本，`metadata`=原元数据 |

**内部逻辑**（对应实现顺序）：
1. `_build_index()`：懒加载，首次调用才从向量库拉全量建索引
2. `jieba.lcut(query)` 分词
3. `get_scores()` 全量打分
4. `sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]` 取高分下标
5. 过滤 `scores[i] <= 0`（无命中的不要）→ 拼 Document

### `add_documents(chunks)` —— 索引维护

| 入参 | 说明 |
|------|------|
| `chunks` | 新增的切片列表 |

收到新文档 → 全量重建索引（当前文档量小，毫秒级，够用）。索引维护策略详见 `02-RAG进阶` 笔记的"方案 A/B/C"。
