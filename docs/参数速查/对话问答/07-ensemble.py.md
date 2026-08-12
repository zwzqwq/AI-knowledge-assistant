# 07 · ensemble.py 参数

> 文件：`src/rag/ensemble.py`（既有），类 `HybridRetriever(BaseRetriever)`
> 功能：**对话链路检索执行器**——`retrieve_node` 调 `retriever.invoke(query)` 时，内部完成"向量+BM25 两路召回 → 合并去重 → Rerank 精排"。
> 完整实现见 [RAG进阶-混合检索与重排序/03-ensemble.py](../RAG进阶-混合检索与重排序/03-ensemble.py.md)。本页只写**对话链路消费的点**。

## 构造参数

| 参数 | 默认值 | 大白话 | 技术性 |
|------|--------|--------|--------|
| `vector_retriever` | 必传 | 语义路检索器 | 由 `get_hybrid_retriever` 传入（k=RETRIEVER_CANDIDATES） |
| `bm25_retriever` | 必传 | 关键词路检索器 | `BM25Retriever(top_n=RETRIEVER_CANDIDATES)` |
| `reranker` | 必传 | 精排器 | `Reranker()`（top_k=RETRIEVER_RERANK_TOP_K） |

## 读取的 config（只列本功能链路消费的）

| config 参数 | 默认值 | 大白话 | 技术性 | 谁消费 |
|------------|--------|--------|--------|--------|
| 无 | — | 参数在构造时由上层传入 | `HybridRetriever` 不直接读 config | — |

## 核心方法

### `invoke(query)`（继承 `BaseRetriever`）→ `_get_relevant_documents(query)`（`ensemble.py`）

```
vector_docs = vector_retriever.invoke(query)   # 语义路，最多 RETRIEVER_CANDIDATES 条
bm25_docs   = bm25_retriever.invoke(query)     # 关键词路，最多 RETRIEVER_CANDIDATES 条
merged[page_content] = doc                     # 按文本内容去重（两路可能重叠）
candidates = list(merged.values())
if not candidates: return []                   # 两路全空 → 返回空（调用方会提示未找到）
return reranker.rerank(query, candidates)      # 精排 → top_k 条
```

> **设计要点（对话链路角度）**
> - `retrieve_node` 拿到的是**精排后 top-3 文档**（`reranker.rerank` 输出），再格式化"片段+来源"喂给 generate。
> - **空结果语义**：`invoke` 返回 `[]` → `retrieve_node` 产出一个占位 ToolMessage"（知识库中未找到相关内容）"→ 后续 router 按 prompt 规则决定是否改调 web_search。【推断】证据：`nodes.py:259-260`。
> - 去重键是 `page_content`（文本内容），不是 doc.id——因为两路检索可能返回同一文本但不同 metadata 的文档。

## 该文件在链路中的位置

```
retrieve_node → retriever.invoke(query) → HybridRetriever（合并两路→去重→精排）→ top-k 文档 → generate 使用
```
