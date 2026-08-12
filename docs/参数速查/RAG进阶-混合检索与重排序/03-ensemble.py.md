# 03 · ensemble.py 参数

> 文件：`src/rag/ensemble.py`（✅ 已实现）
> 功能：混合编排器（"指挥"）——并行召回（向量 top-N + BM25 top-N）→ 按内容去重合并候选池 → 交给 Reranker 精排 → 返回 top-k。
> 自己不召回、不打分，只做"合并 + 去重 + 编排顺序"。

## 构造参数

| 参数 | 默认值 | 大白话 | 技术性 |
|------|--------|--------|--------|
| `vector_retriever` | 无（必传） | 向量检索器 | LangChain Retriever（语义路，`.invoke(query)` → list[Document]） |
| `bm25_retriever` | 无（必传） | BM25 检索器 | `BM25Retriever` 实例（认死理路，同签名） |
| `reranker` | 无（必传） | 精排器 | `Reranker` 实例（`.rerank(query, candidates)` → list[Document]） |

> **依赖注入**：构造传三个组件，ensemble 不 new 它们——只依赖"有 `.invoke()` / `.rerank()` 方法"这个抽象接口。好处：测试能换 mock、以后能换实现，编排逻辑不动。

## 读取的 config

| config 参数 | 默认值 | 大白话 | 技术性 |
|------------|--------|--------|--------|
| （本文件不直接读 config 参数） | — | — | 召回 N、精排 k 都在**上游组件内部**消化（vector/BM25 吃 `RETRIEVER_CANDIDATES`，Reranker 吃 `RETRIEVER_RERANK_TOP_K`）。ensemble 只做编排，不碰数字 |

> 对比规划稿：当时以为 `invoke()` 里要读 `RETRIEVER_CANDIDATES`。落地时发现候选数 N 由 vector/BM25 检索器自己控制（各自构造时已设 k=N），ensemble 只取"它们给多少"，更松耦合。

## 核心方法

### `invoke(query: str) -> list[Document]`

```
query
  ├─ self._vector.invoke(query) → top-N   语义路（若 vector 为空则跳过）
  ├─ self._bm25.invoke(query)   → top-N   认死理路（若 bm25 为空则跳过）
  │
  ▼
  merged: dict[str, Document]   ← 以 page_content 为 key 去重（保序）
  │                             两路重叠的同一切片只留一次
  ▼
  空候选池 → return []           ← 保护：两路都没捞到不崩，打 warning
  │
  ▼
  self._reranker.rerank(query, candidates) → top-k 精准文档
```

### 关键设计点（落地确认）

1. **去重键用 `page_content`**：同一切片 → 文本必然相同 → 去重；同文件不同页 → 文本不同 → 都保留。比用 `metadata["source"]` 准（source 会误杀同源不同页）。
2. **去重保序**：`dict` 以内容为 key，`list(merged.values())` 顺序 = 向量路在前、BM25 补后，稳定。
3. **空组件保护**：`if self._vector else []`——组件可为 None，不崩。空候选池单独打 warning 返回空。
4. **接口契约**：返回 `list[Document]`，每篇带 `page_content` + `metadata["source"]`，下游 `retrieve_node`（nodes.py:252）零改动。

---

> ✅ 本页已按实际代码补全。README 状态更新。
