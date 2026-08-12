# 04 · retriever.py 参数

> 文件：`src/rag/retriever.py`（✅ 已改造），类 `Retriever`
> 功能：向量库管理（创建/加载/追加/删除/统计）+ 检索入口。RAG 进阶后新增 `get_hybrid_retriever()` 组装混合检索三件套。

## 构造参数（未变）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `embedding_function` | `EmbeddingManager().get()` | 从 config 读 embedding 模型 |
| `persist_directory` | `config.CHROMA_DB_DIR` | 向量库目录 |

## 读取的 config

| config 参数 | 默认值 | 大白话 | 技术性 | 谁消费 |
|------------|--------|--------|--------|--------|
| `CHROMA_DB_DIR` | `./chroma_db` | 向量库放哪 | Chroma 持久化目录 | 所有入库/检索 |
| `RETRIEVER_SEARCH_TYPE` | `similarity` | 相似度还是 MMR | 向量检索策略 | `get_retriever` + `get_hybrid_retriever` 的向量路 |
| `RETRIEVER_K` | `3` | 单路检索返回几条 | 单路 top-k | `get_retriever`（纯向量入口） |
| `RETRIEVER_CANDIDATES` | `10` | 混合检索每路海选捞几篇 | 召回 N，须 > 精排 k | `get_hybrid_retriever` 传给向量路和 BM25 路 |
| `RETRIEVER_RERANK_TOP_K` | `3` | 精排后留几篇 | 最终 top-k | `Reranker()` 内部默认读取 |

## 两个检索入口（关键决策）

> ⚠️ **落地决策 vs 规划稿**：规划稿写的是"`get_retriever()` 改为返回混合检索器"。落地时改为**保留 `get_retriever()` 纯向量 + 新增 `get_hybrid_retriever()`**。原因：
> 1. `get_retriever(search_type)` 的 similarity/mmr 开关语义上属于"单路向量"，硬改成混合会名不副实
> 2. 保留纯向量入口方便**对比单路 vs 混合**（调试/面试演示都用得上）
> 3. 只改调用处一行（chat_service.py:180），代价最小，语义最清晰

### `get_retriever(search_type=None)` — 纯向量单路（保留）

```python
vs.as_retriever(search_type=effective_type, search_kwargs={"k": config.RETRIEVER_K})
# mmr 时额外加 fetch_k = RETRIEVER_K*4, lambda_mult = 0.7
```

### `get_hybrid_retriever()` — 混合主入口（新增）

```
get_hybrid_retriever()
  │  向量库未初始化则 raise
  ▼
  ① vector_retriever = vs.as_retriever(similarity, k=RETRIEVER_CANDIDATES=10)  语义路
  ② bm25_retriever   = BM25Retriever(vectorstore=vs, top_n=RETRIEVER_CANDIDATES) 认死理路
  ③ reranker         = Reranker()                                                 评委（k 内部取 config）
  ▼
  return HybridRetriever(①, ②, ③)   # 编排：并行召回 → 去重 → 精排
```

## 入库去重（本轮新加，防 96% 重复事故重演）

| 方法 | 行为 |
|------|------|
| `_unique_chunks(chunks, existing)` | 静态助手：按 `page_content` 去重，返回 `(新文档去重后列表unique, 跳过的切片数)`。`existing` 传库内已有文本集合时，连"库中已存在"的一起跳过 |
| `create(chunks)` | 创建前去重（剔除入参自身重复的切片） |
| `add(chunks)` | 追加前去重：`existing = set(vs._collection.get()["documents"])`，跳过内容已存在的，全重复则返回不追加 |

> 背景：向量库曾积累 96% 重复切片（422 篇只剩 51 篇唯一），根源是同一文档多次导入没有去重。清理脚本见 `scripts/deduplicate_store.py`。

## 对外接口（未变，下游零改动）

`retrieve_node` 仍调 `retriever.invoke(query) -> list[Document]`，`generate_node` 零改动。换检索策略，接口契约不变。

---

> ✅ 本页已按实际代码补全。
