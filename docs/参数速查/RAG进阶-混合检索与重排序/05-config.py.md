# 05 · config.py 参数总表（全局默认值）

> 文件：`src/config.py`，类 `AppConfig`。所有可调参数集中管理，**改参数不用动代码**（配置与逻辑分离）。
> 支持 `.env` 环境变量覆盖的参数已标注。相对路径自动解析为基于项目根的绝对路径。

## 一、RAG 检索参数（本次改造核心）

| config 参数 | 默认值 | env 覆盖 | 大白话 | 技术性 | 谁消费 |
|------------|--------|----------|--------|--------|--------|
| `RETRIEVER_CANDIDATES` | `10` | — | 海选阶段每路捞 10 篇 | 召回 N，须 > k=3，给 Rerank 空间 | ensemble.py |
| `RETRIEVER_RERANK_TOP_K` | `3` | — | 评委最后只留 3 篇 | 精排截断数，喂给 LLM 的数量 | reranker.py |
| `RETRIEVER_K` | `3` | — | 单路检索返回几条 | 单路 top-k（改造后由 Rerank 接管最终 k） | retriever.py |
| `RETRIEVER_SEARCH_TYPE` | `similarity` | — | 相似度还是 MMR | `similarity`（纯相似）/ `mmr`（平衡多样性） | retriever.py |
| `CHUNK_SIZE` | `500` | — | 每个切片最多多少字 | 切片长度，太大降精度、太小丢上下文 | loader 切片 |
| `CHUNK_OVERLAP` | `50` | — | 相邻切片重叠多少字 | 保句子完整，防语义断裂 | loader 切片 |

## 二、Rerank 模型参数

| config 参数 | 默认值 | env 覆盖 | 大白话 | 技术性 | 谁消费 |
|------------|--------|----------|--------|--------|--------|
| `RERANK_MODEL` | `BAAI/bge-reranker-base` | `RERANK_MODEL` | 评委模型名 | BGE cross-encoder，中文好、轻量 | reranker.py |
| `RERANK_CACHE_DIR` | `./rerank_model` | `RERANK_CACHE_DIR` | 模型放本地哪 | 缓存目录（相对路径自动转项目根绝对路径） | reranker.py |
| `RERANK_MAX_LENGTH` | `512` | `RERANK_MAX_LENGTH` | 拼太长的文档截断 | token 上限 | reranker.py |
| `RERANK_BATCH_SIZE` | `32` | `RERANK_BATCH_SIZE` | 一次给模型 32 篇 | 批量推理 batch_size | reranker.py |

## 三、Embedding 参数（改造前已有）

| config 参数 | 默认值 | env 覆盖 | 大白话 | 技术性 | 谁消费 |
|------------|--------|----------|--------|--------|--------|
| `EMBEDDING_MODEL` | `BAAI/bge-small-zh-v1.5` | `EMBEDDING_MODEL` | 向量模型名 | 中文 embedding，bi-encoder | embedder.py |
| `EMBEDDING_CACHE_DIR` | `./bge_model` | `EMBEDDING_CACHE_DIR` | 模型放本地哪 | 缓存目录 | embedder.py |

## 四、调优口诀

- `RETRIEVER_CANDIDATES`(N) ↑ → 召回更全，但 Rerank 打分更慢
- `RETRIEVER_RERANK_TOP_K`(k) ↑ → 上下文更多，但可能掺噪声
- **N 必须 > k**：否则 Rerank 没得挑
- `RERANK_BATCH_SIZE` ↑ → 快但吃内存（CPU 建议 32~64）
- `RERANK_MAX_LENGTH` ↑ → 长文档信息完整但更慢（切片 500 字，512 token 够用）

## 更新记录

- 2026-08-11：初建。RAG 进阶改造（混合检索 + Rerank）新增参数整理。
  - 待补：`ensemble.py` 落地后如有合并相关参数，补进本表。
