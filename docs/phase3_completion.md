# Phase 3 完成纪要

## 新增文件

| 文件 | 功能 |
|------|------|
| `src/kg/extractor.py` | **知识抽取器** — 用 LLM 从文档片段自动抽取实体关系三元组 [(实体A, 关系, 实体B), ...] |
| `src/kg/graph_store.py` | **图谱存储** — 基于 NetworkX DiGraph，支持添加三元组、实体查询、JSON 持久化 |
| `src/kg/__init__.py` | 空文件 |

## 修改文件

| 文件 | 改动 |
|------|------|
| `src/agent/tools.py` | 新增 `graph_query` 工具签名 — LLM 可以调用它查实体关系 |
| `src/agent/graph.py` | ① 导入 `graph_query` 和 `GraphStore` ② 新增 `graph_query_node()` ③ 路由条件边加入 `graph_query` ④ Router prompt 更新 — 引导 LLM 同步使用检索和图谱 ⑤ MAX_ITERATIONS 从 3 升到 5 |
| `src/services/chat_service.py` | `add_document()` 新增 `_build_knowledge_graph()` — 入库时自动抽取实体关系并构建图谱 |

## 完整链路（上传→检索→回答）

```
用户上传文档（.md/.txt）
  │
  ├── 切片 → ChromaDB（向量检索，已有）
  └── LLM 抽取实体关系 → NetworkX DiGraph → JSON 持久化（Phase 3 新增）
         │
用户提问 "InnoDB支持哪些特性？"
  │
  ▼
router: LLM 决定同时调 retrieve + graph_query
  ├── retrieve("InnoDB 特性") → 3 个相关文档片段
  └── graph_query("InnoDB") → 图谱返回 {支持事务, 支持行级锁, 支持外键, 是存储引擎}
         │
         ▼
router: LLM 看结果 → 信息够了 → 直接回答
  │
  ▼
generate: 融合向量检索 + 图谱查询 → 最终回答
```

## 数据文件

- `data/knowledge_graph.json` — 图谱持久化文件，自动创建
- 格式：`{"nodes": {"InnoDB": {"count": 19}}, "edges": [{"source":..., "target":..., "relation":..., "weight":...}]}`

## 你需要理解的部分（按优先级）

1. **`src/kg/extractor.py`** — LLM 怎么从" InnoDB 支持事务和行级锁"这样的文本里抽出 `(InnoDB, 支持, 事务)` 三元组。关键是 prompt 设计和 JSON 解析
2. **`src/kg/graph_store.py`** — `add_triples()`（节点计数+边权重）、`query()`（出边+入边=双向邻居）、`query_to_text()`（转成给 LLM 看的文本）
3. **`src/agent/graph.py`** — `graph_query_node()` 跟 `retrieve_node()` 结构一模一样：取 tool_call → 执行 → ToolMessage 回填
4. **`src/services/chat_service.py:__init__` + `_build_knowledge_graph`** — 文档入库时同时触发 KG 构建

## 运行验证

上传 MySQL 文档 → 自动抽取 12 个三元组 → 问"InnoDB 支持哪些特性"→ LLM 同时调 retrieve + graph_query → 图谱返回"InnoDB支持事务、行级锁、外键"→ 融合回答 768 字，知识库来源
