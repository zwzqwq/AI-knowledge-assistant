# 03 · graph_store.py 参数

> 文件：`src/kg/graph_store.py`（既有），类 `GraphStore`（单例）
> 功能：**图谱存储 + 查询**——NetworkX 有向图，count/weight 加权，JSON 持久化，双向查找 + 模糊匹配。

## 构造参数

| 参数 | 默认值 | 大白话 | 技术性 |
|------|--------|--------|--------|
| `__new__()` 单例 | 全进程一个实例 | 图谱只加载一次 | `cls._instance` 判空，重复 new 返回同一实例 |
| `__init__()` | 无参 | 建图 + 从 JSON 加载 | `self._graph = _init_graph()`（DiGraph）+ `self._load()`；`_initialized` 防重复初始化 |

> 模块级常量 `KG_FILE = <项目根>/data/knowledge_graph.json`——路径由文件自身位置推出，不依赖工作目录。

## 读取的 config（只列本功能链路消费的）

| config 参数 | 默认值 | 大白话 | 技术性 | 谁消费 |
|------------|--------|--------|--------|--------|
| 无 | — | 存储层不读参数 | 路径常量硬编码在模块 | — |

## 核心方法

### `add_triples(triples)`（`graph_store.py:64`）—— 建图侧

```
for source, relation, target in triples:
    source/target 已存在 → count += 1；不存在 → add_node(count=1)
    边已存在 → weight += 1；不存在 → add_edge(relation=..., weight=1)
self._save()   # 每次添加后立即落盘
```

> 节点 count = 实体出现次数（越重要），边 weight = 关系重复叠加——**加权是为了查询排序**。

### `query(entity, max_neighbors=5)`（`graph_store.py:100`）—— 查询侧

```
entity 不在图中 → 模糊匹配：节点名包含 entity 的集合里取 count 最高者
  → 仍无 → 返回空结果
出边（entity → others）+ 入边（others → entity）双向收集
按 weight 降序排序，截断 max_neighbors=5
```

### `query_to_text(entity)`（`graph_store.py:155`）—— 喂给 LLM

```
result = query(entity)
total_connections == 0 → "（知识图谱中未找到与「X」相关的实体）"   # 占位，来源判定会排除
否则 → "知识图谱查询结果 ——「X」: 共 N 个关联 / 关联到的实体 / 被以下实体关联"
```

### `_save()` / `_load()`（`graph_store.py:178,198`）—— 持久化

```
_save: {"nodes": {name: {count}}, "edges": [{source, target, relation, weight}]} → JSON
_load: 文件不存在 → 空图开始；解析失败 → except 重建空 DiGraph（不崩溃）
```

> **设计要点**
> - **单例保证全进程一份图谱**：建图侧 `add_triples` 和查询侧 `query` 操作同一个实例，不用传参共享。【推断】证据：`__new__` 单例实现（`graph_store.py:43-49`）。
> - **延迟导入 networkx**：`_init_graph` 内部 import——避免 Streamlit 启动时的潜在模块顺序错误（`graph_store.py:22-23` 注释）。
> - **`max_neighbors=5` 是查询宽度上限**：防止一个高连节点把 ToolMessage 撑爆，控制喂给 LLM 的 token。
> - **模糊匹配是"抽取名 vs 用户问法"不一致的容错**：LLM 抽的实体名和用户输入不完全一致时仍能命中。【推断】理由：`query` 里显式实现包含匹配 + 取最重要节点。证据：`graph_store.py:113-117`。

## 该文件在链路中的位置

```
建图侧：extractor.extract → graph_store.add_triples → _save → JSON
查询侧：graph_query_node → graph_store.query_to_text → ToolMessage → generate
```
