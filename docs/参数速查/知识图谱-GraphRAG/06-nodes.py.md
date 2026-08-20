# 06 · nodes.py 参数（知识图谱角度）

> 文件：`src/agent/nodes.py`（既有），模块级函数集
> 功能：**graph_query_node 工具执行**——从消息里找未执行的 graph_query 调用，执行图谱查询，返回 ToolMessage。
> 本页只写图谱链路；完整节点见 [对话问答/10-nodes.py.md](../对话问答/10-nodes.py.md)。

## 构造参数

模块级函数，无类。图谱相关函数：

| 函数 | 参数 | 大白话 | 技术性 |
|------|------|--------|--------|
| `graph_query_node(state)` | 必传 | 执行图谱查询 | 找未执行 tool_call → `GraphStore().query_to_text(entity)` |

## 读取的 config（只列本功能链路消费的）

| config 参数 | 默认值 | 大白话 | 技术性 | 谁消费 |
|------------|--------|--------|--------|--------|
| 无 | — | 节点不读图谱参数 | 查询参数在 graph_store 内部 | — |

## 核心方法

### `graph_query_node(state)`（`nodes.py:285`）

```
tc_to_run = _find_pending_tool_call(state["messages"], "graph_query")
if tc_to_run is None: return {}                    # 本轮没有 graph_query 调用
entity = tc_to_run["args"].get("entity", "")       # 取 LLM 传的实体名
logger.info(f"GraphQuery: '{entity}'")
store = GraphStore()                                # 单例（内部已从 JSON 加载）
context = store.query_to_text(entity)               # 双向查找+模糊匹配 → 文本
return {"messages": [ToolMessage(content=context, tool_call_id=tc_to_run["id"], name="graph_query")]}
```

### `_find_pending_tool_call(messages, "graph_query")`（`nodes.py:311`）

```
只扫当前轮（最后一条 HumanMessage 之后）：
  找到带 tool_calls 的 AIMessage → 找 name == "graph_query" 的调用
  → 检查该 tool_call_id 是否已有对应 ToolMessage（执行过没有）
  → 未执行 → 返回该调用
```

> **设计要点**
> - **每次查询 new 一个 GraphStore() 但拿到的是同一个单例**——`__new__` 保证内存中只有一份图，查询不走磁盘。【推断】证据：单例实现 + `query_to_text` 直接查内存图。
> - **ToolMessage 带 `name="graph_query"`**：chat_service 的来源判定靠它识别"回答用了图谱"（`chat_service.py:246-249`）——查询结果非空（>100 字且非占位）时来源标 `knowledge_graph`。
> - **查询空结果 → 占位文本 → 来源判定排除**：`query_to_text` 返回"（知识图谱中未找到…）"时，chat_service 不计入 knowledge_graph 来源，降级到 web_search/LLM。

## 该文件在链路中的位置

```
pick_next_tool 分发 → graph_query_node → GraphStore().query_to_text → ToolMessage → should_continue 回 router
```
