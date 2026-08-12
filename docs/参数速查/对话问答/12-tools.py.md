# 12 · tools.py 参数

> 文件：`src/agent/tools.py`（既有），模块级 `@tool` 函数
> 功能：**工具签名声明**——用 `@tool` 生成 JSON Schema 给 LLM 的 function calling 协议看。**声明 ≠ 执行**，实际执行在 nodes.py 的工具节点。

## 构造参数

| 参数 | 默认值 | 大白话 | 技术性 |
|------|--------|--------|--------|
| `retrieve(query: str)` | 必传 query | 告诉 LLM：这工具要一个查询词 | `@tool` 从签名+docstring 生成 JSON Schema |
| `web_search(query: str)` | 必传 query | 联网搜索 | 同上 |
| `graph_query(entity: str)` | 必传 entity | 查知识图谱要一个实体名 | 同上 |

## 读取的 config（只列本功能链路消费的）

| config 参数 | 默认值 | 大白话 | 技术性 | 谁消费 |
|------------|--------|--------|--------|--------|
| 无 | — | 工具只是签名声明 | 不读 config，返回空字符串占位 | — |

## 核心方法

三个 `@tool` 函数（`tools.py:17,33,47`）——**函数体返回 `""`**，仅 docstring 描述适用场景给 LLM 理解。

模块级常量 `TOOLS = [retrieve, web_search, graph_query]`，被 `router_node` 的 `bind_tools(TOOLS)` 消费。

> **设计要点**
> - **声明与执行分离是本链路的关键架构**：`tools.py` 的返回值永远为空串，它唯一的产出是**函数签名 + docstring → LLM 可见的 JSON Schema**。LLM 据此决定调哪个、传什么参数（如 `{"query": "MySQL 索引是什么"}`）。
> - **真正执行**：`nodes.retrieve_node` 用 `_find_pending_tool_call` 找到这个声明，读 `args`，执行 `retriever.invoke(query)` 等真实逻辑。
> - `graph_query` 的参数名是 `entity`（而非 `query`）——因为它是"查实体关系"，docstring 给了示例值（"InnoDB"、"事务"）帮助 LLM 理解，这是工具设计上的命名讲究。

## 该文件在链路中的位置

```
router_node → llm.bind_tools(TOOLS) → LLM 输出 tool_calls → 工具节点执行（nodes.py）→ ToolMessage
```
