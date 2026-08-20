# 04 · tools.py 参数（知识图谱角度）

> 文件：`src/agent/tools.py`（既有），模块级 `@tool` 函数
> 功能：**graph_query 工具签名声明**——告诉 LLM"图谱查询需要什么参数"。声明 ≠ 执行，执行在 nodes.py。
> 本页只写 graph_query；retrieve/web_search 见 [对话问答/12-tools.py.md](../对话问答/12-tools.py.md)。

## 构造参数

| 参数 | 默认值 | 大白话 | 技术性 |
|------|--------|--------|--------|
| `graph_query(entity: str)` | 必传 entity | 查图谱要一个实体名 | `@tool` 从签名+docstring 生成 JSON Schema |

## 读取的 config（只列本功能链路消费的）

| config 参数 | 默认值 | 大白话 | 技术性 | 谁消费 |
|------------|--------|--------|--------|--------|
| 无 | — | 工具只是签名声明 | 函数体返回 `""` 占位 | — |

## 核心方法

`graph_query(entity)`（`tools.py:47`）：

```
docstring 告知 LLM：
  - 适用场景：用户想知道某个概念/技术/实体与其他概念之间的关系
  - 参数示例：entity="InnoDB"、"事务"、"MySQL"
函数体：return ""   # 实际执行在 nodes.py 的 graph_query_node
```

> **设计要点**
> - **参数名是 `entity` 而非 `query`**——语义区分：检索工具查"问题"，图谱工具查"实体"。docstring 给了示例值帮助 LLM 理解该传什么。
> - 图谱查询只在 router 判定"关系型问题"时被调用（prompts.py 规则），LLM 靠这份声明知道"什么时候该用这个工具"。

## 该文件在链路中的位置

```
router（bind_tools(TOOLS)）→ LLM 决定调 graph_query → graph_query_node 执行真实查询
```
