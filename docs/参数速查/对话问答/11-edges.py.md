# 11 · edges.py 参数

> 文件：`src/agent/edges.py`（既有），模块级函数集
> 功能：**条件边（路由决策）**——纯函数，只读 state 决定"下一步去哪个节点"，不做任何修改。是 Agent 循环的分流逻辑。

## 构造参数

模块级常量 + 函数，无类：

| 常量 | 默认值 | 大白话 | 技术性 |
|------|--------|--------|--------|
| `MAX_ITERATIONS` | 12 | 决策最多 12 轮 | 防死循环安全阀 |

## 读取的 config（只列本功能链路消费的）

| config 参数 | 默认值 | 大白话 | 技术性 | 谁消费 |
|------------|--------|--------|--------|--------|
| 无 | — | 迭代上限硬编码在模块 | 不读 config | — |

## 核心方法

### `route_after_router(state)`（`edges.py:17`）—— Router 之后的第一个分叉

```
if iteration >= MAX_ITERATIONS: return "generate"     # 安全阀：决策到顶，强制出结果
last_msg = messages[-1]
if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
    return "pick_next_tool"        # LLM 想调工具 → 去岔路口
return "generate"                  # 没调工具 → 直接生成
```

### `pick_next_tool(state)`（`edges.py:34`）—— 岔路口：挑下一个未执行的工具

```
倒序遍历本轮 AIMessage 的 tool_calls：
  找第一个"还没有对应 ToolMessage"的调用（即未执行）
  返回它的工具名 → graph 分发到 retrieve/web_search/graph_query
全已执行 → return "generate"
```

### `should_continue(state)`（`edges.py:57`）—— 工具执行后的回指

```
当前轮还有未执行的 tool_call → return "pick_next_tool"   # 继续执行下一个工具
全部执行完 → return "router"                              # 回 router 再决策（可能决定再搜或 generate）
```

> **设计要点**
> - **三个纯函数只读 state，无副作用**——这是 LangGraph 条件边的约定：函数返回目标节点名字符串，`graph.py` 用字典映射成实际节点。
> - **`should_continue` 回 `router` 而非 `generate`** 是实现"多轮工具调用"的关键：retrieve 完可能还要 web_search（如对比型问题缺一方），回 router 让 LLM 看新结果再决定。全走完才 generate。
> - `pick_next_tool` 的扫描逻辑与 `nodes._find_pending_tool_call` **重复实现**——两处都在找未执行 tool_call（edges 返回"名字"，nodes 返回"整个调用"）。这是轻度重复，可抽公共函数，属可优化项。【推断】理由：两者判断逻辑几乎一致。证据：`edges.py:40-53` vs `nodes.py:311-339`。

## 该文件在链路中的位置

```
router → route_after_router → pick_next_tool → (工具) → should_continue → 回 pick_next_tool/router → ... → generate
```
