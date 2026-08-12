# 09 · state.py 参数

> 文件：`src/agent/state.py`（既有），`AgentState(TypedDict)`
> 功能：**图状态定义**——定义节点间传递的数据结构，`messages` 用 add_messages reducer 自动追加。

## 构造参数

| 参数 | 默认值 | 大白话 | 技术性 |
|------|--------|--------|--------|
| `messages` | 必填 | 消息列表，节点返回的新消息自动追加 | `Annotated[list[BaseMessage], add_messages]` |
| `iteration` | 0 | 记录决策了几轮，防死循环 | `int`，router 每轮 +1 |
| `final_answer` | `""` | 最终回答 | `str`，generate 节点写入 |
| `conversation_summary` | `""` | 旧对话的压缩摘要 | `str`，summarize 节点增量更新 |

## 读取的 config（只列本功能链路消费的）

| config 参数 | 默认值 | 大白话 | 技术性 | 谁消费 |
|------------|--------|--------|--------|--------|
| 无 | — | 状态结构硬编码 | 不读 config | — |

## 核心方法

无方法。关键机制在**类型注解**：

```
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
```

> **设计要点**
> - **`add_messages` reducer 是 LangGraph 的"合并规则"**：节点返回 `{"messages": [新消息]}` 时，LangGraph 自动把新消息**追加**到既有列表，而不是整体覆盖。这是 `chat_service` 能收集"所有节点产出的消息"（`chat_service.py:221`）的基础。
> - **`RemoveMessage` 的配合**：`summarize` 节点要删旧消息，靠返回 `RemoveMessage(id=...)` 让 reducer 按 id 删除——不直接改 state 列表（`nodes.py:160`）。
> - 注释明示设计："state['messages'] 是唯一数据源，不再有平行的 tool_calls 数组或 context 字段"（`state.py:6`）——曾经的设计有平行字段，后来收敛到单一消息流。【推断】证据：`state.py:1-9` 文档字符串。

## 该文件在链路中的位置

```
initial_state（chat_service 构造）→ StateGraph 按 AgentState 类型追踪 → 各节点读写 → 最终 final_answer
```
