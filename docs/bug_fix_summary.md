# knowledge_assistant Bug 修复记录 (2026-07-30)

## 项目背景

基于 LangGraph 的 RAG + GraphRAG 知识库问答系统。用户上传文档 → 切片入库 + 知识图谱抽取 → 提问时 Agent 自主决策调用三个工具（retrieve 查 ChromaDB / graph_query 查 NetworkX 图谱 / web_search 查 Bing）→ SSE 流式输出。

技术栈: LangGraph + DeepSeek API + ChromaDB + NetworkX + FastAPI + Streamlit

---

## Bug 1: Router 工具循环调用 —— 同一种工具被反复追加，其他工具永远没机会执行

### 现象

LLM 一会能联网搜索，一会又傻傻地一直只查知识库。控制台显示 retrieve 被调了 N 次，web_search 和 graph_query 从未执行就达到了 MAX_ITERATIONS。

### 根因

`router_node` 的 tool_calls 合并逻辑只用 `name + args + result` 三者完全相同才判定"已执行过"。

```python
# 旧代码
already_done = any(
    old["name"] == new_tc["name"] and old["args"] == new_tc["args"] and old["result"]
    for old in merged_calls
)
```

LLM 每次微调 query 参数（如 `"mysql"` → `"mysql vs redis"` → `"redis 缓存对比"`），args 不同 → 绕过检测 → 新调用被追加。MAX_ITERATIONS=5 时，三次 retrieve 就消耗了 6 次迭代（router + retrieve × 3），其他工具根本没机会。

更深层的根因：系统用两个平行数据结构存储同一件事 —— `state["messages"]` 和 `state["tool_calls"]`。前者被 router 的 `return {"messages": [response]}` 直接覆盖，后者被手动管理。双源不同步导致了合并逻辑成为必需，而合并逻辑又引入了这个 bug。

### 修复方式 —— 最终方案：架构重构

彻底放弃了 `state["tool_calls"]` 数组，改用 LangGraph 原生的 `add_messages` reducer：

- `state["messages"]` 成为唯一数据源（`Annotated[list[BaseMessage], add_messages]`）
- 每个节点只返回 `{"messages": [新消息]}`，由框架自动追加，永不覆盖
- Router 直接传 `state["messages"]` 给 LLM，不再手动提取用户问题或重建消息对
- 工具节点从 messages 的最后一条 AIMessage.tool_calls 中找到自己未执行的调用
- 工具是否已执行通过 ToolMessage.tool_call_id 与 AIMessage.tool_calls[].id 配对来判断
- 一次 Router 决策产生的所有 tool_calls 连续执行完才回到 Router（LLM 不能中途改主意）
- 迭代计数只在 Router 节点 +1，工具节点不碰

### 涉及文件

- `src/agent/state.py` — 新写，用 add_messages reducer
- `src/agent/graph.py` — 重写，从 538 行缩减到 ~100 行
- `src/agent/nodes.py` — 新写，5 个节点函数
- `src/agent/edges.py` — 新写，3 个条件边纯函数
- `src/agent/prompts.py` — 新写，Router prompt + Generate prompt
- `src/agent/tools.py` — 微调，新增 TOOLS 模块常量
- `src/services/chat_service.py` — 适配，initial_state 去掉 tool_calls/context 字段，来源检测改为扫描 ToolMessage.name

---

## Bug 2: MAX_ITERATIONS 不足 —— 正常流程都走不完

### 现象

三个工具的 happy path 需要 6 次迭代（router + retrieve + router + graph_query + router + web_search），但 MAX_ITERATIONS=5，连正常流程都被截断。

### 修复

`MAX_ITERATIONS = 5` → `MAX_ITERATIONS = 12`

12 的分配：3 工具 × max 3 次执行（1 次初始 + 最多 2 次重试）+ router 间插。

---

## Bug 3: 来源标签只显示一个

### 现象

回答融合了知识库和联网搜索的信息，但前端只标注了一个来源。

### 根因

两处：后端 `chat_service.py` 用 `if/elif` 链判断来源（retrieve 命中后不再检查 web_search）；SSE 只推送单个 `source` 字符串；前端只认单个 `source` 字段。

### 修复

- 后端：`if/elif` 改为独立 `if`，收集所有命中来源到 `sources: list[str]`
- SSE：推送 `{"sources": ["knowledge_base", "knowledge_graph", "web_search"]}`
- 前端：改为接收 `sources` 列表，用 ` + ` 拼接展示多个标签
- 工具节点创建 ToolMessage 时加上 `name` 字段，供 chat_service 区分来源

---

## Bug 4: Router 吐出废话 "用户还没有提问" 且工具结果全部丢失

### 现象

控制台显示 retrieve 和 web_search 都执行了、有结果，但前端输出 "用户还没有提问，等待用户的问题"。

### 根因

Router 节点无 tool_calls 分支的旧代码有三个致命操作：

```python
return {
    "final_answer": response.content,  # LLM 的废话被设为最终回答
    "tool_calls": [],                   # 所有已执行的工具结果被清空
    "messages": [response],
}
```

`route_after_router` 看到 final_answer 不为空 → 直接走 generate。generate_node 进入 final_answer 透传分支。前端收到的是 router LLM 的废话，真正的检索结果全部丢失。

### 修复

在无 tool_calls 分支加了保护：如果还有待执行的工具，忽略 LLM 的"不调工具"决定，不设置 final_answer，继续循环。

（架构重构后，这个问题自然消失 —— 因为不再有无 tool_calls 就直接设置 final_answer 的逻辑。无 tool_calls 的 AIMessage 会走到 generate_node，由 generate_node 根据是否有工具结果来决定行为。）

---

## Bug 5: Generate 回答策略导致对比型问题答非所问

### 现象

用户问 "mysql 与 redis 的对比"，回答是 400 字的 MySQL 百科，Redis 只顺带了一句。

### 根因

旧 generate prompt 的策略是"知识是主菜，检索是佐料" —— 指导 LLM 先用自身知识写百科条目，再拼检索结果。这对单实体定义型问题有效，但对对比型问题完全不适用。知识库只有 MySQL 文档 → LLM 写 MySQL 百科 → 用户得不到对比。

### 修复

`build_generate_prompt()` 根据问题类型分两条路径：

- **对比型**（含"对比/区别/vs"等关键词）：三段式结构引导（A 方特征 → B 方特征 → 对比总结）
- **非对比型**：先充分回答核心问题，再自然地用类比或联想延伸（不禁止联想，但前提是先答到点子上）

---

## Bug 6: 回答过于简短

### 现象

问 "MongoDB 是什么"，回答只有一句话。

### 根因

Prompt 中没有明确的"充分展开"指令。LLM 缺乏指令时倾向于选择最省力的回答方式。

### 修复

在非对比型分支中加入展开要求："给出清晰的定义，说明核心特征或原理，列举关键要点"。

---

## Bug 7: Router 消息窗口未隔离 —— 多轮对话时旧工具消息污染 + Token 浪费

### 现象

多轮对话时，Router 传给 LLM 的消息中包含前几轮的 ToolMessage（几百字的检索结果、图谱查询结果），这些内容占用了大量 Token 但对当前轮的决策毫无价值。同时，当前轮已执行的 ToolMessage 完整内容（通常 300-500 字）被完整传给 Router，而 Router 只需要判断"有没有结果、是否相关"这个粗粒度信号。

### 根因分析

这是两个子问题的叠加：

**子问题 A：历史工具消息全部保留**

```python
# 旧代码
history_to_keep = [
    m for m in history_messages if not _is_plain_message(m)
] + recent_plain
```

`not _is_plain_message(m)` 匹配了 AIMessage(tool_calls) 和 ToolMessage —— 旧轮的工具消息在跨轮场景下完全无用（`_find_pending_tool_call`、`pick_next_tool`、`should_continue`、`generate_node` 全都只扫描当前轮范围），但每条 ToolMessage 动辄 300-500 字，白白占用 Token。

**子问题 B：当前轮 ToolMessage 完整内容全量传给 Router**

Router 的任务是决策"下一步调用哪个工具"或"直接回答"，它只需要粗粒度信号：
- retrieve 有没有命中？（前 50 字足矣）
- web_search 返回了什么主题？（前 80 字足矣）
- 知识图谱有没有找到实体？（前 80 字足矣）

Router 不需要读完 "MySQL 的 InnoDB 引擎支持 ACID 事务，默认隔离级别是 REPEATABLE READ..." 才能判断"检索到了 MySQL 相关信息，不需要再检索了"。

但 Generate 需要完整内容来写出好回答。**Router 和 Generate 对 ToolMessage 内容的需求完全不同，但代码把同一条完整的 ToolMessage 喂给了两者。**

### 工程反思

Router 是路由节点，他的目的是判断调用哪个工具，真正需要全部调用结果的应该是 Generate 节点。一股脑地将所有数据都传给 LLM 是不负责任并且浪费 Token 的。

每个节点只应拿到它做决策所需的最小信息量：
- Router 需要：对话上下文 + 当前轮工具执行结果的**粗粒度摘要**（有没有、是否相关）
- Generate 需要：对话上下文 + 当前轮工具执行结果的**完整内容**（具体数据、细节）

参考 `agentic-rag-for-dummies`（3800+ star）的企业级设计，其 orchestrator 只看到检索结果的返回片段来决定要不要继续搜，完整文档块只在下游的 generate 阶段展开。这是 Agent 系统设计中"信息最小化原则"的体现。

### 修复

**修复 A**：`history_to_keep` 只保留 `recent_plain`（最近 N 条纯对话），旧轮工具消息全部丢弃。

**修复 B**：在当前轮消息中，对 ToolMessage 做截断拷贝传给 Router：

```python
max_tool_chars = config.ROUTER_TOOL_RESULT_MAX_CHARS  # 默认 200
truncated_current: list = []
for msg in current_round_messages:
    if isinstance(msg, ToolMessage) and len(msg.content) > max_tool_chars:
        truncated = msg.model_copy()
        truncated.content = msg.content[:max_tool_chars] + "..."
        truncated_current.append(truncated)
    else:
        truncated_current.append(msg)
```

关键设计：用 `msg.model_copy()` 创建浅拷贝，只截断拷贝的 content。state 里的原始 ToolMessage 不受影响 —— Generate 从 state 读取时拿到的仍是完整内容。

**修复 C**：`_build_router_messages` 返回值从 `tuple[list, dict|None]` 简化为 `list`（一并删除了之前遗留的未使用的 `trim_updates` 逻辑和 `old_plain` 变量）。

### 涉及文件

- `src/config.py` — 新增 `ROUTER_TOOL_RESULT_MAX_CHARS = 200`
- `src/agent/nodes.py` — `_build_router_messages` 重构：历史工具消息丢弃 + 当前轮 ToolMessage 截断 + 返回值简化

---

## 设计疑问记录：should_continue 边界 / 隐式重试 / 旧工具消息驻留

在与 DeepSeek 交流源码后，发现三个设计疑问，其中第三点确认为真实 bug 并已修复。

### 疑问 1：纯对话时 should_continue 会不会死循环？

**结论：不会，当前设计正确。**

`pick_next_tool` 扫描当前轮，若没有任何未执行的 tool_calls，返回 `"generate"`。`route_after_router` 对无 tool_calls 的 AIMessage 也返回 `"generate"`。两条路径都导向 generate → END，纯对话不会进入 `should_continue → router` 的循环。

**值得记录的设计思路**：状态图的两个条件边函数职责分离——
- `route_after_router`：Router 输出后，**有 tool_calls → 执行工具，无 → 直接生成**
- `should_continue`：工具执行后，**当前轮还有未执行 tool_call → 继续，全部完成 → 回 Router 再决策**

纯对话由 `route_after_router` 短路到 generate，永远不会经过工具执行分支。这个"职责分离"设计让每条边只回答一个问题，逻辑清晰。

### 疑问 2：代码里为什么没有显式的工具重试机制？

**结论：这是"隐式重试"设计，正确，不需要改。**

代码只判断 `ToolMessage.tool_call_id` 是否已存在，不判断内容是否"成功"。如果工具执行抛异常或返回不符合预期的内容，它仍会作为一条 ToolMessage 进入 state。

**重试发生的方式**：下一轮 Router 看到这条不理想的结果后，依赖 LLM 的智能判断，很可能再次要求调用同一工具（换一个 query 参数）——这就是重试。上一轮修复 Bug 1 时加的工具调用合并逻辑（上限保护 MAX_RETRIES_PER_TOOL=3）正是为了给这种隐式重试设上限。

**值得记录的设计思路**：不要把重试逻辑写死在代码里（失败 → 自动重试同参数），那会让 Agent 丧失灵活性。把"要不要重试、怎么重试"交给 Router LLM 基于上下文判断，代码层面只需要：
1. 失败结果作为普通 ToolMessage 传给下一轮
2. 用工具调用上限防止无限重试

### 疑问 3（真实 bug）：旧轮工具消息驻留在 state 里，未被清理

**现象**：`_summarize_old_messages` 只对超出 max_recent 的**纯对话**做摘要+删除，但旧轮的 AIMessage(tool_calls) 和 ToolMessage（每条几百字）一直驻留在 `state["messages"]` 里，越积越多，浪费内存且每次 graph 序列化都要处理它们。

**根因**：`removals` 只遍历了 `old_plain`，没包含历史中的工具消息。

```python
# 旧代码
removals = [
    RemoveMessage(id=m.id) for m in old_plain if getattr(m, "id", None)
]
```

**修复**：清理两类历史消息——

```python
# 历史中的纯对话消息（超限部分压缩进摘要后删除）
plain_history = [m for m in history_messages if _is_plain_message(m)]
old_plain = plain_history[:-max_recent] if len(plain_history) > max_recent else []

# 历史中的工具消息 —— 跨轮无用，全部删除
old_tool_messages = [
    m for m in history_messages if not _is_plain_message(m)
]

removals = [
    RemoveMessage(id=m.id)
    for m in (old_plain + old_tool_messages)
    if getattr(m, "id", None)
]
```

同时优化了结构：先计算 `removals`，仅在 `removals` 非空时才更新 messages；仅当 `old_plain` 存在时才调用 LLM 生成摘要（否则跳过，省一次 LLM 调用）。

**涉及文件**：`src/agent/nodes.py` — `_summarize_old_messages` 重构

---

## 历史消息清理加固：对象级保留 + 真正增量摘要

在进一步审查 `_summarize_old_messages` 时，发现两个需要加固的细节。

### 加固 1：对象级保留判断 —— 避免无 ID 消息误删

**问题**：用 `keep_ids`（id 集合）+ `discard(None)` 判断保留，id 缺失的消息会被排除在保留集合外，即使它属于最近 N 条对话也会被误删。

```python
# 旧代码
keep_ids = {getattr(m, "id", None) for m in recent_plain}
keep_ids.discard(None)

if getattr(m, "id", None) and m.id in keep_ids:
    continue  # 保留
```

**修复**：改用**对象级相等判断**，不依赖 id。

```python
if any(m == keep_msg for keep_msg in recent_plain):
    continue  # 保留
```

关键点：
- `BaseMessage` 不可哈希，无法用 `set(recent_plain)`，但列表成员判断 `in`/`any` 走的是 `__eq__`
- `BaseMessage.__eq__` 定义为 "content + id 都相同才算相等"
- 效果：**保留范围内的无 id 消息**（内容与 recent_plain 匹配）→ 正确保留，0 误删
- **超限的无 id 消息** → 无法生成 RemoveMessage → 触发警告并跳过

```python
if getattr(m, "id", None):
    removals.append(RemoveMessage(id=m.id))
else:
    logger.warning(
        f"历史消息缺少 id，无法删除: {type(m).__name__} "
        f"(content前50字: {str(getattr(m, 'content', ''))[:50]})"
    )
```

**验证矩阵**：

| 场景 | 结果 |
|------|------|
| 保留范围内、有 id | 正确保留 |
| 保留范围内、无 id、内容匹配 | 正确保留（0 误删） |
| 超限、有 id | 删除 |
| 超限、无 id | 警告 + 跳过 |
| 纯对话 ≤ max_recent | 无删除、无摘要调用 |

### 加固 2：真正增量摘要 —— 早期事实不被反复压缩丢失

**问题**：之前"已有摘要 + 新增内容 → LLM 重写全部摘要"，早期事实每次都被重新压缩一遍，多次迭代后信息逐轮丢失（压缩损失累积）。

```python
# 旧代码 —— 已有摘要进入 LLM 调用，被重新压缩
summary_prompt = f"""...已有摘要：{existing_summary}...本次新增的旧对话：{old_text}...合并后摘要控制在 50-100 字内..."""
```

**修复**：LLM 只压缩本次新增的旧对话 → 生成独立 30 字片段，然后直接拼接在已有摘要后面。**已有摘要内容不进入本次 LLM 调用**。

```python
# LLM 只处理新增部分，压缩成独立片段
summary_prompt = f"""...本次新增的旧对话：{old_text}...压缩为 30 字以内的摘要片段..."""
new_fragment = response.content.strip()

# 拼接 + 截断：已有摘要原样保留，新片段追加在后面
existing_summary = state.get("conversation_summary", "").strip()
merged = f"{existing_summary} | {new_fragment}".strip(" |")
updates["conversation_summary"] = merged[:config.ROUTER_SUMMARY_MAX_CHARS]
```

关键点：
- **已有摘要不进 LLM 调用** → 早期事实永远原样保留，不会被反复重压缩
- **拼接 + 截断** → `ROUTER_SUMMARY_MAX_CHARS = 800` 防止摘要无限膨胀
- 每次 LLM 只处理"新超出的对话"，生成一个独立片段

**涉及文件**：
- `src/agent/nodes.py` — `_summarize_old_messages` 两处加固
- `src/config.py` — 新增 `ROUTER_SUMMARY_MAX_CHARS = 800`

---

## 重构后的最终架构

```
state.py    AgentState { messages(Annotated[list, add_messages]), iteration, final_answer, conversation_summary }
prompts.py  ROUTER_SYSTEM_PROMPT + build_generate_prompt(question, tool_results)
edges.py    route_after_router / pick_next_tool / should_continue (所有扫描函数限定当前轮范围)
nodes.py    router_node / summarize / retrieve_node / web_search_node / graph_query_node / generate_node
            _build_router_messages() — Router 消息窗口管理（历史截断 + ToolMessage 截断 + 摘要注入）
            _find_pending_tool_call() — 当前轮范围查找未执行 tool_call
graph.py    build_agent_graph() — START → summarize → router → pick_next_tool → tools → should_continue

Graph 流转:
  START → summarize → router → (有 tool_calls?) → pick_next_tool → retrieve/web_search/graph_query
                                → should_continue → 还有未执行? → pick_next_tool (继续执行本轮)
                                                   → 全部执行完? → router
                                → (无 tool_calls?) → generate → END

核心原则:
  1. messages IS the state — 单源真理，add_messages 自动追加
  2. 一次 Router 决策的所有 tool_calls 连续执行完，中间不经过 Router
  3. 迭代只在 Router 计数
  4. 所有扫描函数限定当前轮范围（最后一条 HumanMessage 之后）
  5. Router 只拿决策所需的最小信息：对话摘要 + 最近 N 条纯对话 + 截断后的 ToolMessage
  6. Generate 从 state 读取完整的 ToolMessage，不受 Router 截断影响
  7. 旧轮纯对话压缩进摘要，旧轮工具消息直接删除（跨轮无用）
  8. 条件边职责分离：route_after_router 短路纯对话，should_continue 只管工具执行后的去向
  9. 工具重试是隐式的 —— 失败结果传给下一轮 Router，由 LLM 判断是否重试，代码只管设上限
```
