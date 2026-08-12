# 13 · prompts.py 参数

> 文件：`src/agent/prompts.py`（既有），模块级常量 + 函数
> 功能：**Prompt 组装**——Router 的决策规则（告诉 LLM 怎么用工具）+ Generate 的三种场景回答模板。

## 构造参数

| 常量/函数 | 参数 | 大白话 | 技术性 |
|------|------|--------|--------|
| `ROUTER_SYSTEM_PROMPT` | 无 | 决策模型的系统指令 | 模块级常量，含工具列表 + 决策规则 |
| `build_generate_prompt(user_question, tool_results_text)` | 2 个字符串 | 生成回答的提示词 | 按问题类型分支 |

## 读取的 config（只列本功能链路消费的）

| config 参数 | 默认值 | 大白话 | 技术性 | 谁消费 |
|------------|--------|--------|--------|--------|
| 无 | — | prompt 内容硬编码 | 不读 config | — |

## 核心方法

### `ROUTER_SYSTEM_PROMPT`（`prompts.py:3`）—— 决策规则（关键）

核心决策规则（优先级序）：
```
1. 首次收到用户消息 → 必须调 retrieve + graph_query，不输出文字
2. 收到检索结果后 → 判断是否相关：
     不相关 → 必须调 web_search
     对比型问题（A vs B）→ 检查检索是否覆盖双方，缺一方就 web_search
3. 收到 web_search 结果 → 直接 generate
4. 重试纪律：同工具最多重试 2 次（共 3 次），重试必须显著改查询参数
5. 禁止：不反问/不引导/不打招呼，唯一表达方式是 tool_calls
```

### `build_generate_prompt(user_question, tool_results_text)`（`prompts.py:27`）

```
is_comparison = 问题含 ["对比","区别","不同","哪个好","优劣","比较"," vs ","VS"]
分支：
  ① 无工具结果   → "请根据你的自身知识回答。如果不知道就说不知道，不要编造"
  ② 对比型       → 分三结构：各方核心特征 → 核心维度区别 → 缺信息用自身知识补并诚实说明
  ③ 普通有结果   → 先充分回答核心问题 → 再自然过渡延伸 → 无关检索结果直接忽略
```

> **设计要点**
> - **Router 的"人格"是纯调度器**：prompt 反复强调"你不是 chatbot，不能直接回答用户问题""唯一表达方式是 tool_calls"——防止决策模型提前开始说人话、绕过工具。
> - **决策规则 = 检索兜底链**：规则 1 强制先查知识库+图谱，规则 2 的相关性判断决定要不要联网——这是"知识库 → 图谱 → 联网 → 自身知识"降级链的 prompt 侧实现。
> - **对比型问题是专门场景**：检测关键词后生成结构化对比模板，说明项目对"对比/区别"类问题有专门设计。
> - **防编造**：Generate prompt 统一带"不知道就说不知道，不要编造"。
> - 【推断】对比型关键词用中文为主（"对比/区别/哪个好"）+ 一个英文 `vs`，说明面向中文用户。证据：`prompts.py:31` 关键词列表。

## 该文件在链路中的位置

```
router_node → ROUTER_SYSTEM_PROMPT（SystemMessage）→ 决策
generate_node → build_generate_prompt(user_question, tool_results_text) → SystemMessage → 生成
```
