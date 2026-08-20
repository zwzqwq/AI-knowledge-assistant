# 02 · extractor.py 参数

> 文件：`src/kg/extractor.py`（既有），类 `KnowledgeExtractor`
> 功能：**实体关系抽取**——LLM 从 chunk 文本抽三元组 (实体A, 关系, 实体B)，严格 JSON 输出 + 容错解析。

## 构造参数

| 参数 | 默认值 | 大白话 | 技术性 |
|------|--------|--------|--------|
| `__init__()` | 无参 | 建一个抽取用的 LLM | `ChatOpenAI(model=LLM_MODEL, ...)`，复用 DeepSeek 通道 |

## 读取的 config（只列本功能链路消费的）

| config 参数 | 默认值 | 大白话 | 技术性 | 谁消费 |
|------------|--------|--------|--------|--------|
| `LLM_MODEL` | `deepseek-chat` | 抽取用哪个模型 | `__init__` 的 ChatOpenAI |
| `LLM_API_KEY` / `LLM_BASE_URL` | 见 07-config | 连哪个 API | `__init__` |
| `LLM_TIMEOUT` / `LLM_MAX_RETRIES` | 60 / 2 | 超时/重试 | `__init__` |

> 注意：抽取 LLM **没传 temperature**（走模型默认）——抽取要"稳定输出 JSON"，此处用模型默认而非 0.7。【推断】证据：`extractor.py:46-52` 无 temperature 参数。

## 核心方法

### `extract(text)`（`extractor.py:54`）★ 主入口

```
if not text or len(text) < 20: return []        # 太短不抽
text_snippet = text[:2000]                       # 截断，防超 token 窗口
llm.invoke([SystemMessage(EXTRACTION_PROMPT), HumanMessage(...)])
  → 异常 → logger.error + return []              # LLM 失败不致命
triples = _parse_json(content)                   # 解析
return triples                                   # [(source, relation, target), ...]
```

### `_parse_json(content)`（`extractor.py:100`）—— 容错解析

```
① 清理 ```json 代码块标记（正则）
② json.loads 直接解析 → 失败
③ 正则提取第一个 JSON 对象 `\{[\s\S]*\}` → 再解析 → 仍失败 return []
④ 取 relations 列表，source/target/relation 三字段非空才收
```

### `EXTRACTION_PROMPT`（`extractor.py:21`）—— 抽取规则

```
- 实体：重要概念/技术术语/工具名/特性（名词或专有名词）
- 关系：简短动词或短语（"属于""支持""包含"）
- 忽略宽泛实体（"系统""数据"）
- 无明确关系 → 返回空的 {"entities": [], "relations": []}
- 每个关系必须连接两个已列出的实体
- 输出格式：严格 JSON，不要其他文字
```

> **设计要点**
> - **"严格 JSON + 不要其他文字"是防解析失败的 prompt 工程**：LLM 常加解释文字导致 json.loads 失败，所以 prompt 强制 + 解析双层容错（代码块清理 + 正则提取）。
> - **失败全部降级为 `[]`**：抽取是增强能力，一个 chunk 抽失败就跳过，不中断整个建图流程。
> - 为什么需要图谱（模块注释标准答案）：向量检索找"相似"文本，图谱找"相关"实体——文档写"InnoDB支持事务"，用户问"哪些引擎有事务特性"，图谱的 `InnoDB → 支持 → 事务` 边能直接回答。

## 该文件在链路中的位置

```
_build_knowledge_graph → extractor.extract(text) → triples → graph_store.add_triples
```
