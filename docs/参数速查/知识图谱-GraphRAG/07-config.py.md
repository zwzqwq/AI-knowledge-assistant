# 07 · config.py 参数（知识图谱角度）

> 文件：`src/config.py`（既有），类 `AppConfig` + 单例 `config`
> 功能：**本链路参数总表**——知识图谱链路消费的 config 参数汇总。环境变量可覆盖默认值。
> 完整参数见 [对话问答/15-config.py.md](../对话问答/15-config.py.md)。

## 构造参数

`AppConfig` 类属性在定义时从环境变量读取，无 `__init__` 参数。单例 `config = AppConfig()`（`config.py:77`）。

## 读取的 config（本链路消费的参数总表）

| config 参数 | 默认值 | 大白话 | 技术性 | 谁消费 |
|------------|--------|--------|--------|--------|
| `LLM_MODEL` | `deepseek-chat` | 抽取用哪个模型 | ChatOpenAI model | extractor `__init__` |
| `LLM_API_KEY` | `DEEPSEEK_API_KEY` 环境变量 | 密钥 | 认证 | extractor `__init__` |
| `LLM_BASE_URL` | `https://api.deepseek.com/v1` | API 地址 | OpenAI 兼容端点 | extractor `__init__` |
| `LLM_TIMEOUT` | `60` | 抽取请求超时（秒） | — | extractor `__init__` |
| `LLM_MAX_RETRIES` | `2` | 抽取失败重试次数 | — | extractor `__init__` |

> 链外参数（Embedding/Chroma/RAG/对话窗口）**不写**——那是别的功能的页面的事。

## 核心方法

`AppConfig` 无方法。本链路相关的机制：

```
路径解析：KG_FILE 不是 config 参数——graph_store.py 用文件自身位置推导
  （os.path.dirname × 3 → 项目根 → data/knowledge_graph.json），不依赖工作目录
```

> **设计要点**
> - **图谱链路只消费 LLM_* 5 个参数**（全部在 extractor 的 ChatOpenAI）——graph_store 不读 config，路径硬编码。
> - 抽取 LLM 未传 `temperature`（走模型默认）——与对话链路"决策 0 / 回答 0.7"不同，抽取要稳定 JSON，此处用默认即可。【推断】证据：`extractor.py:46-52` 无 temperature。

## 该文件在链路中的位置

```
config 是链上文件的参数来源 → extractor 的 ChatOpenAI（LLM_*）消费
```
