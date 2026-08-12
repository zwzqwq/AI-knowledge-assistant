# 15 · config.py 参数

> 文件：`src/config.py`（既有），类 `AppConfig` + 单例 `config`
> 功能：**对话链路参数总表**——本链路所有 `config.XXX` 消费点汇总。环境变量可覆盖默认值（`.env` / 系统环境变量）。

## 构造参数

`AppConfig` 类属性在**定义时从环境变量读取**，无 `__init__` 参数。单例 `config = AppConfig()`（`config.py:77`）。

## 读取的 config（本链路消费的参数总表）

按消费方分组的对话链路参数：

| config 参数 | 默认值 | 大白话 | 技术性 | 谁消费 |
|------------|--------|--------|--------|--------|
| `LLM_MODEL` | `deepseek-chat` | 决策+生成都用这个模型 | ChatOpenAI model | nodes `_make_llm`/`_make_router_llm` |
| `LLM_API_KEY` | `DEEPSEEK_API_KEY` 环境变量 | 密钥 | 认证 | 同上 |
| `LLM_BASE_URL` | `https://api.deepseek.com/v1` | API 地址 | OpenAI 兼容端点 | 同上 |
| `LLM_TEMPERATURE` | `0.7` | 生成回答的随机度 | 生成 LLM 用 | `_make_llm` |
| `LLM_MAX_TOKENS` | `2048` | 回答长度上限 | — | `_make_llm` |
| `LLM_TIMEOUT` | `60` | 单次请求超时（秒） | — | 两个工厂 |
| `LLM_MAX_RETRIES` | `2` | 失败重试次数 | — | 两个工厂 |
| `ROUTER_MAX_RECENT_MESSAGES` | `8` | 决策窗口保留最近几条纯对话 | 窗口管理 | nodes `_build_router_messages` |
| `ROUTER_TOOL_RESULT_MAX_CHARS` | `200` | 决策时工具结果截断长度 | 省 Token | nodes `_build_router_messages` |
| `ROUTER_SUMMARY_MAX_CHARS` | `800` | 摘要上限，防膨胀 | — | nodes `_summarize_old_messages` |
| `HISTORY_MAX_TURNS` | `6` | 会话历史保留几轮 | 跨请求记忆 | history `__init__` |
| `RETRIEVER_SEARCH_TYPE` | `"similarity"` | 召回策略 | — | retriever `get_hybrid_retriever` |
| `RETRIEVER_CANDIDATES` | `10` | 召回阶段每路 N | N>k 给 Rerank 空间 | retriever `get_hybrid_retriever` |
| `RETRIEVER_RERANK_TOP_K` | `3` | 精排后 k | — | Reranker（经 retriever） |
| `CHROMA_DB_DIR` | `./chroma_db` | 向量库磁盘位置 | — | retriever `_get_or_load` |

> 链外参数（Embedding/Rerank/Chunk 等）**不写**——那是文档上传、RAG 进阶等别的功能的页面的事。

## 核心方法

`AppConfig` 无方法，关键机制是**路径解析**：

```
config.py 自身位置 → 推出项目根目录 _PROJECT_ROOT（不依赖工作目录）
_resolve_path()：相对路径拼上根目录转绝对路径（如 ./chroma_db → <根>/chroma_db）
load_dotenv(_PROJECT_ROOT/.env)：显式指定 .env 位置，不从工作目录找
```

### 文件日志（对齐项目二 `src/utils/logger.py`）

```
logging.basicConfig(...) → 控制台（根 handler，级别 = LOG_LEVEL 默认 INFO）
logger.setLevel(DEBUG)
logger.addHandler(TimedRotatingFileHandler(_LOG_DIR/"app.log",
                 when="midnight", interval=1, backupCount=7, encoding="utf-8"))
```

> **设计要点**
> - **控制台与文件级别独立**：根 handler 显式设为 `LOG_LEVEL`（避免 DEBUG 刷屏控制台），文件 handler 固定 DEBUG（方便排查）——需要根 handler 的 NOTSET 陷阱：`basicConfig` 的 handler 默认 NOTSET 会放行所有级别，须显式 `setLevel`。
> - **`_resolve_path("./logs")`**：日志固定落在项目根目录，不随工作目录漂移（比项目二的 `Path("logs")` 更稳）。
> - **按天轮转**：当天写 `logs/app.log`，午夜后自动改名带日期，保留 7 份后删最旧。
> - `logs/` 已加入 `.gitignore`，运行时不进版本库。

> **设计要点**
> - **相对路径基于项目根而非工作目录**：无论从 `scripts/`、`tests/` 还是项目根运行，`./chroma_db` 都指向同一处——避免"换个目录跑就找不到向量库"的坑。证据：`config.py:1-17` 注释。
> - **所有模块只读 `config.XXX`，不直接读 `os.environ`**：参数集中在单一来源，改一处全局生效。
> - `logging.basicConfig` 在 config 导入时执行（`config.py:80`），`logger = logging.getLogger("knowledge_assistant")` 供全项目使用。

## 该文件在链路中的位置

```
config 是所有链上文件的参数来源 → nodes / retriever / history / chat_service 通过 config.XXX 取值
```
