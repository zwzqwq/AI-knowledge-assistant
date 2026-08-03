# AI 知识库助手

基于 RAG + LangGraph Agent + GraphRAG 的企业级知识库问答系统。上传文档后，通过自然语言提问，Agent 自动组合向量检索、知识图谱查询和联网搜索三种策略生成回答。

## 核心能力

- **多策略检索**：向量检索（语义匹配）+ 知识图谱（实体关系）+ 联网搜索（Bing），Agent 自动选择最优组合
- **SSE 流式对话**：逐字输出，支持多轮追问，带对话历史感知
- **知识图谱自动构建**：上传文档 → LLM 自动抽取实体关系 → NetworkX 存储 → 双向查询
- **前后端分离**：FastAPI 提供 RESTful API + SSE，Streamlit 纯 HTTP 客户端渲染
- **统一错误处理**：LLM 超时/限流/认证/网络异常分类返回可操作的错误信息

## 系统架构

```
用户 ←→ Streamlit (app.py) ──httpx HTTP/SSE──▶ FastAPI (run_api.py)
                                                      │
                                               ChatService (单例)
                                                      │
                                              ┌───────┴───────┐
                                              │ LangGraph Agent │
                                              │  router ─→ retrieve    │
                                              │     ↑       ↓          │
                                              │     └─ graph_query     │
                                              │     ↑       ↓          │
                                              │     └─ web_search      │
                                              │             ↓          │
                                              │          generate      │
                                              └───────┬───────┘
                                                      │
                                        ┌─────────────┼─────────────┐
                                        ▼             ▼             ▼
                                   ChromaDB    NetworkX 图谱    Bing 搜索
                                  (向量检索)   (实体关系)     (联网兜底)
```

**一次对话的完整数据流**：

1. 用户输入 → Streamlit 通过 httpx POST 到 `/chat/stream`
2. ChatService 构建 AgentState，注入对话历史上下文
3. LangGraph Agent 启动决策循环：
   - **router**：LLM（temperature=0）判断是调工具还是直接回答
   - **工具节点**（retrieve/graph_query/web_search）：执行后结果以 ToolMessage 回填
   - **router 再次判断**：信息够了 → 进入 generate，不够 → 继续调其他工具
   - **generate**：LLM（temperature=0.7）汇总所有工具结果生成最终回答
4. 回答以 SSE 流逐字推回 Streamlit 渲染

## 三阶段演进（面试核心亮点）

### Phase 1：FastAPI + 服务层拆分

**问题**：最初 Streamlit 直调 Agent，UI 渲染和 LLM 推理耦合在同一个进程。Streamlit 的 rerun 机制每执行一次操作就重新运行整个脚本，导致 LLM 推理被中断。

**方案**：引入 FastAPI 作为专用后端，ChatService 作为业务层单例。Streamlit 通过 httpx 发 HTTP 请求，`astream()` 实现节点级流式输出。

### Phase 2：LangGraph Agent 替代硬编码路由

**问题**：Phase 1 的检索→回答流程是硬编码的——无论什么问题都先检索向量库，检索不到才降级到 LLM 自身知识。无法处理"先查向量库再查图谱再搜网页"这种多工具组合场景。

**方案**：用 LangGraph StateGraph 构建 Agent 决策循环。router 节点（独立 LLM，temperature=0）根据检索结果动态决定下一步：继续调工具 or 生成回答。三工具（retrieve / graph_query / web_search）+ 最多 5 轮循环。关键设计决策：Router 和 Generate 使用独立 LLM 实例，Router 用 temperature=0 保证决策确定性，Generate 用 temperature=0.7 保证回答多样性。

**踩坑**：tool_calls 被 LangGraph 的"最后写入者胜"机制覆盖，导致已执行的工具结果丢失 → 在 router_node 中实现了合并逻辑；Router 有"反问用户"的 LLM 本能 → prompt 重写，角色从"助手"明确定位为"路由器"，搭配 temperature=0 根治。

### Phase 3：GraphRAG 知识图谱

**问题**：纯向量检索只能做语义相似度匹配，无法回答"MySQL 和 InnoDB 有什么关系？""这个系统有哪些组件？"这类关系型问题。向量检索找到的是相似文本片段，不是结构化关系。

**方案**：引入 GraphRAG。上传文档时 LLM 自动抽取（实体, 关系, 实体）三元组 → NetworkX DiGraph 存储 → `graph_query` 工具支持双向关系查询。查询时支持模糊匹配（如输"Inno"能匹配到"InnoDB"），返回入边和出边。

**权衡**：三元组抽取是昂贵操作（每次调 LLM），通过 filename 去重集合避免重复抽取。

## 快速开始

### 环境要求

- Python 3.10+
- 网络能访问 DeepSeek API（api.deepseek.com）
- 首次运行需下载 Embedding 模型（BAAI/bge-small-zh-v1.5，约 100MB，通过 ModelScope 自动缓存）

### 1. 克隆 & 安装

```bash
git clone https://github.com/zwzqwq/AI-knowledge-assistant.git
cd AI-knowledge-assistant
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env`，填入你的 DeepSeek API Key（在 [platform.deepseek.com](https://platform.deepseek.com) 获取）：

```env
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxx
```

其他配置项保持默认即可。

### 3. 启动

需要两个终端：

```bash
# 终端 1：FastAPI 后端（端口 8000）
uvicorn run_api:app --reload --port 8000

# 终端 2：Streamlit 前端（端口 8501）
streamlit run app.py
```

浏览器打开 `http://localhost:8501` 即可使用。

### 4. 验证

```bash
# 健康检查
curl http://localhost:8000/health

# 上传测试文档
curl -X POST http://localhost:8000/documents \
  -H "Content-Type: application/json" \
  -d '{"content": "MySQL 是一个开源的关系型数据库管理系统。InnoDB 是 MySQL 的默认存储引擎，支持事务和行级锁。", "filename": "mysql_intro.txt"}'

# 查看统计
curl http://localhost:8000/stats

# 流式对话
curl -N -X POST http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "MySQL 的默认存储引擎是什么？"}'
```

## API 文档

启动 FastAPI 后访问 `http://localhost:8000/docs` 查看 Swagger 交互式文档。

| 端点 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 健康检查，返回向量库是否就绪 |
| `/stats` | GET | 系统统计（切片数/文档数/实体数/关系数） |
| `/chat/stream` | POST | SSE 流式对话，请求体 `{"message": "...", "session_id": "..."}` |
| `/documents` | POST | 上传文档内容，请求体 `{"content": "...", "filename": "..."}` |
| `/sessions` | GET | 列出所有会话 ID |
| `/sessions` | POST | 创建新会话，返回 session_id |
| `/session/{id}/history` | GET | 获取指定会话的对话历史 |
| `/session/{id}/history` | DELETE | 清空指定会话的对话历史 |
| `/session/{id}` | DELETE | 彻底删除指定会话 |

### SSE 事件格式

`/chat/stream` 返回 `text/event-stream`，包含三种事件：

```
event: source
data: {"source": "knowledge_base"}     ← 回答来源（knowledge_base / web_search / llm）

event: token
data: {"content": "M"}                 ← 逐字输出

event: done
data: {}                               ← 生成完成
```

## 项目结构

```
knowledge_assistant/
├── .env.example              # 环境变量模板
├── requirements.txt          # Python 依赖
├── run_api.py                # FastAPI 启动入口
├── app.py                    # Streamlit 启动入口
├── data/
│   └── knowledge_graph.json  # 知识图谱持久化文件
├── chroma_db/                # ChromaDB 向量库（自动生成）
├── bge_model/                # Embedding 模型缓存（自动下载）
├── docs/
│   ├── 开发实战问题记录.md     # 20+ 条实战问题 + 面试话术
│   ├── 面试准备.md            # 项目介绍 + 高频追问
│   └── phase3_completion.md  # Phase 3 完成纪要
└── src/
    ├── config.py             # 配置中心（所有可调参数）
    ├── agent/
    │   ├── graph.py          # ★ LangGraph Agent 决策循环
    │   ├── state.py          # AgentState 类型定义
    │   ├── tools.py          # 工具 Function Calling 签名
    │   └── web_search.py     # Bing 搜索 + HTML 解析
    ├── api/
    │   ├── server.py         # FastAPI 端点定义
    │   └── schemas.py        # Pydantic 请求/响应模型
    ├── services/
    │   └── chat_service.py   # ★ 业务服务层（文档/会话/SSE 流式对话）
    ├── rag/
    │   ├── loader.py         # 文档加载 & RecursiveCharacterTextSplitter 切片
    │   ├── embedder.py       # HuggingFaceEmbeddings 管理（单例 + 本地缓存优先）
    │   ├── retriever.py      # ChromaDB 向量存储 + similarity/mmr 检索策略
    │   └── chain.py          # LCEL 管道（检索→格式化→Prompt→LLM）
    ├── kg/
    │   ├── extractor.py      # LLM 实体关系三元组抽取
    │   └── graph_store.py    # NetworkX DiGraph 存储 + 双向查询 + JSON 持久化
    ├── memory/
    │   └── history.py        # 多轮对话管理（自动截断到最近 N 轮）
    └── ui/
        └── app.py            # Streamlit HTTP 客户端（纯渲染层）
```

## 技术栈 & 选型理由

| 技术 | 为什么选它 |
|------|-----------|
| **LangChain + LCEL** | RAG 行业标准框架；LCEL 管道声明式组装（`|` 运算符），比手动拼接 prompt + LLM 调用更易维护 |
| **LangGraph** | Agent 需要非线性的决策循环（router → 工具 → router → 工具 → ...），LangGraph 的 StateGraph 天然支持条件边和循环，比 LangChain 的 AgentExecutor 更可控 |
| **DeepSeek** | 国产模型，中文能力强，OpenAI 兼容接口（LangChain 零改动接入），成本远低于 GPT-4 |
| **ChromaDB** | 嵌入式向量数据库，零配置启动，适合单机部署和原型验证。不选 Pinecone/Weaviate 是因为不需要分布式 |
| **BAAI/bge-small-zh** | 中文 Embedding 开源标杆，ModelScope 国内下载快，small 版本在精度和速度之间平衡好 |
| **NetworkX** | 纯 Python 图算法库，相比 Neo4j 零运维成本。当前数据量可控（万级节点以内），不需要图数据库 |
| **FastAPI** | 异步原生支持，自带 Swagger 文档，SSE 流式响应开箱即用 |
| **Streamlit** | 纯 Python 写 UI，适合原型和内部工具。不选 Gradio 是因为 Streamlit 的状态管理和布局更灵活 |

## 配置参数

所有可调参数集中在 `src/config.py`，部分可通过 `.env` 覆盖：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `LLM_TEMPERATURE` | 0.7 | Generate LLM 温度（Router 固定为 0） |
| `LLM_MAX_TOKENS` | 2048 | 回答最大 token 数 |
| `LLM_TIMEOUT` | 60 | LLM 请求超时（秒） |
| `LLM_MAX_RETRIES` | 2 | LLM 请求失败重试次数（指数退避） |
| `CHUNK_SIZE` | 500 | 文档切片大小 |
| `CHUNK_OVERLAP` | 50 | 切片重叠区间 |
| `RETRIEVER_K` | 3 | 向量检索返回片段数 |
| `RETRIEVER_SEARCH_TYPE` | similarity | 检索策略（similarity / mmr） |
| `HISTORY_MAX_TURNS` | 6 | 对话历史保留轮数 |

## 部署指南

### 方案一：直接部署（Linux 服务器）

```bash
# 安装依赖
pip install -r requirements.txt

# 后端：使用 gunicorn + uvicorn worker（生产级）
pip install gunicorn
gunicorn run_api:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000

# 前端：nohup 后台运行
nohup streamlit run app.py --server.port 8501 --server.address 0.0.0.0 &
```

### 方案二：Docker（推荐）

```dockerfile
# 后端 Dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "run_api:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -t knowledge-assistant .
docker run -d -p 8000:8000 --env-file .env knowledge-assistant
```

### 注意事项

- `.env` 文件包含 API Key，不要提交到 Git 或打包进 Docker 镜像。Docker 部署使用 `--env-file` 或 Docker secrets
- ChromaDB 数据目录（`chroma_db/`）和 Embedding 缓存（`bge_model/`）需要持久化存储，Docker 部署时挂载 volume
- 首次启动会自动从 ModelScope 下载 Embedding 模型（约 100MB），需要网络连接。后续启动使用缓存，不需要网络
- DeepSeek API 需要稳定的网络连接，国内服务器建议配置 HTTP 代理

## 已知局限

1. **流式非 token 级**：当前用 LangGraph 的 `astream()` 实现节点级流式，TTFB（首字节时间）= Agent 完整运行时间。真正的 token 级流式需要 `astream_events` + 节点内部 `llm.stream()`，留给二期
2. **文档粒度限制**：知识库只有某领域的部分文档时，无法回答该领域的其他问题。web_search 是兜底但结果质量不可控
3. **知识图谱规模**：NetworkX 内存图适合万级节点以内，超大规模需迁移到 Neo4j
4. **Router 偶发反问**：检索结果不相关时，router 有小概率输出文字反问而非调 web_search。已通过 prompt + temperature=0 大幅减少但未彻底根除
