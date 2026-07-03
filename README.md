# AI 知识库助手

基于 LangChain + DeepSeek + ChromaDB 的 RAG 知识库问答系统。

## 功能

- 上传 .txt / .md 文档，自动切片入库
- 基于文档内容的多轮对话问答
- 对话历史管理
- 可配置的检索策略（相似度 / MMR）

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env，填入你的 DeepSeek API Key

# 3. 启动
streamlit run app.py
```

## 项目结构

```
src/
├── config.py          # 配置中心
├── rag/
│   ├── loader.py      # 文档加载 & 切片
│   ├── embedder.py    # Embedding 管理
│   ├── retriever.py   # 检索策略
│   └── chain.py       # LCEL RAG 链组装
├── memory/
│   └── history.py     # 对话记忆
└── ui/
    └── app.py         # Streamlit 界面
```

## 技术栈

- Python 3.10
- LangChain (LCEL)
- DeepSeek (OpenAI 兼容接口)
- ChromaDB
- BAAI/bge-small-zh-v1.5 (ModelScope)
- Streamlit
