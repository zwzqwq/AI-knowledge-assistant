"""
Streamlit 用户界面
本体只负责 UI 渲染和事件处理——核心逻辑全部委托给 src 其他模块。
"""
import streamlit as st

from src.config import config, logger
from src.rag.loader import DocumentLoader
from src.rag.retriever import Retriever
from src.rag.chain import build_chain
from src.memory.history import ConversationHistory


# ─── 页面配置 ──────────────────────────────────────────────

st.set_page_config(page_title="AI 知识库助手", page_icon="📚", layout="wide")
st.title("📚 AI 知识库助手")


# ─── 初始化（只执行一次）───   ────────────────────────────────

@st.cache_resource
def init_retriever():
    """缓存 Retriever 实例，Streamlit 重渲染时不复读硬盘"""
    return Retriever()


retriever_mgr = init_retriever()

if "history" not in st.session_state:
    st.session_state.history = ConversationHistory()

if "chain" not in st.session_state:
    try:
        retriever = retriever_mgr.get_retriever()
        st.session_state.chain = build_chain(retriever)
        st.session_state.db_ready = True
    except RuntimeError:
        st.session_state.db_ready = False

history = st.session_state.history


# ─── 侧边栏：文档上传 + 检索配置 ───────────────────────────────

with st.sidebar:
    st.header("📄 文档管理")
    uploaded_files = st.file_uploader(
        "上传文档（.txt / .md / .pdf）",
        type=["txt", "md", "pdf"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        loader = DocumentLoader()
        for f in uploaded_files:
            try:
                if f.name.lower().endswith(".pdf"):
                    # PDF 需要先保存到临时文件，再用 PyPDFLoader 加载
                    import tempfile, os
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(f.read())
                        tmp_path = tmp.name
                    try:
                        from langchain_community.document_loaders import PyPDFLoader
                        pdf_loader = PyPDFLoader(tmp_path)
                        pdf_docs = pdf_loader.load()
                        # 统一用 splitter 切片
                        chunks = loader.splitter.split_documents(pdf_docs)
                        retriever_mgr.add(chunks)
                        st.success(f"✅ {f.name} — 已入库 {len(chunks)} 个片段")
                    finally:
                        os.unlink(tmp_path)
                else:
                    content = f.read().decode("utf-8")
                    chunks = loader.load_text(content, source_name=f.name)
                    retriever_mgr.add(chunks)
                    st.success(f"✅ {f.name} — 已入库 {len(chunks)} 个片段")
                # 新文档入库后重建 Chain
                st.session_state.chain = build_chain(retriever_mgr.get_retriever(
                    search_type=st.session_state.get("search_type", config.RETRIEVER_SEARCH_TYPE)
                ))
                st.session_state.db_ready = True
            except Exception as e:
                logger.error(f"上传文件失败: {f.name}: {e}")
                st.error(f"❌ {f.name} 上传失败: {e}")

    st.divider()
    st.header("⚙️ 检索配置")

    # 检索策略切换 —— 修改后立即重建 Chain
    search_type = st.radio(
        "检索策略",
        options=["similarity", "mmr"],
        format_func=lambda x: {"similarity": "🔍 相似度搜索", "mmr": "🎲 多样性搜索 (MMR)"}[x],
        index=0 if st.session_state.get("search_type", "similarity") == "similarity" else 1,
        key="search_type_radio",
    )
    if search_type != st.session_state.get("search_type"):
        st.session_state.search_type = search_type
        if st.session_state.get("db_ready"):
            retriever = retriever_mgr.get_retriever(search_type=search_type)
            st.session_state.chain = build_chain(retriever)
        st.rerun()

    if st.session_state.get("db_ready"):
        current_type = st.session_state.get("search_type", "similarity")
        type_label = "相似度搜索" if current_type == "similarity" else "多样性搜索 (MMR)"
        st.info(f"📊 向量库就绪 • 策略: {type_label}")

    st.divider()
    if st.button("🗑️ 清空对话"):
        history.clear()
        st.rerun()


# ─── 主区域：对话 ───────────────────────────────────────────

# 渲染已有消息
for msg in history.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# 输入框
prompt = st.chat_input("输入你的问题...")

if prompt:
    # 显示用户消息
    history.add_user(prompt)
    with st.chat_message("user"):
        st.write(prompt)

    # 调用 RAG Chain
    if not st.session_state.get("db_ready"):
        st.error("向量库尚未初始化，请先上传文档")
    else:
        with st.chat_message("assistant"):
            with st.spinner("思考中..."):
                try:
                    history_str = history.format()
                    result = st.session_state.chain.invoke({
                        "history": history_str,
                        "input": prompt,
                    })
                    answer = result.content
                    st.write(answer)
                    history.add_assistant(answer)
                except Exception as e:
                    logger.error(f"LLM 调用失败: {e}")
                    error_msg = f"抱歉，回答生成失败：{e}"
                    st.error(error_msg)
                    history.add_assistant(error_msg)
