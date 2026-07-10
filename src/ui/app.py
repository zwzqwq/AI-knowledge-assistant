"""
Streamlit 用户界面 —— 纯 HTTP 客户端 / 渲染层

Phase 4 关键变化：
  之前：UI 直接 import ChatService，和业务逻辑耦合在同一进程
  现在：UI 只通过 httpx 异步 HTTP 客户端调 FastAPI 端点

为什么这样设计：
  1. 前后端分离 — Streamlit 的 rerun 不会阻塞 LLM 推理
  2. 各自独立部署/压测 — FastAPI 专注于 /chat/stream 的吞吐
  3. HTTP 是最稳定的集成协议 — 以后换前端（React/Vue）零改动后端

API 端点一览：
  GET  /health              → 健康检查 + 向量库状态
  POST /chat/stream         → SSE 流式对话 ★
  POST /documents           → 上传文档
  POST /sessions            → 新建会话
  GET  /sessions            → 列出所有会话
  GET  /session/{id}/history → 获取对话历史
  DELETE /session/{id}       → 删除会话
  DELETE /session/{id}/history → 清空历史
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st
import httpx
import json

# API 底层地址 — Streamlit 和 FastAPI 都在本机
API_BASE = "http://127.0.0.1:8000"


# ─── 辅助函数 ─────────────────────────────────────────────────

def api_url(path: str) -> str:
    """拼完整 URL"""
    return f"{API_BASE}{path}"


# ─── 页面配置 ──────────────────────────────────────────────────

st.set_page_config(page_title="AI 知识库助手", page_icon="📚", layout="wide")
st.title("📚 AI 知识库助手")


# ─── 初始化 session 状态 ────────────────────────────────────────

if "session_id" not in st.session_state:
    import uuid
    st.session_state.session_id = uuid.uuid4().hex[:8]

if "messages" not in st.session_state:
    st.session_state.messages = []

if "db_ready" not in st.session_state:
    st.session_state.db_ready = False


# ─── 侧边栏：文档上传 + 会话管理 ──────────────────────────────────

with st.sidebar:
    st.header("📄 文档管理")

    uploaded_files = st.file_uploader(
        "上传文档（.txt / .md）",
        type=["txt", "md"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        for f in uploaded_files:
            try:
                content = f.read().decode("utf-8")
                resp = httpx.post(
                    api_url("/documents"),
                    json={"filename": f.name, "content": content},
                    timeout=30,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    st.success(f"✅ {f.name} — 已入库 {data['chunks']} 个片段")
                else:
                    st.error(f"❌ {f.name} 上传失败: {resp.text}")
            except Exception as e:
                st.error(f"❌ {f.name} 上传失败: {e}")

    # 检查向量库状态
    try:
        resp = httpx.get(api_url("/health"), timeout=5)
        if resp.status_code == 200:
            db_ready = resp.json().get("db_ready", False)
            st.session_state.db_ready = db_ready
            if db_ready:
                st.success("📊 向量库就绪")
            else:
                st.warning("📊 向量库为空，请上传文档")
    except Exception:
        st.warning("⚠️ 后端未启动，请先运行: uvicorn run_api:app --reload --port 8000")

    st.divider()
    st.header("💬 会话管理")

    st.caption(f"当前: `{st.session_state.session_id[:8]}...`")

    if st.button("🆕 新建会话", use_container_width=True):
        try:
            resp = httpx.post(api_url("/sessions"), timeout=5)
            if resp.status_code == 200:
                st.session_state.session_id = resp.json()["session_id"]
                st.session_state.messages = []
                st.rerun()
        except Exception:
            # fallback: 本地生成
            import uuid
            st.session_state.session_id = uuid.uuid4().hex[:8]
            st.session_state.messages = []
            st.rerun()

    # 从后端列会话
    try:
        resp = httpx.get(api_url("/sessions"), timeout=5)
        if resp.status_code == 200:
            all_sessions = resp.json().get("sessions", [])
        else:
            all_sessions = []
    except Exception:
        all_sessions = []

    if all_sessions:
        selected = st.selectbox(
            "切换会话",
            all_sessions,
            index=all_sessions.index(st.session_state.session_id)
            if st.session_state.session_id in all_sessions else 0,
        )
        if selected != st.session_state.session_id:
            st.session_state.session_id = selected
            # 从后端拉取历史
            try:
                resp = httpx.get(
                    api_url(f"/session/{selected}/history"), timeout=5
                )
                if resp.status_code == 200:
                    st.session_state.messages = resp.json().get("messages", [])
            except Exception:
                st.session_state.messages = []
            st.rerun()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ 清空对话", use_container_width=True):
            try:
                httpx.delete(
                    api_url(f"/session/{st.session_state.session_id}/history"),
                    timeout=5,
                )
            except Exception:
                pass
            st.session_state.messages.clear()
            st.rerun()
    with col2:
        if st.button("❌ 删除会话", use_container_width=True):
            try:
                httpx.delete(
                    api_url(f"/session/{st.session_state.session_id}"),
                    timeout=5,
                )
            except Exception:
                pass
            st.session_state.messages = []
            try:
                resp = httpx.post(api_url("/sessions"), timeout=5)
                if resp.status_code == 200:
                    st.session_state.session_id = resp.json()["session_id"]
            except Exception:
                import uuid
                st.session_state.session_id = uuid.uuid4().hex[:8]
            st.rerun()


# ─── 主区域：对话 ────────────────────────────────────────────────

# 渲染已有消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("source"):
            source_label = {
                "knowledge_base": "📚 来自知识库",
                "llm": "💡 基于 AI 自身知识",
            }.get(msg["source"], f"🔗 {msg['source']}")
            st.caption(source_label)

# 输入框
prompt = st.chat_input("输入你的问题...")

if prompt:
    # 显示用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    if not st.session_state.db_ready:
        st.error("向量库尚未初始化，请先上传文档")
    else:
        with st.chat_message("assistant"):
            try:
                status_placeholder = st.empty()
                answer_placeholder = st.empty()

                full_answer = ""
                source = "unknown"

                # ── HTTP SSE 流式请求 ──
                with httpx.stream(
                    "POST",
                    api_url("/chat/stream"),
                    json={
                        "message": prompt,
                        "session_id": st.session_state.session_id,
                    },
                    timeout=120,
                ) as response:
                    for line in response.iter_lines():
                        if line.startswith("data: "):
                            data = json.loads(line[6:])

                            if "source" in data:
                                source = data["source"]
                                source_label = {
                                    "knowledge_base": "📚 知识库",
                                    "web_search": "🌐 联网搜索",
                                    "llm": "💡 AI 自身知识",
                                }.get(source, "")
                                status_placeholder.caption(f"来源: {source_label}")

                            elif "content" in data:
                                full_answer += data["content"]
                                answer_placeholder.write(full_answer + "▌")

                            elif "error" in data:
                                st.error(data["error"])

                            elif "done" in data:
                                pass

                answer_placeholder.write(full_answer)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_answer,
                    "source": source,
                })

            except Exception as e:
                st.error(f"回答生成失败: {e}")
