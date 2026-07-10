"""
FastAPI 服务入口 —— 把对话能力暴露为 HTTP API

核心概念：
  - FastAPI = Starlette + Pydantic 自动校验 + Swagger 自动文档
  - @app.post("/path")  →  定义 POST 接口
  - StreamingResponse   →  返回流式数据（SSE），适合 LLM 逐字输出

启动方式：
  uvicorn src.api.server:app --reload --port 8000

FastAPI 自带 Swagger 文档：
  启动后访问 http://localhost:8000/docs 可以在网页上直接测试所有接口
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from src.api.schemas import ChatRequest, DocumentUploadRequest
from src.services.chat_service import chat_service


# ═══════════════════════════════════════════════════════════
# 创建 FastAPI 应用
# ═══════════════════════════════════════════════════════════

app = FastAPI(
    title="AI 知识库助手 API",
    version="1.0.0",
    description="基于 RAG + Agent 的知识库问答系统 —— 文档上传 / SSE 流式对话 / 向量检索",
)

# CORS：允许浏览器从任何域名调这个 API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ═══════════════════════════════════════════════════════════
# 健康检查
# ═══════════════════════════════════════════════════════════

@app.get("/health")
async def health_check():
    """最简单的健康检查 —— GET /health"""
    return {
        "status": "ok",
        "db_ready": chat_service.is_db_ready(),
    }


# ═══════════════════════════════════════════════════════════
# 对话接口 ★ 最核心
# ═══════════════════════════════════════════════════════════

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    SSE 流式对话

    请求体（JSON）：
        {"message": "什么是 RAG？", "session_id": "abc123"}

    响应：text/event-stream 流，事件格式：
        event: source  →  回答来源（knowledge_base / llm / web_search）
        event: token   →  回答内容（逐字）
        event: done    →  生成完成

    测试方式：
        curl -N -X POST http://localhost:8000/chat/stream \
          -H "Content-Type: application/json" \
          -d '{"message": "什么是MySQL？"}'
    """
    return StreamingResponse(
        chat_service.stream_chat(
            message=request.message,
            session_id=request.session_id,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ═══════════════════════════════════════════════════════════
# 文档管理
# ═══════════════════════════════════════════════════════════

@app.post("/documents")
async def upload_document(request: DocumentUploadRequest):
    """上传文档内容并切片入库"""
    if not request.content.strip():
        raise HTTPException(status_code=400, detail="文档内容不能为空")

    try:
        chunk_count = chat_service.add_document(request.content, request.filename)
        return {
            "status": "ok",
            "chunks": chunk_count,
            "filename": request.filename,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """彻底删除指定 session_id 的会话"""
    try:
        chat_service.delete_session(session_id)
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sessions")
async def list_sessions():
    """列出所有会话 ID"""
    return {"sessions": chat_service.list_sessions()}


@app.post("/sessions")
async def create_session():
    """创建新会话，返回会话 ID"""
    sid = chat_service.create_session()
    return {"session_id": sid}


@app.get("/session/{session_id}/history")
async def get_history(session_id: str):
    """获取指定会话的对话历史"""
    return {"messages": chat_service.get_history(session_id)}


@app.delete("/session/{session_id}/history")
async def clear_history(session_id: str):
    """清空指定会话的对话历史"""
    chat_service.clear_history(session_id)
    return {"status": "ok"}


