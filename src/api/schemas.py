"""
Pydantic 请求/响应模型 —— 定义 API 的"数据合同"

Pydantic 的作用：
  前端发来 JSON → Pydantic 校验字段类型是否正确 → 交给 Service 处理
  Service 返回数据 → Pydantic 序列化为 JSON → 返回给前端

为什么需要它：
  没有校验的话，`data["message"]` 可能不存在、可能是数字、可能是 None，
  Pydantic 在进入业务逻辑之前就把这些问题拦住了。
"""
from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════════
# 请求模型（前端 → 后端）
# ═══════════════════════════════════════════════════════════

class ChatRequest(BaseModel):
    """发送消息的请求体"""
    message: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="用户输入的问题",
        examples=["什么是 RAG？"],
    )
    session_id: str = Field(
        default="default",
        description="会话 ID，用于区分不同用户/会话的对话历史",
    )


class DocumentUploadRequest(BaseModel):
    """上传文档的请求体"""
    filename: str = Field(
        ...,
        description="文件名（含后缀）",
        examples=["mysql_guide.md"],
    )
    content: str = Field(
        ...,
        min_length=1,
        description="文档的文本内容",
    )

class DocumentInfo(BaseModel):
    """文档信息"""
    filename: str = Field(
        description="文件名（含后缀）",
        examples=["mysql_guide.md"],
    )
    chunks: int = Field(
        description="文档切片数量",
    )


class DocumentListResponse(BaseModel):
    """文档列表响应体"""
    documents: list[DocumentInfo] = Field(
        description="文档列表",
    )

class StatusResponse(BaseModel):
    """状态响应体"""
    status: str = Field(
        description="操作状态",
        examples=["ok"],
    )


# ═══════════════════════════════════════════════════════════
# 响应模型（后端 → 前端）
# ═══════════════════════════════════════════════════════════

class ChatResponse(BaseModel):
    """非流式对话的响应体（Phase 2 LangGraph 阶段用）"""
    answer: str
    source: str = Field(
        description="回答来源：knowledge_base / llm / web_search",
    )


class ErrorResponse(BaseModel):
    """统一的错误响应格式"""
    error: str
    detail: str = ""
