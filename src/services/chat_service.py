"""
对话服务层 —— 从 Streamlit UI 中抽出来的纯业务逻辑

为什么要抽离：
  之前的 ui/app.py 既管界面渲染又管调用 AgentChain，两件事耦合在一起。
  以后如果要换前端（比如做 Web 页面），业务逻辑没法复用。

Service 层的职责：
  - 管理对话历史（每个 session 独立）
  - 调用 AgentChain 生成回答
  - 返回结构化的结果（不关心前端怎么展示）

SSE 事件格式：
  标准 SSE 是 text/event-stream，每条消息格式为：
    event: <事件名>
    data: <JSON 数据>

  我们定义三种事件：
    token    → LLM 每生成一个词就推一次（打字机效果）
    source   → 告诉前端回答来源（knowledge_base / llm / web_search）
    done     → 生成完成，前端可以关闭连接
"""
import json
from typing import AsyncGenerator

from openai import (
    APIError,
    APITimeoutError,
    APIConnectionError,
    AuthenticationError,
    RateLimitError,
)

from src.config import config, logger
from src.rag.loader import DocumentLoader
from src.rag.retriever import Retriever
from src.agent.graph import build_agent_graph
from src.kg.graph_store import GraphStore
from src.kg.extractor import KnowledgeExtractor
from src.memory.history import ConversationHistory
from langchain_core.messages import HumanMessage



class ChatService:
    """
    对话服务

    使用方式：
        service = ChatService()
        async for event in service.stream_chat("用户问题", "session_id"):
            # event 是 SSE 格式的字符串，可以直接 yield 到 HTTP 响应
            pass
    """

    def __init__(self):
        self._loader = DocumentLoader()
        self._retriever_mgr = Retriever()
        self._sessions: dict[str, ConversationHistory] = {}
        self._kg_built = False  # 标记是否已构建过知识图谱
        self._kg_doc_names: set = set()  # 已抽取过 KG 的文档名集合

    # ── 文档管理 ──────────────────────────────────────────────

    def add_document(self, content: str, filename: str) -> int:
        """
        将文本内容切片、入库、构建知识图谱（仅新文件时抽取实体）

        返回：入库的切片数量
        """
        chunks = self._loader.load_text(content, source_name=filename)
        self._retriever_mgr.add(chunks)

        # Phase 3: 知识图谱只在首次入库该文件时构建
        # 防止同一文件被重复上传 / 重复抽取的问题
        # 正常情况下，用户上传新文件 → 一次性的实体抽取
        if filename not in self._kg_doc_names:
            self._build_knowledge_graph(chunks, filename)

        return len(chunks)

    def _build_knowledge_graph(self, chunks: list, filename: str = ""):
        """仅首次入库该文件时抽取实体关系并更新图谱"""

        try:
            store = GraphStore()
            extractor = KnowledgeExtractor()

            new_triples = 0
            for i, chunk in enumerate(chunks):
                text = chunk.page_content if hasattr(chunk, 'page_content') else str(chunk)
                if len(text) < 20:
                    continue
                triples = extractor.extract(text)
                if triples:
                    store.add_triples(triples)
                    new_triples += len(triples)
                if (i + 1) % 5 == 0:
                    logger.info(f"KG 进度: {i+1}/{len(chunks)} 片段已处理")

            if new_triples:
                self._kg_built = True
                self._kg_doc_names.add(filename)
                logger.info(f"知识图谱已更新: +{new_triples} 三元组 (文件: {filename})")
            else:
                logger.info(f"KG: {filename} 未抽取到新的实体关系")
        except Exception as e:
            logger.warning(f"知识图谱构建跳过（非致命错误）: {e}")

    def is_db_ready(self) -> bool:
        """检查向量库是否已初始化"""
        return bool(self._retriever_mgr.exists)

    # ── 对话 ─────────────────────────────────────────────────

    def _get_or_create_history(self, session_id: str) -> ConversationHistory:
        """获取或创建指定 session 的对话历史"""
        if session_id not in self._sessions:
            self._sessions[session_id] = ConversationHistory()
        return self._sessions[session_id]

    def clear_history(self, session_id: str):
        """清空指定 session 的对话历史"""
        if session_id in self._sessions:
            self._sessions[session_id].clear()

    def delete_session(self, session_id: str):
        """彻底删除指定会话（从 _sessions 字典中移除）"""
        if session_id in self._sessions:
            del self._sessions[session_id]

    def get_history(self, session_id: str) -> list[dict]:
        """获取指定会话的历史消息列表"""
        if session_id in self._sessions:
            return self._sessions[session_id].messages.copy()
        return []

    async def stream_chat(
        self,
        message: str,
        session_id: str = "default",
    ) -> AsyncGenerator[str, None]:
        """
        SSE 流式对话 —— 真正的逐 token 输出

        Process:
          agent.astream_events(state, version="v2")
            → 监听 on_chat_model_stream → 拿到 LLM 实时生成的每个 token
            → 监听 on_chain_end → 收集工具执行结果（用于判断回答来源）
        """
        history = self._get_or_create_history(session_id)
        history.add_user(message)

        try:
            retriever = self._retriever_mgr.get_retriever()
            agent = build_agent_graph(retriever)

            # ── 注入对话上下文（仅追问场景） ──
            context_message = message
            if history.messages:
                last_answer = ""
                for m in reversed(history.messages):
                    if m["role"] == "assistant":
                        last_answer = m["content"]
                        break
                last_user_msg = ""
                for m in reversed(history.messages):
                    if m["role"] == "user":
                        last_user_msg = m["content"]
                        break
                if last_answer and len(last_answer) > 20:
                    context_message = (
                        f"对话背景——\n"
                        f"上轮用户问题：「{last_user_msg}」\n"
                        f"上轮助手回答：\n{last_answer}\n\n"
                        f"用户现在追问：「{message}」"
                    )

            initial_state = {
                "messages": [HumanMessage(content=context_message)],
                "tool_calls": [],
                "context": "",
                "final_answer": "",
                "iteration": 0,
            }

            # ── astream：逐个节点等待，收集工具结果和最终答案 ──
            # astream 每次 yield {节点名: {字段更新}}，不是完整 state
            # 例: {"generate": {"final_answer": "...", "messages": [...]}}
            final_answer = ""
            all_tool_calls = []

            async for chunk in agent.astream(initial_state):
                print(chunk)
                for node_name, node_output in chunk.items():
                    if node_name in ("retrieve", "web_search", "graph_query"):
                        # 工具节点：收集 tool_calls（含 result）
                        tool_calls = node_output.get("tool_calls", [])
                        if tool_calls:
                            all_tool_calls = tool_calls

                    elif node_name == "generate":
                        # generate 节点：提取最终回答
                        final_answer = node_output.get("final_answer", "")

                    elif node_name == "router":
                        # router 节点：可能直接生成回答（无工具调用场景）
                        router_answer = node_output.get("final_answer", "")
                        if router_answer:
                            final_answer = router_answer

            if not final_answer:
                yield self._sse_event("error", {"error": "Agent 未返回结果"})
                return

            # ── 判断回答来源 ──
            source = "llm"
            has_retrieve = any(
                tc["name"] == "retrieve" and len(tc.get("result", "")) > 100
                and "（知识库中未找到" not in tc.get("result", "")
                for tc in all_tool_calls if tc["result"]
            )
            has_search = any(
                tc["name"] == "web_search" and len(tc.get("result", "")) > 100
                and "（联网搜索未找到" not in tc.get("result", "")
                for tc in all_tool_calls if tc["result"]
            )
            has_graph = any(
                tc["name"] == "graph_query" and len(tc.get("result", "")) > 100
                and "（知识图谱中未找到" not in tc.get("result", "")
                for tc in all_tool_calls if tc["result"]
            )

            if has_retrieve:
                source = "knowledge_base"
            elif has_graph:
                source = "knowledge_base"
            elif has_search:
                source = "web_search"

            yield self._sse_event("source", {"source": source})

            # ── 逐字输出 ──
            for char in final_answer:
                yield self._sse_event("token", {"content": char})

            history.add_assistant(final_answer)
            yield self._sse_event("done", {})

        except AuthenticationError as e:
            logger.error(f"LLM 认证失败 (API Key 无效): {e}")
            yield self._sse_event("error", {"error": "API 密钥无效，请检查 .env 配置"})

        except RateLimitError as e:
            logger.error(f"LLM 请求被限流: {e}")
            yield self._sse_event("error", {"error": "请求过于频繁，请稍后重试"})

        except APITimeoutError as e:
            logger.error(f"LLM 请求超时: {e}")
            yield self._sse_event("error", {"error": "请求超时，请稍后重试"})

        except APIConnectionError as e:
            logger.error(f"LLM 网络连接失败: {e}")
            yield self._sse_event("error", {"error": "无法连接到 AI 服务，请检查网络"})

        except APIError as e:
            logger.error(f"LLM API 错误: status={e.status_code}, message={e.message}")
            yield self._sse_event("error", {"error": f"AI 服务异常 (错误码: {e.status_code})"})

        except Exception as e:
            logger.error(f"对话生成失败 (未分类异常): {e}", exc_info=True)
            yield self._sse_event("error", {"error": "对话生成失败，请稍后重试"})

    @staticmethod
    def _sse_event(event: str, data: dict) -> str:
        """把事件名和数据组装成标准 SSE 字符串"""
        return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
    
    def list_sessions(self) -> list[str]:
        """列出所有会话的ID列表"""
        return list(self._sessions.keys())

    def create_session(self, session_id: str = None) -> str:
        """创建一个新会话，返回会话 ID"""
        if session_id is None:
            import uuid
            session_id = uuid.uuid4().hex[:8]
        self._sessions[session_id] = ConversationHistory()
        return session_id

# 全局单例（FastAPI 的所有请求共享同一个 ChatService 实例）
chat_service = ChatService()
