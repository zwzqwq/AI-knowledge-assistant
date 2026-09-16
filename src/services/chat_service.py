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
import asyncio
import hashlib
import os
import uuid
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
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    ToolMessage,
    BaseMessage,
    messages_to_dict,
    messages_from_dict,
)



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
        # 方向 A：以 LangGraph state["messages"] 为唯一对话数据源。
        # _sessions 是磁盘会话的内存缓存：sid -> {"messages": [BaseMessage], "summary": str}
        # 真正的持久化在 data/sessions/{sid}.json，重启不丢、list_sessions 仍可列出。
        self._sessions: dict[str, dict] = {}
        self._sessions_dir = config.SESSIONS_DIR
        os.makedirs(self._sessions_dir, exist_ok=True)
        self._kg_built = False  # 标记是否已构建过知识图谱
        # 文档级内容指纹表（MD5 → {"filename", "chunks"}），持久化到向量库目录。
        # 作用：① 文档级防重 —— 内容相同的文档即使改名也只入库一次
        #       ② 重启不失效 —— 旧版是内存 set，重启后重传相同内容会重复抽三元组
        # 存放位置在 CHROMA_DB_DIR 内：手动删除向量库时指纹表随之消失，天然保持一致
        self._fingerprint_path = os.path.join(config.CHROMA_DB_DIR, "doc_fingerprints.json")
        self._doc_fingerprints: dict = self._load_fingerprints()

    # ── 文档指纹持久化 ────────────────────────────────────────

    def _load_fingerprints(self) -> dict:
        """启动时从磁盘加载指纹表；文件缺失/损坏按空表处理（不阻断服务启动）"""
        if not os.path.exists(self._fingerprint_path):
            return {}
        try:
            with open(self._fingerprint_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"指纹文件读取失败（按空表处理）: {e}")
            return {}

    def _save_fingerprints(self):
        """指纹表写回磁盘（全量覆盖 —— 表很小，全量写比增量更新简单可靠）"""
        os.makedirs(config.CHROMA_DB_DIR, exist_ok=True)
        with open(self._fingerprint_path, "w", encoding="utf-8") as f:
            json.dump(self._doc_fingerprints, f, ensure_ascii=False, indent=2)

    # ── 文档管理 ──────────────────────────────────────────────

    def add_document(self, content: str, filename: str) -> int:
        """
        将文本内容切片、入库、构建知识图谱

        三级防重体系（从入口到兜底）：
          ① 文档级（本方法）：MD5 内容指纹短路 —— 内容完全相同的文档（即使改名）
             直接拒绝，不切片、不向量化、不建图。指纹持久化，重启不失效。
          ② 切片级（retriever.add）：部分重复的文档只追加真正新增的切片。
          ③ 离线兜底（scripts/deduplicate_store.py）：清理历史存量重复。

        返回：实际新增入库的切片数量（重复上传返回 0，前端如实提示）
        """
        # ── ① 文档级指纹判定：先算指纹再切片，命中即短路（省掉切片+向量化开销）──
        fingerprint = hashlib.md5(content.encode("utf-8")).hexdigest()
        if fingerprint in self._doc_fingerprints:
            dup_name = self._doc_fingerprints[fingerprint].get("filename", "?")
            logger.info(f"文档级去重：'{filename}' 与已入库的 '{dup_name}' 内容完全相同，跳过")
            return 0

        # ── 未命中指纹 → 切片入库（retriever 内部还有切片级防重做第二道防线）──
        chunks = self._loader.load_text(content, source_name=filename)
        added = self._retriever_mgr.add(chunks)

        if added == 0:
            # 罕见路径：指纹表丢了（如文件被手删）但切片都已在库。
            # 补登记指纹、跳过建图 —— 内容没变，重复抽三元组只会叠加边权重。
            logger.info(f"'{filename}' 全部切片已存在（指纹表缺登记），补登记不建图")
            self._doc_fingerprints[fingerprint] = {"filename": filename, "chunks": len(chunks)}
            self._save_fingerprints()
            return 0

        # ── 指纹未命中 = 首次见此内容 → 建知识图谱 ──
        # 部分重复（内容有更新）指纹也不同 → 会走到这里重新抽取（视为新信息）
        self._build_knowledge_graph(chunks, filename)
        self._doc_fingerprints[fingerprint] = {"filename": filename, "chunks": len(chunks)}
        self._save_fingerprints()

        return added

    def delete_document(self, filename: str) -> dict:
        """删除指定文档的向量库切片，并同步清理其内容指纹

        返回: {"filename", "deleted_chunks", "graph_affected"}
          graph_affected 恒为 False — 知识图谱的实体关系无法按文档精确回滚
          （NetworkX 图不记录三元组来自哪个文件），删除只清向量库切片。
          指纹表按文件名反查清理：每个指纹只登记一个文件名（重复内容在入口
          就被文档级去重拒绝），所以命中的指纹就是"这个文件"的指纹。
          不清理的话，删除后重传相同内容会被指纹短路，内容永远进不了库。
        """
        deleted_chunks = self._retriever_mgr.delete_by_source(filename)

        removed_fps = [
            fp for fp, info in self._doc_fingerprints.items()
            if info.get("filename") == filename
        ]
        for fp in removed_fps:
            del self._doc_fingerprints[fp]
        if removed_fps:
            self._save_fingerprints()
            logger.info(f"已清理 {len(removed_fps)} 条文档指纹")

        return {
            "filename": filename,
            "deleted_chunks": deleted_chunks,
            "graph_affected": False,
        }

    def _build_knowledge_graph(self, chunks: list, filename: str = ""):
        """抽取实体关系并更新图谱（防重由调用方 add_document 按内容指纹控制）"""

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
                logger.info(f"知识图谱已更新: +{new_triples} 三元组 (文件: {filename})")
            else:
                logger.info(f"KG: {filename} 未抽取到新的实体关系")
        except Exception as e:
            logger.warning(f"知识图谱构建跳过（非致命错误）: {e}")

    def is_db_ready(self) -> bool:
        """检查向量库是否已初始化"""
        return bool(self._retriever_mgr.exists)

    def get_stats(self) -> dict:
        """返回向量库和知识图谱的统计信息"""
        store = GraphStore()
        return {
            "vector_store": self._retriever_mgr.get_stats(),
            "knowledge_graph": store.get_stats(),
        }

    # ── 对话 ─────────────────────────────────────────────────

    def _session_path(self, session_id: str) -> str:
        """查询单个会话的磁盘文件路径：data/sessions/{sid}.json"""
        return os.path.join(self._sessions_dir, f"{session_id}.json")

    def _load_session(self, session_id: str) -> dict:
        """从磁盘加载单个会话状态；文件缺失/损坏按空会话处理（不阻断服务）

        返回: {"messages": [BaseMessage], "summary": str}
        内存缓存优先命中，避免每次 query 都读盘。
        """
        #先判断当前对象的会话缓存中是否包含当前会话内容
        if session_id in self._sessions:
            return self._sessions[session_id]
        #不存在则尝试从磁盘读取
        path = self._session_path(session_id)
        #磁盘也没找到只能返回空会话
        if not os.path.exists(path):
            return {"messages": [], "summary": ""}
        #磁盘有文件但读取失败也按空会话处理，无异常则执行读取磁盘会话的操作
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            state = {
                "messages": messages_from_dict(data.get("messages", [])),
                "summary": data.get("summary", ""),
            }
            self._sessions[session_id] = state  # 回填缓存：下次同一 sid 直接命中，避免重复读盘
            return state
        except (json.JSONDecodeError, OSError, ValueError) as e:
            logger.warning(f"会话 {session_id} 读取失败，按空会话处理: {e}")
            # 故意不回填缓存：文件损坏按空会话处理是临时兜底，
            # 若缓存死，后续即使磁盘文件被修复/重建，也读不到真实内容
            return {"messages": [], "summary": ""}

    def _save_session(self, session_id: str, messages: list, summary: str):
        """全量写回会话状态（messages 已被 summarize 裁剪，规模可控）

        同步更新内存缓存 + 磁盘文件，保证两者一致。
        """
        self._sessions[session_id] = {"messages": messages, "summary": summary}
        payload = {"messages": messages_to_dict(messages), "summary": summary}
        try:
            with open(self._session_path(session_id), "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except OSError as e:
            logger.warning(f"会话 {session_id} 写盘失败（内存缓存仍有效）: {e}")

    def _attach_sources(self, messages: list, sources: list):
        """把来源徽标挂到最后一条纯 AI 消息的 additional_kwargs，供下次 get_history 恢复

        用 model_copy + 列表原地替换：规避 LangChain 消息 Pydantic 模型可能 frozen 的风险。
        """
        for i in range(len(messages) - 1, -1, -1):
            m = messages[i]
            if isinstance(m, AIMessage) and not m.tool_calls:
                new_kwargs = {**(m.additional_kwargs or {}), "sources": sources}
                messages[i] = m.model_copy(update={"additional_kwargs": new_kwargs})
                break

    def clear_history(self, session_id: str):
        """清空指定 session 的对话历史（保留会话，重置为空）"""
        self._save_session(session_id, [], "")

    def delete_session(self, session_id: str):
        """彻底删除指定会话（内存缓存 + 磁盘文件一并清除）"""
        self._sessions.pop(session_id, None)
        path = self._session_path(session_id)
        if os.path.exists(path):
            try:
                os.remove(path)
            except OSError as e:
                logger.warning(f"会话 {session_id} 磁盘文件删除失败: {e}")

    def get_history(self, session_id: str) -> list[dict]:
        """获取指定会话的历史消息列表（UI 用）

        只暴露纯对话（HumanMessage + 无 tool_calls 的 AIMessage），过滤掉
        Router AIMessage(tool_calls) 和 ToolMessage —— 这些是图内部状态，不该给前端。
        顺带从 additional_kwargs 恢复来源徽标。
        """
        state = self._load_session(session_id)
        out: list[dict] = []
        for m in state["messages"]:
            if isinstance(m, HumanMessage):
                out.append({"role": "user", "content": m.content})
            elif isinstance(m, AIMessage) and not m.tool_calls:
                entry = {"role": "assistant", "content": m.content}
                srcs = (m.additional_kwargs or {}).get("sources")
                if srcs:
                    entry["sources"] = srcs
                out.append(entry)
        return out

    def _determine_sources(self, all_node_messages: list) -> list[str]:
        """扫描 ToolMessage 判断回答来源"""
        sources: list[str] = []
        retrieve_msgs = [m for m in all_node_messages
                         if isinstance(m, ToolMessage) and len(m.content) > 100
                         and "（知识库中未找到" not in m.content
                         and m.name == "retrieve"]
        search_msgs = [m for m in all_node_messages
                       if isinstance(m, ToolMessage) and len(m.content) > 100
                       and "（联网搜索未找到" not in m.content
                       and m.name == "web_search"]
        graph_msgs = [m for m in all_node_messages
                      if isinstance(m, ToolMessage) and len(m.content) > 100
                      and "（知识图谱中未找到" not in m.content
                      and m.name == "graph_query"]

        if retrieve_msgs:
            sources.append("knowledge_base")
        if graph_msgs:
            sources.append("knowledge_graph")
        if search_msgs:
            sources.append("web_search")
        if not sources:
            sources.append("llm")
        return sources

    async def _run_graph(self, agent, initial_state, shared_msgs: list) -> dict:
        """后台跑图，返回 {final_answer, messages, conversation_summary}

        用 stream_mode="values"：每步产出完整 state 快照（add_messages reducer 已应用）。
        最后一份 snapshot = summarize 裁剪后的规范 messages + 最终 summary + final_answer。

        为什么不用默认的 "updates" 模式：
          updates 模式吐的是每节点的"增量动作"，shared_msgs.extend 会把 summarize
          返回的 RemoveMessage（删除指令）也累积进列表 → 持久化会写入脏数据
          （删除指令本身 + 本该被删的旧消息）。values 模式产出"动作执行后的结果"，
          reducer 已真正删掉旧消息，拿到的就是干净的可写盘列表。

        shared_msgs 仍与 stream_chat 共享同一引用：每步用 [:]= 原地替换为最新快照，
        前台凭它在首个 token 到达时判定来源（此时最近一份快照含全部 ToolMessage）。
        （类比 Java：把一个共享的 ArrayList 传给两个线程，一边写一边读 ——
          旧版是 add 逐个追加，现在是 set 整体替换，"共享引用"的思路不变）
        """
        final_state: dict | None = None
        async for snapshot in agent.astream(initial_state, stream_mode="values"):
            final_state = snapshot
            shared_msgs[:] = list(snapshot.get("messages", []))  # 原地替换，保持引用不变
        return {
            "final_answer": (final_state or {}).get("final_answer", ""),
            "messages": shared_msgs,
            "conversation_summary": (final_state or {}).get("conversation_summary", ""),
        }

    async def stream_chat(
        self,
        message: str,
        session_id: str = "default",
    ) -> AsyncGenerator[str, None]:
        """
        SSE 流式对话 —— 真正的逐 token 输出

        使用 asyncio.Queue 桥接：generate 节点执行时逐 token 推入队列，
        chat_service 从队列读取并实时 yield SSE token 事件。
        后台跑图收集 final_answer / messages，前台读队列推 token。
        """
        # ── 加载会话历史（方向 A：state["messages"] 是唯一跨轮数据源） ──
        # 历史消息 + 摘要从磁盘/缓存读出，新问题追加到尾部喂给图。
        # 不再用 ConversationHistory.add_user，也不再拼 context_message 文本 hack ——
        # 多轮上下文现在由 state["messages"] 携带，summarize 节点自动裁剪旧工具消息、
        # 保留最近 N 条纯对话、增量压缩更老的进 conversation_summary。
        session_state = self._load_session(session_id)
        prior_messages: list = session_state["messages"]
        prior_summary: str = session_state["summary"]

        try:
            retriever = self._retriever_mgr.get_hybrid_retriever()
            token_queue = asyncio.Queue()
            agent = build_agent_graph(retriever, token_queue=token_queue)

            # ── 历史 + 新问题 拼成完整 messages ──
            initial_state = {
                "messages": prior_messages + [HumanMessage(content=message)],
                "iteration": 0,
                "final_answer": "",
                "conversation_summary": prior_summary,
            }

            # ── 后台跑图 + 前台读 queue 推 token ──
            # shared_msgs：与 _run_graph 共享的同一个列表对象（引用传递），
            # 前台凭它判定回答来源，全程不需要"等图跑完"
            # （values 模式下每步 [:]= 原地替换为最新完整快照，不再是旧版的逐节点 extend）
            shared_msgs: list = []
            graph_task = asyncio.create_task(
                self._run_graph(agent, initial_state, shared_msgs)
            )

            source_sent = False
            sources: list[str] = []

            while True:
                token = await token_queue.get()
                if token is None:
                    break  # generate 结束（哨兵值）
                if not source_sent:
                    # 关键时序：能收到 token 说明 generate 已开始，
                    # 而图拓扑保证工具节点在此之前全部跑完 → shared_msgs 里已有 ToolMessage，
                    # 直接判定来源即可，绝不能在这里 await graph_task（会把流式卡成一次性输出）
                    sources = self._determine_sources(shared_msgs)
                    yield self._sse_event("source", {"sources": sources})
                    source_sent = True
                yield self._sse_event("token", {"content": token})

            # ── token 吐完再收图：此刻 generate 已推送哨兵，图只剩最后收尾，
            # 这个 await 几乎瞬时完成，只用于取图收尾结果存历史（final_answer +
            # 裁剪后的 messages/summary 写回会话），不影响流式体验 ──
            graph_result = await graph_task
            final_answer = graph_result["final_answer"]
            final_messages = graph_result["messages"]
            final_summary = graph_result["conversation_summary"]

            # 兜底：LLM 返回空内容时循环里一个 token 都没发过，不会走到判断来源的逻辑，这里补一个 source判断
            if not source_sent:
                sources = self._determine_sources(shared_msgs)
                yield self._sse_event("source", {"sources": sources})

            if not final_answer:
                yield self._sse_event("error", {"error": "Agent 未返回结果"})
                return

            # ── 把来源徽标挂到最终 AI 消息，持久化裁剪后的会话状态 ──
            self._attach_sources(final_messages, sources)
            self._save_session(session_id, final_messages, final_summary)
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
        """列出所有会话的 ID 列表

        扫描磁盘 .json 文件名，重启后仍可列出旧会话（不再只依赖内存）。
        """
        try:
            return [
                os.path.splitext(f)[0]
                for f in os.listdir(self._sessions_dir)
                if f.endswith(".json")
            ]
        except OSError:
            return []

    def create_session(self, session_id: str = None) -> str:
        """创建一个新会话，返回会话 ID（空会话直接落盘）"""
        if session_id is None:
            session_id = uuid.uuid4().hex[:8]
        self._save_session(session_id, [], "")
        return session_id

# 全局单例（FastAPI 的所有请求共享同一个 ChatService 实例）
chat_service = ChatService()
