"""
项目配置中心 —— 所有可调参数集中在这里

路径策略：
  环境变量 / .env 中可以用绝对路径，也可以用相对路径。
  相对路径统一基于项目根目录（config.py 往上两级）解析为绝对路径，
  这样无论从哪个目录运行都不会跑偏。
"""
import os
import logging
from logging.handlers import TimedRotatingFileHandler
from dotenv import load_dotenv

# 项目根目录 —— 根据 config.py 自身位置算出，不依赖工作目录
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# load_dotenv 默认从工作目录找 .env，这里显式指定项目根目录下的 .env
load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))


def _resolve_path(value: str) -> str:
    """如果是相对路径，拼上项目根目录转成绝对路径；绝对路径原样返回"""
    if not os.path.isabs(value):
        return os.path.normpath(os.path.join(_PROJECT_ROOT, value))
    return value


class AppConfig:
    """应用的全局配置。所有模块从这里取值，不直接读 os.environ。"""

    # ── LLM ──
    LLM_API_KEY: str = os.environ.get("DEEPSEEK_API_KEY", "")
    LLM_BASE_URL: str = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    LLM_MODEL: str = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
    LLM_TEMPERATURE: float = float(os.environ.get("LLM_TEMPERATURE", "0.7"))
    LLM_MAX_TOKENS: int = int(os.environ.get("LLM_MAX_TOKENS", "2048"))
    LLM_TIMEOUT: int = int(os.environ.get("LLM_TIMEOUT", "60"))
    LLM_MAX_RETRIES: int = int(os.environ.get("LLM_MAX_RETRIES", "2"))

    # ── Embedding ──
    EMBEDDING_MODEL: str = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
    EMBEDDING_CACHE_DIR: str = _resolve_path(
        os.environ.get("EMBEDDING_CACHE_DIR", "./bge_model")
    )

    # ── ChromaDB ──
    CHROMA_DB_DIR: str = _resolve_path(
        os.environ.get("CHROMA_DB_DIR", "./chroma_db")
    )

    # ── RAG 参数 ──
    CHUNK_SIZE: int = 500  # 每个文档切片的最大字符数（太大降低检索精度，太小丢失上下文）
    CHUNK_OVERLAP: int = 50  # 相邻切片的重叠字符数（保证句子完整，避免语义断裂）
    RETRIEVER_K: int = 3  # 检索返回的文档片段数量（越多越全，但也可能引入噪声）
    RETRIEVER_SEARCH_TYPE: str = "similarity"  # 检索策略：similarity（纯相似度）| mmr（最大边际相关性，平衡多样性）
    RETRIEVER_CANDIDATES: int = 10  # 混合检索召回阶段每路的候选数 N（须 > 精排 k，给 Rerank 足够的挑选空间）
    RETRIEVER_RERANK_TOP_K: int = 3  # 精排后最终保留的文档数（喂给 LLM 的精准 top-k）

    # ── Rerank ──
    RERANK_MODEL: str = os.environ.get("RERANK_MODEL", "BAAI/bge-reranker-base")
    RERANK_CACHE_DIR: str = _resolve_path(
        os.environ.get("RERANK_CACHE_DIR", "./rerank_model")
    )
    RERANK_MAX_LENGTH: int = int(os.environ.get("RERANK_MAX_LENGTH", "512"))  # 拼入模型的文本最大长度（截断超长文档）
    RERANK_BATCH_SIZE: int = int(os.environ.get("RERANK_BATCH_SIZE", "32"))  # 打分时每批过模型的文档数（越大越快但吃内存）

    # ── 对话 ──
    HISTORY_MAX_TURNS: int = 6  # 保留最近 N 轮对话
    ROUTER_MAX_RECENT_MESSAGES: int = 8  # Router 传给 LLM 时，最多保留最近 N 条消息
    ROUTER_COMPRESS_OLD_MESSAGES: bool = True  # 旧消息压缩为摘要（true）还是直接丢弃（false）
    ROUTER_TOOL_RESULT_MAX_CHARS: int = 200  # Router 传给 LLM 时，ToolMessage 内容最大字符数（截断尾部）
    ROUTER_SUMMARY_MAX_CHARS: int = 800  # 对话摘要（conversation_summary）最大字符数，防止无限膨胀

    # ── 日志 ──
    LOG_LEVEL: str = os.environ.get("LOG_LEVEL", "INFO")


config = AppConfig()

# 应用启动时配置日志
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("knowledge_assistant")

# basicConfig 的根 handler 默认 NOTSET（放行所有级别），显式设为 LOG_LEVEL，
# 否则 DEBUG 会污染控制台。文件 handler 固定收 DEBUG，两者独立。
_console_level = getattr(logging, config.LOG_LEVEL.upper(), logging.INFO)
for _h in logging.getLogger().handlers:
    _h.setLevel(_console_level)

# ── 文件日志：按天轮转，保留 7 天（对齐项目二 src/utils/logger.py） ──
_LOG_DIR = _resolve_path("./logs")
os.makedirs(_LOG_DIR, exist_ok=True)
_file_handler = TimedRotatingFileHandler(
    os.path.join(_LOG_DIR, "app.log"),
    when="midnight",
    interval=1,
    backupCount=7,
    encoding="utf-8",
)
_file_handler.setLevel(logging.DEBUG)
_file_handler.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
))
logger.setLevel(logging.DEBUG)  # 让 DEBUG 能到达文件 handler；控制台级别独立由 _console_level 控制
logger.addHandler(_file_handler)
