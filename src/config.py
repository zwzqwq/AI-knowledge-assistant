"""
项目配置中心 —— 所有可调参数集中在这里
"""
import os
import logging
from dotenv import load_dotenv

load_dotenv()


class AppConfig:
    """应用的全局配置。所有模块从这里取值，不直接读 os.environ。"""

    # ── LLM ──
    LLM_API_KEY: str = os.environ.get("DEEPSEEK_API_KEY", "")
    LLM_BASE_URL: str = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    LLM_MODEL: str = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

    # ── Embedding ──
    EMBEDDING_MODEL: str = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
    EMBEDDING_CACHE_DIR: str = os.environ.get("EMBEDDING_CACHE_DIR", "./bge_model")

    # ── ChromaDB ──
    CHROMA_DB_DIR: str = os.environ.get("CHROMA_DB_DIR", "./chroma_db")

    # ── RAG 参数 ──
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    RETRIEVER_K: int = 3
    RETRIEVER_SEARCH_TYPE: str = "similarity"  # "similarity" | "mmr"

    # ── 对话 ──
    HISTORY_MAX_TURNS: int = 6  # 保留最近 N 轮对话

    # ── 日志 ──
    LOG_LEVEL: str = os.environ.get("LOG_LEVEL", "INFO")


config = AppConfig()

# 应用启动时配置日志
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("knowledge_assistant")
