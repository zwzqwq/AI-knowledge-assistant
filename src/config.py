"""
项目配置中心 —— 所有可调参数集中在这里

路径策略：
  环境变量 / .env 中可以用绝对路径，也可以用相对路径。
  相对路径统一基于项目根目录（config.py 往上两级）解析为绝对路径，
  这样无论从哪个目录运行都不会跑偏。
"""
import os
import logging
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
