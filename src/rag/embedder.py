"""
Embedding 模型管理器
"""
import os
from langchain_huggingface import HuggingFaceEmbeddings
from modelscope.hub.snapshot_download import snapshot_download

from src.config import config, logger


class EmbeddingManager:
    """管理 Embedding 模型的下载与实例化，单例模式避免重复加载"""

    _instance = None

    def __new__(cls):
        # 只有类的_instance实例不存在时，才进行对象的创建以及初始化
        if cls._instance is None:
            # 调用父类的 __new__ 方法创建实例，返回一个空对象
            cls._instance = super().__new__(cls)
            # __init__在每次创建新对象时都会自动运行，设置一个标志位来避免重复初始化模型
            cls._instance._initialized = False
        return cls._instance

    #self拿到的就是__new__返回的_instance实例
    def __init__(self):
        if self._initialized:
            return
        self._embeddings = None
        self._initialized = True

    def _find_local_model(self) -> str | None:
        """在缓存目录中查找已下载的模型路径"""
        import glob
        #glob 是 Python 标准库中的 文件匹配模块 ， glob.glob() 返回所有匹配指定模式的文件路径列表。
        cache_dir = config.EMBEDDING_CACHE_DIR
        # ModelScope 将模型名中的特殊字符转为下划线: "bge-small-zh-v1.5" -> "bge-small-zh-v1*"
        # 用通配符匹配
        patterns = [
            # os.path.join() 会根据操作系统自动选择 / 或 \ ，保证跨平台兼容性。不同平台文件路径的分隔符不同
            os.path.join(cache_dir, "BAAI", "bge-small-zh-v1*"),
            os.path.join(cache_dir, "**", "bge-small-zh-v1*"),
        ]
        for pattern in patterns:
            # 第一个参数为匹配的路径字符串，第二个参数recursive=True，当模式中包含 ** 时， recursive=True 会递归匹配所有子目录
            matches = glob.glob(pattern, recursive=True)
            if matches:
                # 可能有多个匹配结果（比如不同版本的模型），这里取第一个匹配的路径。
                return matches[0]
        return None

    def get(self) -> HuggingFaceEmbeddings:
        """获取（或懒加载）Embedding 实例，优先使用本地缓存"""
        if self._embeddings is None:
            logger.info(f"正在加载 Embedding 模型: {config.EMBEDDING_MODEL}")
            try:
                # 优先本地缓存，避免不必要的联网
                model_path = self._find_local_model()
                if model_path:
                    logger.info(f"使用本地缓存: {model_path}")
                else:
                    logger.info("本地缓存未找到，从 ModelScope 下载...")
                    model_path = snapshot_download(
                        config.EMBEDDING_MODEL,
                        cache_dir=config.EMBEDDING_CACHE_DIR,
                    )
                self._embeddings = HuggingFaceEmbeddings(model_name=model_path)
                logger.info("Embedding 模型加载完成")
            except Exception as e:
                logger.error(f"Embedding 模型加载失败: {e}")
                raise RuntimeError(f"无法加载 Embedding 模型: {e}")
        return self._embeddings
