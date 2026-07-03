"""
文档加载与切片模块
"""
import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import config, logger


class DocumentLoader:
    """加载本地 .txt / .md 文档，切片后返回 chunk 列表"""

    SUPPORTED_EXT = {".txt", ".md"}

    def __init__(self, chunk_size: int | None = None, chunk_overlap: int | None = None):
        self.chunk_size = chunk_size or config.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or config.CHUNK_OVERLAP
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", "。", ".", " ", ""],
        )

    def load_file(self, file_path: str) -> list:
        """从文件路径加载文档并切片"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        #获取文件后缀名并转换为小写处理
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in self.SUPPORTED_EXT:
            raise ValueError(f"不支持的文件格式: {ext}，支持: {self.SUPPORTED_EXT}")

        loader = TextLoader(file_path, encoding="utf-8")
        docs = loader.load()
        chunks = self.splitter.split_documents(docs)
        logger.info(f"已加载 {file_path} -> {len(chunks)} 个切片 (chunk_size={self.chunk_size})")
        return chunks

    def load_text(self, text: str, source_name: str = "upload") -> list:
        """从字符串直接加载并切片（用于 Streamlit 上传场景）"""
        from langchain_core.documents import Document

        doc = Document(page_content=text, metadata={"source": source_name})
        chunks = self.splitter.split_documents([doc])
        logger.info(f"已从文本创建 {len(chunks)} 个切片")
        return chunks
