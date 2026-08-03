"""
文档删除功能测试

用假的 Chroma 对象替换真实向量库，验证 delete_by_source 的删除逻辑：
  - 假 Chroma 的 _collection 记录 count / delete(where=...) 调用
  - 不碰真实向量库和真实 embedding 模型

方法：monkeypatch 把 Retriever._get_or_load 替换成返回假对象
"""
from src.rag.retriever import Retriever


class FakeChromaCollection:
    """模拟 ChromaDB 的 collection：只记录 count 和 delete 调用"""

    def __init__(self, chunks_by_source: dict):
        self._data = chunks_by_source  # {"a.txt": [c1, c2], "b.txt": [c3]}
        self.delete_calls = []  # 记录每次 delete 的 where 参数

    def count(self) -> int:
        return sum(len(v) for v in self._data.values())

    def delete(self, where: dict) -> None:
        self.delete_calls.append(where)
        source = where.get("source")
        if source in self._data:
            del self._data[source]


class FakeVectorStore:
    """模拟 Chroma 对象：只有测试需要的 _collection 属性"""

    def __init__(self, chunks_by_source: dict):
        self._collection = FakeChromaCollection(chunks_by_source)


def test_delete_existing_source_removes_all_chunks(monkeypatch):
    """删除已存在的文档 → 返回删除数量，且该文档切片清空"""
    fake_vs = FakeVectorStore({"a.txt": ["c1", "c2"], "b.txt": ["c3"]})
    monkeypatch.setattr(Retriever, "_get_or_load", lambda self: fake_vs)

    r = Retriever()
    deleted = r.delete_by_source("a.txt")

    assert deleted == 2
    # a.txt 已被删除，b.txt 保留
    assert fake_vs._collection.count() == 1
    # where 过滤条件必须精确匹配文档名
    assert fake_vs._collection.delete_calls == [{"source": "a.txt"}]


def test_delete_returns_zero_for_missing_source(monkeypatch):
    """删除不存在的文档 → 返回 0，向量库无变化"""
    fake_vs = FakeVectorStore({"a.txt": ["c1", "c2"]})
    monkeypatch.setattr(Retriever, "_get_or_load", lambda self: fake_vs)

    r = Retriever()
    deleted = r.delete_by_source("not_exist.txt")

    assert deleted == 0
    assert fake_vs._collection.count() == 2


def test_delete_other_doc_untouched(monkeypatch):
    """删除文档 A 不应影响文档 B 的切片"""
    fake_vs = FakeVectorStore({"a.txt": ["c1"], "b.txt": ["c2", "c3", "c4"]})
    monkeypatch.setattr(Retriever, "_get_or_load", lambda self: fake_vs)

    r = Retriever()
    deleted = r.delete_by_source("b.txt")

    assert deleted == 3
    assert fake_vs._collection.count() == 1  # 只剩 a.txt


def test_delete_when_vectorstore_missing(monkeypatch):
    """向量库从未创建（_get_or_load 返回 None）→ 返回 0，不抛异常"""
    monkeypatch.setattr(Retriever, "_get_or_load", lambda self: None)

    r = Retriever()
    deleted = r.delete_by_source("a.txt")

    assert deleted == 0
