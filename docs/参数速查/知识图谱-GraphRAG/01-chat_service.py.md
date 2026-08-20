# 01 · chat_service.py 参数（知识图谱角度）

> 文件：`src/services/chat_service.py`（既有），类 `ChatService`
> 功能：**图谱构建的编排触发点**——文档入库时首次建图、防重复抽取、删除文档旁支。
> 本页只写知识图谱链路消费的部分；对话编排见 [对话问答/04-chat_service.py.md](../对话问答/04-chat_service.py.md)。

## 构造参数

| 参数 | 默认值 | 大白话 | 技术性 |
|------|--------|--------|--------|
| `__init__()` | 无参 | 备好图谱标记 | `self._kg_built = False`、`self._kg_fingerprints: set = set()` |

## 读取的 config（只列本功能链路消费的）

| config 参数 | 默认值 | 大白话 | 技术性 | 谁消费 |
|------------|--------|--------|--------|--------|
| 无 | — | 建图编排不直接读参数 | 参数在 extractor/graph_store 内消费 | — |

## 核心方法

### `add_document(content, filename)`（`chat_service.py:65`）

```
chunks = loader.load_text(content, filename)   # 切片（向量库路径，本链不管）
retriever_mgr.add(chunks)                      # 向量库入库
fingerprint = hashlib.md5(content.encode("utf-8")).hexdigest()
if fingerprint not in self._kg_fingerprints:   # 内容指纹防重：内容相同只建一次图
    self._build_knowledge_graph(chunks, filename)
    self._kg_fingerprints.add(fingerprint)
return len(chunks)
```

### `_build_knowledge_graph(chunks, filename)`（`chat_service.py:99`）

```
try:
    store = GraphStore()                       # 单例
    extractor = KnowledgeExtractor()
    for i, chunk in enumerate(chunks):
        text = chunk.page_content
        if len(text) < 20: continue            # 太短的片段不抽（无实体可言）
        triples = extractor.extract(text)      # LLM 抽取
        if triples: store.add_triples(triples) # 入图（内部自动持久化）
        (i+1) % 5 == 0 → 打进度日志
    if new_triples: _kg_built=True   # 防重由调用方 add_document 按指纹控制
except Exception as e:
    logger.warning(f"知识图谱构建跳过（非致命错误）: {e}")   # ← 不阻塞入库
```

### `delete_document(filename)`（`chat_service.py:82`）—— 旁支

```
deleted = retriever_mgr.delete_by_source(filename)   # 只清向量库切片
# 指纹集合同样不回滚（图谱残留旧三元组，重传同内容不重复抽取）
return {"graph_affected": False}                     # 图谱无法按文档回滚
```

> **设计要点**
> - **`_kg_fingerprints` 是防重复抽取的关键**：按**内容指纹**（MD5 of content）判断，内容完全相同的文档（即使改名重传）只建一次图；比原"文件名防重"强，能挡"改名重传"。
> - **建图失败不阻塞入库**：整个 `_build_knowledge_graph` 在 try/except 里——图谱是增强能力，向量库才是主路径。【推断】证据：`chat_service.py:124-125` 明确注释"非致命错误"。
> - **图谱无法按文档回滚**：NetworkX 不记录三元组来源，删除只清向量库。这是存储简单 vs 删除精确的权衡。【推断】证据：`chat_service.py:83-88` 注释明示设计意图。

## 该文件在链路中的位置

```
add_document（入口）→ _build_knowledge_graph → extractor.extract → graph_store.add_triples → JSON
delete_document（旁支）→ 清向量库 + discard 文件名
```
