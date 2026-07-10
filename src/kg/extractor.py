"""
知识抽取器 —— 用 LLM 从文本中抽取实体和关系

输入：文档文本片段（chunk）
输出：三元组列表 [(实体A, 关系, 实体B), ...]

为什么需要这个：
  RAG 的向量检索只能找到"相似"文本，找不到"相关"实体。
  比如文档写"InnoDB支持事务"，用户问"哪些引擎有事务特性"——
  向量检索找"InnoDB"和"事务"，但图谱知道「InnoDB → supports → 事务」这条边，
  可以直接回答。图谱 = 结构化语义。
"""
import json
import re
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from src.config import config, logger


EXTRACTION_PROMPT = """你是一个知识图谱构建专家。请从以下文本中提取实体和关系，输出 JSON 格式。

规则：
1. 实体：文本中提到的重要概念、技术术语、工具名称、特性等（名词或专有名词）
2. 关系：实体之间的语义连接，用简短动词或短语表示（如"属于""支持""包含""依赖""实现"等）
3. 忽略过于宽泛的实体（如"系统""数据""信息"）
4. 如果文本中没有明确的实体关系，返回空的 {"entities": [], "relations": []}
5. 每个关系必须连接两个已列出的实体

输出格式（严格 JSON，不要其他文字）：
{
  "entities": [
    {"name": "InnoDB", "type": "存储引擎"},
    {"name": "事务", "type": "特性"}
  ],
  "relations": [
    {"source": "InnoDB", "target": "事务", "relation": "支持"}
  ]
}"""


class KnowledgeExtractor:
    """用 LLM 从文本片段中抽取实体关系三元组"""

    def __init__(self):
        self.llm = ChatOpenAI(
            model=config.LLM_MODEL,
            api_key=config.LLM_API_KEY,
            base_url=config.LLM_BASE_URL,
            timeout=config.LLM_TIMEOUT,
            max_retries=config.LLM_MAX_RETRIES,
        )

    def extract(self, text: str) -> list[tuple[str, str, str]]:
        """
        从一段文本中抽取实体关系三元组

        返回: [(source, relation, target), ...]
          例: [("InnoDB", "支持", "事务"), ("MySQL", "属于", "关系型数据库")]
        """
        if not text or len(text) < 20:
            return []

        # 截断过长文本（LLM token 窗口有限）
        text_snippet = text[:2000]

        messages = [
            SystemMessage(content=EXTRACTION_PROMPT),
            HumanMessage(content=f"请从以下文本中提取实体和关系：\n\n{text_snippet}"),
        ]

        try:
            response = self.llm.invoke(messages)
            content = response.content.strip()
            logger.info(f"KG 实体抽取完成，LLM 返回 {len(content)} 字符")
        except Exception as e:
            logger.error(f"KG 实体抽取失败: {e}")
            return []

        triples = self._parse_json(content)
        logger.info(f"KG: 抽取到 {len(triples)} 个三元组")
        return triples

    def extract_from_chunks(self, chunks: list) -> list[tuple[str, str, str]]:
        """
        从多个文档片段中批量抽取（合并去重）
        """
        all_triples = []
        for i, chunk in enumerate(chunks):
            text = chunk.page_content if hasattr(chunk, 'page_content') else str(chunk)
            if len(text) < 20:
                continue
            triples = self.extract(text)
            all_triples.extend(triples)
            if (i + 1) % 5 == 0:
                logger.info(f"KG 进度: {i+1}/{len(chunks)} 个片段已处理")
        logger.info(f"KG 批量抽取完成: {len(all_triples)} 个三元组（{len(chunks)} 个片段）")
        return all_triples

    @staticmethod
    def _parse_json(content: str) -> list[tuple[str, str, str]]:
        """从 LLM 返回内容中解析 JSON → 三元组列表"""
        # 清理 markdown 代码块标记
        content = content.strip()
        content = re.sub(r'^```(?:json)?\s*', '', content)
        content = re.sub(r'\s*```$', '', content)

        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            # 尝试提取第一个 JSON 对象
            match = re.search(r'\{[\s\S]*\}', content)
            if match:
                try:
                    data = json.loads(match.group())
                except json.JSONDecodeError:
                    logger.warning(f"KG 提取: 无法解析 LLM 返回的 JSON，内容前 200 字符: {content[:200]}")
                    return []
            else:
                logger.warning(f"KG 提取: 未找到 JSON 对象，内容前 200 字符: {content[:200]}")
                return []

        triples = []
        relations = data.get("relations", [])
        for rel in relations:
            source = rel.get("source", "").strip()
            target = rel.get("target", "").strip()
            relation = rel.get("relation", "").strip()
            if source and target and relation:
                triples.append((source, relation, target))

        return triples
