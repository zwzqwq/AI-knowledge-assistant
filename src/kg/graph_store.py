"""
知识图谱存储 —— 基于 NetworkX 构建和查询实体关系网络

核心数据结构：
  - 节点（Node）：实体，如 "InnoDB"、"事务"、"MySQL"
  - 边（Edge）：关系，如 ("InnoDB", "事务") 标注为 "支持"
  - 图类型：有向图（DiGraph），关系有方向性（A 支持 B ≠ B 支持 A）

持久化：
  JSON 文件（{project_root}/data/knowledge_graph.json）
  结构：{"nodes": {...}, "edges": [...]}

查询能力：
  给定实体名 → 返回所有相邻实体 + 关系类型
  支持双向查找（入边 + 出边）
"""
import os
import json

from src.config import config, logger

# ⚠️ NetworkX 只有在需要时才会被导入，避免 Streamlit 启动时的潜在导入错误
# 实际导入在 __init__ 中的 _init_graph() 里完成

# 图谱 JSON 文件路径
KG_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "knowledge_graph.json"
)


class GraphStore:
    """
    知识图谱存储 —— 单例模式

    方法：
      add_triples(triples)  →  批量添加实体关系
      query(entity)         →  返回 {实体, 邻居: [(关系, 目标实体), ...]}
      stats()               →  返回 节点数/边数
      save() / load()       →  持久化
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._graph = self._init_graph()
        self._load()  # 从 JSON 加载已有数据
        self._initialized = True

    @staticmethod
    def _init_graph():
        """延迟导入 NetworkX，防止意外的模块顺序错误"""
        import networkx as nx
        return nx.DiGraph()

    def add_triples(self, triples: list[tuple[str, str, str]]):
        """
        批量添加三元组

        triples: [(source, relation, target), ...]
          例: [("InnoDB", "支持", "事务"), ("MySQL", "属于", "关系型数据库")]
        """
        added = 0
        for source, relation, target in triples:
            source = source.strip()
            target = target.strip()
            relation = relation.strip()
            if not source or not target or not relation:
                continue
            # 添加节点（带计数——出现次数越多越重要）
            if source in self._graph:
                self._graph.nodes[source]["count"] = self._graph.nodes[source].get("count", 1) + 1
            else:
                self._graph.add_node(source, count=1)

            if target in self._graph:
                self._graph.nodes[target]["count"] = self._graph.nodes[target].get("count", 1) + 1
            else:
                self._graph.add_node(target, count=1)

            # 添加边（重复边叠加权重）
            if self._graph.has_edge(source, target):
                self._graph[source][target]["weight"] += 1
            else:
                self._graph.add_edge(source, target, relation=relation, weight=1)
            added += 1

        logger.info(f"KG: 已添加 {added} 条边（当前: {self._graph.number_of_nodes()} 节点, "
                     f"{self._graph.number_of_edges()} 边）")
        self._save()

    def query(self, entity: str, max_neighbors: int = 5) -> dict:
        """
        查询实体及其邻居

        返回:
          {"entity": "InnoDB",
           "neighbors": [("支持", "事务"), ("是", "存储引擎"), ...],
           "in_edges": [("MySQL", "包含")],  # 哪些实体指向它
           "total_connections": 8}
        """
        entity = entity.strip()

        if entity not in self._graph:
            # 模糊匹配：尝试在节点中查找包含该词的实体
            matches = [n for n in self._graph.nodes() if entity.lower() in n.lower()]
            if matches:
                # 取最重要（出现次数最多）的匹配
                entity = max(matches, key=lambda n: self._graph.nodes[n].get("count", 0))
                logger.info(f"KG 查询 '{entity}': 模糊匹配到 '{entity}'")
            else:
                return {"entity": entity, "neighbors": [], "in_edges": [], "total_connections": 0}

        # 出边（entity → others）
        neighbors = []
        for _, target, data in self._graph.out_edges(entity, data=True):
            neighbors.append((data.get("relation", "关联"), target, data.get("weight", 1)))

        # 入边（others → entity）
        in_edges = []
        for source, _, data in self._graph.in_edges(entity, data=True):
            in_edges.append((source, data.get("relation", "关联"), data.get("weight", 1)))

        # 按权重排序（重要关系优先），截断到 max_neighbors
        neighbors.sort(key=lambda x: x[2], reverse=True)
        in_edges.sort(key=lambda x: x[2], reverse=True)
        neighbors = neighbors[:max_neighbors]
        in_edges = in_edges[:max_neighbors]

        total = self._graph.degree(entity)
        logger.info(f"KG 查询 '{entity}': {len(neighbors)} 出边, {len(in_edges)} 入边 (共 {total} 连接)")

        return {
            "entity": entity,
            "neighbors": [(r, t) for r, t, w in neighbors],
            "in_edges": [(s, r) for s, r, w in in_edges],
            "total_connections": total,
        }

    def query_to_text(self, entity: str) -> str:
        """查询结果转为 LLM 可用的文本"""
        result = self.query(entity)
        if result["total_connections"] == 0:
            return f"（知识图谱中未找到与「{entity}」相关的实体）"

        lines = [f"知识图谱查询结果 —— 「{result['entity']}」:"]
        lines.append(f"  共 {result['total_connections']} 个关联")

        if result["neighbors"]:
            lines.append("  关联到的实体:")
            for relation, target in result["neighbors"]:
                lines.append(f"    - {result['entity']} {relation} {target}")

        if result["in_edges"]:
            lines.append("  被以下实体关联:")
            for source, relation in result["in_edges"]:
                lines.append(f"    - {source} {relation} {result['entity']}")

        return "\n".join(lines)

    def stats(self) -> dict:
        """图谱统计"""
        return {
            "nodes": self._graph.number_of_nodes(),
            "edges": self._graph.number_of_edges(),
            "top_entities": sorted(
                self._graph.nodes(data=True),
                key=lambda x: x[1].get("count", 0),
                reverse=True
            )[:10],
        }

    # ── 持久化 ─────────────────────────────────────────────────

    def _save(self):
        """保存为 JSON"""
        os.makedirs(os.path.dirname(KG_FILE), exist_ok=True)

        nodes_data = {}
        for node, attrs in self._graph.nodes(data=True):
            nodes_data[node] = {"count": attrs.get("count", 1)}

        edges_data = []
        for source, target, attrs in self._graph.edges(data=True):
            edges_data.append({
                "source": source,
                "target": target,
                "relation": attrs.get("relation", "关联"),
                "weight": attrs.get("weight", 1),
            })

        with open(KG_FILE, "w", encoding="utf-8") as f:
            json.dump({"nodes": nodes_data, "edges": edges_data}, f, ensure_ascii=False, indent=2)

    def _load(self):
        """从 JSON 加载"""
        if not os.path.exists(KG_FILE):
            logger.info("KG: 未找到已保存的图谱文件，从空图开始")
            return

        try:
            with open(KG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)

            for node, attrs in data.get("nodes", {}).items():
                self._graph.add_node(node, count=attrs.get("count", 1))

            for edge in data.get("edges", []):
                self._graph.add_edge(
                    edge["source"],
                    edge["target"],
                    relation=edge.get("relation", "关联"),
                    weight=edge.get("weight", 1),
                )

            logger.info(f"KG 已加载: {self._graph.number_of_nodes()} 节点, "
                         f"{self._graph.number_of_edges()} 边 (文件: {KG_FILE})")
        except Exception as e:
            logger.error(f"KG 加载失败: {e}")
            import networkx as nx
            self._graph = nx.DiGraph()
