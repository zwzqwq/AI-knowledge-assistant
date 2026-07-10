"""
Bing 联网搜索工具 —— 当 RAG 检索不到相关内容时自动回退
"""
import re, urllib3
import requests
from src.config import logger

urllib3.disable_warnings()

SEARCH_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
}


def search_bing(query: str, max_results: int = 3) -> list[dict]:
    """
    通过 Bing 搜索并解析结果页

    返回: [{"title": "...", "snippet": "...", "link": "..."}, ...]
    """
    try:
        resp = requests.get(
            "https://www.bing.com/search",
            params={"q": query},
            headers=SEARCH_HEADERS,
            timeout=15,
            verify=False,
        )
        resp.encoding = "utf-8"
    except Exception as e:
        logger.error(f"Bing 搜索请求失败: {e}")
        return []

    # 解析 Bing 搜索结果（多种兼容模式）
    results = []

    # 提取每个 <li class="b_algo"> 块
    algo_pattern = re.compile(
        r'<li class="b_algo".*?</li>',
        re.DOTALL,
    )
    for li in algo_pattern.finditer(resp.text):
        if len(results) >= max_results:
            break
        li_html = li.group()

        # 提取链接和标题（<a href="...">title</a> 在 h2 或直接嵌套）
        a_match = re.search(r'<a[^>]*href="(https?://[^"]+)"[^>]*>(.*?)</a>', li_html, re.DOTALL)
        if not a_match:
            continue
        link = a_match.group(1)
        title = re.sub(r"<[^>]+>", "", a_match.group(2)).strip()
        # 去掉 title 里的额外 URL 文本（Bing 偶尔会在标题里插入域名）
        title = re.sub(r"^https?://\S+\s*", "", title).strip()
        if not title:
            continue

        # 提取摘要：找第一个 <p>
        p_match = re.search(r"<p[^>]*>(.*?)</p>", li_html, re.DOTALL)
        snippet = re.sub(r"<[^>]+>", "", p_match.group(1)).strip() if p_match else ""
        # 清理 HTML 实体并截断，注意 GBK 环境下的特殊字符
        import html as html_mod
        snippet = html_mod.unescape(snippet)
        snippet = snippet.encode("utf-8", errors="ignore").decode("utf-8")[:200]

        results.append({
            "title": title,
            "snippet": snippet[:200],  # 截断过长摘要
            "link": link,
        })

    logger.info(f"Bing 搜索 \"{query}\" → {len(results)} 条结果")
    return results


def format_search_results(results: list[dict]) -> str:
    """把搜索结果格式化为 LLM 可用的文本"""
    if not results:
        return "（联网搜索未找到相关内容）"
    parts = []
    for i, r in enumerate(results, 1):
        parts.append(f"[来源 {i}] {r['title']}\n{r['snippet']}")
    return "\n\n".join(parts)
