# 14 · web_search.py 参数

> 文件：`src/agent/web_search.py`（既有），模块级函数
> 功能：**联网搜索实现**——`web_search_node` 调它抓 Bing 结果页、正则解析成结构化结果，供 generate 使用。

## 构造参数

模块级常量 + 函数，无类：

| 常量 | 默认值 | 大白话 | 技术性 |
|------|--------|--------|--------|
| `SEARCH_HEADERS` | Chrome UA | 伪装成浏览器访问 Bing | 反爬 UA（`verify=False` 关闭 SSL 校验） |

## 读取的 config（只列本功能链路消费的）

| config 参数 | 默认值 | 大白话 | 技术性 | 谁消费 |
|------------|--------|--------|--------|--------|
| 无 | — | 搜索参数在函数入参 | 不读 config | — |

## 核心方法

### `search_bing(query, max_results=3)`（`web_search.py:19`）

```
resp = requests.get("https://www.bing.com/search", params={"q": query}, headers=UA, timeout=15, verify=False)
异常 → return []
解析：
  <li class="b_algo">...</li> 块（正则，最多 max_results 条）
    提取 <a href="http...">title</a> → link + title（去标签/去多余URL）
    提取第一个 <p> → snippet（去标签、unescape 实体、截断 200 字）
return [{"title","snippet","link"}, ...]
```

### `format_search_results(results)`（`web_search.py:80`）

```
空 → "（联网搜索未找到相关内容）"
有 → "[来源 1] title\nsnippet\n\n[来源 2] ..."
```

> **设计要点**
> - **零 API key 的联网方案**：直接抓 Bing 搜索结果页用正则解析，不依赖任何搜索 API 服务。代价是**依赖 Bing 页面结构**——页面改版正则可能失效（脆弱点）。【推断】理由：`re.compile` 硬编码 `class="b_algo"`。证据：`web_search.py:42`。
> - **`verify=False` + 关闭警告**（`urllib3.disable_warnings()`）：为绕过本地 SSL 证书校验，简化部署，但牺牲 TLS 校验（中间人风险），适合学习项目。属**明确权衡**，面试可讲。
> - **失败语义**：请求异常返回 `[]` → `format_search_results` 产出占位"（联网搜索未找到相关内容）" → `chat_service` 来源判定时**不计入 web_search**（占位串被排除，`chat_service.py:244`）→ 回答来源可能落 llm。
> - `max_results=3` 是 web_search_node 调用时的固定值（`nodes.py:277`），控制喂给 LLM 的结果数。

## 该文件在链路中的位置

```
web_search_node → search_bing(query, 3) → format_search_results → ToolMessage → generate
```
