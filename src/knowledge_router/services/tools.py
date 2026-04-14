"""示例工具集合：模拟 GitHub / Notion / Slack 检索能力。"""

from langchain.tools import tool
from knowledge_router.core.config import TAVILY_API_KEY
from tavily import TavilyClient



@tool
def search_code(query: str, repo: str = "main") -> str:
    """在 GitHub 仓库中搜索代码。"""

    return f"在 {repo} 仓库中找到与“{query}”相关的代码：src/auth.py 中的认证中间件。"


@tool
def search_issues(query: str) -> str:
    """搜索 GitHub 的 Issue 与 Pull Request。"""

    return f"找到 3 条与“{query}”相关的记录：#142（API 认证文档）、#89（OAuth 流程）、#203（Token 刷新）。"


@tool
def search_prs(query: str) -> str:
    """搜索 Pull Request 中的实现细节。"""

    return f"PR #156 新增了 JWT 认证，PR #178 更新了 OAuth 权限范围。"


@tool
def search_notion(query: str) -> str:
    """在 Notion 工作区中搜索文档。"""

    return "找到文档：《API 认证指南》，覆盖 OAuth2 流程、API Key 与 JWT Token。"


@tool
def get_page(page_id: str) -> str:
    """根据页面 ID 获取 Notion 页面内容。"""

    return f"页面 {page_id} 内容：分步骤的认证接入说明。"


@tool
def search_slack(query: str) -> str:
    """在 Slack 消息与线程中搜索讨论。"""

    return f"在 #engineering 中找到讨论：API 认证建议使用 Bearer Token，并参考文档中的刷新流程。"


@tool
def get_thread(thread_id: str) -> str:
    """根据线程 ID 获取 Slack 线程内容。"""

    return f"线程 {thread_id} 讨论了 API Key 轮换的最佳实践。"


@tool
def search_web(query: str, max_results: int = 5) -> str:
    """使用 Tavily 进行互联网搜索。"""

    try:
        client = TavilyClient(api_key=TAVILY_API_KEY)
        response = client.search(
            query=query,
            max_results=max_results,
            search_depth="advanced",
        )
    except Exception as exc:
        return f"Tavily 搜索失败：{exc}"

    results = response.get("results", [])
    if not results:
        return "联网检索完成，但未找到相关结果。"

    lines = []
    for idx, item in enumerate(results[:max_results], start=1):
        title = item.get("title", "无标题")
        url = item.get("url", "")
        content = (item.get("content", "") or "").replace("\n", " ").strip()
        snippet = content[:180]
        lines.append(f"{idx}. {title}\n链接：{url}\n摘要：{snippet}")

    return "Tavily 搜索结果：\n" + "\n\n".join(lines)
