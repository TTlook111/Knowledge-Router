"""创建各知识源子代理。"""

from langchain.agents import create_agent

from knowledge_router.models import model
from knowledge_router.tools import (
    get_page,
    get_thread,
    search_code,
    search_issues,
    search_notion,
    search_prs,
    search_slack,
)

github_agent = create_agent(
    model,
    tools=[search_code, search_issues, search_prs],
    system_prompt=(
        "你是 GitHub 专家。请通过仓库代码、Issue 与 Pull Request 回答实现细节、"
        "API 参考与工程实践相关问题。"
    ),
)

notion_agent = create_agent(
    model,
    tools=[search_notion, get_page],
    system_prompt=(
        "你是 Notion 专家。请通过团队文档、流程与制度内容回答组织内部知识问题。"
    ),
)

slack_agent = create_agent(
    model,
    tools=[search_slack, get_thread],
    system_prompt=(
        "你是 Slack 专家。请通过团队讨论与历史线程提炼经验、背景与可执行结论。"
    ),
)
