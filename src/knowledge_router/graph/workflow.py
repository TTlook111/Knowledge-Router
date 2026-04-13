"""路由工作流定义。"""

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from knowledge_router.core.models import router_llm
from knowledge_router.core.schemas import AgentInput, ClassificationResult, RouterState
from knowledge_router.services.agents import github_agent, notion_agent, slack_agent


def classify_query(state: RouterState) -> dict:
    """分类用户问题，并决定调用哪些知识源代理。"""

    structured_llm = router_llm.with_structured_output(ClassificationResult)
    result = structured_llm.invoke(
        [
            {
                "role": "system",
                "content": (
                    "请分析用户问题，并判断应查询哪些知识源。\n"
                    "对每个相关知识源，生成一个针对该来源的子问题。\n\n"
                    "可用知识源：\n"
                    "- github：代码、API 参考、实现细节、Issue、Pull Request\n"
                    "- notion：内部文档、流程制度、团队知识库\n"
                    "- slack：团队讨论、经验沉淀、近期上下文\n\n"
                    "只返回与问题相关的知识源。"
                ),
            },
            {"role": "user", "content": state["query"]},
        ]
    )
    return {"classifications": result.classifications}


def route_to_agents(state: RouterState) -> list[Send]:
    """根据分类结果并行分发到各子代理。"""

    return [Send(c["source"], {"query": c["query"]}) for c in state["classifications"]]


def query_github(state: AgentInput) -> dict:
    """调用 GitHub 子代理。"""

    result = github_agent.invoke({"messages": [{"role": "user", "content": state["query"]}]})
    return {"results": [{"source": "github", "result": result["messages"][-1].content}]}


def query_notion(state: AgentInput) -> dict:
    """调用 Notion 子代理。"""

    result = notion_agent.invoke({"messages": [{"role": "user", "content": state["query"]}]})
    return {"results": [{"source": "notion", "result": result["messages"][-1].content}]}


def query_slack(state: AgentInput) -> dict:
    """调用 Slack 子代理。"""

    result = slack_agent.invoke({"messages": [{"role": "user", "content": state["query"]}]})
    return {"results": [{"source": "slack", "result": result["messages"][-1].content}]}


def synthesize_results(state: RouterState) -> dict:
    """综合多个知识源结果，输出最终答案。"""

    if not state["results"]:
        return {"final_answer": "未从任何知识源检索到结果。"}

    formatted = [f"【{r['source'].title()}】\n{r['result']}" for r in state["results"]]
    synthesis_response = router_llm.invoke(
        [
            {
                "role": "system",
                "content": (
                    f"请基于以下检索结果回答原始问题：{state['query']}\n\n"
                    "- 融合多源信息，避免重复\n"
                    "- 优先给出最相关、可执行的信息\n"
                    "- 若来源间有冲突，请明确指出\n"
                    "- 表达简洁、结构清晰"
                ),
            },
            {"role": "user", "content": "\n\n".join(formatted)},
        ]
    )
    return {"final_answer": synthesis_response.content}


workflow = (
    StateGraph(RouterState)
    .add_node("classify", classify_query)
    .add_node("github", query_github)
    .add_node("notion", query_notion)
    .add_node("slack", query_slack)
    .add_node("synthesize", synthesize_results)
    .add_edge(START, "classify")
    .add_conditional_edges("classify", route_to_agents, ["github", "notion", "slack"])
    .add_edge("github", "synthesize")
    .add_edge("notion", "synthesize")
    .add_edge("slack", "synthesize")
    .add_edge("synthesize", END)
    .compile()
)
