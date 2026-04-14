"""路由工作流定义。"""

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from knowledge_router.core.models import router_llm
from knowledge_router.core.schemas import AgentInput, ClassificationResult, RouterState
from knowledge_router.services.agents import github_agent, notion_agent, slack_agent


def prepare_memory_context(state: RouterState) -> dict:
    """图入口节点：从历史会话中提炼本轮可用记忆。"""

    history_queries = state.get("history_queries", [])
    if history_queries:
        recent = history_queries[-5:]
        raw_history = "\n".join(f"- {item}" for item in recent)
        try:
            # 用模型把“历史问题 + 当前问题”压缩成可直接复用的上下文
            summary_response = router_llm.invoke(
                [
                    {
                        "role": "system",
                        "content": (
                            "你是会话记忆整理助手。请基于历史问题提炼与当前问题最相关的信息，"
                            "输出简洁要点，避免复述无关内容。"
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"当前问题：{state['query']}\n\n"
                            f"最近历史问题：\n{raw_history}\n\n"
                            "请输出：\n"
                            "1) 与当前问题直接相关的背景\n"
                            "2) 可能影响回答的约束/偏好\n"
                            "3) 若无明显相关信息，请明确写“无有效历史线索”"
                        ),
                    },
                ]
            )
            memory_context = f"历史记忆提炼：\n{summary_response.content}"
        except Exception:
            # 回退逻辑：模型异常时仍保证流程可用
            memory_context = f"历史问题摘要（最近 5 条）：\n{raw_history}"
    else:
        memory_context = "无历史会话记忆。"

    # 通过 operator.add 聚合，把当前 query 追加到历史里供下轮使用
    return {"memory_context": memory_context, "history_queries": [state["query"]]}


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
            {
                "role": "user",
                "content": f"{state.get('memory_context', '无历史会话记忆。')}\n\n当前问题：{state['query']}",
            },
        ]
    )
    return {"classifications": result.classifications}


def route_to_agents(state: RouterState) -> list[Send]:
    """根据分类结果并行分发到各子代理。"""

    # classifications 由 classify 节点给出，形如：
    # [{"source": "github", "query": "..."}, {"source": "notion", "query": "..."}]
    # 这里返回 list[Send]，LangGraph 会把每个 Send 当作一条独立分支并发执行。
    return [
        Send(
            # 目标节点名（必须与 add_node 的名字一致）
            c["source"],
            {
                # 该知识源的子问题（已由 classify 做过针对性改写）
                "query": c["query"],
                # 共享的会话记忆上下文：每个并行分支都会收到同一份
                "memory_context": state.get("memory_context", "无历史会话记忆。"),
            },
        )
        # 为每个分类结果生成一个并行任务
        for c in state["classifications"]
    ]


def query_github(state: AgentInput) -> dict:
    """调用 GitHub 子代理。"""

    result = github_agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": f"{state['memory_context']}\n\n请回答：{state['query']}",
                }
            ]
        }
    )
    return {"results": [{"source": "github", "result": result["messages"][-1].content}]}


def query_notion(state: AgentInput) -> dict:
    """调用 Notion 子代理。"""

    result = notion_agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": f"{state['memory_context']}\n\n请回答：{state['query']}",
                }
            ]
        }
    )
    return {"results": [{"source": "notion", "result": result["messages"][-1].content}]}


def query_slack(state: AgentInput) -> dict:
    """调用 Slack 子代理。"""

    result = slack_agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": f"{state['memory_context']}\n\n请回答：{state['query']}",
                }
            ]
        }
    )
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


_checkpointer = InMemorySaver()

workflow = (
    StateGraph(RouterState)
    .add_node("prepare_memory", prepare_memory_context)
    .add_node("classify", classify_query)
    .add_node("github", query_github)
    .add_node("notion", query_notion)
    .add_node("slack", query_slack)
    .add_node("synthesize", synthesize_results)
    .add_edge(START, "prepare_memory")
    .add_edge("prepare_memory", "classify")
    .add_conditional_edges("classify", route_to_agents, ["github", "notion", "slack"])
    .add_edge("github", "synthesize")
    .add_edge("notion", "synthesize")
    .add_edge("slack", "synthesize")
    .add_edge("synthesize", END)
    .compile(checkpointer=_checkpointer)
)
