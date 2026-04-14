"""状态与结构化输出的数据模型定义。"""

import operator
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, Field


class AgentInput(TypedDict):
    """每个子代理接收的输入。"""

    query: str
    memory_context: str


class AgentOutput(TypedDict):
    """每个子代理返回的输出。"""

    source: str
    result: str


class Classification(TypedDict):
    """单次路由决策：给哪个代理发送什么查询。"""

    source: Literal["github", "notion", "slack"]
    query: str


class RouterState(TypedDict):
    """路由流程全局状态。"""

    # 用户本轮原始问题（入口输入）
    query: str
    # 分类节点输出的任务清单：每个知识源对应一个子问题
    classifications: list[Classification]
    # 各并行子代理的检索/回答结果；operator.add 表示多分支结果做列表聚合
    results: Annotated[list[AgentOutput], operator.add]
    # 会话历史问题列表；operator.add 用于每轮把当前 query 追加进历史
    history_queries: Annotated[list[str], operator.add]
    # 从历史中提炼出的“本轮可用记忆”，供 classify 与各子代理复用
    memory_context: str
    # 综合节点产出的最终回答（流程最终输出）
    final_answer: str


class ClassificationResult(BaseModel):
    """查询分类后的结构化输出。"""

    classifications: list[Classification] = Field(description="需要调用的知识源与对应子问题列表")
