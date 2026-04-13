"""状态与结构化输出的数据模型定义。"""

import operator
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, Field


class AgentInput(TypedDict):
    """每个子代理接收的输入。"""

    query: str


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

    query: str
    classifications: list[Classification]
    results: Annotated[list[AgentOutput], operator.add]
    final_answer: str


class ClassificationResult(BaseModel):
    """查询分类后的结构化输出。"""

    classifications: list[Classification] = Field(description="需要调用的知识源与对应子问题列表")
