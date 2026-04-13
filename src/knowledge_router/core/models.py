"""模型初始化：统一创建项目使用的 LLM。"""

from langchain_community.chat_models.tongyi import ChatTongyi

from knowledge_router.core.config import DASHSCOPE_API_KEY

# 主模型：用于各子代理回答
model = ChatTongyi(model="qwen3-max", api_key=DASHSCOPE_API_KEY)

# 路由模型：用于分类与结果综合
router_llm = ChatTongyi(model="qwen3-max", api_key=DASHSCOPE_API_KEY)
