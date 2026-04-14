# Knowledge-Router

一个基于 LangGraph 的多智能体知识路由系统。  
它会先理解用户问题，再把子问题并行分发给不同知识源代理（GitHub / Notion / Slack），最后聚合为结构化回答。

## 功能特性

- 多源路由：根据问题自动选择需要查询的知识源
- 并行检索：多个子代理并发执行，提升响应效率
- 会话记忆：结合历史问题生成当前轮可复用上下文
- 结果综合：统一融合各来源结论并输出最终答案
- 易于扩展：可按同样模式增加新的知识源代理

## 技术栈

- Python 3.13+
- LangChain
- LangGraph
- Pydantic
- DashScope / Qwen（通过 `langchain-community`）

## 快速开始

### 1. 安装依赖

推荐使用 `uv`：

```bash
uv sync
```

或使用 `pip`：

```bash
pip install -e .
```

### 2. 配置环境变量

在项目根目录创建 `.env` 文件：

```env
DASHSCOPE_API_KEY=your_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

- `DASHSCOPE_API_KEY` 为必填，未配置时程序会在启动时报错。
- `TAVILY_API_KEY` 用于联网搜索（Web 知识源），未配置时仅该能力不可用。

### 3. 运行示例

```bash
python -m knowledge_router.main
```

运行后会输出：

- 原始问题
- 路由分类结果（发送到哪些知识源）
- 综合后的最终回答

## 工作流说明

核心流程位于 `src/knowledge_router/graph/workflow.py`：

1. `prepare_memory`：提炼最近会话历史，生成记忆上下文
2. `classify`：判定问题应发送到哪些知识源
3. `github/notion/slack`：并行执行子代理检索与回答
4. `synthesize`：融合多源结果，输出最终答案

其中 `results` 与 `history_queries` 使用 `operator.add` 在并行分支间自动聚合。

## 项目结构

```text
src/knowledge_router/
  core/
    config.py      # 环境变量与配置
    models.py      # LLM 初始化
    schemas.py     # 状态与结构化模型
  graph/
    workflow.py    # LangGraph 主流程
  services/
    agents.py      # 各知识源子代理
    tools.py       # 示例检索工具
  main.py          # 运行入口
```

## 扩展建议

- 替换 `services/tools.py` 中的模拟工具为真实 API 调用
- 已内置 `search_web`（Tavily）联网检索工具，可作为外部知识补充来源
- 新增知识源时，补充对应工具、子代理和路由分支
- 将 `InMemorySaver` 替换为持久化 Checkpointer（如 SQLite）以支持跨进程记忆

## License

MIT
