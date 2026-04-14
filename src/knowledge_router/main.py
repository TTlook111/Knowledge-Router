"""项目运行入口。"""

from knowledge_router.graph.workflow import workflow


if __name__ == "__main__":
    result = workflow.invoke(
        {"query": "如何进行 API 请求认证？"},
        config={"configurable": {"thread_id": "demo-thread-1"}},
    )

    print("原始问题：", result["query"])
    print("\n路由分类：")
    for c in result["classifications"]:
        print(f"  {c['source']}: {c['query']}")
    print("\n" + "=" * 60 + "\n")
    print("最终回答：")
    print(result["final_answer"])
