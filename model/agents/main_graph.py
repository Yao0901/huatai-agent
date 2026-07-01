"""
主图：串联三张子图，处理跨图回退与 Human-in-the-loop

将 intent_agent、query_agent、security_agent 三张子图组装成完整的取数流水线，
并实现跨图重试回退（Security 报错 → 反思 → 自动回到 Query 重新检索生成）。

核心设计决策：
- 三张子图作为主图的三个节点（用 add_node 注册子图编译后的 graph）
- 跨图重试用 conditional edge 实现：Security 结束 → 判断 retry_count → 回退到 Query 或终止
- Human-in-the-loop 用 interrupt 实现：遇到追问/确认时暂停，等待外部输入后 resume
"""

from langgraph.graph import StateGraph, END

from agents.state import AgentState

# 导入三张子图的构建函数
from agents.intent_agent import build_intent_agent
from agents.query_agent import build_query_agent
from agents.security_agent import build_security_agent


# ---------------------------------------------------------------------------
# 路由函数
# ---------------------------------------------------------------------------

def _should_retry(state: AgentState) -> str:
    """
    判断子图 3 结束后是「回退到子图 2 重试」还是「彻底终止」。

    读取 state 中的关键字段做决策：
    - execution_result.success=True → 正常结束
    - retry_count < 3 且 reflect_node 有产出 → 回退到子图2重试
    - 其他 → 终止

    Args:
        state: 全局状态

    Returns:
        str: "retry_to_query"（回退）或 "__end__"（终止）
    """
    # 情况 1：执行成功 → 正常结束
    exec_result = state.get("execution_result", {})
    if exec_result.get("success"):
        return "__end__"

    # 情况 2：执行失败但还有重试次数 → 回退到子图2
    retry_count = state.get("retry_count", 0)
    if retry_count < 3 and state.get("reflection_feedback"):
        return "retry_to_query"

    # 情况 3：其他（权限拒绝、禁止操作、超出重试上限等）→ 终止
    return "__end__"


# ---------------------------------------------------------------------------
# 主图构建
# ---------------------------------------------------------------------------

def build_main_graph() -> StateGraph:
    """
    构建并编译主图：串联 intent_agent → query_agent → security_agent。

    图结构（高层视角）：
        ┌──────────┐    ┌──────────┐    ┌───────────┐
        │ Agent 1  │───▶│ Agent 2  │───▶│ Agent 3   │
        │ 意图识别  │    │ 查询生成  │    │ 安全评估   │
        └──────────┘    └──────────┘    └───────────┘
                              ▲               │
                              │  重试回退      │
                              └───────────────┘

    Returns:
        StateGraph: 编译后的主图（已 .compile()，可直接 invoke）
    """
    main_graph = StateGraph(AgentState)

    # ------------------------------------------------------------------
    # 第 1 步：编译三张子图
    # ------------------------------------------------------------------

    # 注意：每张子图都使用相同的 AgentState TypedDict
    # LangGraph 的 subgraph 机制保证了状态的自由流通
    intent_subgraph = build_intent_agent().compile()
    query_subgraph = build_query_agent().compile()
    security_subgraph = build_security_agent().compile()

    # ------------------------------------------------------------------
    # 第 2 步：将三张子图注册为主图的三个节点
    # ------------------------------------------------------------------

    main_graph.add_node("agent1_intent", intent_subgraph)
    main_graph.add_node("agent2_query", query_subgraph)
    main_graph.add_node("agent3_security", security_subgraph)

    # ------------------------------------------------------------------
    # 第 3 步：定义节点间的串联边
    # ------------------------------------------------------------------

    # 入口 → Agent1（意图识别）
    main_graph.set_entry_point("agent1_intent")

    # Agent1 → Agent2（意图识别 → 查询生成）
    main_graph.add_edge("agent1_intent", "agent2_query")

    # Agent2 → Agent3（查询生成 → 安全评估）
    main_graph.add_edge("agent2_query", "agent3_security")

    # Agent3 → 判断是否回退
    #   成功或不可重试错误 → END
    #   可重试错误       → 回退到 Agent2（携带 reflection_feedback）
    main_graph.add_conditional_edges(
        "agent3_security",
        _should_retry,
        {
            "__end__": END,
            "retry_to_query": "agent2_query",  # 回退到子图2重新检索/生成
        },
    )

    # ------------------------------------------------------------------
    # 第 4 步：编译主图
    # ------------------------------------------------------------------

    return main_graph.compile()
