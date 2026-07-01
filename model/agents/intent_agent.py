"""
子图 1：意图识别 Agent (Agent1)

负责：
1. 接收用户自然语言输入
2. 判断是否存在模糊口径 / 模糊目标（节点 3）
3. 如有歧义 → 在业务口径数据库中检索标准定义
4. 如找不到 → 触发 Human-in-the-loop 追问

图结构：
    check_ambiguity(节点3)
        ├── 否 → __end__（进入子图2）
        └── 是 → search_metric → CheckFound
                    ├── 找到 → __end__（进入子图2）
                    └── 找不到 → ask_user → (等待人类输入) → 回到 check_ambiguity
"""

from langgraph.graph import StateGraph, END
from langgraph.types import interrupt

from agents.state import AgentState

from nodes.ambiguity_node import check_ambiguity
from nodes.metric_search_node import search_metric


# ---------------------------------------------------------------------------
# 追问用户节点（Human-in-the-loop）
# ---------------------------------------------------------------------------

def ask_user_node(state: AgentState) -> AgentState:
    """
    暂停图执行，向用户发起追问，等待用户补充信息后继续。

    触发场景：
    - 口径检索失败：用户说的某个指标在知识库中找不到
    - 表结构不贴合：检索到的表无法回答用户问题

    原理：
    调用 LangGraph 的 interrupt() 函数，图会在此处暂停，
    将追问信息返回给 main.py。main.py 收集用户回复后，
    调用 graph.invoke(..., config) 恢复执行，从本节点之后继续。

    用户回复会自动追加到 state["user_input"]，
    然后路由回到 check_ambiguity 重新分析。

    Args:
        state: 全局状态

    Returns:
        AgentState: 合并了用户补充信息后的状态
    """
    unresolved = state.get("unresolved_terms", [])
    if unresolved:
        question = (
            f"以下业务指标我不确定含义，请补充说明：\n"
            + "\n".join(f"  - {t}" for t in unresolved)
            + "\n\n请用更具体的指标名称描述，例如'总资产'而非'资产'。"
        )
    else:
        question = "当前查询不够明确，请补充更多细节（如时间范围、具体指标名称）。"

    # interrupt 暂停图执行，返回追问内容给 main.py
    user_reply = interrupt(question)

    # 用户回复后从这继续：把回复注入 state，下次循环用
    return {
        "user_input": state.get("user_input", "") + "\n（补充：" + user_reply + "）",
    }


# ---------------------------------------------------------------------------
# 路由函数：判断是否存在模糊口径
# ---------------------------------------------------------------------------

def _route_after_ambiguity_check(state: AgentState) -> str:
    """
    根据节点 3 的结果决定下一步。

    - ambiguity_flag=True  → 走「寻找业务口径数据库」
    - ambiguity_flag=False → 结束子图1，进入子图2的节点4

    Args:
        state: 全局状态

    Returns:
        str: "search_metric" 或 "__end__"
    """
    if state.get("ambiguity_flag"):
        return "search_metric"
    return "__end__"


# ---------------------------------------------------------------------------
# 路由函数：判断口径检索结果
# ---------------------------------------------------------------------------

def _route_after_metric_search(state: AgentState) -> str:
    """
    根据口径检索结果决定下一步。

    - metric_found=True  → 结束子图1，进入子图2
    - metric_found=False → 追问用户（Human-in-the-loop）

    Args:
        state: 全局状态

    Returns:
        str: "__end__" 或 "ask_user"
    """
    if state.get("metric_found"):
        return "__end__"
    return "ask_user"


# ---------------------------------------------------------------------------
# 子图构建
# ---------------------------------------------------------------------------

def build_intent_agent() -> StateGraph:
    """
    构建「意图识别 Agent」子图。

    图结构：
        check_ambiguity
            ├── False → END
            └── True  → search_metric
                            ├── 全找到 → END
                            └── 有遗漏 → ask_user → 回到 check_ambiguity

    Returns:
        StateGraph: 编译后的子图
    """
    graph = StateGraph(AgentState)

    # 注册节点
    graph.add_node("check_ambiguity", check_ambiguity)
    graph.add_node("search_metric", search_metric)
    graph.add_node("ask_user", ask_user_node)

    # 入口：check_ambiguity
    graph.set_entry_point("check_ambiguity")

    # check_ambiguity → 根据歧义判断路由
    graph.add_conditional_edges(
        "check_ambiguity",
        _route_after_ambiguity_check,
        {
            "search_metric": "search_metric",
            "__end__": END,
        },
    )

    # search_metric → 根据检索结果路由
    graph.add_conditional_edges(
        "search_metric",
        _route_after_metric_search,
        {
            "__end__": END,
            "ask_user": "ask_user",
        },
    )

    # ask_user → 回到 check_ambiguity 重新分析
    graph.add_edge("ask_user", "check_ambiguity")

    return graph
