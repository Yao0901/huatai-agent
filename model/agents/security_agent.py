"""
子图 3：安全评估 Agent (Agent3)

负责：
1. SQL 操作安全分级（节点 7）：sqlglot AST → forbidden / dangerous / safe
2. dangerous 操作触发 Human-in-the-loop 确认（确认→继续 / 拒绝→终止）
3. 数据权限校验与行级过滤注入（节点 7.1）
4. 执行 SQL 并捕获结果（节点 8）
5. 报错时 LLM 智能反思 → 回退 Agent2 重试（最多 3 次）

图结构：
    evaluate_security (节点7)
        ├── forbidden → END（直接终止）
        ├── dangerous → confirm_dangerous
        │                  ├── 确认 → check_permission
        │                  └── 拒绝 → END
        └── safe → check_permission (节点7.1)
                      ├── denied → END
                      ├── filtered → execute_sql
                      └── passed → execute_sql (节点8)
                                      ├── 成功 → END
                                      └── 失败 → reflect → END (回退 Agent2)
"""

from langgraph.graph import StateGraph, END
from langgraph.types import interrupt

from agents.state import AgentState

from nodes.security_node import evaluate_security, check_permission_and_inject
from nodes.execution_node import execute_sql_statement
from nodes.reflect_node import reflect_on_error


# ---------------------------------------------------------------------------
# 节点：危险操作确认
# ---------------------------------------------------------------------------

def confirm_dangerous_node(state: AgentState) -> AgentState:
    """
    SQL 被判定为 dangerous（UPDATE/DELETE/INSERT）时，暂停流程，
    向用户展示 SQL 并请求确认。

    Args:
        state: 全局状态

    Returns:
        AgentState: 更新 user_confirmed_dangerous 字段
    """
    sql = state.get("generated_sql", "")
    detail = state.get("security_detail", {})

    message = (
        f"⚠ 检测到写操作 SQL（类型: {detail.get('operation_type', '未知')}）\n\n"
        f"SQL 内容:\n{sql[:500]}\n\n"
        f"涉及的表: {', '.join(detail.get('tables_involved', []))}\n\n"
        f"是否继续执行？输入 y/yes/确认 继续，其他任意字符取消。"
    )

    user_reply = interrupt(message)

    confirmed = user_reply.strip().lower() in ("y", "yes", "确认", "是", "ok", "继续")
    return {"user_confirmed_dangerous": confirmed}


# ---------------------------------------------------------------------------
# 路由函数
# ---------------------------------------------------------------------------

def _route_after_security_check(state: AgentState) -> str:
    level = state.get("security_level", "safe")
    if level == "forbidden":
        return "end_forbidden"
    elif level == "dangerous":
        return "confirm_dangerous"
    else:
        return "check_permission"


def _route_after_dangerous_confirm(state: AgentState) -> str:
    if state.get("user_confirmed_dangerous"):
        return "check_permission"
    return "end_user_cancelled"


def _route_after_permission_check(state: AgentState) -> str:
    if state.get("permission_check") == "denied":
        return "end_permission_denied"
    return "execute_sql"


def _route_after_execution(state: AgentState) -> str:
    result = state.get("execution_result", {})
    if result.get("success"):
        return "output"
    return "check_retry"


def _route_retry(state: AgentState) -> str:
    if state.get("retry_count", 0) < 3:
        return "reflect"
    return "end_max_retry"


# ---------------------------------------------------------------------------
# 子图构建
# ---------------------------------------------------------------------------

def build_security_agent() -> StateGraph:
    """
    构建「安全评估 Agent」子图。

    图结构：
        evaluate_security
            ├── forbidden → END
            ├── dangerous → confirm_dangerous
            │                  ├── 确认 → check_permission
            │                  └── 拒绝 → END
            └── safe → check_permission
                          ├── denied → END
                          └── passed/filtered → execute_sql
                                                    ├── 成功 → END
                                                    └── 失败 → reflect → END
    """
    graph = StateGraph(AgentState)

    graph.add_node("evaluate_security", evaluate_security)
    graph.add_node("confirm_dangerous", confirm_dangerous_node)
    graph.add_node("check_permission", check_permission_and_inject)
    graph.add_node("execute_sql", execute_sql_statement)
    graph.add_node("reflect", reflect_on_error)

    graph.set_entry_point("evaluate_security")

    # 安全分级路由
    graph.add_conditional_edges(
        "evaluate_security",
        _route_after_security_check,
        {
            "end_forbidden": END,
            "confirm_dangerous": "confirm_dangerous",
            "check_permission": "check_permission",
        },
    )

    # 危险确认 → 继续或取消
    graph.add_conditional_edges(
        "confirm_dangerous",
        _route_after_dangerous_confirm,
        {
            "check_permission": "check_permission",
            "end_user_cancelled": END,
        },
    )

    # 权限校验 → 拒绝或执行
    graph.add_conditional_edges(
        "check_permission",
        _route_after_permission_check,
        {
            "end_permission_denied": END,
            "execute_sql": "execute_sql",
        },
    )

    # 执行结果 → 成功或反思
    graph.add_conditional_edges(
        "execute_sql",
        _route_after_execution,
        {
            "output": END,
            "check_retry": "reflect",
        },
    )

    # 反思 → 重试次数判断
    graph.add_conditional_edges(
        "reflect",
        _route_retry,
        {
            "reflect": END,         # 继续重试 → 回退 Agent2
            "end_max_retry": END,   # 达上限 → 终止
        },
    )

    return graph
