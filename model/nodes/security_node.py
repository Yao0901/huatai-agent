"""
节点 7 & 7.1：安全评估节点

对生成的 SQL 做两层审查：
1. 操作安全审查（节点 7）：AST 分析 → 分级拦截（禁止/危险/安全）
2. 数据权限审查（节点 7.1）：提取涉及的表 → 注入行级过滤条件
"""

from typing import Dict, Any

from tools.ast_parser import analyze_sql, inject_permission_filter


# ---------------------------------------------------------------------------
# 节点 7：操作安全审查
# ---------------------------------------------------------------------------

def evaluate_security(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    对生成的 SQL 做操作安全分级审查。

    调用时机：SQL 生成后（节点 6 → 节点 7）。

    三级分类：
    - forbidden: 禁止操作（DROP、TRUNCATE、ALTER 等）→ 直接终止
    - dangerous: 危险操作（UPDATE、DELETE 等）→ 需用户确认
    - safe: 安全操作（SELECT）→ 直接进入权限校验

    Args:
        state: 全局状态字典，需含 generated_sql

    Returns:
        dict: 更新字段:
              - security_level (str): "safe"|"dangerous"|"forbidden"
              - security_detail (dict): AST 分析详情（含 tables_involved）
    """
    sql = state.get("generated_sql", "")
    analysis = analyze_sql(sql)

    return {
        "security_level": analysis["security_level"],
        "security_detail": analysis,
    }


# ---------------------------------------------------------------------------
# 节点 7.1：数据权限校验与过滤注入
# ---------------------------------------------------------------------------

def check_permission_and_inject(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    校验数据权限，必要时向 SQL 注入行级过滤条件。

    调用时机：节点 7 判定 SQL 安全后。

    权限校验逻辑：
    1. 从 AST 分析结果提取涉及的表
    2. 模拟查询用户的表/行级权限
    3. 如需过滤 → 自动注入 WHERE 条件
    4. 如权限充足 → 原 SQL 通过

    当前实现：模拟权限（假设所有表可读，含 org_id 列的表注入示例行过滤）。
    生产环境需对接实际权限系统。

    Args:
        state: 全局状态字典

    Returns:
        dict: 更新字段:
              - permission_check (str): "denied"|"filtered"|"passed"
              - injected_sql (str): 注入过滤条件后的 SQL
    """
    sql = state.get("generated_sql", "")
    analysis = state.get("security_detail", {})
    tables = analysis.get("tables_involved", [])

    # 模拟：用户只能看特定营业部数据
    # 实际环境从 session/权限系统读取
    user_accessible_orgs = ["XX00000054"]

    # 判断哪些表需要行级过滤（含 org_id 列的表）
    tables_needing_filter = {"ads_cust_info_d", "dim_branch"}

    needs_filter = any(t in tables_needing_filter for t in tables)

    if needs_filter:
        # 构建行级过滤条件（注入到 SQL 的 WHERE 子句）
        quoted_orgs = ", ".join(f"'{org}'" for org in user_accessible_orgs)
        filter_condition = f"org_id IN ({quoted_orgs})"
        filtered_sql = inject_permission_filter(sql, filter_condition)
        return {
            "permission_check": "filtered",
            "injected_sql": filtered_sql,
        }

    return {
        "permission_check": "passed",
        "injected_sql": sql,
    }
