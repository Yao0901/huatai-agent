"""
节点 8：执行 SQL 语句

将经过安全审查和权限注入的 SQL 发送到 SQLite 数据库执行，捕获结果或报错。
"""

from typing import Dict, Any

from tools.db_connector import execute_sql


# ---------------------------------------------------------------------------
# 节点主函数
# ---------------------------------------------------------------------------

def execute_sql_statement(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    安全执行 SQL 并返回结果。

    调用时机：节点 7（安全审查）和 7.1（权限注入）完成后。

    使用的 SQL 来源：
    - 如果 permission_check=="filtered"，使用 injected_sql（含权限过滤）
    - 否则使用 generated_sql（原始 SQL）

    Args:
        state: 全局状态字典，需含:
               - generated_sql (str): 原始 SQL
               - injected_sql (str): 注入过滤后的 SQL
               - permission_check (str): 权限校验结果

    Returns:
        dict: 更新字段:
              - execution_result (dict): {"success": bool, "data": list, "error": str, "row_count": int}
              - error_message (str|None): 简化版错误信息（供反思节点使用）
    """
    # 选择正确的 SQL 执行
    permission_check = state.get("permission_check", "passed")
    sql = (
        state.get("injected_sql", "")
        if permission_check == "filtered"
        else state.get("generated_sql", "")
    )

    result = execute_sql(sql)

    return {
        "execution_result": result,
        "error_message": result.get("error") if not result.get("success") else None,
    }
