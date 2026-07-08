"""
run_sql — 执行 SELECT 查询并返回结果

供 ReAct Agent 用于验证 SQL 和获取最终数据。
"""

from tools.db_connector import execute_sql


def run_sql(sql: str) -> str:
    """
    执行 SELECT 查询并返回结果。用于验证 SQL 是否正确、查看实际数据。
    如果结果不对，可以修改 SQL 后重新调用此工具。
    禁止任何写操作（INSERT/UPDATE/DELETE/DROP）。
    """
    sql_upper = sql.strip().upper()
    if not sql_upper.startswith("SELECT") and not sql_upper.startswith("WITH"):
        return f"[BLOCKED] 只允许 SELECT/WITH 查询。你发送的是: {sql[:60]}"

    result = execute_sql(sql)
    if result["success"]:
        rows = result["data"]
        row_count = result["row_count"]
        if not rows:
            return "[EMPTY] 查询成功但返回 0 行。可能是条件太严格或日期不匹配。"
        if row_count > 30:
            rows = rows[:30]
            extra = f"\n... (还有 {row_count - 30} 行省略)"
        else:
            extra = ""
        cols = list(rows[0].keys())
        lines = [f"返回 {row_count} 行:", " | ".join(cols)]
        for row in rows:
            lines.append(" | ".join(str(row.get(c, "")) for c in cols))
        return "\n".join(lines) + extra
    else:
        return f"[SQL ERROR] {result.get('error', '未知错误')}\n请检查 SQL 并修正后重试。"
