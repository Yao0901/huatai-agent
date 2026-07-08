"""
轻量探索查询（SQL Agent 补充探索用）。
"""

from tools.db_connector import execute_sql


def query_database(db_name: str, sql: str) -> str:
    """
    在指定数据库中执行只读查询，用于补充探索。
    仅当缺少必要信息（列名记不清、日期范围不确定）时使用。
    """
    sql_upper = sql.strip().upper()
    if not (sql_upper.startswith("SELECT") or sql_upper.startswith("PRAGMA")):
        return f"[BLOCKED] 只允许 SELECT 和 PRAGMA。"
    if "sqlite_master" in sql_upper:
        return "[BLOCKED] 禁止查系统表"

    result = execute_sql(db_name, sql)
    if result["success"]:
        rows = result["data"]
        if not rows:
            return "[EMPTY]"
        if len(rows) > 30:
            rows = rows[:30]
            extra = f"\n... (还有 {result['row_count'] - 30} 行)"
        else:
            extra = ""
        cols = list(rows[0].keys())
        lines = [" | ".join(cols)]
        for row in rows:
            lines.append(" | ".join(str(row.get(c, "")) for c in cols))
        return "\n".join(lines) + extra
    else:
        return f"[ERROR] {result.get('error', '')}"
