"""
在指定数据库中执行只读探索查询（带 Schema 缓存）。

同张表结构只查一次，后续命中缓存直接返回。
"""

import re
from tools.db_connector import execute_sql
from tools import schema_cache


def _extract_table_name(sql: str) -> str:
    """从 SQL 中提取表名。"""
    # PRAGMA table_info('表名')
    m = re.search(r"table_info\s*\(\s*'(\w+)'\s*\)", sql, re.IGNORECASE)
    if m:
        return m.group(1)
    # FROM 表名
    m = re.search(r'\bFROM\s+(\w+)', sql, re.IGNORECASE)
    if m:
        return m.group(1)
    return ""


def _extract_col_name(sql: str) -> str:
    """从 SELECT DISTINCT col 中提取列名。"""
    m = re.search(r'DISTINCT\s+(\w+)', sql, re.IGNORECASE)
    if m:
        return m.group(1)
    return ""


def query_database(db_name: str, sql: str) -> str:
    """
    在指定数据库中执行只读 SQL 探索数据库结构。同张表结构自动缓存。
    """
    sql_upper = sql.upper().strip()
    if not (sql_upper.startswith("SELECT") or sql_upper.startswith("PRAGMA")):
        return f"[BLOCKED] 只允许 SELECT 和 PRAGMA。"
    if "sqlite_master" in sql_upper:
        return "[BLOCKED] 禁止查系统表"

    table = _extract_table_name(sql)

    # ---- PRAGMA table_info 缓存 ----
    if "PRAGMA" in sql_upper and table:
        cached = schema_cache.get_columns(db_name, table)
        if cached:
            lines = [" | ".join(["cid", "name", "type", "notnull", "dflt_value", "pk"])]
            for i, col in enumerate(cached):
                lines.append(f"{i} | {col['name']} | {col['type']} | 0 |  | 0")
            return "\n".join(lines) + f"\n[缓存命中] {table} 已探索过"

    # ---- 日期范围缓存 ----
    if ("MIN(" in sql_upper or "MAX(" in sql_upper) and "data_dt" in sql_upper and table:
        cached = schema_cache.get_date_range(db_name, table)
        if cached:
            return f"min_dt | max_dt\n{cached}\n[缓存命中]"

    # ---- 枚举值缓存 ----
    if "DISTINCT" in sql_upper and table:
        col = _extract_col_name(sql)
        if col:
            cached = schema_cache.get_enum(db_name, table, col)
            if cached:
                lines = [col]
                for v in cached:
                    lines.append(str(v))
                return "\n".join(lines) + f"\n[缓存命中] 共 {len(cached)} 种"

    # ---- COUNT 缓存 ----
    if sql_upper.startswith("SELECT COUNT(*)") and "FROM" in sql_upper and "WHERE" not in sql_upper and table:
        cached = schema_cache.get_row_count(db_name, table)
        if cached is not None:
            return f"cnt\n{cached}\n[缓存命中]"

    # ---- 执行查询 ----
    result = execute_sql(db_name, sql)
    if not result["success"]:
        return f"[ERROR] {result.get('error', '未知')}"

    rows = result["data"]
    if not rows:
        return "[EMPTY] 查询无结果"

    # ---- 缓存结果 ----
    if "PRAGMA" in sql_upper and table:
        schema_cache.set_columns(db_name, table, rows)
    elif ("MIN(" in sql_upper or "MAX(" in sql_upper) and "data_dt" in sql_upper and table and len(rows) == 1:
        row = rows[0]
        vals = [str(v) for v in row.values()]
        if len(vals) == 2:
            schema_cache.set_date_range(db_name, table, f"{vals[0]} | {vals[1]}")
        elif len(vals) == 1:
            schema_cache.set_date_range(db_name, table, str(vals[0]))
    elif "DISTINCT" in sql_upper and table and len(rows) <= 50:
        col = _extract_col_name(sql)
        if col and rows:
            values = [list(r.values())[0] for r in rows]
            schema_cache.set_enum(db_name, table, col, values)
    elif sql_upper.startswith("SELECT COUNT(*)") and "WHERE" not in sql_upper and table and len(rows) == 1:
        schema_cache.set_row_count(db_name, table, list(rows[0].values())[0])

    # ---- 格式化输出 ----
    if len(rows) > 50:
        rows = rows[:50]
        extra = f"\n... (还有 {result['row_count'] - 50} 行)"
    else:
        extra = ""
    cols = list(rows[0].keys())
    lines = [" | ".join(cols)]
    for row in rows:
        lines.append(" | ".join(str(row.get(c, "")) for c in cols))
    return "\n".join(lines) + extra
