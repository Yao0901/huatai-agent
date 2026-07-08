"""
query_database — 数据库结构探索工具（带 Schema 缓存）

供 ReAct Agent 调用，用于探索表结构、列名、枚举值、样本数据。
同一张表的同类型查询只执行一次，后续命中缓存。
"""

import re
from typing import Optional

from tools.db_connector import execute_sql
from tools import schema_cache


# ---------------------------------------------------------------------------
# 已知表名白名单（用于从 SQL 中精确提取表名）
# ---------------------------------------------------------------------------

_ALL_TABLES = {
    "ads_cust_info_d", "dws_cust_aset_d", "dwd_cust_tran_d",
    "dwd_cust_hold_d", "dws_cust_fin_d", "dim_product",
    "dim_branch", "dim_public",
}


# ---------------------------------------------------------------------------
# SQL 解析辅助
# ---------------------------------------------------------------------------

def _extract_table(sql: str) -> Optional[str]:
    """从 SQL 中提取操作的表名。"""
    sql_upper = sql.upper()

    # PRAGMA table_info('表名')
    m = re.search(r"table_info\s*\(\s*'(\w+)'\s*\)", sql_upper)
    if m:
        return m.group(1).lower()

    # FROM 表名（取第一个匹配的已知表）
    for m in re.finditer(r'\bFROM\s+(\w+)', sql_upper):
        t = m.group(1).lower()
        if t in _ALL_TABLES:
            return t

    return None


def _detect_query_type(sql: str) -> str:
    """检测 SQL 的探索类型：columns / enum / date_range / row_count / sample / other。"""
    sql_upper = sql.upper()

    if "PRAGMA" in sql_upper and "TABLE_INFO" in sql_upper:
        return "columns"

    if "COUNT(*)" in sql_upper and "DISTINCT" not in sql_upper:
        return "row_count"

    if "MIN(" in sql_upper and "MAX(" in sql_upper and "DATA_DT" in sql_upper:
        return "date_range"

    if "DISTINCT" in sql_upper:
        return "enum"

    if "LIMIT" in sql_upper and "SELECT" in sql_upper:
        return "sample"

    return "other"


# ---------------------------------------------------------------------------
# 缓存读写
# ---------------------------------------------------------------------------

def _check_cache(table: str, qtype: str, sql: str) -> Optional[str]:
    """检查缓存，命中则返回格式化字符串，否则返回 None。"""
    if qtype == "columns":
        cols = schema_cache.get_columns(table)
        if cols:
            names = [c["name"] for c in cols] if isinstance(cols[0], dict) else [str(c) for c in cols]
            types = [c.get("type", "") for c in cols] if isinstance(cols[0], dict) else [""] * len(cols)
            lines = ["[缓存命中] cid | name | type"]
            for i, (n, t) in enumerate(zip(names, types)):
                lines.append(f"{i} | {n} | {t}")
            return "\n".join(lines)

    elif qtype == "enum":
        m = re.search(r'DISTINCT\s+(\w+)', sql, re.IGNORECASE)
        col = m.group(1).lower() if m else None
        if col:
            vals = schema_cache.get_enum(table, col)
            if vals:
                lines = [f"[缓存命中] {col}"]
                for v in vals:
                    lines.append(str(v))
                return "\n".join(lines)

    elif qtype == "date_range":
        dr = schema_cache.get_date_range(table)
        if dr:
            return f"[缓存命中] data_dt\n{dr}"

    elif qtype == "row_count":
        rc = schema_cache.get_row_count(table)
        if rc is not None:
            return f"[缓存命中] COUNT(*)\n{rc}"

    return None


def _write_cache(table: str, qtype: str, rows: list[dict], sql: str):
    """将查询结果写入缓存。"""
    if qtype == "columns":
        schema_cache.set_columns(table, rows)

    elif qtype == "enum":
        m = re.search(r'DISTINCT\s+(\w+)', sql, re.IGNORECASE)
        col = m.group(1).lower() if m else None
        if col and rows:
            col_key = list(rows[0].keys())[0]
            vals = [row[col_key] for row in rows]
            schema_cache.set_enum(table, col, vals)

    elif qtype == "date_range":
        if rows:
            keys = list(rows[0].keys())
            if len(keys) >= 2:
                dr = f"{rows[0][keys[0]]} ~ {rows[0][keys[1]]}"
            else:
                dr = str(rows[0][keys[0]])
            schema_cache.set_date_range(table, dr)

    elif qtype == "row_count":
        if rows:
            row = rows[0]
            val = None
            for key in row.keys():
                if "count" in key.lower() or "cnt" in key.lower():
                    val = row[key]
                    break
            if val is None:
                val = list(row.values())[-1]
            schema_cache.set_row_count(table, str(val))


# ---------------------------------------------------------------------------
# 公开工具
# ---------------------------------------------------------------------------

def query_database(sql: str) -> str:
    """
    执行只读 SQL 探索数据库结构。用于：
    - PRAGMA table_info('表名') 查看列名和类型
    - SELECT DISTINCT 列名 FROM 表名 LIMIT 30 查看枚举值
    - SELECT * FROM 表名 LIMIT 3 查看样本数据
    - SELECT MIN(data_dt), MAX(data_dt) FROM 表名 查看日期范围
    禁止 INSERT/UPDATE/DELETE/DROP。

    内置 Schema 缓存：同一张表的同类型查询只执行一次。
    """
    sql_upper = sql.strip().upper()
    if not (sql_upper.startswith("SELECT") or sql_upper.startswith("PRAGMA")):
        return f"[BLOCKED] 只允许 SELECT 和 PRAGMA。你发送的是: {sql[:60]}"
    if "sqlite_master" in sql_upper:
        return "[BLOCKED] 禁止查系统表"

    # ---- Schema 缓存检查 ----
    table = _extract_table(sql)
    qtype = _detect_query_type(sql)

    if table and qtype != "other":
        cached = _check_cache(table, qtype, sql)
        if cached is not None:
            return cached

    # ---- 缓存未命中，执行 SQL ----
    result = execute_sql(sql)
    if result["success"]:
        rows = result["data"]
        if not rows:
            return "[EMPTY] 查询无结果"
        if len(rows) > 50:
            rows = rows[:50]
            extra = f"\n... (还有 {result['row_count'] - 50} 行，已截断)"
        else:
            extra = ""
        cols = list(rows[0].keys())
        lines = [" | ".join(cols)]
        for row in rows:
            lines.append(" | ".join(str(row.get(c, "")) for c in cols))
        output = "\n".join(lines) + extra

        # ---- 写入缓存 ----
        if table:
            _write_cache(table, qtype, rows, sql)

        return output
    else:
        return f"[ERROR] {result.get('error', '未知')}"
