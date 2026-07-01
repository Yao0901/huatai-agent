"""
SQL 语法树解析工具

使用 sqlglot 库对生成的 SQL 进行真实 AST 级别分析：
1. 识别操作类型（SELECT / DROP / DELETE / UPDATE 等）
2. 安全分级：forbidden（写结构）/ dangerous（写数据）/ safe（只读）
3. 提取涉及的表名和列名（供权限校验使用）
4. 向 SQL 注入行级权限过滤条件
"""

from typing import Dict, Any, List

import sqlglot
from sqlglot import exp


# ---------------------------------------------------------------------------
# 危险操作分类
# ---------------------------------------------------------------------------

FORBIDDEN_KINDS = {
    "DROP", "TRUNCATE", "ALTER", "CREATE",
}

DANGEROUS_KINDS = {
    "DELETE", "UPDATE", "INSERT", "MERGE", "REPLACE",
}


def analyze_sql(sql: str) -> Dict[str, Any]:
    """
    用 sqlglot 解析 SQL AST，做安全分级并提取元数据。

    Args:
        sql: 待分析的 SQL 字符串

    Returns:
        dict:
            - security_level: "forbidden" | "dangerous" | "safe"
            - operation_type: 检测到的操作类型
            - tables_involved: SQL 涉及的表名列表
            - columns_involved: SQL 涉及的列名列表
            - has_where_clause: 是否包含 WHERE 条件
    """
    try:
        parsed = sqlglot.parse_one(sql)
    except Exception:
        # 解析失败时保守处理：标记为 safe  SELECT 并尝试正则提取表名
        return _fallback_analyze(sql)

    if parsed is None:
        return _fallback_analyze(sql)

    # 获取根操作类型
    kind = _get_statement_kind(parsed)

    # 安全分级
    if kind in FORBIDDEN_KINDS:
        return {
            "security_level": "forbidden",
            "operation_type": kind,
            "tables_involved": [],
            "columns_involved": [],
            "has_where_clause": False,
        }
    elif kind in DANGEROUS_KINDS:
        return {
            "security_level": "dangerous",
            "operation_type": kind,
            "tables_involved": _extract_tables(parsed),
            "columns_involved": _extract_columns(parsed),
            "has_where_clause": _has_where(parsed),
        }
    else:
        return {
            "security_level": "safe",
            "operation_type": kind or "SELECT",
            "tables_involved": _extract_tables(parsed),
            "columns_involved": _extract_columns(parsed),
            "has_where_clause": _has_where(parsed),
        }


def _get_statement_kind(parsed: exp.Expression) -> str:
    """从 AST 根节点提取操作类型关键字。"""
    # 用 sqlglot 自带的类型名
    kind_map = {
        "Select": "SELECT",
        "Insert": "INSERT",
        "Update": "UPDATE",
        "Delete": "DELETE",
        "Drop": "DROP",
        "Create": "CREATE",
        "Alter": "ALTER",
        "Truncate": "TRUNCATE",
        "Merge": "MERGE",
        "Replace": "REPLACE",
        "Union": "SELECT",   # UNION 本质还是查询
        "CTE": "SELECT",     # WITH ... SELECT
    }
    key = type(parsed).__qualname__.replace("sqlglot.expressions.", "")
    return kind_map.get(key, "UNKNOWN")


def _extract_tables(parsed: exp.Expression) -> List[str]:
    """
    从 AST 中遍历所有 Table 节点，提取表名。

    处理 FROM、JOIN、子查询中的表引用。
    """
    tables = set()
    for node in parsed.walk():
        if isinstance(node, exp.Table):
            name = node.name
            # 忽略 sqlglot 生成的占位符
            if name and name != "":
                tables.add(name)
    return sorted(tables)


def _extract_columns(parsed: exp.Expression) -> List[str]:
    """
    从 AST 中遍历所有 Column 节点，提取列名。

    用于权限校验——判断 SQL 访问了哪些敏感列。
    """
    columns = set()
    for node in parsed.walk():
        if isinstance(node, exp.Column):
            col = node.name
            if col and col != "" and "*" not in col:
                columns.add(col)
    return sorted(columns)


def _has_where(parsed: exp.Expression) -> bool:
    """检测 AST 中是否存在 WHERE 子句。"""
    for node in parsed.walk():
        if isinstance(node, exp.Where):
            return True
    return False


# ---------------------------------------------------------------------------
# 兜底：sqlglot 解析失败时的正则降级
# ---------------------------------------------------------------------------

def _fallback_analyze(sql: str) -> Dict[str, Any]:
    """sqlglot 解析失败时的保守分析。"""
    import re
    sql_upper = sql.upper().strip()

    # 整词匹配检测危险关键字
    for kw in FORBIDDEN_KINDS:
        if re.search(r"\b" + kw + r"\b", sql_upper):
            return {
                "security_level": "forbidden",
                "operation_type": kw,
                "tables_involved": [],
                "columns_involved": [],
                "has_where_clause": False,
            }
    for kw in DANGEROUS_KINDS:
        if re.search(r"\b" + kw + r"\b", sql_upper):
            return {
                "security_level": "dangerous",
                "operation_type": kw,
                "tables_involved": _extract_tables_regex(sql),
                "columns_involved": [],
                "has_where_clause": "WHERE" in sql_upper,
            }

    return {
        "security_level": "safe",
        "operation_type": "SELECT",
        "tables_involved": _extract_tables_regex(sql),
        "columns_involved": [],
        "has_where_clause": "WHERE" in sql_upper,
    }


def _extract_tables_regex(sql: str) -> List[str]:
    """正则提取 FROM/JOIN 后的表名（降级用）。"""
    import re
    tables = set()
    for pattern in [r"\bFROM\s+([a-zA-Z_][a-zA-Z0-9_]*)", r"\bJOIN\s+([a-zA-Z_][a-zA-Z0-9_]*)"]:
        for m in re.findall(pattern, sql, re.IGNORECASE):
            if m.upper() not in ("SELECT", "WHERE", "ON", "AS", "AND", "OR", "EXISTS"):
                tables.add(m)
    return sorted(tables)


# ---------------------------------------------------------------------------
# 权限过滤注入
# ---------------------------------------------------------------------------

def inject_permission_filter(sql: str, filter_condition: str) -> str:
    """
    向 SQL 中注入行级权限过滤条件。

    注入位置规则：
    - 已有 WHERE  → 追加 AND
    - 无 WHERE     → 在 GROUP BY / ORDER BY / LIMIT 之前插入 WHERE
    - 纯 SELECT   → 末尾追加 WHERE

    Args:
        sql: 原始 SQL
        filter_condition: 过滤条件（如 "org_id IN ('XX00000054')"）

    Returns:
        str: 注入后的 SQL
    """
    import re

    sql_stripped = sql.strip()

    # 情况1：已有 WHERE → 追加 AND
    if re.search(r"\bWHERE\b", sql_stripped, re.IGNORECASE):
        return re.sub(
            r"\bWHERE\b",
            f"WHERE {filter_condition} AND ",
            sql_stripped,
            count=1,
            flags=re.IGNORECASE,
        )

    # 情况2：无 WHERE → 在这些子句前插入
    boundary = re.search(
        r"\b(GROUP\s+BY|ORDER\s+BY|LIMIT\s+\d+|HAVING\s+)",
        sql_stripped,
        re.IGNORECASE,
    )
    if boundary:
        pos = boundary.start()
        return sql_stripped[:pos] + f" WHERE {filter_condition} " + sql_stripped[pos:]

    # 情况3：纯查询，末尾追加
    return sql_stripped + f" WHERE {filter_condition}"
