"""
Schema 缓存层 — 避免重复探索同一张表的结构。

跨轮次持久化：同一个数据库连接中，表结构只查一次。
单数据库版本，key 为 table_name。
"""

from typing import Optional

# 缓存结构: {table_name: {"columns": [...], "enums": {...}, "dates": str, "rows": int}}
_cache: dict = {}


def get_columns(table_name: str) -> Optional[list]:
    """获取已缓存的列信息，每列 {name, type}。"""
    entry = _cache.get(table_name.lower())
    return entry["columns"] if entry else None


def set_columns(table_name: str, columns: list):
    """缓存列信息。"""
    k = table_name.lower()
    if k not in _cache:
        _cache[k] = {}
    _cache[k]["columns"] = columns


def get_enum(table_name: str, col_name: str) -> Optional[list]:
    """获取已缓存的枚举值。"""
    entry = _cache.get(table_name.lower())
    return entry.get("enums", {}).get(col_name) if entry else None


def set_enum(table_name: str, col_name: str, values: list):
    """缓存枚举值（最多 50 个）。"""
    k = table_name.lower()
    if k not in _cache:
        _cache[k] = {"columns": None, "enums": {}}
    if "enums" not in _cache[k]:
        _cache[k]["enums"] = {}
    _cache[k]["enums"][col_name] = values[:50]


def get_date_range(table_name: str) -> Optional[str]:
    """获取已缓存的日期范围。"""
    entry = _cache.get(table_name.lower())
    return entry.get("dates") if entry else None


def set_date_range(table_name: str, range_str: str):
    """缓存日期范围。"""
    k = table_name.lower()
    if k not in _cache:
        _cache[k] = {}
    _cache[k]["dates"] = range_str


def get_row_count(table_name: str) -> Optional[str]:
    """获取已缓存的行数。"""
    entry = _cache.get(table_name.lower())
    return entry.get("rows") if entry else None


def set_row_count(table_name: str, count: str):
    """缓存行数。"""
    k = table_name.lower()
    if k not in _cache:
        _cache[k] = {}
    _cache[k]["rows"] = count


