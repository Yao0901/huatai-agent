"""
Schema 缓存层 — 避免重复探索同一张表的结构。

跨轮次持久化：同一个数据库连接中，表结构只查一次。
"""

from typing import Optional

# 缓存结构: {(db_name, table_name): {"columns": [...], "enums": {...}, "dates": str, "rows": int}}
_cache: dict = {}


def _key(db_name: str, table_name: str) -> tuple:
    return (db_name, table_name.lower())


def get_columns(db_name: str, table_name: str) -> Optional[list]:
    """获取已缓存的列信息，每列 {name, type}。"""
    entry = _cache.get(_key(db_name, table_name))
    return entry["columns"] if entry else None


def set_columns(db_name: str, table_name: str, columns: list):
    """缓存列信息。"""
    k = _key(db_name, table_name)
    if k not in _cache:
        _cache[k] = {}
    _cache[k]["columns"] = columns


def get_enum(db_name: str, table_name: str, col_name: str) -> Optional[list]:
    """获取已缓存的枚举值。"""
    entry = _cache.get(_key(db_name, table_name))
    return entry.get("enums", {}).get(col_name) if entry else None


def set_enum(db_name: str, table_name: str, col_name: str, values: list):
    """缓存枚举值（最多 50 个）。"""
    k = _key(db_name, table_name)
    if k not in _cache:
        _cache[k] = {"columns": None, "enums": {}}
    if "enums" not in _cache[k]:
        _cache[k]["enums"] = {}
    _cache[k]["enums"][col_name] = values[:50]


def get_date_range(db_name: str, table_name: str) -> Optional[str]:
    """获取已缓存的日期范围。"""
    entry = _cache.get(_key(db_name, table_name))
    return entry.get("dates") if entry else None


def set_date_range(db_name: str, table_name: str, range_str: str):
    """缓存日期范围。"""
    k = _key(db_name, table_name)
    if k not in _cache:
        _cache[k] = {}
    _cache[k]["dates"] = range_str


def get_row_count(db_name: str, table_name: str) -> Optional[int]:
    """获取已缓存的行数。"""
    entry = _cache.get(_key(db_name, table_name))
    return entry.get("rows") if entry else None


def set_row_count(db_name: str, table_name: str, count: int):
    """缓存行数。"""
    k = _key(db_name, table_name)
    if k not in _cache:
        _cache[k] = {}
    _cache[k]["rows"] = count


def is_cached(db_name: str, table_name: str) -> bool:
    """表结构是否已缓存。"""
    return _key(db_name, table_name) in _cache and _cache[_key(db_name, table_name)].get("columns") is not None


def summary() -> str:
    """返回已缓存的表摘要。"""
    if not _cache:
        return "（空）"
    lines = []
    for (db, table), entry in _cache.items():
        cols = entry.get("columns")
        if cols:
            col_names = [c["name"] for c in cols] if isinstance(cols[0], dict) else cols
            dates = entry.get("dates", "")
            rows = entry.get("rows", "")
            lines.append(f"  [{db}] {table}: {len(col_names)} 列, {rows} 行, 日期 {dates}")
    return "\n".join(lines) if lines else "（空）"
