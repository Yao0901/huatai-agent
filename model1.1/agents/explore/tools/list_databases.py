"""
列出所有可用数据库及其表名。
"""

from tools.db_connector import get_available_databases


def list_databases() -> str:
    """列出所有可用数据库及其包含的表。探索前应先调用此工具。"""
    dbs = get_available_databases()
    if not dbs:
        return "没有可用数据库"
    lines = [f"共 {len(dbs)} 个数据库:\n"]
    for db_name, info in dbs.items():
        tables = info.get("tables", [])
        lines.append(f"[{db_name}]  {len(tables)} 张表: {', '.join(tables)}")
    return "\n".join(lines)
