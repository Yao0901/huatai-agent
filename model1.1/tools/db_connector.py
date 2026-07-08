"""
数据库连接与执行工具（多数据库版）

启动时自动扫描 data/ 下每个含 CSV 的子文件夹，各自建独立 SQLite 数据库。
不同文件夹的数据库互不干扰。
"""

import os
import csv
import glob
import sqlite3
from typing import Dict, Any, Optional


# ---------------------------------------------------------------------------
# 路径解析
# ---------------------------------------------------------------------------

def _resolve_paths() -> tuple:
    """推测项目路径：从当前文件位置向上找 model1.1/ 目录。"""
    try:
        tool_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        tool_dir = os.path.join(os.getcwd(), "tools")
    model_dir = os.path.dirname(tool_dir)
    project_dir = os.path.dirname(model_dir)
    return model_dir, project_dir


_MODEL_DIR, _PROJECT_DIR = _resolve_paths()
_DATA_ROOT = os.path.join(_PROJECT_DIR, "data")


# ---------------------------------------------------------------------------
# CSV 工具
# ---------------------------------------------------------------------------

def _extract_table_name(filename: str) -> str:
    """从文件名提取表名，自动识别末尾时间戳（如 _202606031625）。"""
    base = filename.replace(".csv", "")
    parts = base.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return base


def _build_csv_map(csv_dir: str) -> dict:
    """扫描指定目录下所有 CSV，返回 {完整路径: 表名} 映射。"""
    mapping = {}
    for f in glob.glob(os.path.join(csv_dir, "*.csv")):
        table = _extract_table_name(os.path.basename(f))
        mapping[f] = table
    return mapping


# ---------------------------------------------------------------------------
# 多数据库管理
# ---------------------------------------------------------------------------

# 全局连接池：{db_name: sqlite3.Connection}
_connections: Dict[str, sqlite3.Connection] = {}
# 数据库信息：{db_name: {"path": str, "csv_dir": str, "tables": [str]}}
_db_info: Dict[str, dict] = {}


def _discover_databases() -> Dict[str, str]:
    """
    扫描 data/ 下所有子文件夹，发现含 CSV 的目录。
    跳过"业务词汇匹配文件夹"等无 CSV 的目录。

    Returns:
        dict: {db_name: csv_dir_path}
    """
    result = {}
    if not os.path.isdir(_DATA_ROOT):
        return result

    for entry in os.scandir(_DATA_ROOT):
        if not entry.is_dir():
            continue
        csv_files = glob.glob(os.path.join(entry.path, "*.csv"))
        if csv_files:
            result[entry.name] = entry.path

    return result


def _init_database(db_name: str, csv_dir: str, force_reload: bool = False) -> sqlite3.Connection:
    """
    为一个 CSV 目录建 SQLite 数据库。

    Args:
        db_name: 数据库名称（即 data/ 下的文件夹名）
        csv_dir: CSV 文件所在目录路径
        force_reload: 是否强制重建

    Returns:
        sqlite3.Connection
    """
    db_path = os.path.join(_MODEL_DIR, f"{db_name}.db")

    if force_reload and os.path.exists(db_path):
        os.remove(db_path)

    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row

    csv_map = _build_csv_map(csv_dir)
    tables = []

    for csv_path, table_name in csv_map.items():
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            columns = next(reader)

        col_defs = ", ".join(f'"{c}" TEXT' for c in columns)
        create_sql = f'CREATE TABLE IF NOT EXISTS "{table_name}" ({col_defs})'

        cursor = conn.execute(
            f"SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='{table_name}'"
        )
        table_exists = cursor.fetchone()[0] > 0

        if not table_exists or force_reload:
            if force_reload:
                conn.execute(f'DROP TABLE IF EXISTS "{table_name}"')
            conn.execute(create_sql)

            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                next(reader)
                placeholders = ", ".join("?" for _ in range(len(columns)))
                insert_sql = f'INSERT INTO "{table_name}" VALUES ({placeholders})'
                conn.executemany(insert_sql, reader)

            conn.commit()
            print(f"[DB:{db_name}] 导入 {table_name}: {os.path.basename(csv_path)}")

        tables.append(table_name)

    _db_info[db_name] = {"path": db_path, "csv_dir": csv_dir, "tables": tables}
    return conn


def init_all_databases(force_reload: bool = False):
    """扫描 data/ 并初始化所有数据库。main.py 启动时调用一次。"""
    global _connections

    databases = _discover_databases()
    if not databases:
        print("[DB] data/ 下未发现含 CSV 的文件夹，跳过建库")
        return

    print(f"[DB] 发现 {len(databases)} 个数据库: {', '.join(databases.keys())}")

    for db_name, csv_dir in databases.items():
        if db_name in _connections and not force_reload:
            continue
        _connections[db_name] = _init_database(db_name, csv_dir, force_reload)

    print(f"[DB] 全部数据库就绪")


# ---------------------------------------------------------------------------
# 获取连接
# ---------------------------------------------------------------------------

def get_connection(db_name: str) -> Optional[sqlite3.Connection]:
    """获取指定数据库的连接（懒初始化）。"""
    return _connections.get(db_name)


def get_available_databases() -> Dict[str, dict]:
    """
    返回所有可用数据库的摘要信息。

    Returns:
        dict: {db_name: {"tables": [str], "csv_dir": str}}
    """
    return dict(_db_info)


# ---------------------------------------------------------------------------
# SQL 执行
# ---------------------------------------------------------------------------

def execute_sql(db_name: str, sql: str) -> Dict[str, Any]:
    """
    在指定数据库上执行 SELECT 语句并返回结果。

    Args:
        db_name: 数据库名称
        sql: 只读 SQL 语句

    Returns:
        dict: {success, data, error, row_count}
    """
    conn = get_connection(db_name)
    if conn is None:
        return {
            "success": False,
            "data": [],
            "error": f"数据库'{db_name}'不存在。可用: {list(_connections.keys())}",
            "row_count": 0,
        }

    try:
        cursor = conn.execute(sql)
        rows = [dict(row) for row in cursor.fetchall()]
        return {
            "success": True,
            "data": rows,
            "error": None,
            "row_count": len(rows),
        }
    except Exception as e:
        return {
            "success": False,
            "data": [],
            "error": str(e),
            "row_count": 0,
        }
