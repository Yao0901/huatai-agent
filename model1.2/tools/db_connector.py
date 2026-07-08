"""
数据库连接与执行工具

基于 SQLite 实现，启动时自动从 CSV 文件初始化数据库。
生产环境可替换为 PostgreSQL / MySQL 连接。
"""

import os
import csv
import sqlite3
from typing import Dict, Any, Optional

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

# 数据库文件路径
# 优先使用环境变量，其次用当前工作目录推测
def _resolve_paths() -> tuple:
    """推测项目路径：从当前文件位置向上找 model/ 目录。"""
    try:
        tool_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        tool_dir = os.path.join(os.getcwd(), "tools")
    model_dir = os.path.dirname(tool_dir)  # tools/ 的上一级即 model/
    project_dir = os.path.dirname(model_dir)  # model/ 的上一级即项目根目录
    return model_dir, project_dir

_MODEL_DIR, _PROJECT_DIR = _resolve_paths()
DB_PATH = os.path.join(_MODEL_DIR, "huatai.db")

# CSV 数据源目录 — 递归扫描 data/ 下所有 CSV 文件
def _extract_table_name(filename: str) -> str:
    """从文件名提取表名，自动识别末尾时间戳。"""
    base = filename.replace(".csv", "")
    parts = base.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return base


def _build_csv_map() -> dict:
    """递归扫描 data/ 下所有 CSV，返回 {完整路径: 表名} 映射。"""
    import glob
    data_root = os.path.join(_PROJECT_DIR, "data")
    mapping = {}
    for f in glob.glob(os.path.join(data_root, "**", "*.csv"), recursive=True):
        table = _extract_table_name(os.path.basename(f))
        mapping[f] = table
    return mapping

# 全局连接实例（单例模式，避免重复建库）
_connection: Optional[sqlite3.Connection] = None


# ---------------------------------------------------------------------------
# 数据库初始化
# ---------------------------------------------------------------------------

def _init_database(force_reload: bool = False) -> sqlite3.Connection:
    """
    初始化 SQLite 数据库：从 CSV 文件建表并导入数据。

    首次调用时自动执行，后续调用返回已有连接。
    设置 force_reload=True 可强制重建。

    Args:
        force_reload: 是否强制删除已有数据库并重建

    Returns:
        sqlite3.Connection: 数据库连接对象
    """
    global _connection

    if _connection is not None and not force_reload:
        return _connection

    # 如果强制重建，删除旧文件
    if force_reload and os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    _connection = sqlite3.connect(DB_PATH, check_same_thread=False)
    _connection.row_factory = sqlite3.Row  # 让查询结果可以用列名访问

    for csv_path, table_name in _build_csv_map().items():
        # 第1步：读取 CSV 表头推断列类型
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            columns = next(reader)

        # 第2步：建表（所有列用 TEXT 存储，SQLite 动态类型足够灵活）
        col_defs = ", ".join(f'"{c}" TEXT' for c in columns)
        create_sql = f'CREATE TABLE IF NOT EXISTS "{table_name}" ({col_defs})'

        # 检查表是否已有数据（非空则跳过导入）
        cursor = _connection.execute(
            f"SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='{table_name}'"
        )
        table_exists = cursor.fetchone()[0] > 0

        if not table_exists or force_reload:
            if force_reload:
                _connection.execute(f'DROP TABLE IF EXISTS "{table_name}"')
            _connection.execute(create_sql)

            # 第3步：导入 CSV 数据
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                next(reader)  # 跳过表头
                placeholders = ", ".join("?" for _ in range(len(columns)))
                insert_sql = f'INSERT INTO "{table_name}" VALUES ({placeholders})'
                _connection.executemany(insert_sql, reader)

            _connection.commit()
            print(f"[DB] 导入 {table_name}: {os.path.basename(csv_path)}")

    return _connection


# ---------------------------------------------------------------------------
# 获取连接
# ---------------------------------------------------------------------------

def get_connection() -> sqlite3.Connection:
    """
    获取数据库连接（懒初始化，首次调用时自动建库导入CSV）。

    所有需要执行 SQL 的模块都应通过此函数获取连接，
    而不是直接访问全局变量。

    Returns:
        sqlite3.Connection
    """
    return _init_database()


# ---------------------------------------------------------------------------
# SQL 执行
# ---------------------------------------------------------------------------

def execute_sql(sql: str) -> Dict[str, Any]:
    """
    安全执行一条 SELECT 语句并返回结果。

    核心职责：
    1. 接收经过安全校验的 SQL
    2. 在 SQLite 中执行
    3. 返回结构化结果（成功含数据 / 失败含错误信息）

    Args:
        sql: 已经过安全校验的 SQL 语句（只读 SELECT）

    Returns:
        dict:
            - success (bool): 是否执行成功
            - data (list[dict]): 查询结果行列表（成功时）
            - error (str): 报错信息（失败时）
            - row_count (int): 返回行数
    """
    conn = get_connection()
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


