"""
业务口径与码值检索工具（精简版）

仅保留码值映射查询功能。口径定义匹配由 Supervisor 在调度层处理。
"""

import os
from typing import List, Dict, Any, Optional


# ---------------------------------------------------------------------------
# 路径解析
# ---------------------------------------------------------------------------

def _resolve_data_dir() -> str:
    tools_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.dirname(tools_dir)
    project_dir = os.path.dirname(model_dir)
    return os.path.join(project_dir, "data", "业务词汇匹配文件夹")


_DATA_DIR = _resolve_data_dir()
_DEFINITIONS_PATH = os.path.join(_DATA_DIR, "metric_definitions.json")


# ---------------------------------------------------------------------------
# 码值映射查询
# ---------------------------------------------------------------------------

def search_code_mapping(
    user_value: str,
    db_name: str,
    code_type_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    查询码值映射：将用户自然语言中的枚举值翻译为数据库编码。

    Args:
        user_value: 用户描述的值（如 "男" / "钻石卡" / "博士"）
        db_name: 目标数据库名称
        code_type_id: 限定码值类型（如 "500"=性别），None 表示不限

    Returns:
        list[dict]: 匹配的码值列表 [{code, describe, code_type_id}]
    """
    from tools.db_connector import get_connection

    conn = get_connection(db_name)
    if conn is None:
        return []

    results = []

    if code_type_id:
        cursor = conn.execute(
            "SELECT code, describe, code_type_id FROM dim_public "
            "WHERE code_type_id=? AND describe LIKE ?",
            (code_type_id, f"%{user_value}%"),
        )
    else:
        cursor = conn.execute(
            "SELECT code, describe, code_type_id FROM dim_public "
            "WHERE describe LIKE ?",
            (f"%{user_value}%",),
        )

    for row in cursor.fetchall():
        results.append({
            "code": row["code"],
            "describe": row["describe"],
            "code_type_id": row["code_type_id"],
        })

    # 没匹配到 describe 则尝试精确匹配 code
    if not results:
        cursor = conn.execute(
            "SELECT code, describe, code_type_id FROM dim_public WHERE code=?",
            (user_value,),
        )
        for row in cursor.fetchall():
            results.append({
                "code": row["code"],
                "describe": row["describe"],
                "code_type_id": row["code_type_id"],
            })

    return results


def get_metric_definitions_text() -> str:
    """获取全部口径定义的文本描述，供 Supervisor 参考。"""
    import json
    if not os.path.exists(_DEFINITIONS_PATH):
        return ""
    with open(_DEFINITIONS_PATH, "r", encoding="utf-8") as f:
        defs = json.load(f)
    lines = []
    for d in defs:
        aliases = ", ".join(d.get("aliases", []))
        lines.append(
            f"- {d['metric_name']}（别名: {aliases}）: {d.get('description', '')}"
            f" | 涉及表: {', '.join(d.get('involved_tables', []))}"
        )
    return "\n".join(lines)
