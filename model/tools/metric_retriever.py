"""
业务口径与指标检索工具

从用户自然语言中提取指标关键词，匹配标准口径定义。
使用 LLM 语义匹配，不再依赖本地 BERT 向量模型。

两大功能：
1. 码值映射查询：从 dim_public 表中查询编码 ↔ 中文描述
2. 口径定义匹配：LLM 根据数据库真实结构 + 口径描述判断用户说的是哪个指标
"""

import json
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

# 口径定义（懒加载）
_definitions = None


def _load_definitions():
    global _definitions
    if _definitions is None:
        with open(_DEFINITIONS_PATH, "r", encoding="utf-8") as f:
            _definitions = {d["metric_name"]: d for d in json.load(f)}
    return _definitions


def get_metric_definitions_text() -> str:
    """获取全部口径定义的文本描述，供 LLM 参考。"""
    defs = _load_definitions()
    lines = []
    for name, d in defs.items():
        aliases = ", ".join(d.get("aliases", []))
        lines.append(
            f"- {name}（别名: {aliases}）: {d.get('description', '')}"
            f" | 涉及表: {', '.join(d.get('involved_tables', []))}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 码值映射查询（查询 dim_public 表）
# ---------------------------------------------------------------------------

def search_code_mapping(user_value: str, code_type_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    查询码值映射：将用户自然语言中的枚举值翻译为数据库编码。

    Args:
        user_value: 用户描述的值（如 "男" / "钻石卡" / "博士"）
        code_type_id: 限定码值类型（如 "500"=性别），None 表示不限

    Returns:
        list[dict]: 匹配的码值列表
    """
    from tools.db_connector import get_connection

    conn = get_connection()
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


# ---------------------------------------------------------------------------
# 口径定义匹配（LLM 语义匹配，替代原来的 BERT 向量搜索）
# ---------------------------------------------------------------------------

def search_metric_definition(user_term: str) -> Optional[Dict[str, Any]]:
    """
    用 LLM 匹配用户词到标准口径定义。

    LLM 看着完整的口径定义列表，判断用户说的词最匹配哪个定义。
    如果明显不匹配，返回 None。

    Args:
        user_term: 用户提到的指标关键词

    Returns:
        dict | None: 匹配到的指标定义
    """
    from tools.llm_config import chat

    user_term = user_term.strip()
    if not user_term:
        return None

    defs = _load_definitions()
    metrics_text = get_metric_definitions_text()

    prompt = (
        f"以下是系统中定义的标准业务口径：\n\n{metrics_text}\n\n"
        f"用户提到了一个指标词：「{user_term}」\n\n"
        f"请判断这个词最匹配哪个口径定义。\n"
        f"- 如果明显匹配某个定义，输出该定义的 metric_name\n"
        f"- 如果都不匹配，输出 none\n"
        f"只输出 metric_name 或 none，不要其他文字。"
    )

    try:
        response = chat("你是金融指标匹配专家", prompt, temperature=0.0)
        matched_name = response.strip()
        if matched_name and matched_name.lower() != "none":
            return defs.get(matched_name)
        return None
    except Exception:
        return None
