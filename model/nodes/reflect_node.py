"""
反思节点：LLM 分析 SQL 执行报错原因，生成修正建议。

当执行报错且重试次数未达上限（<3次）时触发。
将报错信息 + 失败 SQL + 表结构一起发给 LLM，让它做智能诊断，
输出结构化的修正建议供 sql_gen_node 在下一次重试时使用。
"""

import json
from typing import Dict, Any

from tools.llm_config import chat


# ---------------------------------------------------------------------------
# 节点主函数
# ---------------------------------------------------------------------------

def reflect_on_error(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LLM 分析 SQL 执行报错，生成诊断结果和修正建议。

    调用时机：节点 8 执行失败且 retry_count < 3 时。

    Args:
        state: 全局状态字典

    Returns:
        dict: 更新 error_type, reflection_feedback, retry_count
    """
    error_message = state.get("error_message", "")
    generated_sql = state.get("generated_sql", "")
    retry_count = state.get("retry_count", 0)
    table_schemas = state.get("table_schemas", [])

    # ---- 组装表名列名清单 ----
    col_info_lines = []
    for t in table_schemas:
        tname = t.get("table_name", "")
        tdesc = t.get("description", "")
        cols = [c["name"] for c in t.get("columns", [])]
        col_info_lines.append(
            f"  {tname}（{tdesc[:60]}）: {', '.join(cols[:25])}"
        )
    col_info = "\n".join(col_info_lines) if col_info_lines else "（无）"

    # ---- LLM 诊断 ----
    system_prompt = (
        "你是 SQLite 数据库调试专家。分析下面这条 SQL 执行失败的原因，给出修正建议。\n\n"
        "输出一个 JSON 对象：\n"
        "  - error_type: 错误类型（column_not_found / table_not_found / syntax_error / "
        "type_mismatch / ambiguous_column / other）\n"
        "  - diagnosis: 一句话诊断（中文）\n"
        "  - fix_suggestion: 具体的修改建议（中文，可直接交给 LLM 用于修正 SQL）\n"
        "只输出 JSON，不要其他文字。"
    )

    user_prompt = (
        f"失败的 SQL:\n```sql\n{generated_sql}\n```\n\n"
        f"报错信息: {error_message}\n\n"
        f"可用表及列（只有这些是真实存在的）:\n{col_info}\n\n"
        f"这是第 {retry_count + 1} 次重试。"
    )

    try:
        response = chat(system_prompt, user_prompt, temperature=0.0)
        response = response.strip()
        if response.startswith("```"):
            response = response.split("\n", 1)[1].rsplit("\n```", 1)[0]
        result = json.loads(response)
        error_type = result.get("error_type", "other")
        diagnosis = result.get("diagnosis", "未知错误")
        fix_suggestion = result.get("fix_suggestion", "请检查 SQL 语法")
    except Exception:
        # LLM 不可用，降级为关键词分类
        error_type = _keyword_classify(error_message)
        diagnosis = f"关键词分类: {error_type}"
        fix_suggestion = _build_fallback_feedback(error_message, generated_sql, col_info_lines)

    # ---- 组装反馈 ----
    reflection_feedback = (
        f"上次生成的 SQL 执行失败（第 {retry_count + 1} 次重试）。\n\n"
        f"失败的 SQL:\n```sql\n{generated_sql}\n```\n\n"
        f"报错信息: {error_message}\n\n"
        f"诊断: {diagnosis}\n\n"
        f"修正建议: {fix_suggestion}\n\n"
        f"可用表和列（只使用这些）:\n{col_info}"
    )

    return {
        "error_type": error_type,
        "reflection_feedback": reflection_feedback,
        "retry_count": retry_count + 1,
    }


# ---------------------------------------------------------------------------
# 降级：关键词分类
# ---------------------------------------------------------------------------

def _keyword_classify(error_message: str) -> str:
    msg = error_message.lower()
    if "no such column" in msg:
        return "column_not_found"
    elif "no such table" in msg:
        return "table_not_found"
    elif "syntax error" in msg or "near" in msg:
        return "syntax_error"
    elif "ambiguous column" in msg:
        return "ambiguous_column"
    elif "mismatch" in msg or "cannot" in msg:
        return "type_mismatch"
    return "other"


def _build_fallback_feedback(
    error_message: str, generated_sql: str, col_info_lines: list
) -> str:
    """LLM 不可用时生成基础修正建议。"""
    col_text = "\n".join(col_info_lines) if col_info_lines else "（无）"
    return (
        f"SQL 执行报错: {error_message}\n"
        f"请检查列名是否在以下可用列中:\n{col_text}\n"
        f"并确认表名和 JOIN 条件正确。"
    )
