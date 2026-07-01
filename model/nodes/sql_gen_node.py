"""
节点 6：生成 SQL 语句

调用 DeepSeek LLM，将结构化的取数意图 + 表结构信息 + 业务口径定义
转化为可执行的 PostgreSQL/SQLite SQL 语句。

这是整个 Agent 中唯一直接生成 SQL 的节点。
"""

from typing import Dict, Any

from tools.llm_config import chat


# ---------------------------------------------------------------------------
# Prompt 构建
# ---------------------------------------------------------------------------

def _build_sql_prompt(state: Dict[str, Any]) -> tuple:
    """
    组装送给 LLM 的 SQL 生成 Prompt。

    包含四部分关键信息：
    1. 角色设定
    2. 可用表结构（列信息 + 样本数据，帮助 LLM 理解码值字段）
    3. 业务口径定义（用户说的词 → 标准 SQL 表达式）
    4. 码值映射（用户说的"钻石卡" → 数据库编码）
    5. 用户原始问题

    Args:
        state: 全局状态字典

    Returns:
        tuple: (system_prompt, user_message)
    """
    user_input = state.get("user_input", "")
    table_schemas = state.get("table_schemas", [])
    resolved_metrics = state.get("resolved_metrics", [])
    resolved_intent = state.get("resolved_intent", {})
    reflection_feedback = state.get("reflection_feedback", "")

    # ---- 拼接跨轮对话历史 ----
    messages = state.get("messages", [])
    history_preamble = ""
    if messages:
        history_lines = []
        for m in messages[-6:]:
            role_label = "用户" if m.get("role") == "user" else "助手"
            content = m.get("content", "")[:200]
            history_lines.append(f"[{role_label}] {content}")
        history_preamble = "【对话历史】\n" + "\n".join(history_lines) + "\n\n"

    # ---- 构建表结构提示 ----
    schema_lines = []
    for t in table_schemas:
        schema_lines.append(f"【{t['table_name']}】{t.get('description', '')}")
        schema_lines.append(f"  行数: {t.get('row_count', '?')}")
        col_names = [c["name"] for c in t.get("columns", [])]
        schema_lines.append(f"  列: {', '.join(col_names[:30])}")
        # 附加样本数据（前2行，帮助 LLM 理解码值字段的实际内容）
        sample = t.get("sample_data", [])
        if sample:
            for i, row in enumerate(sample[:2]):
                vals = {k: v for k, v in row.items() if k in col_names[:8]}
                schema_lines.append(f"  样本{i+1}: {vals}")
        schema_lines.append("")

    schema_text = "\n".join(schema_lines)

    # ---- 构建口径提示 ----
    metric_lines = []
    for m in resolved_metrics:
        definition = m.get("definition", {})
        metric_lines.append(
            f"- 关键词「{m['keyword']}」→ {definition.get('metric_name')}: "
            f"{definition.get('description')}\n"
            f"  SQL表达式: {definition.get('formula')}\n"
            f"  涉及表: {', '.join(definition.get('involved_tables', []))}"
        )
    metric_text = "\n".join(metric_lines) if metric_lines else "（无特殊口径定义）"

    # ---- 构建码值映射提示 ----
    code_lines = []
    for term, mappings in resolved_intent.get("code_mappings", {}).items():
        for m in mappings:
            code_lines.append(
                f"- 用户说的「{term}」→ {m['code_type_id']}类码值 "
                f"code='{m['code']}', describe='{m['describe']}'"
            )
    code_text = "\n".join(code_lines) if code_lines else "（无需码值翻译）"

    # ---- 组装系统提示词 ----
    system_prompt = (
        "你是一名金融证券领域的数据查询专家。请根据以下信息生成一条可以在 SQLite 上执行的 SQL 语句。\n\n"
        "【SQL 编写规范】\n"
        "1. 只输出 SELECT 语句，禁止任何写操作\n"
        "2. 聚合计算使用 COALESCE 处理 NULL: COALESCE(SUM(amt), 0)\n"
        "3. 码值字段通过 dim_public 表做 JOIN 翻译，不要硬编码 code 值\n"
        "   （例如性别：LEFT JOIN dim_public g ON a.gender_cd=g.code AND g.code_type_id='500'）\n"
        "4. 复杂查询使用 WITH (CTE) 子句分步写\n"
        "5. 表名和列名只能来自【可用表结构】中列出的表，禁止编造表名\n"
        "6. 禁止查询 sqlite_master、information_schema 等系统表，这些表里没有业务数据\n"
        "7. 时间过滤条件用 data_dt 字段，格式 YYYYMMDD\n"
        "8. 如果有 Q1/Q2 等季度表述，Q1=01-01至03-31，Q2=04-01至06-30\n\n"
        "【输出格式】\n"
        "只输出一条完整的 SQL 语句。不要用代码块包裹，不要加解释。"
    )

    # ---- 组装用户消息 ----
    user_message = (
        f"{history_preamble}"
        f"【可用表结构】\n{schema_text}\n"
        f"【业务口径定义】\n{metric_text}\n\n"
        f"【码值映射】\n{code_text}\n\n"
        f"【用户问题】\n{user_input}"
    )

    # 如果有上次重试的反馈，追加进去
    if reflection_feedback:
        user_message += f"\n\n【上次错误反馈】\n{reflection_feedback}\n请修正上述问题。"

    return system_prompt, user_message


# ---------------------------------------------------------------------------
# 节点主函数
# ---------------------------------------------------------------------------

def generate_sql(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    调用 DeepSeek LLM 生成 SQL 语句。

    调用时机：节点 5（贴合度校验）判定表结构贴合用户意图后。

    Args:
        state: 全局状态字典，需包含:
               - user_input: 用户原始输入
               - table_schemas: 候选表结构
               - resolved_metrics: 已消歧的指标定义
               - resolved_intent: 结构化意图（含码值映射）
               - reflection_feedback: 上次重试的反馈（可选）

    Returns:
        dict: 更新字段:
              - generated_sql (str): LLM 生成的 SQL
    """
    system_prompt, user_message = _build_sql_prompt(state)

    try:
        response = chat(system_prompt, user_message, temperature=0.0, max_tokens=2048)
        sql = response.strip()
        # 清理可能的代码块包裹
        if sql.startswith("```"):
            lines = sql.split("\n")
            # 去掉 ```sql 和结尾的 ```
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            sql = "\n".join(lines).strip()
        return {"generated_sql": sql}
    except Exception as e:
        # LLM 不可用时的兜底：返回一个基础查询模板
        user_input = state.get("user_input", "")
        return {
            "generated_sql": (
                f"-- [LLM不可用，返回基础模板]\n"
                f"-- 原始问题: {user_input}\n"
                f"SELECT * FROM ads_cust_info_d LIMIT 10"
            )
        }
