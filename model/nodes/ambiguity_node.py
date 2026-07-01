"""
节点 3：模糊口径与目标判断节点

分析用户输入，判断是否存在以下歧义：
1. 口径歧义：同一指标有多种计算方式（如"资产"可能是"总资产"或"净产"）
2. 目标歧义：用户诉求不明确（如"看看数据"这类笼统表述）

使用 LLM（DeepSeek）做语义分析，判断是否需要进一步消歧。
若分析歧义功能失效，直接跳过该节点
"""

from typing import Dict, Any

from tools.llm_config import chat


# ---------------------------------------------------------------------------
# 节点主函数
# ---------------------------------------------------------------------------

def check_ambiguity(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    调用 LLM 判断用户输入中是否存在模糊口径或模糊目标。

    调用时机：用户输入后立即执行（节点 3）。

    逻辑说明：
    1. 从全局状态取出 user_input
    2. 用 LLM 分析是否存在需要翻译的指标词或不够明确的分析目标
    3. 返回 ambiguity_flag 驱动后续路由：
       - True  → 走「寻找业务口径数据库」分支（search_metric）
       - False → 走「检索表结构」分支（跳过口径消歧，进入节点4）

    Args:
        state: 全局状态字典

    Returns:
        dict: 更新字段:
              - ambiguity_flag (bool): 是否存在模糊口径/目标
              - ambiguity_detail (str): 歧义说明
    """
    user_input = state.get("user_input", "")

    if not user_input.strip():
        return {
            "ambiguity_flag": False,
            "ambiguity_detail": "输入为空",
        }

    # ---- 拼接跨轮对话历史（让 LLM 理解"那上个月的呢"这类指代） ----
    messages = state.get("messages", [])
    history_preamble = ""
    if messages:
        history_lines = []
        for m in messages[-6:]:  # 最近 3 轮问答
            role_label = "用户" if m.get("role") == "user" else "助手"
            content = m.get("content", "")[:200]  # 截断防止 prompt 过长
            history_lines.append(f"[{role_label}] {content}")
        history_preamble = "【对话历史】\n" + "\n".join(history_lines) + "\n\n"

    # 调用 LLM 做歧义分析
    system_prompt = (
        "你是一个金融数据查询分析助手。请判断用户的自然语言问题中是否包含模糊的"
        "指标口径（如'盈亏''资产'可能有多种计算方式）、时期口径（如果用户未提及是近一周还是一年）或不明确的分析目标。\n"
        "特殊规则：如果用户表示不想继续消歧、让你随便选、多次追问后不耐烦、或明确表示放弃澄清，"
        "则 has_ambiguity 应为 false，表示不需要再追问，直接按默认理解继续。\n"
        "输出要求：只输出一个 JSON 对象，包含两个字段：\n"
        "  - has_ambiguity: true/false\n"
        "  - detail: 一句话描述歧义在哪里（无歧义或用户放弃时填'清晰'或'用户放弃消歧'）\n"
        "不要输出任何其他文字。"
    )

    try:
        response = chat(system_prompt, f"{history_preamble}用户问题：{user_input}", temperature=0.0)
        import json
        # 尝试从回复中提取 JSON
        response = response.strip()
        if response.startswith("```"):
            response = response.split("\n", 1)[1].rsplit("\n```", 1)[0]
        result = json.loads(response)
        return {
            "ambiguity_flag": result.get("has_ambiguity", False),
            "ambiguity_detail": result.get("detail", ""),
        }
    except Exception:
        # LLM 调用失败时的兜底策略：关键词匹配
        return _fallback_check(user_input)


def _fallback_check(user_input: str) -> Dict[str, Any]:
    """
    LLM 不可用时的兜底策略。

    不尝试做模糊判断（硬编码词表不靠谱），直接告知用户检测出问题，
    跳过消歧步骤，让流程继续往下走（进入子图2），由后续的 SQL 生成
    和错误重试机制兜底。

    Args:
        user_input: 用户原始输入

    Returns:
        dict: ambiguity_flag=False（跳过消歧，直接继续）
    """
    print("[WARN] 模糊语义检测 API 调用失败，跳过消歧步骤，直接进入查询生成")
    return {
        "ambiguity_flag": False,
        "ambiguity_detail": "LLM 调用失败，跳过模糊检测",
    }
