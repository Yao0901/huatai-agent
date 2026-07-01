"""
NOTE:做法是用模糊词做向量化，再与定义的数据库做匹配，取最相似的
寻找业务口径数据库节点

当节点 3 判定存在模糊口径时触发。负责：
1. 从用户输入中抽取指标关键词
2. 调用 tools/metric_retriever 检索标准定义 + 码值映射
3. 判断是否找到 → 驱动后续路由
"""

from typing import Dict, Any

from tools.metric_retriever import search_metric_definition, search_code_mapping
from tools.llm_config import chat


# ---------------------------------------------------------------------------
# 关键词抽取（LLM辅助）
# ---------------------------------------------------------------------------

def _extract_key_terms(user_input: str, messages: list = None) -> dict:
    """
    用 LLM 从用户输入中抽取关键要素：指标词、实体词、维度词、码值词。

    比纯关键词匹配更精准——LLM 能区分"查资产"（指标）和"资产表"（表名）。

    Args:
        user_input: 用户原始输入
        messages:   跨轮对话历史 [{"role":..., "content":...}, ...]

    Returns:
        dict: {"metrics": [...], "entities": [...], "dimensions": [...], "code_terms": [...]}
    """
    # ---- 拼接跨轮对话历史 ----
    history_preamble = ""
    if messages:
        history_lines = []
        for m in messages[-6:]:
            role_label = "用户" if m.get("role") == "user" else "助手"
            content = m.get("content", "")[:200]
            history_lines.append(f"[{role_label}] {content}")
        history_preamble = "【对话历史】\n" + "\n".join(history_lines) + "\n\n"

    system_prompt = (
        "从以下用户问题中抽取金融数据查询的关键要素。"
        "输出一个 JSON，包含四个数组字段：\n"
        "  - metrics: 需要计算的指标（如 资产、盈亏、交易量、市值）\n"
        "  - entities: 涉及的实体（如 客户、产品、营业部）\n"
        "  - dimensions: 分组维度（如 年龄段、省份、营业部）\n"
        "  - code_terms: 需要翻译的码值描述（如 钻石卡、男、本科、科创板）\n"
        "每个数组元素是一个字符串。只输出 JSON，不要其他文字。"
    )

    try:
        response = chat(system_prompt, f"{history_preamble}用户问题：{user_input}", temperature=0.0)
        import json
        response = response.strip()
        if response.startswith("```"):
            response = response.split("\n", 1)[1].rsplit("\n```", 1)[0]
        return json.loads(response)
    except Exception:
        return {"metrics": [], "entities": [], "dimensions": [], "code_terms": []}


# ---------------------------------------------------------------------------
# 节点主函数
# ---------------------------------------------------------------------------

def search_metric(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    在业务口径数据库中检索用户提到的指标定义和码值映射。

    调用时机：节点 3 判定 ambiguity_flag=True 后。

    流程：
    1. 用 LLM 从用户输入中抽取指标词、实体词、码值词
    2. 逐个检索指标公式库（search_metric_definition）
    3. 逐个检索码值映射表（search_code_mapping，查 dim_public）
    4. 返回结果，驱动分支判断

    Args:
        state: 全局状态字典，需含 user_input

    Returns:
        dict: 更新字段:
              - metric_found (bool): 是否成功找到全部口径
              - resolved_metrics (list[dict]): 已消歧的指标定义
              - unresolved_terms (list[str]): 未找到定义的关键词
              - resolved_intent (dict): 结构化的意图表示（含 code_mappings）
    """
    user_input = state.get("user_input", "")
    messages = state.get("messages", [])
    terms = _extract_key_terms(user_input, messages)

    # 第1步：检索指标定义
    resolved = []
    unresolved = []
    metric_keywords = terms.get("metrics", [])

    for kw in metric_keywords:
        definition = search_metric_definition(kw)
        if definition:
            resolved.append({"keyword": kw, "definition": definition})
        else:
            unresolved.append(kw)

    # 第2步：检索码值映射（用户说的"钻石卡"→数据库编码'1000003'）
    code_mappings = {}
    for term in terms.get("code_terms", []):
        mappings = search_code_mapping(term)
        if mappings:
            code_mappings[term] = mappings

    # 第3步：组装结构化的意图对象
    resolved_intent = {
        "entities": terms.get("entities", []),
        "metrics": [m["definition"]["metric_name"] for m in resolved],
        "dimensions": terms.get("dimensions", []),
        "code_mappings": code_mappings,
    }

    return {
        "metric_found": len(unresolved) == 0,
        "resolved_metrics": resolved,
        "unresolved_terms": unresolved,
        "resolved_intent": resolved_intent,
    }
