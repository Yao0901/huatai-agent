"""
节点 4：检索表和列结构

根据消除歧义后的结构化意图，从数据库和描述库中检索相关的表结构（Schema）。

这是连接「用户意图」与「数据库物理模型」的关键桥梁。
"""

from typing import Dict, Any

from tools.schema_retriever import retrieve_schemas


# ---------------------------------------------------------------------------
# 上下文组装
# ---------------------------------------------------------------------------

def _build_search_context(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    从全局状态中组装表结构检索所需的上下文。

    合并三部分信息：
    - resolved_intent 中的实体、指标、维度
    - resolved_metrics 中指标定义自带的 involved_tables
    - user_input 中的原始关键词（兜底）

    Args:
        state: 全局状态字典

    Returns:
        dict: 用于 schema 检索的意图对象
    """
    intent = state.get("resolved_intent", {})

    # 从已消歧指标中提取涉及的表名
    suggested_tables = set()
    for item in state.get("resolved_metrics", []):
        for t in item.get("definition", {}).get("involved_tables", []):
            suggested_tables.add(t)

    return {
        "entities": intent.get("entities", []),
        "metrics": intent.get("metrics", []),
        "dimensions": intent.get("dimensions", []),
        "suggested_tables": list(suggested_tables),
    }


# ---------------------------------------------------------------------------
# 节点主函数
# ---------------------------------------------------------------------------

def retrieve_table_schemas(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    检索与当前意图最相关的数据库表结构。

    调用时机：
    - 路径 A：节点 3 判定无歧义 → 直接进入
    - 路径 B：口径检索完成 → 进入

    流程：
    1. 从 state 组装检索上下文
    2. 调用 tools/schema_retriever 做关键词匹配 + 关联扩散
    3. 返回候选表结构列表

    Args:
        state: 全局状态字典

    Returns:
        dict: 更新字段:
              - candidate_tables (list[str]): 候选表名
              - table_schemas (list[dict]): 每张表的 columns, sample_data, description
    """
    search_context = _build_search_context(state)

    # 召回表结构（LLM 优先 → 关键词兜底）
    schema_results = retrieve_schemas(
        search_context,
        top_k=10,
        user_input=state.get("user_input", ""),
    )

    return {
        "candidate_tables": [r["table_name"] for r in schema_results],
        "table_schemas": schema_results,
    }
