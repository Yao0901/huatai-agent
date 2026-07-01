"""
全局状态定义

定义了贯穿三个 Agent 子图的状态字典结构。
LangGraph 使用 TypedDict 定义 State，所有节点通过读写 State 字段来传递数据。

设计原则：
- 每个字段有明确的"谁写入、谁读取"
- 路由决策字段（如 ambiguity_flag、security_level）在 agent 层的 conditional edge 中使用
"""

from typing import TypedDict, List, Dict, Any, Optional


class AgentState(TypedDict, total=False):
    """
    贯穿意图识别 → 查询生成 → 安全评估全流程的全局状态。

    字段按使用阶段分为四组。
    """

    # =========================================================================
    # 第 0 组：贯穿全流程的基础字段
    # =========================================================================

    # 用户原始自然语言输入，由 main.py 在启动时写入
    user_input: str

    # 对话消息历史（可选，用于多轮交互场景）
    messages: List[Dict[str, Any]]

    # =========================================================================
    # 第 1 组：意图识别 Agent (Agent1) 写入的字段
    # =========================================================================

    # 是否存在模糊口径/目标 → 节点 3 写入，Agent1 conditional edge 读取
    ambiguity_flag: bool

    # 歧义说明文本（调试/日志用）→ 节点 3 写入
    ambiguity_detail: str

    # 是否成功在口径库中找到所有指标 → metric_search_node 写入
    metric_found: bool

    # 已成功消歧的指标列表，每项 {"keyword": str, "definition": dict}
    # → metric_search_node 写入，sql_gen_node 读取
    resolved_metrics: List[Dict[str, Any]]

    # 未能消歧的关键词列表 → metric_search_node 写入
    unresolved_terms: List[str]

    # 结构化的意图表示（实体、维度、条件等）→ 意图识别阶段写入
    resolved_intent: Dict[str, Any]

    # =========================================================================
    # 第 2 组：查询生成 Agent (Agent2) 写入的字段
    # =========================================================================

    # 候选表名列表 → schema_node 写入
    candidate_tables: List[str]

    # 候选表详细结构（含 DDL、列信息）→ schema_node 写入，sql_gen_node 读取
    table_schemas: List[Dict[str, Any]]

    # 表结构是否贴合用户意图 → fit_check_node 写入，Agent2 conditional edge 读取
    fit_check: bool

    # 贴合度校验详情（用户放弃时说明大模型如何选表）→ fit_check_node 写入
    fit_check_detail: str

    # LLM 生成的 SQL 语句 → sql_gen_node 写入
    generated_sql: str

    # =========================================================================
    # 第 3 组：安全评估 Agent (Agent3) 写入的字段
    # =========================================================================

    # SQL 安全级别："safe" | "dangerous" | "forbidden"
    # → security_node 写入，Agent3 conditional edge 读取
    security_level: str

    # 用户是否确认执行危险 SQL → confirm_dangerous_node 写入
    user_confirmed_dangerous: bool

    # AST 安全分析详情 → security_node 写入
    security_detail: Dict[str, Any]

    # 权限校验结果："denied" | "filtered" | "passed"
    # → security_node 写入
    permission_check: str

    # 注入权限过滤条件后的 SQL（行级安全）
    # → security_node 写入，execution_node 读取
    injected_sql: str

    # SQL 执行结果 → execution_node 写入
    execution_result: Dict[str, Any]

    # 执行报错信息（成功时为 None）→ execution_node 写入
    error_message: Optional[str]

    # 错误分类标签 → reflect_node 写入，Agent3 conditional edge 读取
    error_type: Optional[str]

    # 反思反馈（给下次 SQL 生成的修正建议）→ reflect_node 写入
    reflection_feedback: Optional[str]

    # 当前重试次数（每次反思递增 1）→ reflect_node 写入
    retry_count: int

    # =========================================================================
    # 第 4 组：最终输出
    # =========================================================================

    # 最终返回给用户的结果文本
    final_output: Optional[str]
