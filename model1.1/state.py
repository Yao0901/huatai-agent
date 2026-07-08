"""
多 Agent 共享状态定义

Supervisor → Explore / SQL / Analysis 三个 Agent 通过此 State 通信。
"""

from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages


class MultiAgentState(TypedDict):
    # 全图共享消息历史（add_messages 自动合并）
    messages: Annotated[list, add_messages]

    # Supervisor 路由决策
    next_agent: str                    # "explore" | "sql" | "analysis" | "FINISH"
    supervisor_instruction: str        # 给目标 Agent 的指令

    # 循环控制（防止无限循环）
    round_count: int                   # 当前第几轮
    explore_count: int                 # Explore Agent 已调用次数
    sql_count: int                     # SQL Agent 已调用次数
    analysis_count: int                # Analysis Agent 已调用次数

    # 多数据库支持
    available_dbs: list[str]           # 所有可用数据库名称
    current_db: str                    # 当前查询使用的数据库

    # Explore Agent 产出
    explored_schemas: str              # Schema 探索报告（结构化文本）

    # SQL Agent 产出
    final_sql: str                     # 最终执行的 SQL
    final_result: str                  # SQL 执行返回的原始数据

    # Analysis Agent 产出
    analysis: str                      # 自然语言解读结果
