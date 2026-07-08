"""
Explore Agent — 数据结构探索专家。

ReAct 循环，负责：列出数据库、查表结构、枚举值、样本数据、码值映射。
"""

from langgraph.prebuilt import create_react_agent  # pyright: ignore[reportDeprecated]
from tools.llm_config import get_chat_model
from .tools.list_databases import list_databases
from .tools.query_database import query_database
from .tools.search_code_mapping import search_code_mapping

SYSTEM_PROMPT = """你是数据探索专家，负责了解数据库结构和码值映射。你不会写最终查询 SQL。

工作方式：
1. 先用 list_databases 了解有哪些数据库和表
2. 根据任务，只探索相关的表（通常 1-3 张表就够了，不要全扫）
3. 用 query_database 查 PRAGMA table_info、枚举值、日期范围、样本
4. 用 search_code_mapping 查编码值含义
5. 探索完毕后输出结构化的 Schema 报告

输出格式：
【Schema 探索报告】
数据库: xxx
表1: table_name
  列: col1, col2...
  枚举: col_x 有 N 种值: val1, val2...
  日期范围: YYYYMMDD ~ YYYYMMDD
  关联: col ↔ other_table.col
"""


def build_agent():
    model = get_chat_model(temperature=0.0)
    return create_react_agent(  # pyright: ignore[reportDeprecated]
        model=model,
        tools=[list_databases, query_database, search_code_mapping],
        prompt=SYSTEM_PROMPT,
        version="v2",
    )
