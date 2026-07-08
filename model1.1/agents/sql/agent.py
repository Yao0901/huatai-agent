"""
SQL Agent — SQL 生成与执行验证。

ReAct 循环，负责：根据 Schema 报告生成 SQL、执行、报错自修正。
"""

from langgraph.prebuilt import create_react_agent  # pyright: ignore[reportDeprecated]
from tools.llm_config import get_chat_model
from .tools.run_sql import run_sql
from .tools.query_database import query_database

SYSTEM_PROMPT = """你是 SQL 生成与执行专家。你会拿到 Schema 探索报告，不会凭空猜测表结构。

工作方式：
1. 审阅任务中提供的表结构信息
2. 生成正确的 SQL 查询
3. 用 run_sql 执行，查看结果
4. 报错 → 分析原因 → 修正 → 再执行（最多重试 3 次）
5. 执行成功后汇报最终 SQL 和结果

SQL 规范：
- 日期格式 YYYYMMDD，Q1 = 20260101 ~ 20260331
- 聚合用 COALESCE 处理 NULL
- 编码列用 dim_public JOIN 翻译
- 优先 LEFT JOIN，复杂查询用 CTE
- 严格遵循任务中的分组/筛选条件，不自己重新划分

输出格式：
【SQL 执行结果】
数据库: xxx
SQL:
  SELECT ...
结果:
  (run_sql 返回的数据)
"""


def build_agent():
    model = get_chat_model(temperature=0.0)
    return create_react_agent(  # pyright: ignore[reportDeprecated]
        model=model,
        tools=[run_sql, query_database],
        prompt=SYSTEM_PROMPT,
        version="v2",
    )
