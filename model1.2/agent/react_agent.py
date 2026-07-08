"""
ReAct Agent — 单一 LLM 循环替代三张子图流水线

LLM 可自主调用三个工具：
- query_database: 探索表结构 / 列名 / 枚举值 / 样本数据（带 Schema 缓存）
- run_sql: 执行 SQL 并查看结果（自我验证/修正）
- ask_user: 向用户提问澄清模糊需求
"""

from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

from tools.llm_config import get_chat_model
from tools.query_database import query_database
from tools.run_sql import run_sql
from tools.ask_user import ask_user


# ---------------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """你是「智谋洞见 Agent」，由智谋洞见团队打造。你可以操作一个包含客户、产品、交易、持仓、资产等数据的 SQLite 数据库，帮助用户进行金融数据查询与分析。

## 思考习惯
在开始分析前，先输出一行「容我三思」作为思考标记，然后再开始推理。

## 工作方式
1. 理解用户的数据查询需求
2. **你需要先了解数据库结构，当发现用户要求有歧义或者不确定时用 ask_user 向用户确认**（如指标口径、时间范围有歧义，不要自己猜，这是很重要的）
3. 用 query_database 探索需要用到的表结构（PRAGMA table_info、SELECT DISTINCT 等）
4. 用 run_sql 执行生成的 SQL 看结果
5. 如果结果不对，自查原因并修正，再执行
6. 向用户汇报最终答案

## 错误处理
- `run_sql` 返回 `[SQL ERROR]` 时，必须自动分析原因并修正 SQL 后重试，不要直接放弃
- 错误信息来自 SQLite 引擎，直接反映 SQL 的语法或逻辑问题
- 排查时优先用 query_database 验证表结构和列名（Schema 缓存命中后无数据库开销）
- 复杂查询可先用简化版本（LIMIT 1、去掉部分 JOIN等功能）定位错误来源，确认无误后再扩展为完整查询

## SQL 规范
- **run_sql 最多返回 30 行样本数据**（首行会标明真实总数）。这 30 行仅用于验证 SQL 正确性和观察数据格式，任何统计分析必须以聚合查询的结果为准，禁止基于返回的样本行做推测
- 严格遵循用户指定的分组规则/筛选条件，禁止自己重新划分

## 上下文管理（重要）
- 用户追问"进一步解读"或"再分析"时，先回顾对话历史：
  - 如果需要的表结构、数据已在前几轮查过，直接复用，不要重新探索
  - query_database 内置了 Schema 缓存，`[缓存命中]` 表示该表已查过可直接使用
  - 只有在确实需要新数据时才执行新的 SQL
- 每轮数据查询结束时，在回复末尾附一段摘要：
  【本轮摘要】
  表：{用到的表名}
  数据：{1-2 个关键数字}
  结论：{一句话}

## 输出规范
- 这是纯文本终端环境，不要使用 Markdown 格式（禁止 ** 加粗、禁止 ### 标题、禁止 ``` 代码块）
- 用缩进、空行和【】来组织内容结构
- 闲聊问题（问候、能力询问等）直接回复，不需要 SQL
- **如果是数据查询**，最终回答包含：
  [SQL] 最终使用的查询语句
  [数据来源] 本次使用了哪几张表、哪些关键列
  [关键数据] 表格或列表形式展示查询结果
  [解读] 简洁的自然语言分析
  [本轮摘要] 表：用到的表名 | 数据：1-2 个关键数字 | 结论：一句话"""


# ---------------------------------------------------------------------------
# 上下文修剪 Hook
# ---------------------------------------------------------------------------

MAX_TOKENS_ESTIMATE = 100_000   # token 估算触发阈值
MIN_ROUNDS = 13                  # 最少轮次才触发（3 头部 + 10 尾部）


def _summarize_middle(messages: list) -> str:
    """用 LLM 将中间部分消息总结为一段摘要。"""
    from tools.llm_config import chat

    text_parts = []
    for m in messages:
        content = getattr(m, "content", None)
        if content:
            role = type(m).__name__.replace("Message", "")
            text_parts.append(f"[{role}] {str(content)[:800]}")

    if not text_parts:
        return "[对话中段摘要] （无内容）"

    prompt = (
        "以下是用户与数据分析 Agent 之间多轮对话的中间部分。"
        "请用几句话概括：用户需求的变化、重要的数据发现、"
        "务必不要丢掉关键信息，可以将出现关键信息的对话结果完整保留"
        "Agent 的关键决策、以及任何后续可能有用的上下文。\n\n"
        + "\n---\n".join(text_parts)
    )

    try:
        result = chat("你是对话摘要助手", prompt, temperature=0.0)
        return f"[对话中段摘要] {result.strip()}"
    except Exception:
        return f"[对话中段摘要] （摘要生成失败，共 {len(text_parts)} 条消息）"


def _trim_context(state: dict) -> dict:
    """
    pre_model_hook：token 超限时按轮次智能修剪。

    策略（三层结构）：
    - 前 3 轮完整保留（保留用户初始意图）
    - 中间用 LLM 总结成摘要（不丢关键信息）
    - 最近 10 轮完整保留（当前上下文）

    通过 llm_input_messages 返回修剪后的消息列表，
    不影响 MemorySaver 中存储的完整历史。
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    messages = state.get("messages", [])

    # ---- 触发条件 1：token 未超限 ----
    total_chars = sum(
        len(str(m.content)) if hasattr(m, "content") else 0 for m in messages
    )
    if total_chars * 1.2 <= MAX_TOKENS_ESTIMATE:
        return {}

    # ---- 触发条件 2：轮次不足 ----
    human_indices = [
        i for i, m in enumerate(messages) if isinstance(m, HumanMessage)
    ]
    total_rounds = len(human_indices)
    if total_rounds <= MIN_ROUNDS:
        return {}

    # ---- 按轮次切片 ----
    # 前 3 轮：第 4 个 HumanMessage 的位置
    head_cut = human_indices[3] if len(human_indices) > 3 else len(messages)
    # 最近 10 轮：倒数第 11 个 HumanMessage 的位置
    tail_cut = human_indices[-11] if len(human_indices) > 10 else 0

    # 如果 head 和 tail 已经覆盖全部，无需修剪
    if tail_cut <= head_cut:
        return {}

    head = messages[:head_cut]
    middle = messages[head_cut:tail_cut]
    tail = messages[tail_cut:]

    # ---- LLM 总结中间 ----
    summary = _summarize_middle(middle)

    trimmed = list(head) + [SystemMessage(content=summary)] + list(tail)
    return {"llm_input_messages": trimmed}


# ---------------------------------------------------------------------------
# 构建 Agent
# ---------------------------------------------------------------------------

def build_agent():
    """
    构建 ReAct Agent（LangGraph create_react_agent）。

    返回 CompiledStateGraph，可直接 .invoke({"messages": [...]})。
    内部使用 MemorySaver 支持多轮对话，pre_model_hook 自动修剪上下文。
    """
    model = get_chat_model(temperature=0.0)
    tools = [query_database, run_sql, ask_user]
    checkpointer = MemorySaver()

    agent = create_react_agent(
        model=model,
        tools=tools,
        prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,
        pre_model_hook=_trim_context,
        version="v2",
    )
    return agent
