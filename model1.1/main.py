"""
huatai-agent 1.1 — 多 Agent 协作版

一个主 ReAct Agent，三个子 Agent 作为工具：
- explore_agent: 探索数据库结构（ReAct，3个工具）
- sql_agent:     生成并执行 SQL（ReAct，2个工具）
- analysis_agent: 解读查询结果（纯 LLM）

没有 Supervisor，没有计数器，没有关键词匹配。LLM 自己判断一切。
"""

import re
import sys
import os
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from langgraph.prebuilt import create_react_agent  # pyright: ignore[reportDeprecated]
from langgraph.checkpoint.memory import MemorySaver

from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style
from prompt_toolkit.formatted_text import HTML

from tools.llm_config import get_chat_model
from tools.db_connector import init_all_databases
from agents.explore.agent import build_agent as build_explore_agent
from agents.sql.agent import build_agent as build_sql_agent
from agents.analysis.agent import run as run_analysis


# ---------------------------------------------------------------------------
# 预构建子 Agent（模块级单例，避免重复创建）
# ---------------------------------------------------------------------------

_explore_agent = None
_sql_agent = None


def _get_explore_agent():
    global _explore_agent
    if _explore_agent is None:
        _explore_agent = build_explore_agent()
    return _explore_agent


def _get_sql_agent():
    global _sql_agent
    if _sql_agent is None:
        _sql_agent = build_sql_agent()
    return _sql_agent


# ---------------------------------------------------------------------------
# 子 Agent 工具（主 Agent 的工具函数）
# ---------------------------------------------------------------------------

def explore_agent(task: str) -> str:
    """
    探索数据库结构。传入任务描述（要查哪些表、什么字段），返回 Schema 报告。
    用于：了解有哪些表可用、列名是什么、枚举值有哪些、码值映射。
    不用于：写 SQL、执行查询、解读结果。
    """
    agent = _get_explore_agent()
    result = agent.invoke({"messages": [("user", task)]})
    msgs = result.get("messages", [])
    return str(msgs[-1].content) if msgs else "探索未返回结果"


def sql_agent(task: str) -> str:
    """
    生成并执行 SQL 查询。传入任务描述（包含 Schema 信息、查询要求），
    返回 SQL 语句和查询结果。会自动处理报错并修正。
    不用于：探索表结构、解读结果。
    """
    agent = _get_sql_agent()
    result = agent.invoke({"messages": [("user", task)]})
    msgs = result.get("messages", [])
    return str(msgs[-1].content) if msgs else "SQL 执行未返回结果"


def analysis_agent(task: str) -> str:
    """
    解读 SQL 查询结果。传入解读要求 + 查询结果数据，返回自然语言解读。
    不用于：探索表结构、生成 SQL。
    """
    # analysis_agent 不是 ReAct，是纯 LLM 函数
    # task 里自带上下文
    return run_analysis(task, "")


def ask_user(question: str) -> str:
    """
    向用户提问以澄清模糊需求。仅在确实无法从数据中判断时使用。
    例如：不确定用户说的是总资产还是净资产、不确定时间范围。
    """
    print(f"\n  ? {question}")
    reply = input("  > ").strip()
    return reply if reply else "跳过"


# ---------------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """你是「智谋洞见 Agent」，由智谋洞见团队打造。你操作一个包含客户、产品、交易、持仓、资产等数据的 SQLite 数据库。

## 你可以调用四个工具

- **explore_agent** — 传入探索任务，它会查表结构、枚举值、码值映射，返回 Schema 报告
- **sql_agent** — 传入 SQL 生成任务（含 Schema 信息），它会写出 SQL、执行、报错自修正
- **analysis_agent** — 传入解读任务（含 SQL 结果 + 深度指令），它会把数据转成自然语言。
  必须在 task 中明确深度：普通查询写「简要总结，1-2句话」，只有用户明确要求分析/解读/洞察时才写「深入分析」
- **ask_user** — 向用户提问澄清

## 工作方式

你自己判断要不要用工具、用哪个、用几次：

1. **闲聊/问候**（"你是谁""你好""能做什么"）→ 直接回复，不调工具
2. **简单数据查询**（"客户多少人""钻石卡客户数"）→ explore_agent → sql_agent → 直接汇报，不调 analysis_agent
3. **复杂分析**（"深入分析""洞察""对比差异"）→ explore_agent → sql_agent → analysis_agent
4. **用户追问**（"那上个月呢？"）→ 从对话历史推断上下文，修改条件后调 sql_agent

## 效率规则

- explore_agent 自带缓存，同张表查过一次就不会重复查。放心调用。
- **简单查询不要调 analysis_agent**，拿到 SQL 结果后自己用 1-2 句话汇报即可。只有用户明确要求"分析""解读""洞察""对比"等要求深入分析时才调用 analysis_agent。
- 调用 analysis_agent 时必须在 task 首句写明深度：
  - 普通查询 → "简要总结。1-2句话给核心数字。"
  - 深度分析 → "深入分析。多维度展开，给出洞察建议。"
- 每次回复尽量精简，节省 token。

## 数据库概况
8 张表: ads_cust_info_d(客户信息), dws_cust_aset_d(资产), dwd_cust_tran_d(交易),
dwd_cust_hold_d(持仓), dws_cust_fin_d(资金), dim_product(产品), dim_branch(营业部), dim_public(码值)

## SQL 规范
日期 YYYYMMDD, Q1=20260101~20260331, COALESCE 处理 NULL, LEFT JOIN, 复杂查询用 CTE

## 输出规范
- 纯文本终端，不用 Markdown（禁止 ** / ### / ``` / |表格|）
- 数据查询回复中必须包含：
  1. 最终 SQL（完整展示，不省略）
  2. 数据来源：本次使用了哪几张表、哪些关键列
  3. 关键数字和结论
- 闲聊直接回复，不需要 SQL

"""


# ---------------------------------------------------------------------------
# 主 Agent 构建
# ---------------------------------------------------------------------------

def build_agent():
    """构建主 ReAct Agent，子 Agent 作为工具。"""
    model = get_chat_model(temperature=0.0)
    tools = [explore_agent, sql_agent, analysis_agent, ask_user]
    checkpointer = MemorySaver()

    # pyright: ignore[reportDeprecated]
    return create_react_agent(
        model=model,
        tools=tools,
        prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,
        version="v2",
    )


# ---------------------------------------------------------------------------
# 查询执行
# ---------------------------------------------------------------------------

# 所有可能的用户表名（用于从消息中提取探索过的表）
ALL_TABLES = {
    "ads_cust_info_d", "dws_cust_aset_d", "dwd_cust_tran_d",
    "dwd_cust_hold_d", "dws_cust_fin_d", "dim_product",
    "dim_branch", "dim_public",
}


def _extract_tables(text: str) -> set:
    """从探索结果文本中提取被探索的表名（精确匹配）。"""
    found = set()
    for t in ALL_TABLES:
        # 只在有 Schema 上下文的行中匹配
        if re.search(rf'\b{t}\b', text, re.IGNORECASE):
            found.add(t)
    return found


def _clean_sql(sql: str) -> str:
    """清理 SQL 中的 markdown 标记。"""
    sql = sql.strip()
    sql = re.sub(r'^```(?:sql)?\s*\n?', '', sql)
    sql = re.sub(r'\n?```\s*$', '', sql)
    return sql.strip()


def _extract_sql_from_text(text: str) -> str:
    """从 Agent 输出中提取 SQL 语句。"""
    # 先清理 markdown 代码块
    # 匹配 ```sql ... ``` 块
    m = re.search(r'```(?:sql)?\s*\n([\s\S]*?)\n```', text, re.IGNORECASE)
    if m:
        return _clean_sql(m.group(1))
    # 匹配 SQL: 后面的缩进块
    m = re.search(r'SQL:\s*\n(\s{2,}.*?)(?:\n\n|\n\Z)', text, re.DOTALL | re.IGNORECASE)
    if m:
        return _clean_sql(m.group(1))
    # 匹配 SELECT 开头的独立块（排除自然语言中的 SELECT 字）
    m = re.search(r'(?:^|\n\n)(SELECT\s[\s\S]*?)(?:\n\n|\n\Z)', text, re.IGNORECASE)
    if m:
        sql = m.group(1).strip()
        if len(sql) > 30:
            return _clean_sql(sql)
    return ""


def run_query(graph, user_input: str, thread_id: str) -> str:
    """
    执行一次查询。流式显示推理过程，跟 model/ 一样的输出风格。
    """
    config = {"configurable": {"thread_id": thread_id}}

    final_output = ""
    explored_tables = set()
    final_sql = ""
    first_thought = True

    for event in graph.stream(
        {"messages": [("user", user_input)]},
        config=config,
        stream_mode="updates",
    ):
        for node_name, update in event.items():
            if update is None:
                continue

            new_msgs = update.get("messages", [])
            if not isinstance(new_msgs, list):
                new_msgs = [new_msgs] if new_msgs else []

            for msg in new_msgs:
                if not msg:
                    continue

                content = getattr(msg, "content", "")
                tool_calls = getattr(msg, "tool_calls", None) or []

                # LLM 思考 + 工具选择
                if tool_calls:
                    # 先展示思考（首次有工具调用就显示三思框，不依赖 content）
                    if first_thought:
                        _print_think_box()
                        first_thought = False
                    if content:
                        _print_thought(content)
                    # 再展示工具调用
                    for tc in tool_calls:
                        tn = tc.get("name", "") if isinstance(tc, dict) else getattr(tc, "name", "")
                        args = tc.get("args", {}) if isinstance(tc, dict) else getattr(tc, "args", {})

                        if tn in ("explore_agent", "sql_agent", "analysis_agent"):
                            task = args.get("task", "")
                            _print_tool_call(tn, task)

                # 工具返回结果
                elif getattr(msg, "tool_call_id", None) or getattr(msg, "type", None) == "tool":
                    tool_content = str(content or "")
                    # 从 explore_agent 结果中提取表名
                    if tool_content:
                        tables = _extract_tables(tool_content)
                        explored_tables |= tables
                    # 从 sql_agent 结果中提取 SQL
                    if tool_content:
                        sql = _extract_sql_from_text(tool_content)
                        if sql:
                            final_sql = sql

                # LLM 最终文本回复
                elif content and node_name == "agent":
                    final_output = content

    # ---- 消化后统一输出 ----
    if explored_tables:
        tables_str = ", ".join(sorted(explored_tables))
        print(f"  [已了解] {tables_str}")
    if final_sql:
        _print_sql_box(final_sql)

    return final_output or "（未能获取回复）"


# ---------------------------------------------------------------------------
# 显示函数
# ---------------------------------------------------------------------------

def _print_think_box():
    print(f"\n  ╔══════════════╗")
    print(f"  ║  容我三思   ║")
    print(f"  ╚══════════════╝")


def _print_thought(content: str):
    text = str(content).strip()
    if len(text) > 150:
        text = text[:150] + "..."
    if text:
        print(f"  [思考] {text}")


def _print_tool_call(name: str, task: str):
    labels = {
        "explore_agent": "调用 探索Agent",
        "sql_agent": "调用 SQL Agent",
        "analysis_agent": "调用 分析Agent",
    }
    label = labels.get(name, name)
    brief = task[:120] + "..." if len(task) > 120 else task
    print(f"  [{label}] {brief}")


def _print_sql_box(sql: str):
    print(f"\n{'─' * 50}")
    print(f"  最终 SQL:")
    print(f"{'─' * 50}")
    for line in sql.split("\n"):
        print(f"  {line}")
    print(f"{'─' * 50}")


# ---------------------------------------------------------------------------
# 终端输入模块
# ---------------------------------------------------------------------------

def _create_input_session() -> PromptSession | None:
    """创建类 Claude Code 的输入会话。非 TTY 环境返回 None。"""
    try:
        if not os.isatty(0):
            return None

        bindings = KeyBindings()

        @bindings.add('c-enter')
        def _send(event):
            """Ctrl+Enter 发送"""
            buf = event.current_buffer
            if buf.text.strip():
                buf.validate_and_handle()

        @bindings.add('enter')
        @bindings.add('s-enter')
        @bindings.add('c-j')
        def _newline(event):
            """Enter / Shift+Enter / Ctrl+J 换行"""
            event.current_buffer.insert_text('\n')

        @bindings.add('escape')
        def _clear(event):
            event.current_buffer.text = ''

        history_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            '.huatai_history'
        )

        style = Style.from_dict({
            'prompt':     'ansigreen bold',
            'completion-menu.completion': 'bg:#333333 #ffffff',
            'completion-menu.completion.current': 'bg:#0066cc #ffffff',
        })

        return PromptSession(
            multiline=True,
            key_bindings=bindings,
            history=FileHistory(history_path),
            style=style,
        )

    except Exception as e:
        print(f"  [WARN] 无法创建高级输入模式: {e}")
        return None


def _fallback_input() -> str | None:
    """简化输入：Enter 换行，空行提交。EOF 时返回 None。"""
    lines = []
    print()
    while True:
        prompt = ">>> " if not lines else "... "
        try:
            line = input(prompt)
        except (EOFError, KeyboardInterrupt):
            return None
        if line.strip():
            lines.append(line)
        elif lines:
            break  # 空行提交
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("[INIT] 正在扫描 data/ 目录并初始化数据库...")
    init_all_databases()

    print("[INIT] 正在构建 Agent...")
    _agent = build_agent()
    print("[INIT] 就绪")

    print()
    print("=" * 60)
    print("  huatai-agent 1.1  多 Agent 协作取数系统")
    print("  1 个主 Agent + 3 个子 Agent (探索/SQL/分析)")
    print("=" * 60)

    # ---- 创建输入会话 ----
    session = _create_input_session()

    if session is not None:
        print("  Ctrl+Enter 发送 | Enter/Shift+Enter 换行 | ↑↓ 历史 | Esc 清空")
    else:
        print("  [简化模式] Enter 换行 | 空行提交")

    print("  quit 退出")
    print("=" * 60)

    thread_id = str(uuid.uuid4())

    while True:
        try:
            if session is not None:
                user_input = session.prompt(
                    HTML('\n<ansigreen><b>&gt; </b></ansigreen>')
                )
            else:
                user_input = _fallback_input()
                if user_input is None:
                    print("\n再见")
                    break

        except (EOFError, KeyboardInterrupt):
            print("\n再见")
            break

        user_input = user_input.strip()

        if user_input.lower() in ("quit", "exit", "q"):
            print("再见")
            break

        if not user_input:
            continue

        print()
        response = run_query(_agent, user_input, thread_id=thread_id)
        print(response)
        print("-" * 60)
