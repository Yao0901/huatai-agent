"""
huatai-agent 唯一入口

初始化数据库 → 编译主图 → 处理用户查询 → 输出结果。

支持 Human-in-the-loop：当 Agent 遇到歧义需要追问时，
图会暂停并返回问题，main.py 收集用户回复后继续执行。

使用方式：
    python main.py                                    # 交互模式
    python main.py --query "30岁以下女性客户的持仓市值"   # 单次查询
"""

import sys
import io
from typing import Dict, Any

# 解决 Windows GBK 终端编码问题
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from langgraph.errors import GraphInterrupt
from langgraph.types import Command

from agents.main_graph import build_main_graph
from agents.state import AgentState
from tools.db_connector import _init_database


# ---------------------------------------------------------------------------
# 启动时初始化数据库
# ---------------------------------------------------------------------------

print("[INIT] 正在初始化数据库...")
_init_database()
print("[INIT] 数据库就绪")

# 编译主图（全局单例）
_graph = build_main_graph()
print("[INIT] Agent 图编译完成")


# ---------------------------------------------------------------------------
# 执行查询（支持 Human-in-the-loop 中断与恢复）
# ---------------------------------------------------------------------------

def run_query(
    user_input: str,
    thread_id: str = "default",
    conversation_history: list = None,
) -> AgentState:
    """
    执行一次完整的取数查询流程，自动处理 Human-in-the-loop 中断。

    流水线：
    user_input → Agent1(意图识别) → Agent2(查询生成) → Agent3(安全评估) → 结果

    当 Agent1 遇到无法自动消歧的指标时，图会暂停并返回追问。
    main.py 在此处收集用户回复，然后自动恢复执行。

    Args:
        user_input:          用户的自然语言查询
        thread_id:           会话 ID（用于区分不同用户/会话的中断状态）
        conversation_history: 跨轮对话历史 [{"role":"user"/"assistant", "content":...}, ...]

    Returns:
        AgentState: 包含所有中间结果和最终输出的完整状态
    """
    initial_state: AgentState = {
        "user_input": user_input,
        "messages": list(conversation_history) if conversation_history else [],
        "retry_count": 0,
        "ambiguity_flag": False,
        "metric_found": False,
        "resolved_metrics": [],
        "unresolved_terms": [],
        "resolved_intent": {},
        "candidate_tables": [],
        "table_schemas": [],
        "fit_check": False,
        "fit_check_detail": "",
        "generated_sql": "",
        "security_level": "safe",
        "user_confirmed_dangerous": False,
        "security_detail": {},
        "permission_check": "passed",
        "injected_sql": "",
        "execution_result": {},
        "error_message": None,
        "error_type": None,
        "reflection_feedback": None,
        "final_output": None,
    }

    config = {"configurable": {"thread_id": thread_id}}

    # 循环处理可能的多次中断（多次追问）
    while True:
        try:
            # 尝试执行图（首次用 initial_state，恢复时用 Command）
            if "_resume_count" not in dir():
                result = _graph.invoke(initial_state, config)
            else:
                result = _graph.invoke(Command(resume=_pending_reply), config)

            # 正常结束，返回结果
            return result

        except GraphInterrupt as e:
            # 图被 interrupt() 暂停了，e.args[0] 是追问内容
            question = e.args[0] if e.args else "请补充说明"
            print(f"\n[追问] {question}")
            user_reply = input("[回复] >>> ").strip()
            if not user_reply:
                user_reply = "跳过"
            _pending_reply = user_reply
            _resume_count = True  # 标记：下次走 resume 路径


def _build_turn_summary(state: AgentState) -> str:
    """
    将本轮 Agent 的执行结果压缩为一条简短摘要，供下一轮对话时带入 LLM 上下文。

    只保留关键信息：生成的 SQL、查询是否成功、返回多少行。
    """
    parts = []
    sql = state.get("generated_sql", "")
    if sql:
        parts.append(f"SQL: {sql[:300]}{'...' if len(sql) > 300 else ''}")
    exec_result = state.get("execution_result", {})
    if exec_result.get("success"):
        parts.append(f"结果: 成功，返回 {exec_result.get('row_count', 0)} 行")
    elif state.get("error_message"):
        parts.append(f"结果: 失败，{state.get('error_message', '')[:100]}")
    resolved = state.get("resolved_metrics", [])
    if resolved:
        names = [m.get("definition", {}).get("metric_name", m.get("keyword", "")) for m in resolved]
        parts.append(f"消歧指标: {', '.join(names)}")
    return " | ".join(parts) if parts else ""


def print_result(state: AgentState):
    """格式化打印查询结果。"""
    print()
    print("=" * 60)
    print("  查询结果")
    print("=" * 60)

    # SQL 展示
    sql = state.get("generated_sql", "")
    if sql:
        print(f"\n[SQL]\n{sql}")

    # 执行结果
    result = state.get("execution_result", {})
    if result.get("success"):
        row_count = result.get("row_count", 0)
        print(f"\n[RESULT] 成功，返回 {row_count} 行")

        # 展示前 10 行
        data = result.get("data", [])
        if data:
            print("-" * 40)
            cols = list(data[0].keys())
            print(" | ".join(cols))
            print("-" * 40)
            for row in data[:10]:
                print(" | ".join(str(row.get(c, "")) for c in cols))
            if row_count > 10:
                print(f"... 还有 {row_count - 10} 行")
    else:
        error = state.get("error_message") or result.get("error", "未知错误")
        print(f"\n[RESULT] 执行失败: {error}")
        if state.get("retry_count", 0) > 0:
            print(f"[RETRY] 已重试 {state.get('retry_count')} 次")

    # 选表说明（用户放弃追问时，展示大模型怎么选的）
    fit_detail = state.get("fit_check_detail", "")
    if fit_detail:
        print(f"\n[TABLE SELECTION] {fit_detail}")

    # 路径追踪
    print(f"\n[PATH] ambiguity={state.get('ambiguity_flag')}, "
          f"security={state.get('security_level')}, "
          f"tables={state.get('candidate_tables', [])}")

    print("=" * 60)


# ---------------------------------------------------------------------------
# 命令行入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if "--query" in sys.argv:
        # 单次查询模式
        idx = sys.argv.index("--query")
        query = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else ""
        if query:
            print(f"\n[QUERY] {query}")
            result = run_query(query)
            print_result(result)
        else:
            print("用法: python main.py --query \"你的问题\"")
    else:
        # 交互模式
        print()
        print("=" * 60)
        print("  huatai-agent  取数 Agent 原型系统")
        print("  输入自然语言取数问题，输入 quit 退出")
        print("  示例问题:")
        print("    学历本科及以上的男性客户，年龄大于50岁的有多少个")
        print("    不同客户年龄段资产分布情况")
        print("    查询钻石卡男性客户的总资产")
        print("=" * 60)

        # 跨轮对话历史：[{"role": "user"/"assistant", "content": "..."}, ...]
        conversation_history = []
        MAX_HISTORY = 10  # 保留最近 5 轮问答（10 条消息）

        while True:
            try:
                user_input = input("\n>>> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n再见")
                break

            if user_input.lower() in ("quit", "exit", "q"):
                print("再见")
                break

            if not user_input:
                continue

            result = run_query(user_input, conversation_history=conversation_history)
            print_result(result)

            # ---- 更新跨轮历史 ----
            conversation_history.append({"role": "user", "content": user_input})
            # 构建助手回复摘要（SQL + 结果）
            assistant_summary = _build_turn_summary(result)
            if assistant_summary:
                conversation_history.append({"role": "assistant", "content": assistant_summary})
            # 裁剪历史，防止超出 LLM 上下文窗口
            if len(conversation_history) > MAX_HISTORY:
                conversation_history = conversation_history[-MAX_HISTORY:]
