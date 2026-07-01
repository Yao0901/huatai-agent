"""
huatai-agent — 华泰证券智能问数原型系统

ReAct 架构：LLM 自主探索数据库结构 → 生成 SQL → 执行验证 → 修正 → 输出答案。
基于 LangGraph create_react_agent，替代原来的 3 张子图固定流水线。
"""

import uuid

from tools.db_connector import _init_database
from agents.react_agent import build_agent


# ---------------------------------------------------------------------------
# 启动初始化
# ---------------------------------------------------------------------------

print("[INIT] 正在初始化数据库...")
_init_database()
print("[INIT] 数据库就绪")

print("[INIT] 正在构建 ReAct Agent...")
_agent = build_agent()
print("[INIT] Agent 就绪")


# ---------------------------------------------------------------------------
# 查询执行
# ---------------------------------------------------------------------------

def run_query(user_input: str, thread_id: str = None):
    """
    执行一次查询，流式输出推理过程，返回 Agent 的最终回复文本。

    Args:
        user_input: 用户自然语言输入
        thread_id:  会话 ID（用于多轮对话记忆）

    Returns:
        str: Agent 的最终回复
    """
    if thread_id is None:
        thread_id = str(uuid.uuid4())

    config = {"configurable": {"thread_id": thread_id}}

    # 流式执行，收集探索信息和最终结果
    final_content = ""
    final_sql = ""
    explored_tables = set()  # 探索过的表
    first_thought = True     # 首次思考标记

    for event in _agent.stream(
        {"messages": [("user", user_input)]},
        config=config,
        stream_mode="updates",
    ):
        for node_name, update in event.items():
            if update is None:
                continue
            msg = update.get("messages", [None])[0]

            if node_name == "agent":
                if hasattr(msg, "content") and msg.content:
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        # 收集探索信息
                        for tc in msg.tool_calls:
                            args = tc.get("args", {})
                            sql = args.get("sql", "")
                            if sql:
                                _track_exploration(sql, explored_tables)
                            # 捕获最后一条 run_sql（只要非探索性查询）
                            if tc.get("name") == "run_sql" and sql and len(sql) > 200:
                                final_sql = sql
                        thought = _clean_thought(msg.content)
                        if thought:
                            if first_thought:
                                print(f"\n  ╔══════════════╗")
                                print(f"  ║  容我三思   ║")
                                print(f"  ╚══════════════╝")
                                first_thought = False
                            print(f"  💭 {thought}")
                    else:
                        final_content = msg.content

    # 显示探索摘要
    if explored_tables:
        tables_str = ", ".join(sorted(explored_tables))
        print(f"  📋 已了解: {tables_str}")
    if final_sql:
        print(f"\n{'─' * 50}")
        print(f"  📝 最终 SQL:")
        print(f"{'─' * 50}")
        for line in final_sql.strip().split("\n"):
            print(f"  {line}")
        print(f"{'─' * 50}")

    return final_content or "（Agent 未产生回复）"


def _format_tokens(usage: dict) -> str:
    """格式化 token 消耗为简洁显示。"""
    t = usage.get("total", 0)
    if t < 1000:
        return f"[{t}T]"
    return f"[{t/1000:.1f}K]"


def _track_exploration(sql: str, tables: set, _cols: dict = None):
    """从 SQL 中提取被探索的表名。"""
    import re
    # PRAGMA table_info('表名')
    for m in re.finditer(r"table_info\s*\(\s*'(\w+)'\s*\)", sql, re.IGNORECASE):
        tables.add(m.group(1))
    # FROM 表名
    for m in re.finditer(r'\bFROM\s+(\w+)', sql.upper()):
        t = m.group(1).lower()
        # 过滤掉 CTE 名称（通常大写或全大写）
        if t in ("dws_cust_aset_d", "dwd_cust_tran_d", "dwd_cust_hold_d",
                 "dim_product", "dim_branch", "dim_public", "ads_cust_info_d",
                 "dws_cust_fin_d"):
            tables.add(t)


# ---------------------------------------------------------------------------
# 流式显示辅助
# ---------------------------------------------------------------------------

def _clean_thought(text: str) -> str:
    """提取 Agent 推理文本的关键句。"""
    # 去掉过长的内容，保留前 150 字符
    t = text.strip()
    if len(t) > 150:
        t = t[:150] + "..."
    return t


def _brief_args(args: dict) -> str:
    """从 SQL 参数中提取表名，生成简洁描述。"""
    sql = args.get("sql", "")
    if not sql:
        return ""
    sql_upper = sql.upper()
    # PRAGMA table_info('表名')
    if "PRAGMA" in sql_upper:
        import re
        m = re.search(r"table_info\s*\(\s*'(\w+)'\s*\)", sql, re.IGNORECASE)
        if m:
            return f"表={m.group(1)}"
        return "表结构"
    # FROM 子句
    import re
    tables = re.findall(r'\bFROM\s+(\w+)', sql_upper)
    if tables:
        tables = list(set(tables))
        if len(tables) <= 3:
            return f"表={', '.join(tables)}"
        return f"表={', '.join(tables[:3])}..."
    # DISTINCT / MIN MAX
    if "DISTINCT" in sql_upper:
        return "枚举值"
    if "MIN" in sql_upper and "MAX" in sql_upper:
        return "日期范围"
    return sql[:40]


def _brief_result(tool_name: str, content: str) -> str:
    """简洁显示工具执行结果（枚举值只显示数量，不列内容）。"""
    c = content[:300]
    if tool_name == "query_database":
        if "cid" in c and "name" in c and "type" in c:
            # PRAGMA table_info 输出
            col_count = c.count("\n")
            return f"{col_count} 列"
        if "DISTINCT" in c.upper() or "up_prdt" in c or "prdt_type" in c.lower():
            # 枚举值查询 — 只显示数量，不列值
            count = len([l for l in c.split("\n") if l.strip() and "|" in l])
            return f"共 {count} 种"
        if "MIN" in c and "MAX" in c:
            vals = [x.strip() for x in c.replace("|", "~").split("\n") if "~" in x and "202" in x]
            return vals[0][:50] if vals else "日期范围获取完成"
        if "LIMIT" in c.upper() or ("SELECT" in c.upper() and "*" in c):
            return "样本获取完成"
        return "查询完成"
    elif tool_name == "search_code_mapping":
        if "NOT FOUND" in c:
            return "未找到"
        count = c.count("code=")
        return f"{count} 条匹配"
    elif tool_name == "run_sql":
        lines = c.strip().split("\n")
        for line in lines:
            if "返回" in line and "行" in line:
                return line.split(":")[0].strip()
        if "ERROR" in c:
            return f"执行失败"
        return "执行完成"
    return "完成"


# 旧版函数保留（不再使用）
def _summarize_query(content: str) -> str:
    """将 query_database 的原始输出转为人类可读摘要。"""
    c = content[:200]
    if "PRAGMA" in c or ("cid" in c and "name" in c and "type" in c):
        # PRAGMA 输出，提取表名
        return f"🔍 查看表列结构"
    if "DISTINCT" in c.upper() or "up_prdt" in c or "prdt_type" in c:
        vals = [x.strip() for x in c.split("|") if x.strip()][:8]
        return f"🔍 查枚举值 → {', '.join(vals[:5])}"
    if "MIN" in c and "MAX" in c:
        return f"🔍 查日期范围"
    if "LIMIT" in c.upper() and ("SELECT" in c.upper()):
        return f"🔍 查样本数据"
    return f"🔍 查询数据库"


def _summarize_code(content: str) -> str:
    """将 search_code_mapping 输出转为摘要。"""
    if "NOT FOUND" in content:
        return f"🔍 码值查询 → 未找到"
    lines = content.strip().split("\n")
    count = len([l for l in lines if l.startswith("  code=")])
    sample = ""
    for l in lines:
        if "describe=" in l:
            sample = l.split("describe=")[-1].split(",")[0].strip()
            break
    return f"🔍 码值查询 → 找到 {count} 条 ({sample})"


def _summarize_result(content: str) -> str:
    """将 run_sql 输出转为摘要。"""
    lines = content.strip().split("\n")
    first_line = lines[0] if lines else ""
    if "返回" in first_line:
        count_part = first_line.split(":")[0] if ":" in first_line else first_line
        # 提取列名
        if len(lines) > 1:
            cols = lines[1].count("|")
            return f"📊 {count_part}, {cols+1} 列"
        return f"📊 {count_part}"
    if "ERROR" in content:
        return f"❌ {content[:80]}"
    if "EMPTY" in content:
        return f"📊 查询无结果"
    return f"📊 执行完成"


# ---------------------------------------------------------------------------
# 命令行入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print()
    print("=" * 60)
    print("  huatai-agent  取数 Agent 原型系统")
    print("  输入自然语言取数问题，输入 quit 退出")
    print("  示例问题:")
    print("    学历本科及以上的男性客户，年龄大于50岁的有多少个")
    print("    不同客户年龄段资产分布情况")
    print("    查询钻石卡男性客户的总资产")
    print("=" * 60)

    # 保持会话 ID，支持多轮对话
    thread_id = str(uuid.uuid4())

    while True:
        try:
            first_line = input("\n>>> ")
        except (EOFError, KeyboardInterrupt):
            print("\n再见")
            break

        # 支持多行输入：如果首行以引号开头但没闭合，继续读取直到闭合
        lines = [first_line]
        if first_line.strip().startswith('"') and first_line.count('"') < 2:
            while True:
                try:
                    more = input("... ")  # 续行提示
                except (EOFError, KeyboardInterrupt):
                    break
                lines.append(more)
                if more.strip().endswith('"'):
                    break
        user_input = "\n".join(lines).strip().strip('"').strip()

        if user_input.lower() in ("quit", "exit", "q"):
            print("再见")
            break

        if not user_input:
            continue

        print()  # 空行分隔
        response = run_query(user_input, thread_id=thread_id)
        print(response)
        print("-" * 60)
