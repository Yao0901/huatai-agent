"""
子图 2：查询生成 Agent (Agent2)

负责：
1. 根据结构化意图检索相关表结构（节点 4）
2. LLM 校验表结构与用户意图的贴合度（fit_check_node）
3. 贴合 → 生成 SQL（节点 6）；不贴合 → Human-in-the-loop 追问
4. 用户放弃追问 → 按 LLM 理解继续，并说明选表理由

图结构：
    retrieve_schemas → fit_check_node
        ├── 贴合 / 用户放弃 → generate_sql → END
        └── 不贴合 → ask_user → 回到 retrieve_schemas

特殊处理：
- 当来自子图3的重试时（retry_count > 0），将 reflection_feedback 注入上下文
"""

from langgraph.graph import StateGraph, END
from langgraph.types import interrupt

from agents.state import AgentState

# 导入本子图涉及的节点函数
from nodes.schema_node import retrieve_table_schemas
from nodes.sql_gen_node import generate_sql
from tools.llm_config import chat


# ---------------------------------------------------------------------------
# 节点：LLM 贴合度校验
# ---------------------------------------------------------------------------

def fit_check_node(state: AgentState) -> AgentState:
    """
    用 LLM 评估检索到的表结构是否能回答用户问题。

    三件事一起做：
    1. 判断表结构是否贴合用户意图
    2. 检测用户是否已放弃追问（"随便""你定""算了""就这样"等）
    3. 如果放弃，生成一段说明——告诉用户大模型是怎么理解/选表的

    调用时机：retrieve_schemas 之后。

    Args:
        state: 全局状态

    Returns:
        AgentState: 更新 fit_check, fit_check_detail
    """
    user_input = state.get("user_input", "")
    candidate_tables = state.get("candidate_tables", [])
    table_schemas = state.get("table_schemas", [])
    resolved_metrics = state.get("resolved_metrics", [])

    # ---- 拼表结构摘要 ----
    schema_summary = ""
    for t in table_schemas:
        tname = t.get("table_name", "")
        tdesc = t.get("description", "")
        cols = [c["name"] for c in t.get("columns", [])[:8]]
        schema_summary += f"- {tname}: {tdesc}\n  列: {', '.join(cols)}\n"

    # ---- 拼指标信息 ----
    metric_text = ""
    for m in resolved_metrics:
        d = m.get("definition", {})
        metric_text += f"- {m.get('keyword','')} → {d.get('metric_name','')}: {d.get('description','')}\n"

    system_prompt = (
        "你是金融数据查询专家。评估当前选中的数据库表是否足以回答用户的问题。\n\n"
        "判断标准：\n"
        "1. 用户需要的指标对应的列是否在这些表中\n"
        "2. 用户筛选条件（年龄、性别、产品类型等）需要的列是否在\n"
        "3. JOIN 链路是否完整（事实表+维表+码值翻译表）\n\n"
        "特殊规则：\n"
        "- 如果用户表示不想继续、让你随便选、多次追问后不耐烦、或明确说'算了''就这样''你定'，"
        "  则即使表结构不够完美也应判定为 fit=true，并生成一段说明解释你是怎么理解和选表的\n"
        "- 如果表结构足够，fit=true\n\n"
        "输出一个 JSON 对象：\n"
        "  - fit: true/false（表结构是否贴合或用户已放弃）\n"
        "  - detail: 如果 fit=false，说明缺什么；如果用户放弃，说明你如何理解用户意图、为什么选这些表\n"
        "只输出 JSON，不要其他文字。"
    )

    user_prompt = (
        f"用户问题（含追问历史）：{user_input}\n\n"
        f"已选表：\n{schema_summary}\n"
        f"消歧指标：\n{metric_text if metric_text else '（无）'}"
    )

    try:
        import json
        response = chat(system_prompt, user_prompt, temperature=0.0)
        response = response.strip()
        if response.startswith("```"):
            response = response.split("\n", 1)[1].rsplit("\n```", 1)[0]
        result = json.loads(response)
        return {
            "fit_check": result.get("fit", True),
            "fit_check_detail": result.get("detail", ""),
        }
    except Exception:
        # LLM 挂了 → 默认放行，避免卡住流程
        return {
            "fit_check": True,
            "fit_check_detail": "",
        }


# ---------------------------------------------------------------------------
# 节点：Human-in-the-loop 追问
# ---------------------------------------------------------------------------

def ask_user_for_tables_node(state: AgentState) -> AgentState:
    """
    表结构不贴合时暂停流程，向用户追问更多信息。

    追问内容包含：
    - 当前选中的表及缺陷说明
    - 提示用户补充信息或放弃追问

    Args:
        state: 全局状态

    Returns:
        AgentState: 合并了用户补充信息后的状态
    """
    detail = state.get("fit_check_detail", "当前选中的表可能无法完全回答您的问题")
    tables = state.get("candidate_tables", [])

    question = (
        f"当前选中的表：{', '.join(tables) if tables else '（无）'}\n"
        f"问题：{detail}\n\n"
        f"请补充更多信息（如时间范围、具体指标名称），或回复「随便」「你定」「就这样」按默认理解继续。"
    )

    user_reply = interrupt(question)

    # 把用户回复追加到 user_input，下次 LLM 检索 + 选表时会看到完整上下文
    return {
        "user_input": state.get("user_input", "") + "\n（补充：" + user_reply + "）",
    }


# ---------------------------------------------------------------------------
# 路由函数：根据贴合度校验结果决定下一步
# ---------------------------------------------------------------------------

def _route_after_fit_check(state: AgentState) -> str:
    """
    - fit_check=True  → 进 SQL 生成
    - fit_check=False → 追问用户

    Args:
        state: 全局状态

    Returns:
        str: "gen_sql" 或 "ask_user"
    """
    if state.get("fit_check"):
        return "gen_sql"
    return "ask_user"


# ---------------------------------------------------------------------------
# 子图构建
# ---------------------------------------------------------------------------

def build_query_agent() -> StateGraph:
    """
    构建「查询生成 Agent」子图。

    图结构：
        retrieve_schemas → fit_check
            ├── 贴合 / 用户放弃 → generate_sql → END
            └── 不贴合 → ask_user → 回到 retrieve_schemas

    Returns:
        StateGraph: 编译后的子图
    """
    graph = StateGraph(AgentState)

    # 注册节点
    graph.add_node("retrieve_schemas", retrieve_table_schemas)
    graph.add_node("fit_check", fit_check_node)
    graph.add_node("generate_sql", generate_sql)
    graph.add_node("ask_user", ask_user_for_tables_node)

    # 入口
    graph.set_entry_point("retrieve_schemas")

    # retrieve_schemas → fit_check
    graph.add_edge("retrieve_schemas", "fit_check")

    # fit_check → 贴合进 SQL，不贴合追问
    graph.add_conditional_edges(
        "fit_check",
        _route_after_fit_check,
        {
            "gen_sql": "generate_sql",
            "ask_user": "ask_user",
        },
    )

    # ask_user → 回到 retrieve_schemas 重新检索+选表
    graph.add_edge("ask_user", "retrieve_schemas")

    # generate_sql → END
    graph.add_edge("generate_sql", END)

    return graph
