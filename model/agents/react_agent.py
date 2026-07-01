"""
ReAct Agent — 单一 LLM 循环替代三张子图流水线

LLM 可自主调用四个工具：
- query_database: 探索表结构 / 列名 / 枚举值 / 样本数据
- search_code_mapping: 查 dim_public 码值映射
- run_sql: 执行 SQL 并查看结果（自我验证/修正）
- ask_user: 向用户提问澄清模糊需求
"""

from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

from tools.llm_config import get_chat_model
from tools.db_connector import execute_sql
from tools.metric_retriever import search_code_mapping as _search_code


# ---------------------------------------------------------------------------
# 工具 1: 探索数据库结构
# ---------------------------------------------------------------------------

def query_database(sql: str) -> str:
    """
    执行只读 SQL 探索数据库结构。用于：
    - PRAGMA table_info('表名') 查看列名和类型
    - SELECT DISTINCT 列名 FROM 表名 LIMIT 30 查看枚举值
    - SELECT * FROM 表名 LIMIT 3 查看样本数据
    - SELECT MIN(data_dt), MAX(data_dt) FROM 表名 查看日期范围
    禁止 INSERT/UPDATE/DELETE/DROP。
    """
    sql_upper = sql.strip().upper()
    if not (sql_upper.startswith("SELECT") or sql_upper.startswith("PRAGMA")):
        return f"[BLOCKED] 只允许 SELECT 和 PRAGMA。你发送的是: {sql[:60]}"
    if "sqlite_master" in sql_upper:
        return "[BLOCKED] 禁止查系统表"

    result = execute_sql(sql)
    if result["success"]:
        rows = result["data"]
        if not rows:
            return "[EMPTY] 查询无结果"
        if len(rows) > 50:
            rows = rows[:50]
            extra = f"\n... (还有 {result['row_count'] - 50} 行，已截断)"
        else:
            extra = ""
        cols = list(rows[0].keys())
        lines = [" | ".join(cols)]
        for row in rows:
            lines.append(" | ".join(str(row.get(c, "")) for c in cols))
        return "\n".join(lines) + extra
    else:
        return f"[ERROR] {result.get('error', '未知')}"


# ---------------------------------------------------------------------------
# 工具 2: 码值映射查询
# ---------------------------------------------------------------------------

def search_code_mapping(term: str, code_type_id: str = "") -> str:
    """
    查 dim_public 码值映射表。将用户自然语言翻译为数据库编码。
    例如: term='男' → code='5000002'; term='钻石卡' → code_type_id='100'级。
    code_type_id: 100=客户等级, 500=性别, 600=学历, 700=职业。不指定则全局搜索。
    """
    ctid = code_type_id if code_type_id else None
    results = _search_code(term, ctid)
    if not results:
        return f"[NOT FOUND] 未找到与'{term}'匹配的码值"
    lines = [f"查询'{term}'结果 ({len(results)} 条):"]
    for r in results[:15]:
        lines.append(f"  code={r['code']}, describe={r['describe']}, type={r['code_type_id']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 工具 3: 执行 SQL（用于自我验证和最终查询）
# ---------------------------------------------------------------------------

def run_sql(sql: str) -> str:
    """
    执行 SELECT 查询并返回结果。用于验证 SQL 是否正确、查看实际数据。
    如果结果不对，可以修改 SQL 后重新调用此工具。
    禁止任何写操作（INSERT/UPDATE/DELETE/DROP）。
    """
    sql_upper = sql.strip().upper()
    if not sql_upper.startswith("SELECT") and not sql_upper.startswith("WITH"):
        return f"[BLOCKED] 只允许 SELECT/WITH 查询。你发送的是: {sql[:60]}"

    result = execute_sql(sql)
    if result["success"]:
        rows = result["data"]
        row_count = result["row_count"]
        if not rows:
            return "[EMPTY] 查询成功但返回 0 行。可能是条件太严格或日期不匹配。"
        if row_count > 30:
            rows = rows[:30]
            extra = f"\n... (还有 {row_count - 30} 行省略)"
        else:
            extra = ""
        cols = list(rows[0].keys())
        lines = [f"返回 {row_count} 行:", " | ".join(cols)]
        for row in rows:
            lines.append(" | ".join(str(row.get(c, "")) for c in cols))
        return "\n".join(lines) + extra
    else:
        return f"[SQL ERROR] {result.get('error', '未知错误')}\n请检查 SQL 并修正后重试。"


# ---------------------------------------------------------------------------
# 工具 4: 向用户提问澄清
# ---------------------------------------------------------------------------

def ask_user(question: str) -> str:
    """
    向用户提问以澄清模糊需求。仅在确实无法从数据中判断时使用。
    例如：
    - 用户说"查资产"，不确定是总资产还是净资产
    - 用户说"最近"，不确定时间范围
    - 口径定义中有多种计算方式，需要用户确认
    """
    print(f"\n  ❓ {question}")
    reply = input("  👤 ").strip()
    return reply if reply else "跳过"


# ---------------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """你是「智谋洞见 Agent」，由智谋洞见团队打造。你可以操作一个包含客户、产品、交易、持仓、资产等数据的 SQLite 数据库，帮助用户进行金融数据查询与分析。

## 思考习惯
在开始分析前，先输出一行「容我三思」作为思考标记，然后再开始推理。

## 工作方式
1. 理解用户的数据查询需求
2. **不确定时用 ask_user 向用户确认**（如指标口径、时间范围有歧义，不要自己猜）
3. 用 query_database 探索需要用到的表结构（PRAGMA table_info、SELECT DISTINCT 等）
4. 用 search_code_mapping 查询编码含义（如用户说"钻石卡"，查 dim_public 找到对应 code）
5. 用 run_sql 执行生成的 SQL 看结果
6. 如果结果不对，自查原因并修正，再执行
7. 向用户汇报最终答案

## 数据库概况
8 张表:
- ads_cust_info_d: 客户信息（等级、性别、年龄、学历、职业、营业部）
- dws_cust_aset_d: 客户资产日汇总（nm_tot_aset 普通账户, fc_pur_aset 富裕账户）
- dwd_cust_tran_d: 客户交易流水（buy_amt, sell_amt, buy_cnt, sell_cnt 等）
- dwd_cust_hold_d: 客户持仓明细（hold_cnt, mkt_val）
- dws_cust_fin_d: 客户资金流入流出（cash_in/out, tran_in/out）
- dim_product: 产品维度（prdt_name, prdt_type_name, up_prdt_type_name）
- dim_branch: 营业部维度（org_id, org_name, up_org_name）
- dim_public: 码值映射（code_type_id + code + describe）

## 错误处理
- `run_sql` 返回 `[SQL ERROR]` 时，必须自动分析原因并修正 SQL 后重试，不要直接放弃
- 常见错误及修复：列名拼写→查 PRAGMA 确认；日期不匹配→检查各表日期范围；JOIN 条件错→确认关联键

## SQL 规范
- 日期格式 YYYYMMDD，Q1 = 20260101 ~ 20260331
- 聚合用 COALESCE 处理 NULL
- 不同表 data_dt 范围可能不一致，注意日期对齐
- 编码列用 dim_public JOIN 翻译或 search_code_mapping 查询
- 优先 LEFT JOIN，复杂查询用 CTE
- 严格遵循用户指定的分组规则/筛选条件，禁止自己重新划分

## 上下文管理（重要）
- 用户追问"进一步解读"或"再分析"时，先回顾对话历史：
  - 如果需要的表结构、数据已在前几轮查过，直接复用，不要重新探索
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
- **如果是数据查询**，最终回答必须依次包含：
  1. **最终 SQL**：你执行的最后一条完整 SQL 语句（用代码块包裹）
  2. **数据来源**：本次使用了哪几张表、哪些关键列
  3. **关键数据**：表格或列表形式
  4. **自然语言解读**：简洁分析
  5. 如果结果有问题，说明原因和修正过程"""


# ---------------------------------------------------------------------------
# 上下文修剪 Hook
# ---------------------------------------------------------------------------

MAX_TOKENS_ESTIMATE = 100_000  # 超过此阈值触发修剪


def _is_important(msg) -> bool:
    """判断消息是否重要（数据查询、需求修改、报错修复等）。"""
    if not hasattr(msg, "content") or not msg.content:
        return False
    text = str(msg.content)[:200]
    # 数据查询相关
    keywords = [
        "查询", "统计", "分析", "计算", "分布",
        "SQL", "SELECT", "WHERE", "FROM",
        "修改", "改成", "换成", "换成", "不要",
        "报错", "失败", "错误", "BUG",
        "规则", "条件", "筛选", "分段",
        "盈亏", "资产", "交易", "持仓", "客户",
    ]
    return any(kw in text for kw in keywords)


def _trim_context(state: dict) -> dict:
    """
    pre_model_hook：token 接近上限时智能修剪。

    策略：
    - 前 3 条完整保留（初始上下文）
    - 最近 5 条完整保留（当前上下文）
    - 中间：重要的保留（数据查询/需求/报错），闲聊压缩
    - 使用 llm_input_messages 不影响存储的完整历史
    """
    messages = state.get("messages", [])
    total_chars = sum(len(str(m.content)) if hasattr(m, "content") else 0 for m in messages)
    estimated_tokens = total_chars * 1.2
    if estimated_tokens <= MAX_TOKENS_ESTIMATE:
        return {}

    from langchain_core.messages import SystemMessage

    total = len(messages)
    keep_first = min(3, total // 4)
    keep_last = min(5, total // 4)

    # 中间消息：重要的保留，闲聊的压缩
    middle = messages[keep_first:total - keep_last]
    important_msgs = []
    compressed_count = 0
    for m in middle:
        if _is_important(m):
            important_msgs.append(m)
        else:
            compressed_count += 1

    trimmed = list(messages[:keep_first])
    if compressed_count > 0:
        important_count = len(important_msgs)
        trimmed.append(
            SystemMessage(
                content=f"[中间已压缩 {compressed_count} 条闲聊消息，"
                f"保留了 {important_count} 条重要消息（数据查询/需求修改等）]"
            )
        )
    trimmed.extend(important_msgs)
    trimmed.extend(messages[-keep_last:])

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
    tools = [query_database, search_code_mapping, run_sql, ask_user]
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
