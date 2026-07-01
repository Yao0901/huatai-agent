"""
表结构检索工具

根据用户意图（实体、维度、指标）检索相关的数据库表结构。
基于 SQLite 系统表（sqlite_master + PRAGMA），不需要向量数据库。

检索策略（两层）：
1. LLM 语义选表：将 8 张表的目录发给 DeepSeek，由 LLM 根据用户问题语义判断哪些表相关
2. 关键词兜底：LLM 不可用时退化到关键词匹配 + 关联扩散
3. 关联扩散：选中表后自动拉入 related_tables（保证 JOIN 链路完整）
"""

import os
from typing import List, Dict, Any

from tools.db_connector import get_connection, get_table_schema, get_all_table_names
from tools.llm_config import chat

# ---------------------------------------------------------------------------
# 表描述知识库（中文描述 + 关联关系）
# 这些是"AI 友好的元数据"，用于将用户自然语言映射到物理表
# ---------------------------------------------------------------------------

TABLE_DESCRIPTIONS: Dict[str, Dict[str, Any]] = {
    "ads_cust_info_d": {
        "description": "客户信息表，包含客户等级、性别、年龄、学历、职业、所在省份城市、营业部ID等画像信息",
        "keywords": ["客户", "等级", "性别", "年龄", "学历", "职业", "省份", "城市", "营业部", "钻石卡", "黄金卡", "个人信息", "画像"],
        "related_tables": ["dim_public", "dim_branch", "dws_cust_aset_d"],
        "join_keys": {"org_id": "dim_branch.org_id"},
        "columns": {
            "data_dt": "数据日期，格式YYYYMMDD",
            "pty_id": "客户ID（主键，关联各事实表的pty_id）",
            "sor_pty_id": "源系统客户ID",
            "cust_lvl_cd": "客户等级编码（JOIN dim_public ON code WHERE code_type_id='100'翻译为中文描述）",
            "cust_status": "客户状态",
            "cust_type": "客户类型",
            "prov_name": "省份名称",
            "city_name": "城市名称",
            "birth_dt": "出生日期",
            "cust_age": "客户年龄（数值，可直接比较大小）",
            "name": "客户姓名",
            "gender_cd": "性别编码（JOIN dim_public ON code WHERE code_type_id='500'翻译，如'5000002'=男）",
            "edu_cd": "学历编码（JOIN dim_public ON code WHERE code_type_id='600'翻译，如'6000003'=本科）",
            "prof_cd": "职业编码（JOIN dim_public ON code WHERE code_type_id='700'翻译）",
            "org_id": "所属营业部ID（关联dim_branch.org_id获取营业部名称）",
        },
    },
    "dim_product": {
        "description": "产品维度表，包含产品名称、产品类型（A股/科创板/基金等）、产品大类层级关系",
        "keywords": ["产品", "股票", "基金", "债券", "A股", "科创板", "比亚迪", "招商银行", "中国平安", "名称"],
        "related_tables": ["dwd_cust_hold_d", "dwd_cust_tran_d"],
        "join_keys": {"prdt_id": "dwd_cust_hold_d.prdt_id, dwd_cust_tran_d.prdt_id"},
        "columns": {
            "prdt_id": "产品ID（主键，关联持仓/交易事实表）",
            "prdt_name": "产品名称（如'比亚迪''招商银行''中国平安'，用于模糊搜索）",
            "sor_prdt_id": "源系统产品ID",
            "market_id": "市场ID",
            "prdt_type_id": "产品类型编码",
            "prdt_type_name": "产品类型名称（如A股、科创板、基金、债券）",
            "up_prdt_type_id": "上级产品大类编码",
            "up_prdt_type_name": "上级产品大类名称（如股票、基金、债券）",
        },
    },
    "dwd_cust_hold_d": {
        "description": "客户持仓明细事实表，记录客户持有产品的份额和市值，按日和客户+产品+币种粒度",
        "keywords": ["持仓", "持有", "份额", "市值", "持股", "仓位"],
        "related_tables": ["ads_cust_info_d", "dim_product"],
        "join_keys": {"pty_id": "ads_cust_info_d.pty_id", "prdt_id": "dim_product.prdt_id"},
        "columns": {
            "data_dt": "数据日期，格式YYYYMMDD",
            "pty_id": "客户ID（关联ads_cust_info_d.pty_id）",
            "prdt_id": "产品ID（关联dim_product.prdt_id获取产品名称和类型）",
            "sys_source": "系统来源",
            "ccy": "币种",
            "hold_cnt": "持仓数量（份额）",
            "mkt_val": "持仓市值（金额，可用于聚合求和）",
        },
    },
    "dwd_cust_tran_d": {
        "description": "客户交易流水事实表，记录客户买卖产品的笔数、金额、佣金、手续费等",
        "keywords": ["交易", "买卖", "买入", "卖出", "成交", "佣金", "手续费", "交易量"],
        "related_tables": ["ads_cust_info_d", "dim_product"],
        "join_keys": {"pty_id": "ads_cust_info_d.pty_id", "prdt_id": "dim_product.prdt_id"},
        "columns": {
            "data_dt": "数据日期，格式YYYYMMDD",
            "pty_id": "客户ID（关联ads_cust_info_d.pty_id）",
            "prdt_id": "产品ID（关联dim_product.prdt_id获取产品名称和类型）",
            "sys_source": "系统来源",
            "ccy": "币种",
            "buy_cnt": "买入笔数",
            "buy_mnt": "买入数量（股/份）",
            "buy_rake": "买入佣金",
            "buy_amt": "买入金额",
            "buy_fare": "买入手续费",
            "sell_cnt": "卖出笔数",
            "sell_mnt": "卖出数量（股/份）",
            "sell_rake": "卖出佣金",
            "sell_amt": "卖出金额",
            "sell_fare": "卖出手续费",
        },
    },
    "dws_cust_aset_d": {
        "description": "客户资产日汇总表，记录客户普通账户和富裕账户的总资产、现金资产",
        "keywords": ["资产", "总资产", "现金", "富裕", "账户", "日均资产"],
        "related_tables": ["ads_cust_info_d", "dws_cust_fin_d"],
        "join_keys": {"pty_id": "ads_cust_info_d.pty_id"},
        "columns": {
            "data_dt": "数据日期，格式YYYYMMDD",
            "pty_id": "客户ID（关联ads_cust_info_d.pty_id）",
            "nm_tot_aset": "普通账户总资产（含现金+持仓市值，核心指标）",
            "nm_bal": "普通账户现金余额",
            "fc_pur_aset": "富裕账户总资产",
            "fc_bal": "富裕账户现金余额",
        },
    },
    "dws_cust_fin_d": {
        "description": "客户资金流入流出日事实表，记录现金转入转出、证券转入转出、指定转入转出",
        "keywords": ["资金", "流入", "流出", "转入", "转出", "现金转入", "银证转账", "盈亏"],
        "related_tables": ["dws_cust_aset_d"],
        "join_keys": {"pty_id": "ads_cust_info_d.pty_id"},
        "columns": {
            "data_dt": "数据日期，格式YYYYMMDD",
            "pty_id": "客户ID（关联ads_cust_info_d.pty_id）",
            "sys_source": "系统来源",
            "cash_in": "现金转入金额（银证转账入金）",
            "cash_out": "现金转出金额（银证转账出金）",
            "tran_in": "证券转入金额",
            "tran_out": "证券转出金额",
            "assign_in": "指定转入金额",
            "assign_out": "指定转出金额",
        },
    },
    "dim_public": {
        "description": "公共码值映射表，将编码（如 gender_cd='5000002'）映射为中文描述（如'男'）。包含客户等级(100)、性别(500)、学历(600)、职业(700)等码值类型",
        "keywords": ["码值", "编码", "映射", "字典", "翻译", "男", "女", "博士", "硕士", "学士"],
        "related_tables": ["ads_cust_info_d"],
        "join_keys": {},
        "columns": {
            "code": "编码值（如'5000002'，关联ads_cust_info_d的gender_cd/edu_cd等码值字段）",
            "code_type_id": "码值类型ID：100=客户等级, 500=性别, 600=学历, 700=职业",
            "describe": "编码对应的中文描述（如'男''本科''钻石卡'）",
        },
    },
    "dim_branch": {
        "description": "营业部维度表，包含营业部名称、ID及上级分公司名称",
        "keywords": ["营业部", "分公司", "网点", "分支机构", "部门"],
        "related_tables": ["ads_cust_info_d"],
        "join_keys": {"org_id": "ads_cust_info_d.org_id", "up_org_id": "（自引用）"},
        "columns": {
            "data_dt": "数据日期，格式YYYYMMDD",
            "org_id": "营业部ID（主键，关联ads_cust_info_d.org_id）",
            "org_name": "营业部名称",
            "up_org_id": "上级分公司ID",
            "up_org_name": "上级分公司名称",
        },
    },
}


def _build_table_catalog() -> str:
    """
    构建所有表的简要目录文本，供 LLM 选表时使用。

    每张表包含：表名、描述、主要列（含列描述）、JOIN 关系。

    Returns:
        str: 格式化的表目录文本
    """
    lines = []
    for i, (table_name, info) in enumerate(TABLE_DESCRIPTIONS.items()):
        # 列摘要（截断每列描述，控制总长度）
        col_entries = []
        for col_name, col_desc in info.get("columns", {}).items():
            # 截断过长的列描述
            short_desc = col_desc[:60] + "…" if len(col_desc) > 60 else col_desc
            col_entries.append(f"{col_name}({short_desc})")
        col_text = ", ".join(col_entries[:20])  # 最多展示 20 列

        join_keys = info.get("join_keys", {})
        join_text = ", ".join(f"{k}→{v}" for k, v in join_keys.items()) if join_keys else "无"

        lines.append(
            f"{i + 1}. **{table_name}**\n"
            f"   描述: {info['description']}\n"
            f"   列: {col_text}\n"
            f"   JOIN: {join_text}"
        )

    return "\n".join(lines)


def _llm_select_tables(
    user_input: str, intent: Dict[str, Any], top_k: int
) -> List[str]:
    """
    用 LLM（DeepSeek）根据用户问题语义选择最相关的表。

    将 8 张表的目录 + 用户问题一次发给 LLM，让它判断哪些表能回答这个问题。
    同义词、别名、口语化表述都能自然处理，不再依赖硬编码 keywords。

    Args:
        user_input: 用户原始问题
        intent: 结构化意图（entities, metrics, dimensions）
        top_k: 最多选几张表

    Returns:
        list[str]: 选中的表名列表，失败时返回空列表
    """
    catalog = _build_table_catalog()

    system_prompt = (
        "你是金融数据库查询专家。根据用户的分析需求，从可用表中选择相关的表。\n"
        "选择原则：\n"
        "1. 用户提到的实体/指标/维度对应的表必须选\n"
        "2. 如果某表的列是码值编码（如 gender_cd），必须同时选 dim_public 做翻译\n"
        "3. 如果涉及客户属性（年龄、性别、等级等），选 ads_cust_info_d\n"
        "4. 如果涉及产品名称/类型筛选，选 dim_product\n"
        "5. 如果涉及营业部，选 dim_branch\n"
        "6. 宁多勿少：不确定是否相关时，倾向于选入\n"
        f"只输出表名，逗号分隔。最多选 {top_k} 张。不要输出其他文字。"
    )

    entities = ", ".join(intent.get("entities", [])) or "无"
    metrics = ", ".join(intent.get("metrics", [])) or "无"
    dimensions = ", ".join(intent.get("dimensions", [])) or "无"

    user_prompt = (
        f"用户问题：{user_input}\n"
        f"实体：{entities}\n"
        f"指标：{metrics}\n"
        f"维度：{dimensions}\n\n"
        f"可用表目录：\n{catalog}"
    )

    try:
        response = chat(system_prompt, user_prompt, temperature=0.0)
        # 清洗回复：去换行、去多余空格
        response = response.strip().replace("\n", "").replace(" ", "")
        table_names = [t.strip() for t in response.split(",") if t.strip()]
        # 只保留在 TABLE_DESCRIPTIONS 中存在的表名
        valid = [t for t in table_names if t in TABLE_DESCRIPTIONS]
        if valid:
            return valid
    except Exception:
        pass

    return []  # 失败返回空，让调用方走关键词兜底


# ---------------------------------------------------------------------------
# 主检索函数
# ---------------------------------------------------------------------------


def retrieve_schemas(
    intent: Dict[str, Any],
    top_k: int = 10,
    user_input: str = "",
) -> List[Dict[str, Any]]:
    """
    根据结构化意图检索最相关的表结构。

    检索策略（LLM 优先 + 关键词兜底）：
    1. 用 LLM 语义理解用户问题，从 8 张表中选出相关的表
    2. 如果 LLM 不可用，退化到关键词匹配 + 关联扩散
    3. 对选中的表做关联扩散（related_tables），保证 JOIN 链路完整
    4. 返回每张表的完整 Schema（列信息 + 列描述 + 样本数据 + JOIN 关系）

    Args:
        intent: 结构化意图字典，包含 entities, metrics, dimensions
        top_k: 返回最相关的表数上限
        user_input: 用户原始问题（供 LLM 语义理解用）

    Returns:
        list[dict]: 每张表包含 table_name, description, columns, sample_data, row_count, join_keys
    """
    # ---- 第 0 步：尝试 LLM 语义选表 ----
    selected = set()
    if user_input:
        llm_picks = _llm_select_tables(user_input, intent, top_k)
        if llm_picks:
            selected = set(llm_picks[:top_k])

    # ---- 第 1 步：LLM 失败时，退化为关键词匹配 ----
    if not selected:
        selected = _keyword_fallback(intent, top_k)

    # ---- 第 2 步：关联扩散（保证 JOIN 链路完整） ----
    expanded = set(selected)
    for table_name in selected:
        for related in TABLE_DESCRIPTIONS.get(table_name, {}).get("related_tables", []):
            expanded.add(related)

    # ---- 第 3 步：获取完整 Schema ----
    result = []
    for table_name in expanded:
        schema = get_table_schema(table_name)
        if schema:
            meta = TABLE_DESCRIPTIONS.get(table_name, {})
            schema["description"] = meta.get("description", "")
            schema["join_keys"] = meta.get("join_keys", {})
            col_descs = meta.get("columns", {})
            for col in schema.get("columns", []):
                col["description"] = col_descs.get(col["name"], "")
            result.append(schema)

    return result


def _keyword_fallback(intent: Dict[str, Any], top_k: int) -> set:
    """
    LLM 不可用时的兜底：关键词匹配 + 关联扩散。

    与旧版检索逻辑一致，保留作为可靠性保障。
    """
    # 组装搜索词
    search_terms = set()
    for key in ("entities", "metrics", "dimensions"):
        for item in intent.get(key, []):
            search_terms.add(item.lower())
    for t in intent.get("suggested_tables", []):
        search_terms.add(t.lower())

    # 兜底：无搜索词 → 返回所有表
    if not search_terms:
        return set(TABLE_DESCRIPTIONS.keys())

    # 打分
    scored_tables = []
    for table_name, info in TABLE_DESCRIPTIONS.items():
        score = 0
        for term in search_terms:
            if term in table_name.lower():
                score += 10
            for kw in info["keywords"]:
                if term in kw.lower() or kw.lower() in term:
                    score += 3
            if term in info["description"].lower():
                score += 1
            for col_name, col_desc in info.get("columns", {}).items():
                if term in col_desc.lower():
                    score += 2
                if term in col_name.lower():
                    score += 1
        if score > 0:
            scored_tables.append((table_name, score))

    scored_tables.sort(key=lambda x: x[1], reverse=True)
    return {t for t, _ in scored_tables[:top_k]}
