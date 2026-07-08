"""
查 dim_public 码值映射表。
"""

from tools.metric_retriever import search_code_mapping as _search


def search_code_mapping(term: str, db_name: str, code_type_id: str = "") -> str:
    """
    查 dim_public 码值映射。将用户自然语言翻译为数据库编码。
    例如: term='男' → code='5000002'; term='钻石卡' → code_type_id='100'级。
    code_type_id: 100=客户等级, 500=性别, 600=学历, 700=职业。不指定则全局搜索。
    """
    ctid = code_type_id if code_type_id else None
    results = _search(term, db_name, ctid)
    if not results:
        return f"[NOT FOUND] 在'{db_name}'中未找到与'{term}'匹配的码值"
    lines = [f"查询'{term}'结果 ({len(results)} 条):"]
    for r in results[:15]:
        lines.append(f"  code={r['code']}, describe={r['describe']}, type={r['code_type_id']}")
    return "\n".join(lines)
