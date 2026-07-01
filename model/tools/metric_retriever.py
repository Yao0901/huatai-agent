"""
业务口径与指标检索工具

解决"用户说的词 ≠ 数据库存的词"这一核心问题。

采用 RAG（检索增强生成）两段式匹配：
  Stage 1（本地）: 用户词 → 实时向量化 → 在同义向量库中计算余弦相似度 → top-k 粗筛
  Stage 2（LLM） : 歧义区间（0.4~0.75）时，将 top-3 候选 + 用户原话送给 LLM 精判

两大功能：
1. 码值映射查询：从 dim_public 表中查询编码 ↔ 中文描述
2. 指标公式检索：向量粗筛 + LLM 精判 → 返回标准指标定义
"""

import json
import os
import numpy as np
from typing import List, Dict, Any, Optional


# ---------------------------------------------------------------------------
# 路径解析：找到 data/业务词汇匹配文件夹/
# ---------------------------------------------------------------------------

def _resolve_data_dir() -> str:
    """从当前文件位置推断 data/业务词汇匹配文件夹/ 的路径。"""
    # metric_retriever.py 在 model/tools/ → 上两级到项目根 → 下钻到 data
    tools_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.dirname(tools_dir)
    project_dir = os.path.dirname(model_dir)
    return os.path.join(project_dir, "data", "业务词汇匹配文件夹")


_DATA_DIR = _resolve_data_dir()
_MODEL_PATH = os.path.join(_DATA_DIR, "model")
_VECTORS_PATH = os.path.join(_DATA_DIR, "metric_vectors.json")
_DEFINITIONS_PATH = os.path.join(_DATA_DIR, "metric_definitions.json")


# ---------------------------------------------------------------------------
# 懒加载：首次调用时才加载模型和向量库
# ---------------------------------------------------------------------------

_model = None       # SentenceTransformer 实例
_vectors = None     # {metric_name: {aliases, vector}}
_definitions = None # [{metric_name, aliases, formula, involved_tables, description}, ...]
_sim_threshold_high = 0.75   # 高于此值直接返回，跳过 LLM
_sim_threshold_low  = 0.4    # 低于此值直接放弃
_llm_top_k = 3               # 歧义区间送给 LLM 的候选数


def _ensure_loaded():
    """首次调用时加载本地模型 + 口径定义 + 向量库。"""
    global _model, _vectors, _definitions

    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(_MODEL_PATH)

    if _vectors is None:
        with open(_VECTORS_PATH, "r", encoding="utf-8") as f:
            _vectors = json.load(f)

    if _definitions is None:
        with open(_DEFINITIONS_PATH, "r", encoding="utf-8") as f:
            _definitions = {d["metric_name"]: d for d in json.load(f)}


# ---------------------------------------------------------------------------
# 码值映射查询（查询 dim_public 表）
# ---------------------------------------------------------------------------

CODE_TYPE_MAP = {"100": "客户等级", "500": "性别", "600": "学历", "700": "职业"}


def search_code_mapping(user_value: str, code_type_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    查询码值映射：将用户自然语言中的枚举值翻译为数据库编码。

    Args:
        user_value: 用户描述的值（如 "男" / "钻石卡" / "博士"）
        code_type_id: 限定码值类型（如 "500"=性别），None 表示不限

    Returns:
        list[dict]: 匹配的码值列表
    """
    from tools.db_connector import get_connection

    conn = get_connection()
    results = []

    if code_type_id:
        cursor = conn.execute(
            "SELECT code, describe, code_type_id FROM dim_public "
            "WHERE code_type_id=? AND describe LIKE ?",
            (code_type_id, f"%{user_value}%"),
        )
    else:
        cursor = conn.execute(
            "SELECT code, describe, code_type_id FROM dim_public "
            "WHERE describe LIKE ?",
            (f"%{user_value}%",),
        )

    for row in cursor.fetchall():
        results.append({
            "code": row["code"], "describe": row["describe"], "code_type_id": row["code_type_id"],
        })

    if not results:
        cursor = conn.execute(
            "SELECT code, describe, code_type_id FROM dim_public WHERE code=?",
            (user_value,),
        )
        for row in cursor.fetchall():
            results.append({
                "code": row["code"], "describe": row["describe"], "code_type_id": row["code_type_id"],
            })

    return results


# ---------------------------------------------------------------------------
# 指标公式检索（RAG 两段式）
# ---------------------------------------------------------------------------

def search_metric_definition(user_term: str) -> Optional[Dict[str, Any]]:
    """
    根据用户口语化表述，检索标准指标定义。

    RAG 两段式匹配：
    Stage 1（本地向量): 实时向量化用户词 → 与口径库中心向量算余弦相似度 → top-k 粗筛
    Stage 2（LLM 精判): 歧义区间时，把 top-3 候选 + 用户词送给 LLM 做最终判断

    Args:
        user_term: 用户提到的指标关键词（如"总成交量"、"今天赚了多少"）

    Returns:
        dict | None: 找到则返回指标定义（含 formula, involved_tables, description）
                     未找到返回 None
    """
    user_term = user_term.strip()
    if not user_term:
        return None

    _ensure_loaded()

    # ---- Stage 1: 向量粗筛 ----
    user_vec = _model.encode([user_term])[0]

    candidates = []
    for metric_name, data in _vectors.items():
        centroid = np.array(data["vector"])
        sim = float(
            np.dot(user_vec, centroid)
            / (np.linalg.norm(user_vec) * np.linalg.norm(centroid))
        )
        candidates.append((metric_name, sim))

    candidates.sort(key=lambda x: x[1], reverse=True)
    if not candidates:
        return None

    top_score = candidates[0][1]

    # ---- 阈值分流 ----
    # 高分 → 直接返回，不走 LLM
    if top_score >= _sim_threshold_high:
        return _definitions.get(candidates[0][0])

    # 极低分 → 直接放弃
    if top_score < _sim_threshold_low:
        return None

    # ---- Stage 2: 歧义区间 → LLM 精判 ----
    top_k_candidates = candidates[:_llm_top_k]
    return _llm_disambiguate(user_term, top_k_candidates)


def _llm_disambiguate(
    user_term: str, candidates: list
) -> Optional[Dict[str, Any]]:
    """
    Stage 2：用 LLM 对 top-k 候选做精细判断。

    将用户说的词 + 候选指标的名称/别名/描述一起发给 LLM，
    让它判断最匹配的是哪个，或者都不匹配。

    Args:
        user_term: 用户说的词
        candidates: [(metric_name, similarity_score), ...]

    Returns:
        dict | None: 匹配到的指标定义，或 None
    """
    from tools.llm_config import chat

    # 组装候选列表给 LLM
    candidate_text = ""
    for i, (name, score) in enumerate(candidates):
        definition = _definitions.get(name, {})
        candidate_text += (
            f"{i + 1}. {name}（相似度 {score:.2f}）\n"
            f"   别名: {', '.join(definition.get('aliases', []))}\n"
            f"   含义: {definition.get('description', '')}\n\n"
        )

    prompt = (
        f"用户提到了一个指标词：「{user_term}」\n\n"
        f"以下是向量检索找到的候选指标:\n{candidate_text}"
        f"请判断用户说的最可能是哪个指标。\n"
        f"- 如果用户说的词和某个候选明显是同一含义，输出该候选的编号(1/2/3)\n"
        f"- 如果都不匹配，输出 none\n"
        f"只输出编号数字或 none，不要其他文字。"
    )

    try:
        response = chat("你是金融指标匹配专家", prompt, temperature=0.0)
        response = response.strip().lower()
        if response in ("1", "2", "3"):
            idx = int(response) - 1
            if idx < len(candidates):
                return _definitions.get(candidates[idx][0])
        return None
    except Exception:
        # LLM 挂了，降级：top-1 分数 > 0.6 就用，否则放弃
        if candidates[0][1] >= 0.6:
            return _definitions.get(candidates[0][0])
        return None
