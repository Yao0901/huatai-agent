"""
向量库重建脚本

#NOTE:每次修改 metric_definitions.json 后运行此脚本，重新生成向量库。

用法:
    cd data/业务词汇匹配文件夹
    python rebuild_vectors.py
"""

import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer

# 路径（相对于此脚本所在目录）
HERE = os.path.dirname(os.path.abspath(__file__))
DEFINITIONS_PATH = os.path.join(HERE, "metric_definitions.json")
VECTORS_PATH = os.path.join(HERE, "metric_vectors.json")
MODEL_PATH = os.path.join(HERE, "model")


def rebuild():
    # 1. 加载本地模型
    print(f"[1/4] 加载本地模型: {MODEL_PATH}")
    model = SentenceTransformer(MODEL_PATH)
    print("      模型加载成功")

    # 2. 加载口径定义
    print(f"[2/4] 加载口径定义: {DEFINITIONS_PATH}")
    with open(DEFINITIONS_PATH, "r", encoding="utf-8") as f:
        definitions = json.load(f)
    print(f"      {len(definitions)} 条口径")

    # 3. 向量化：每个指标取所有别名向量的均值 → 一个中心向量
    print(f"[3/4] 正在向量化...")
    vectors = {}
    for item in definitions:
        texts = [item["metric_name"]] + item["aliases"]
        embeddings = model.encode(texts)
        centroid = np.mean(embeddings, axis=0).tolist()
        vectors[item["metric_name"]] = {
            "aliases": texts,
            "vector": centroid,
        }
    print(f"      完成, {len(vectors)} 个向量 (维度 {len(centroid)})")

    # 4. 保存
    print(f"[4/4] 保存到: {VECTORS_PATH}")
    with open(VECTORS_PATH, "w", encoding="utf-8") as f:
        json.dump(vectors, f, ensure_ascii=False, indent=2)
    print("      完成")

    # 5. 验证一下
    print("\n验证匹配效果:")
    model = SentenceTransformer(MODEL_PATH)
    test_terms = ["总成交量", "今天赚了多少", "客户心情指数"]
    for term in test_terms:
        tv = model.encode([term])[0]
        best, best_score = "", -1
        for name, data in vectors.items():
            sim = float(
                np.dot(tv, np.array(data["vector"]))
                / (np.linalg.norm(tv) * np.linalg.norm(np.array(data["vector"])))
            )
            if sim > best_score:
                best, best_score = name, sim
        hit = "hit" if best_score >= 0.6 else "miss"
        print(f"  [{hit}] {term} -> {best} ({best_score:.3f})")


if __name__ == "__main__":
    rebuild()
