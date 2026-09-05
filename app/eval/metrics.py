# -*- coding: utf-8 -*-
"""检索质量指标计算。

指标说明（对外可展示的硬指标）：
  - Recall@k：在返回的前 k 条结果中，是否命中期望文档（命中比例）。
  - MRR（Mean Reciprocal Rank）：首个命中文档的排名倒数之均值。
    全未命中计 0。衡量"正确答案排多靠前"。
  - NDCG@k（Normalized Discounted Cumulative Gain）：
    排序质量指标，奖励相关文档排在前面。这里用 0/1 二元相关性，
    期望命中的文档视为 1，其余为 0。
  - 重排增益：对比"纯向量检索"与"向量+重排"的 Recall@k，量化 reranker 的价值。
"""

from __future__ import annotations

import math


def _is_hit(path: str, expected: list[str]) -> bool:
    """判断一条检索结果的路径是否命中期望文件名集合。"""
    base = path.replace("\\", "/").split("/")[-1]
    return base in expected


def _unique_basename_paths(ranked_paths: list[str]) -> list[str]:
    """按文件名去重并保持首次出现顺序（同一文件的多个 chunk 只算一次命中）。"""
    seen: set[str] = set()
    out: list[str] = []
    for p in ranked_paths:
        base = p.replace("\\", "/").split("/")[-1]
        if base in seen:
            continue
        seen.add(base)
        out.append(p)
    return out


def recall_at_k(ranked_paths: list[str], expected: list[str], k: int) -> float:
    """前 k 条结果中是否至少命中一个期望文档 -> 0.0 / 1.0。"""
    if not expected:
        return 0.0
    for p in _unique_basename_paths(ranked_paths)[:k]:
        if _is_hit(p, expected):
            return 1.0
    return 0.0


def mrr(ranked_paths: list[str], expected: list[str]) -> float:
    """首个命中文档的 1/rank，全未命中为 0。"""
    for i, p in enumerate(_unique_basename_paths(ranked_paths), start=1):
        if _is_hit(p, expected):
            return 1.0 / i
    return 0.0


def dcg_at_k(ranked_paths: list[str], expected: list[str], k: int) -> float:
    """DCG：相关文档的位置越靠前、增益越高（log2 折扣）。"""
    dcg = 0.0
    for i, p in enumerate(_unique_basename_paths(ranked_paths)[:k], start=1):
        if _is_hit(p, expected):
            dcg += 1.0 / math.log2(i + 1)
    return dcg


def _ideal_dcg(expected_count: int, k: int) -> float:
    """理想 DCG：所有 related 文档都排在最前面时的最佳增益。"""
    idcg = 0.0
    for i in range(1, min(expected_count, k) + 1):
        idcg += 1.0 / math.log2(i + 1)
    return idcg


def ndcg_at_k(ranked_paths: list[str], expected: list[str], k: int) -> float:
    if not expected:
        return 0.0
    idcg = _ideal_dcg(len(expected), k)
    if idcg == 0.0:
        return 0.0
    return dcg_at_k(ranked_paths, expected, k) / idcg


def evaluate_retrieval_results(results: list[dict], k_values: tuple[int, ...] = (1, 3, 5)) -> dict:
    """批量评估检索结果。

    results: list[dict]，每个元素 {query, expected, ranked_paths}
    返回：整体在多个 k 下的平均 Recall@k、MRR、NDCG@k，以及单条明细。
    """
    n = len(results)
    if n == 0:
        return {}

    agg: dict = {"n": n, "k_values": list(k_values)}
    for k in k_values:
        agg[f"recall@{k}"] = sum(recall_at_k(r["ranked_paths"], r["expected"], k) for r in results) / n
    agg["mrr"] = sum(mrr(r["ranked_paths"], r["expected"]) for r in results) / n
    for k in k_values:
        agg[f"ndcg@{k}"] = sum(ndcg_at_k(r["ranked_paths"], r["expected"], k) for r in results) / n

    agg["details"] = []
    for r in results:
        agg["details"].append(
            {
                "query": r["query"],
                "expected": r["expected"],
                "recall@1": recall_at_k(r["ranked_paths"], r["expected"], 1),
                "recall@3": recall_at_k(r["ranked_paths"], r["expected"], 3),
                "recall@5": recall_at_k(r["ranked_paths"], r["expected"], 5),
                "mrr": mrr(r["ranked_paths"], r["expected"]),
                "top_paths": [p.replace("\\", "/").split("/")[-1] for p in r["ranked_paths"][:5]],
            }
        )
    return agg
