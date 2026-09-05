from __future__ import annotations

import math

from app.eval.metrics import (
    dcg_at_k,
    mrr,
    ndcg_at_k,
    recall_at_k,
    _unique_basename_paths,
)


def test_unique_basename_paths_dedup_keeps_order():
    ranked = [
        r"E:\kb\a.txt",
        r"E:\kb\b.txt",
        r"E:\kb\a.txt",
        r"E:\kb\c.txt",
    ]
    assert _unique_basename_paths(ranked) == [
        r"E:\kb\a.txt",
        r"E:\kb\b.txt",
        r"E:\kb\c.txt",
    ]


def test_recall_at_k_dedup_same_file_multiple_chunks():
    # 同一文件的多个 chunk 只算一次命中；@3 命中则 @5 也命中
    ranked = [r"E:\kb\a.txt", r"E:\kb\a.txt", r"E:\kb\b.txt"]
    assert recall_at_k(ranked, ["a.txt"], 1) == 1.0
    assert recall_at_k(ranked, ["b.txt"], 3) == 1.0
    assert recall_at_k(ranked, ["b.txt"], 2) == 1.0  # 去重后 b 在第 2 位
    assert recall_at_k(ranked, ["missing.txt"], 5) == 0.0


def test_mrr_uses_first_unique_file():
    ranked = [r"E:\kb\b.txt", r"E:\kb\a.txt", r"E:\kb\a.txt"]
    assert mrr(ranked, ["a.txt"]) == 0.5


def test_ndcg_never_exceeds_one_with_duplicate_chunks():
    # 同一文件 3 个 chunk 排前 3，ground truth 只有 1 个相关文件：
    # 去重后只有 1 次命中，NDCG 必须 <= 1（旧实现会把同一文件重复计分导致 >1）
    ranked = [r"E:\kb\a.txt", r"E:\kb\a.txt", r"E:\kb\a.txt", r"E:\kb\b.txt"]
    assert 0.0 < ndcg_at_k(ranked, ["a.txt"], 5) <= 1.0
    # 单片命中：DCG = 1/log2(2) = 1，IDCG = 1 -> NDCG = 1
    assert math.isclose(ndcg_at_k([r"E:\kb\a.txt"], ["a.txt"], 5), 1.0)


def test_dcg_matches_binary_gain():
    ranked = [r"E:\kb\a.txt", r"E:\kb\b.txt"]
    expected_dcg = 1.0 / math.log2(2)  # 位置1命中
    assert math.isclose(dcg_at_k(ranked, ["a.txt"], 5), expected_dcg)