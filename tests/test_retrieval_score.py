from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from app.src.util.retrieval.retrieval import (
    _classes_score,
    _norm_vector,
    _path_recall,
    _bm25_recall,
    _PATH_SCORE_CAP,
    _BM25_SCORE_MAX,
    _VECTOR_SCORE_MAX,
)


def _chunk(file_path, text, page=None, cid=1):
    return SimpleNamespace(
        file_path=file_path,
        page=page,
        chunk_text=text,
        id=cid,
        source_type="txt",
        doc_topic="other",
        section_title=None,
        section_path=None,
        summary=None,
    )


def test_classes_score_thresholds():
    assert _classes_score(80) == "strong"
    assert _classes_score(70) == "strong"
    assert _classes_score(69.9) == "weak"
    assert _classes_score(30) == "weak"
    assert _classes_score(29.9) == "low"
    assert _classes_score(0) == "low"


def test_norm_vector_caps_at_vector_max():
    assert _norm_vector(0.0) == 0.0
    assert _norm_vector(1.0) == _VECTOR_SCORE_MAX
    # 原始值超 1（重度重排 logits）时封顶到 85
    assert _norm_vector(5.0) == _VECTOR_SCORE_MAX
    # 负数归一为 0
    assert _norm_vector(-0.5) == 0.0
    assert _norm_vector(None) == 0.0
    assert _norm_vector(0.9) < _VECTOR_SCORE_MAX


def test_path_recall_score_capped_and_low():
    # 查询含文件名 work.png，触发路径命中；分数应封顶在低分，绝不超 100
    rows = [
        _chunk(r"E:\docu\work.png", "some ocr content work", cid=1),
        _chunk(r"E:\docu\other.txt", "unrelated", cid=2),
    ]
    hits = _path_recall(rows, "work.png 里有什么", top_k=5)
    # 只有 work.png 命中
    assert [h.path for h in hits] == [r"E:\docu\work.png"]
    assert all(h.score <= _PATH_SCORE_CAP for h in hits)
    assert all(h.score <= 100 for h in hits)
    assert all(h.relatedness == "low" for h in hits)


def test_bm25_recall_score_within_scale():
    rows = [
        _chunk(r"E:\docu\a.txt", "aaaa bbbb cccc", cid=1),
        _chunk(r"E:\docu\b.txt", "xxxx yyyy zzzz", cid=2),
    ]
    # 极小语料下真实 BM25 得分可能归零，这里打桩验证归一化映射本身：
    # 原始分 s>=1 -> 到达 BM25 上限；0<s<1 -> 按比例映射，且恒在 [0, 100]。
    class FakeBM25:
        def __init__(self, corpus):
            self._corpus = corpus

        def get_scores(self, tokens):
            return [0.5, 2.0]

    with patch("app.src.util.retrieval.retrieval.BM25Okapi", FakeBM25):
        hits = _bm25_recall(rows, "aaaa bbbb", top_k=5, reason="bm25")
    # s=2.0 封顶到 _BM25_SCORE_MAX；s=0.5 按比例映射
    assert len(hits) == 2
    assert hits[0].score == _BM25_SCORE_MAX  # 原始 2.0 封顶
    assert 0 < hits[1].score < _BM25_SCORE_MAX  # 原始 0.5 按比例
    for h in hits:
        assert 0 <= h.score <= 100
        assert h.relatedness in {"strong", "weak", "low"}
