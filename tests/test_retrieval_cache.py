from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import sessionmaker

from app.src.common.config.knowledge_db_config import KnowledgeBase
from app.src.domain.entity.knowledge_entity import DocumentChunk
from app.src.util.retrieval import retrieval as r


def chunk(index, content):
    return SimpleNamespace(id=index, file_path=f"{index}.txt", page=None, chunk_text=content)


def test_raw_bm25_rank_survives_display_score_saturation():
    rows = [chunk(0, "alpha beta gamma"), chunk(1, "alpha beta gamma " * 10)]
    rows += [chunk(i + 2, f"irrelevant text {i}") for i in range(20)]
    hits = r._bm25_recall(rows, "alpha beta gamma", 2, "bm25")
    assert [hit.chunk_id for hit in hits] == [1, 0]
    assert hits[0].score == hits[1].score == r._BM25_SCORE_MAX
    assert r._rrf_fuse(hits, top_k=1)[0].chunk_id == 1


def test_bm25_reuses_tokens_and_index_but_respects_filtered_corpus(monkeypatch):
    tokenizer = Mock(side_effect=lambda value: value.split())
    builds = []

    class Index:
        def __init__(self, corpus):
            builds.append(corpus)

    monkeypatch.setattr(r, "_tokenize", tokenizer)
    monkeypatch.setattr(r, "BM25Okapi", Index)
    first = r._bm25_for_texts(["alpha beta", "gamma"])
    assert r._bm25_for_texts(["alpha beta", "gamma"]) is first
    assert len(builds) == 1 and tokenizer.call_count == 2
    assert r._bm25_for_texts(["alpha beta"]) is not first
    assert len(builds) == 2 and tokenizer.call_count == 2


def test_corpus_snapshot_invalidates_for_committed_external_writes(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'knowledge.db'}")
    KnowledgeBase.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    monkeypatch.setattr(r, "KnowledgeSessionLocal", factory)
    with factory() as db:
        db.add(DocumentChunk(file_path="a", chunk_text="old", chunk_hash="one"))
        db.commit()
    queries = []

    @event.listens_for(engine, "before_cursor_execute")
    def track(conn, cursor, statement, *args):
        if statement.startswith("SELECT document_chunks."):
            queries.append(statement)

    try:
        assert r._collect_chunks()[0].chunk_text == "old"
        assert r._collect_chunks()[0].chunk_text == "old"
        assert len(queries) == 1
        # Bypass ORM events: database triggers must still invalidate the snapshot.
        with engine.begin() as conn:
            conn.execute(text("UPDATE document_chunks SET chunk_text='new'"))
        assert r._collect_chunks()[0].chunk_text == "new"
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM document_chunks"))
        assert r._collect_chunks() == []
        assert len(queries) == 3
        with factory() as db:
            db.add(DocumentChunk(file_path="b", chunk_text="inserted", chunk_hash="two"))
            db.commit()
        assert r._collect_chunks()[0].chunk_text == "inserted"
        assert len(queries) == 4
    finally:
        engine.dispose()


@pytest.mark.parametrize("backend", ["bm25", "langchain"])
def test_non_hybrid_query_does_not_call_unused_rewriter(monkeypatch, backend):
    monkeypatch.setattr(r.settings, "rag_backend", backend)
    monkeypatch.setattr(r, "_collect_chunks", lambda: [chunk(1, "alpha")])
    monkeypatch.setattr(r, "_apply_source_filter", lambda rows, query: rows)
    monkeypatch.setattr(r, "_apply_document_summary_filter", lambda rows, query: rows)
    monkeypatch.setattr(r, "_bm25_recall", lambda *args, **kwargs: [])
    monkeypatch.setattr(r, "_search_langchain", lambda *args, **kwargs: [])
    rewrite = Mock(side_effect=AssertionError("unused model request"))
    monkeypatch.setattr(r, "_rewrite_query", rewrite)
    assert r.search_chunks("question") == []
    rewrite.assert_not_called()


def test_hybrid_still_uses_rewritten_vector_query(monkeypatch):
    monkeypatch.setattr(r.settings, "rag_backend", "hybrid")
    monkeypatch.setattr(r, "_collect_chunks", lambda: [chunk(1, "alpha")])
    monkeypatch.setattr(r, "_apply_source_filter", lambda rows, query: rows)
    monkeypatch.setattr(r, "_apply_document_summary_filter", lambda rows, query: rows)
    monkeypatch.setattr(r, "_bm25_recall", lambda *args, **kwargs: [])
    monkeypatch.setattr(r, "_rewrite_query", lambda *args, **kwargs: "expanded")
    vector = Mock(return_value=[])
    monkeypatch.setattr(r, "_vector_recall", vector)
    monkeypatch.setattr(r, "_rerank_fused", lambda *args, **kwargs: [])
    r.search_chunks("question")
    assert [call.args[0] for call in vector.call_args_list] == ["question", "expanded"]
