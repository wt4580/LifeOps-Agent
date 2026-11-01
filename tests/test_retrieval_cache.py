"""Persistent-index regressions replacing the former full-corpus cache tests."""
from unittest.mock import Mock

import pytest
from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import sessionmaker

from app.src.common.config.knowledge_db_config import KnowledgeBase
from app.src.domain.entity.knowledge_entity import Document, DocumentChunk
from app.src.util.retrieval import retrieval as r
from app.src.util.retrieval.sqlite_fts import sync_index


@pytest.fixture
def knowledge(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'knowledge.db'}")
    KnowledgeBase.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    monkeypatch.setattr(r, "KnowledgeSessionLocal", factory)
    monkeypatch.setattr(r.settings, "rag_backend", "bm25")
    yield engine, factory
    engine.dispose()


def add(factory, content, path="a.txt", **kwargs):
    with factory() as db:
        row = DocumentChunk(file_path=path, chunk_text=content, chunk_hash=content, **kwargs)
        db.add(row)
        db.flush()
        ident = row.id
        db.commit()
        return ident


def test_existing_corpus_migrates_and_chinese_query_matches(knowledge):
    _, factory = knowledge
    ident = add(factory, "鸡胸肉富含蛋白质")
    hits = r.search_chunks("鸡胸肉")
    assert [hit.chunk_id for hit in hits] == [ident]
    assert hits[0].snippet == "鸡胸肉富含蛋白质"
    assert 0 < hits[0].score <= r._BM25_SCORE_MAX


def test_raw_sql_changes_and_rollback_keep_index_consistent(knowledge):
    engine, factory = knowledge
    add(factory, "apple")
    assert r.search_chunks("apple")
    with engine.begin() as conn:
        conn.execute(text("UPDATE document_chunks SET chunk_text='banana'"))
    assert not r.search_chunks("apple")
    assert r.search_chunks("banana")
    with pytest.raises(RuntimeError):
        with engine.begin() as conn:
            conn.execute(text("UPDATE document_chunks SET chunk_text='rollback'"))
            raise RuntimeError()
    assert r.search_chunks("banana")
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM document_chunks"))
    assert not r.search_chunks("banana")
    add(factory, "pear")
    assert r.search_chunks("pear")


def test_restart_uses_persisted_index_without_document_retokenization(knowledge, monkeypatch):
    engine, factory = knowledge
    add(factory, "鸡胸肉蛋白质")
    assert r.search_chunks("鸡胸肉")
    restarted = create_engine(engine.url)
    monkeypatch.setattr(r, "KnowledgeSessionLocal", sessionmaker(bind=restarted))
    tokenizer = Mock(wraps=r._tokenize)
    monkeypatch.setattr(r, "_tokenize", tokenizer)
    try:
        assert r.search_chunks("鸡胸肉")
        assert [call.args[0] for call in tokenizer.call_args_list] == ["鸡胸肉"]
    finally:
        restarted.dispose()


def test_queries_never_load_or_rebuild_full_corpus(knowledge, monkeypatch):
    engine, factory = knowledge
    for i in range(30):
        add(factory, f"irrelevant {i}", f"{i}.txt")
    add(factory, "needle")
    sync_index(engine, r._tokenize)
    monkeypatch.setattr(r, "_collect_chunks", Mock(side_effect=AssertionError("full corpus")))
    monkeypatch.setattr(r, "BM25Okapi", Mock(side_effect=AssertionError("in-memory index")))
    statements = []
    def track(conn, cursor, statement, *args):
        if statement.startswith("SELECT document_chunks."):
            statements.append(statement)
    event.listen(engine, "before_cursor_execute", track)
    try:
        assert len(r.search_chunks("needle", top_k=1)) == 1
        assert len(statements) == 1 and "WHERE document_chunks.id IN" in statements[0]
    finally:
        event.remove(engine, "before_cursor_execute", track)


def test_fts_raw_rank_survives_display_conversion(knowledge):
    _, factory = knowledge
    low = add(factory, "alpha beta " + "noise " * 80)
    high = add(factory, "alpha beta " * 10, "high.txt")
    for i in range(20):
        add(factory, f"irrelevant {i}", f"{i}.txt")
    hits = r.search_chunks("alpha beta", top_k=2)
    assert [hit.chunk_id for hit in hits] == [high, low]
    assert hits[0].score >= hits[1].score


def test_partial_summary_filter_and_summary_updates(knowledge):
    engine, factory = knowledge
    for path in ("matched", "other", "pending"):
        add(factory, "alpha text", path)
    with factory() as db:
        db.add_all([Document(file_path="matched", summary="alpha"),
                    Document(file_path="other", summary="beta")])
        db.commit()
    assert {hit.path for hit in r.search_chunks("alpha")} == {"matched", "pending"}
    with engine.begin() as conn:
        conn.execute(text("UPDATE documents SET summary='alpha' WHERE file_path='other'"))
    assert len(r.search_chunks("alpha")) == 3


def test_personal_query_retains_guidelines_as_reference_material(knowledge):
    _, factory = knowledge
    add(factory, "我今天吃了鸡胸肉", "guide", doc_topic="guideline")
    assert r.search_chunks("我今天吃了什么")
    add(factory, "我今天吃了米饭", "diary", doc_topic="personal_log")
    assert {hit.path for hit in r.search_chunks("我今天吃了什么")} == {"diary", "guide"}


def test_query_syntax_is_literal_and_identifier_tokens_are_preserved(knowledge):
    _, factory = knowledge
    add(factory, "gpt-4.1", "model")
    add(factory, "gpt other", "other")
    assert [hit.path for hit in r.search_chunks('"gpt-4.1" OR')] == ["model"]
    assert r.search_chunks("！ ？") == []


def test_parallel_refreshes_from_separate_engines(knowledge):
    from concurrent.futures import ThreadPoolExecutor
    from app.src.util.retrieval.sqlite_fts import ensure_index
    engine, factory = knowledge
    ensure_index(engine)
    for i in range(12):
        add(factory, f"蛋白质 {i}", str(i))
    second = create_engine(engine.url)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(sync_index, item, r._tokenize) for item in (engine, second)]
            for future in futures:
                future.result()
        with engine.connect() as conn:
            assert conn.execute(text("SELECT COUNT(*) FROM lexical_chunk")).scalar_one() == 12
            assert conn.execute(text("SELECT COUNT(*) FROM lexical_chunk_dirty")).scalar_one() == 0
        assert len(r.search_chunks("蛋白质", top_k=12)) == 12
    finally:
        second.dispose()


@pytest.mark.parametrize("backend", ["bm25", "langchain"])
def test_non_hybrid_query_does_not_call_unused_rewriter(monkeypatch, backend):
    monkeypatch.setattr(r.settings, "rag_backend", backend)
    monkeypatch.setattr(r, "_fts_recall", lambda *args: [])
    monkeypatch.setattr(r, "_search_langchain", lambda *args, **kwargs: [])
    rewrite = Mock(side_effect=AssertionError("unused model request"))
    monkeypatch.setattr(r, "_rewrite_query", rewrite)
    assert r.search_chunks("question") == []
    rewrite.assert_not_called()


def test_hybrid_still_uses_rewritten_vector_query(monkeypatch):
    monkeypatch.setattr(r.settings, "rag_backend", "hybrid")
    monkeypatch.setattr(r, "_fts_recall", lambda *args: [])
    monkeypatch.setattr(r, "_rewrite_query", lambda *args, **kwargs: "expanded")
    vector = Mock(return_value=[])
    monkeypatch.setattr(r, "_vector_recall", vector)
    monkeypatch.setattr(r, "_rerank_fused", lambda *args, **kwargs: [])
    r.search_chunks("question")
    assert [call.args[0] for call in vector.call_args_list] == ["question", "expanded"]

@pytest.mark.parametrize('entry', ['recall', 'rerank'])
def test_vector_candidates_must_match_current_sqlite_evidence(knowledge, monkeypatch, entry):
    import sys
    from types import SimpleNamespace
    _, factory = knowledge
    ident = add(factory, 'current text', path='a.txt', source_type='guide')
    def hit(chunk_id=ident, path='a.txt', snippet='current text', page=None):
        return SimpleNamespace(chunk_id=chunk_id, path=path, snippet=snippet, page=page,
            score=0.8, reason='vector', source_type='personal_log', doc_topic='stale',
            section_title='stale', section_path='stale')
    hits = [hit(snippet='deleted old text'), hit(chunk_id=ident+99), hit(path='other.txt'),
            hit(page=99), hit(chunk_id=None), hit(chunk_id=str(ident)), hit()]
    backend = SimpleNamespace(vector_recall=Mock(return_value=hits), search_with_rerank=Mock(return_value=hits))
    monkeypatch.setitem(sys.modules, 'app.src.util.retrieval.rag_langchain', backend)
    result = r._vector_recall('query', 10) if entry == 'recall' else r._search_langchain('query', 10)
    assert len(result) == 1
    assert result[0].snippet == 'current text'
    assert result[0].source_type == 'guide'
    assert result[0].doc_topic is None
    assert result[0].section_title is None


def test_stale_vector_only_results_fall_back_to_current_lexical_index(knowledge, monkeypatch):
    import sys
    from types import SimpleNamespace
    _, factory = knowledge
    ident = add(factory, 'current protein recommendation', path='a.txt')
    hit = SimpleNamespace(chunk_id=ident, path='a.txt', page=None, snippet='deleted recommendation')
    monkeypatch.setattr(r.settings, 'rag_backend', 'langchain')
    monkeypatch.setitem(sys.modules, 'app.src.util.retrieval.rag_langchain',
        SimpleNamespace(search_with_rerank=Mock(return_value=[hit])))
    results = r.search_chunks('protein', top_k=3)
    assert len(results) == 1
    assert results[0].snippet == 'current protein recommendation'

@pytest.mark.parametrize('merge', [r._merge_rank, r._rrf_fuse])
def test_fusion_preserves_distinct_chunk_identity(merge):
    first = r.Citation(path='a.txt', page=None, snippet='same text', score=40, chunk_id=1, section_path='first')
    second = r.Citation(path='a.txt', page=None, snippet='same text', score=45, chunk_id=2, section_path='second')
    duplicate = r.Citation(path='a.txt', page=None, snippet='same text', score=50, chunk_id=1, section_path='first')
    result = merge([first, second], [duplicate], top_k=5)
    assert {hit.chunk_id for hit in result} == {1, 2}
    assert len(result) == 2


def test_rerank_restores_correct_section_for_identical_text(monkeypatch):
    from app.src.util.retrieval import rag_langchain
    monkeypatch.setattr(r.settings, 'rag_use_rerank', True)
    def rerank(query, hits, top_k):
        return list(reversed(hits))
    monkeypatch.setattr(rag_langchain, 'rerank_hits', rerank)
    citations = [r.Citation(path='a.txt', page=None, snippet='same text', score=40, chunk_id=i, section_path=str(i)) for i in [1, 2]]
    result = r._rerank_fused('query', citations, top_k=2)
    assert [hit.chunk_id for hit in result] == [2, 1]
    assert [hit.section_path for hit in result] == ['2', '1']
