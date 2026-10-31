"""Retrieval operational logs must not become a copy of private evidence."""
import logging
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from app.src.util.retrieval import retrieval, rag_langchain
from app.src.service import rag_service

SECRET = 'private-rag-evidence-7832'


@pytest.mark.parametrize('output', [SECRET, SECRET * 20, ' '.join([SECRET] * 21), RuntimeError(SECRET)])
def test_query_rewrite_logs_do_not_include_input_or_output(monkeypatch, caplog, output):
    model = Mock(side_effect=output) if isinstance(output, Exception) else Mock(return_value=output)
    monkeypatch.setattr(retrieval, 'chat_completion', model)
    with caplog.at_level(logging.INFO):
        retrieval._rewrite_query(SECRET)
    assert SECRET not in caplog.text
    assert 'rewrite' in caplog.text


def test_rerank_logs_exclude_paths_snippets_and_reasons(caplog):
    hit = SimpleNamespace(snippet=SECRET, path=SECRET, reason=SECRET, score=.8, page=1, chunk_id=1)
    with caplog.at_level(logging.INFO):
        rag_langchain._log_rerank_results('vector', SECRET, [hit])
    assert SECRET not in caplog.text
    assert 'rank=1' in caplog.text and 'snippet_chars=' in caplog.text


def test_model_source_logs_exclude_local_paths(tmp_path, caplog):
    path = tmp_path / SECRET
    path.mkdir()
    with caplog.at_level(logging.INFO):
        source, selected = rag_langchain._pick_model_source(local_path=str(path), remote_model=SECRET, model_label='embedding')
    assert source == 'local' and selected == str(path)
    assert SECRET not in caplog.text


def test_rag_service_logs_exclude_question_and_index_directory(monkeypatch, caplog):
    monkeypatch.setattr(rag_service, 'search_chunks', Mock(return_value=[]))
    monkeypatch.setattr(rag_service, 'rag_answer', Mock(return_value='answer'))
    monkeypatch.setattr(rag_service, 'index_documents', Mock(return_value={'vector_dir': SECRET, 'inserted': 1}))
    with caplog.at_level(logging.INFO):
        service = rag_service.RAGService()
        service.ask_question(SECRET)
        service.search(SECRET)
        service.index_docs()
    assert SECRET not in caplog.text


@pytest.mark.parametrize('backend', ['bm25', 'hybrid'])
def test_search_summary_logs_exclude_rewritten_question(monkeypatch, caplog, backend):
    monkeypatch.setattr(retrieval.settings, 'rag_backend', backend)
    monkeypatch.setattr(retrieval, '_rewrite_query', Mock(return_value=SECRET))
    monkeypatch.setattr(retrieval, '_fts_recall', Mock(return_value=[]))
    monkeypatch.setattr(retrieval, '_vector_recall', Mock(return_value=[]))
    with caplog.at_level(logging.INFO):
        assert retrieval.search_chunks(SECRET) == []
    assert SECRET not in caplog.text
