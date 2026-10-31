"""Admission control for expensive document indexing and summary jobs."""
from concurrent.futures import Future, ThreadPoolExecutor
import threading
from unittest.mock import Mock
import pytest
from app.src.util.retrieval import retrieval as module
from app.src.util.ingest.limits import IngestionLimitExceeded


def test_overlapping_index_rejected_before_work_and_lock_released(monkeypatch):
    entered, release = threading.Event(), threading.Event()
    def build(**kwargs):
        entered.set()
        assert release.wait(5)
        return kwargs
    monkeypatch.setattr(module, '_build_document_index', build)
    with ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(module.index_documents, reset=True)
        assert entered.wait(5)
        try:
            with pytest.raises(module.IndexingBusyError):
                module.index_documents()
        finally:
            release.set()
        assert first.result(timeout=5)['reset'] is True
    assert module.index_documents()['reset'] is False


def test_index_failure_releases_admission_lock(monkeypatch):
    monkeypatch.setattr(module, '_build_document_index', Mock(side_effect=ValueError('failed')))
    with pytest.raises(ValueError):
        module.index_documents()
    monkeypatch.setattr(module, '_build_document_index', Mock(return_value={'ok': True}))
    assert module.index_documents() == {'ok': True}


def test_duplicate_summary_jobs_do_not_queue(monkeypatch):
    first, second = Future(), Future()
    executor = Mock()
    executor.submit.side_effect = [first, second]
    monkeypatch.setattr(module, '_SUMMARY_EXECUTOR', executor)
    monkeypatch.setattr(module, '_SUMMARY_FUTURE', None)
    monkeypatch.setattr(module, '_SUMMARY_PENDING', False)
    for _ in range(100):
        module._schedule_document_summaries()
    assert executor.submit.call_count == 1
    first.set_result(None)
    assert executor.submit.call_count == 2
    second.set_result(None)
    assert executor.submit.call_count == 2
    assert module._SUMMARY_FUTURE is None


@pytest.mark.parametrize('error,status', [(module.IndexingBusyError('busy'),409), (IngestionLimitExceeded('limit'),422)])
def test_index_api_reports_busy(error, status, monkeypatch):
    from fastapi import HTTPException
    from app.src.controller import rag_controller
    monkeypatch.setattr(rag_controller.rag_service, 'index_docs', Mock(side_effect=error))
    with pytest.raises(HTTPException) as caught:
        rag_controller.index_documents()
    assert caught.value.status_code == status

def test_running_summary_requests_coalesce_into_one_followup(monkeypatch):
    entered, release, followup = threading.Event(), threading.Event(), threading.Event()
    calls = []
    def worker():
        calls.append(True)
        if len(calls) == 1:
            entered.set()
            assert release.wait(5)
        else:
            followup.set()
    monkeypatch.setattr(module, '_SUMMARY_FUTURE', None)
    monkeypatch.setattr(module, '_SUMMARY_PENDING', False)
    monkeypatch.setattr(module, '_generate_document_summaries', worker)
    with ThreadPoolExecutor(max_workers=1) as executor:
        monkeypatch.setattr(module, '_SUMMARY_EXECUTOR', executor)
        module._schedule_document_summaries()
        assert entered.wait(5)
        try:
            for _ in range(100):
                module._schedule_document_summaries()
        finally:
            release.set()
        assert followup.wait(5)
    assert len(calls) == 2
    assert module._SUMMARY_FUTURE is None


def test_already_completed_summary_future_does_not_deadlock(monkeypatch):
    completed = Future()
    completed.set_result(None)
    monkeypatch.setattr(module, '_SUMMARY_FUTURE', None)
    monkeypatch.setattr(module, '_SUMMARY_PENDING', False)
    monkeypatch.setattr(module, '_SUMMARY_EXECUTOR', Mock(submit=Mock(return_value=completed)))
    module._schedule_document_summaries()
    assert module._SUMMARY_FUTURE is None

def test_index_api_does_not_expose_storage_error(monkeypatch, caplog):
    from fastapi import HTTPException
    from app.src.controller import rag_controller
    secret = 'private-path-and-sql-8312'
    monkeypatch.setattr(rag_controller.rag_service, 'index_docs', Mock(side_effect=RuntimeError(secret)))
    with pytest.raises(HTTPException) as caught:
        rag_controller.index_documents()
    assert caught.value.status_code == 500
    assert secret not in str(caught.value.detail) + caplog.text
    assert 'RuntimeError' in caplog.text


def test_vector_failure_reports_degradation_without_raw_exception(monkeypatch, caplog):
    import sys
    from types import SimpleNamespace
    secret = 'private-vector-key-9527'
    vector = Mock(side_effect=RuntimeError(secret))
    monkeypatch.setitem(sys.modules, 'app.src.util.retrieval.rag_langchain',
                        SimpleNamespace(build_vector_index=vector, _load_vector_store=Mock()))
    monkeypatch.setattr(module.settings, 'rag_backend', 'hybrid')
    monkeypatch.setattr(module, '_index_documents_sqlite', Mock(return_value=(3, 0)))
    monkeypatch.setattr(module, '_sync_document_records', Mock())
    factory = Mock()
    factory.return_value.__enter__ = Mock(return_value=Mock())
    factory.return_value.__exit__ = Mock(return_value=False)
    factory.return_value.__enter__.return_value.execute.return_value.scalar_one.return_value = 0
    monkeypatch.setattr(module, 'KnowledgeSessionLocal', factory)
    result = module.index_documents()
    assert result['inserted'] == 3
    assert result['indexing_status'] == 'degraded'
    assert result['vector_error'] == 'vector_index_build_failed'
    assert secret not in str(result) + caplog.text
