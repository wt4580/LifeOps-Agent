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
    for _ in range(100):
        module._schedule_document_summaries()
    assert executor.submit.call_count == 1
    first.set_result(None)
    module._schedule_document_summaries()
    assert executor.submit.call_count == 2


@pytest.mark.parametrize('error,status', [(module.IndexingBusyError('busy'),409), (IngestionLimitExceeded('limit'),422)])
def test_index_api_reports_busy(error, status, monkeypatch):
    from fastapi import HTTPException
    from app.src.controller import rag_controller
    monkeypatch.setattr(rag_controller.rag_service, 'index_docs', Mock(side_effect=error))
    with pytest.raises(HTTPException) as caught:
        rag_controller.index_documents()
    assert caught.value.status_code == status
