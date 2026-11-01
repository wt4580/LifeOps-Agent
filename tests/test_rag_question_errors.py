from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from fastapi import HTTPException
from app.src.controller import rag_controller as module
from app.src.domain.dto.rag_dto import AskRequest
from app.src.common.harness.context import TurnLimitExceeded


def test_knowledge_question_failure_redacts_dependency_details(monkeypatch, caplog):
    secret = 'private-database-path-and-key'
    monkeypatch.setattr(module.rag_service, 'ask_question', Mock(side_effect=RuntimeError(secret)))
    with pytest.raises(HTTPException) as caught:
        module.ask_question(AskRequest(question='question'), SimpleNamespace(state=SimpleNamespace(username='alice')))
    assert caught.value.status_code == 500
    assert secret not in str(caught.value.detail) + caplog.text
    assert 'RuntimeError' in caplog.text


def test_knowledge_question_budget_returns_no_citations_and_budget_trace(monkeypatch):
    monkeypatch.setattr(module.rag_service, 'ask_question', Mock(side_effect=TurnLimitExceeded('budget')))
    response = module.ask_question(AskRequest(question='question'), SimpleNamespace(state=SimpleNamespace(username='alice')))
    assert response.result['citations'] == []
    summary = response.result['trace']['steps'][0]['output']
    assert summary['status'] == 'budget_exceeded'

def test_question_capacity_rejects_before_model_and_recovers(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    import threading
    entered, release = threading.Event(), threading.Event()
    monkeypatch.setattr(module, '_ASK_SLOTS', threading.BoundedSemaphore(1))
    def ask(**kwargs):
        entered.set()
        assert release.wait(5)
        return {'answer': 'ok', 'citations': []}
    model = Mock(side_effect=ask)
    monkeypatch.setattr(module.rag_service, 'ask_question', model)
    req = AskRequest(question='question')
    identity = SimpleNamespace(state=SimpleNamespace(username='alice'))
    with ThreadPoolExecutor(max_workers=1) as pool:
        active = pool.submit(module.ask_question, req, identity)
        assert entered.wait(2)
        try:
            with pytest.raises(HTTPException) as caught:
                module.ask_question(req, identity)
            assert caught.value.status_code == 503
            assert caught.value.headers['Retry-After'] == '1'
            assert model.call_count == 1
        finally:
            release.set()
        assert active.result(5).result['answer'] == 'ok'
    assert module.ask_question(req, identity).result['answer'] == 'ok'


def test_question_failure_releases_capacity(monkeypatch):
    import threading
    slots = threading.BoundedSemaphore(1)
    monkeypatch.setattr(module, '_ASK_SLOTS', slots)
    monkeypatch.setattr(module.rag_service, 'ask_question', Mock(side_effect=RuntimeError('failed')))
    with pytest.raises(HTTPException):
        module.ask_question(AskRequest(question='question'), SimpleNamespace(state=SimpleNamespace(username='alice')))
    assert slots.acquire(blocking=False)
    slots.release()

def test_http_question_overload_preserves_retry_header_and_recovers(monkeypatch):
    import threading
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    app = FastAPI()
    @app.middleware('http')
    async def identity(request, call_next):
        request.state.username = 'alice'
        return await call_next(request)
    app.include_router(module.router)
    slots = threading.BoundedSemaphore(1)
    monkeypatch.setattr(module, '_ASK_SLOTS', slots)
    model = Mock(return_value={'answer': 'current answer', 'citations': []})
    monkeypatch.setattr(module.rag_service, 'ask_question', model)
    assert slots.acquire(blocking=False)
    with TestClient(app) as client:
        try:
            response = client.post('/api/ask', json={'question': 'question'})
            assert response.status_code == 503
            assert response.headers['retry-after'] == '1'
            model.assert_not_called()
        finally:
            slots.release()
        response = client.post('/api/ask', json={'question': 'question'})
        assert response.status_code == 200
        assert response.json()['result']['answer'] == 'current answer'
        assert response.json()['result']['trace']['steps'][0]['output']['status'] == 'success'
    model.assert_called_once_with(question='question', session_id='alice')
