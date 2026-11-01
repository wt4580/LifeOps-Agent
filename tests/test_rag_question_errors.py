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
