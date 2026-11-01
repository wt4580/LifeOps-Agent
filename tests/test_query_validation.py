import pytest
from pydantic import ValidationError
from app.src.domain.dto.rag_dto import AskRequest
from app.src.domain.vo.chat_vo import ChatVo


@pytest.mark.parametrize('value', ['', '   ', '\n\t', 'x' * 1001, 123, True, [], {}])
@pytest.mark.parametrize('model,field', [(AskRequest, 'question'), (ChatVo, 'message')])
def test_invalid_query_values_are_validation_errors(model, field, value):
    with pytest.raises(ValidationError):
        model.model_validate({field: value})


@pytest.mark.parametrize('model,field', [(AskRequest, 'question'), (ChatVo, 'message')])
def test_valid_queries_trim_whitespace_and_accept_boundary(model, field):
    assert getattr(model.model_validate({field: ' hello '}), field) == 'hello'
    assert len(getattr(model.model_validate({field: 'x' * 1000}), field)) == 1000

@pytest.mark.parametrize('value', ['', '  ', 'x' * 1001, 123, True, [], {}])
@pytest.mark.parametrize('path,field', [('/api/chat/stream', 'message'), ('/api/ask', 'question')])
def test_http_validation_rejects_before_service_or_database(monkeypatch, path, field, value):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from unittest.mock import Mock
    from app.src.controller import chat_controller, rag_controller
    from app.src.service import chat_service
    app = FastAPI()
    @app.middleware('http')
    async def identity(request, call_next):
        request.state.username = 'validation-user'
        return await call_next(request)
    app.include_router(chat_controller.router)
    app.include_router(rag_controller.router)
    stream = Mock(side_effect=AssertionError('invalid input reached execution'))
    ask = Mock(side_effect=AssertionError('invalid input reached retrieval'))
    database = Mock(side_effect=AssertionError('invalid input reached persistence'))
    monkeypatch.setattr(chat_controller.chat_service, 'chat_stream', stream)
    monkeypatch.setattr(rag_controller.rag_service, 'ask_question', ask)
    monkeypatch.setattr(chat_service, 'SessionLocal', database)
    with TestClient(app) as client:
        response = client.post(path, json={field: value})
    assert response.status_code == 422
    stream.assert_not_called()
    ask.assert_not_called()
    database.assert_not_called()
