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
