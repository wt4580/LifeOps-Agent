from unittest.mock import Mock

from app.src.agent.memory import memory_extractor as module


def test_generated_assistant_facts_never_enter_extraction_input(monkeypatch):
    model = Mock(return_value='{"items":[]}')
    monkeypatch.setattr(module, "chat_completion", model)
    module.extract_memory_from_dialogue([
        {"role": "assistant", "content": "user ate a cake"},
        {"role": "user", "content": "I did not eat cake"},
        {"role": "system", "content": "generated summary"},
    ])
    messages = model.call_args.args[0]
    assert messages[1:] == [{"role": "user", "content": "I did not eat cake"}]


def test_assistant_only_dialogue_does_not_call_model(monkeypatch):
    model = Mock()
    monkeypatch.setattr(module, "chat_completion", model)
    assert module.extract_memory_from_dialogue([{"role": "assistant", "content": "suggestion"}]).items == []
    model.assert_not_called()


def test_evidence_limit_does_not_slice_user_statement(monkeypatch):
    model = Mock(return_value='{"items":[]}')
    monkeypatch.setattr(module, "chat_completion", model)
    module.extract_memory_from_dialogue([{"role": "user", "content": "x" * 6001},
                                        {"role": "user", "content": "recent statement"}])
    assert model.call_args.args[0][1:] == [{"role": "user", "content": "recent statement"}]


import pytest
from pydantic import ValidationError
from app.src.domain.dto.memory_dto import MemoryItem, MemoryExtraction


@pytest.mark.parametrize("patch", [{"kind": "invented"}, {"title": "   "}, {"title": "x" * 256},
    {"confidence": float("nan")}, {"insight_type": "verified"}, {"insight_type": "inferred", "confidence": 1.0}])
def test_invalid_memory_contract_is_rejected(patch):
    with pytest.raises(ValidationError):
        MemoryItem.model_validate({"kind": "event", "title": "event", **patch})


def test_valid_inference_remains_uncertain_and_batch_is_bounded():
    item = MemoryItem(kind="pattern", title="pattern", insight_type="inferred", confidence=0.7)
    assert item.insight_type == "inferred" and item.confidence == 0.7
    with pytest.raises(ValidationError):
        MemoryExtraction(items=[item] * 21)


def test_candidate_api_preserves_uncertainty_and_identity(tmp_path, monkeypatch):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from app.src.common.config.db_config import Base
    from app.src.domain.entity.chat_entity import MemoryCandidate
    from app.src.service import memory_service as module
    engine = create_engine(f"sqlite:///{(tmp_path / 'memory.db').as_posix()}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    monkeypatch.setattr(module, "SessionLocal", factory)
    try:
        with factory() as db:
            db.add_all([MemoryCandidate(session_id="alice", kind="pattern", title="uncertain", confidence=0.6, insight_type="inferred"),
                        MemoryCandidate(session_id="bob", kind="event", title="private")])
            db.commit()
        rows = module.MemoryService().get_memory_candidates("alice")
        assert len(rows) == 1
        assert rows[0]["confidence"] == 0.6
        assert rows[0]["insight_type"] == "inferred"
        assert rows[0]["verified"] is False
        assert rows[0]["created_at"]
    finally:
        engine.dispose()
