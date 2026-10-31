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

@pytest.mark.parametrize("response", [
    '{"items":[{"category":"diet","title":"private-memory-7834"}]}',
    '{"items":[{"category":"diet","title":"event","amount":"private-memory-7834"}]}',
])
def test_personal_memory_logs_exclude_extracted_content(response, monkeypatch, caplog):
    import logging
    from app.src.agent.memory import personal_memory as module
    monkeypatch.setattr(module, "chat_completion", Mock(side_effect=[response, "private-memory-7834"]))
    with caplog.at_level(logging.INFO):
        module.extract_personal_events("private-user-input-7925", [], now_iso="2026-06-18T12:00:00")
    assert "private-memory-7834" not in caplog.text
    assert "private-user-input-7925" not in caplog.text
    assert "context=personal_event_extraction" in caplog.text or "personal_event_extraction" in caplog.text


@pytest.mark.parametrize("response", ['[]', 'null', '42', '{"reason":"private-memory-7834"}', 'private-memory-7834'])
def test_invalid_diagnosis_safely_skips_without_logging_content(response, monkeypatch, caplog):
    from app.src.agent.memory import personal_memory as module
    monkeypatch.setattr(module, "chat_completion", Mock(return_value=response))
    assert module._diagnose_failure("personal_event_extraction", "error", "output", {}, None) == "insufficient_data"
    assert "private-memory-7834" not in caplog.text

@pytest.mark.parametrize("patch", [
    {"category": "invented"}, {"importance": "critical"}, {"title": "   "},
    {"title": "x" * 256}, {"amount": float("nan")}, {"amount": float("inf")},
    {"tags": ["tag"] * 21},
])
def test_personal_event_rejects_invalid_model_values(patch):
    from app.src.domain.dto.memory_dto import PersonalEventItem
    with pytest.raises(ValidationError):
        PersonalEventItem.model_validate({"category": "diet", "title": "event", **patch})


def test_personal_event_batch_bound_and_signed_amount():
    from app.src.domain.dto.memory_dto import PersonalEventItem, PersonalEventExtraction
    event = PersonalEventItem(category="finance", title="refund", amount=-10)
    assert event.amount == -10
    with pytest.raises(ValidationError):
        PersonalEventExtraction(items=[event] * 21)


@pytest.mark.parametrize("patch", [
    {"height_cm": 0}, {"weight_kg": -1}, {"age": -1},
    {"height_cm": float("inf")}, {"weight_kg": float("nan")},
])
def test_profile_rejects_nonfinite_or_negative_measurements(patch):
    from app.src.domain.dto.memory_dto import ProfileFactPatch
    with pytest.raises(ValidationError):
        ProfileFactPatch.model_validate(patch)

@pytest.mark.parametrize('function,response', [('extract_personal_events', '{"items":[]}'), ('extract_profile_facts', '{}')])
def test_personal_fact_extractors_only_receive_user_testimony(monkeypatch, function, response):
    import json
    from app.src.agent.memory import personal_memory as module
    model = Mock(return_value=response)
    monkeypatch.setattr(module, 'chat_completion', model)
    getattr(module, function)('我没有吃蛋糕', [
        {'role': 'assistant', 'content': 'private assistant invented cake event'},
        {'role': 'system', 'content': 'private generated summary'},
        {'role': 'tool', 'content': 'private guide recommends cake'},
        {'role': 'user', 'content': '我昨天吃了米饭'},
    ], now_iso='2026-06-18T12:00:00')
    messages = model.call_args.args[0]
    payload = json.loads(messages[1]['content'])
    assert payload['user_message'] == '我没有吃蛋糕'
    assert payload['recent_dialogue'] == [{'role': 'user', 'content': '我昨天吃了米饭'}]
    assert 'private' not in messages[1]['content']


@pytest.mark.parametrize('function', ['extract_personal_events', 'extract_profile_facts'])
@pytest.mark.parametrize('current', ['', 'x' * 6001])
def test_empty_or_oversized_personal_statement_skips_without_truncation(monkeypatch, function, current):
    from app.src.agent.memory import personal_memory as module
    model = Mock()
    monkeypatch.setattr(module, 'chat_completion', model)
    result = getattr(module, function)(current, [{'role': 'user', 'content': 'old event'}], now_iso='2026-06-18T12:00:00')
    assert result is not None
    model.assert_not_called()


def test_personal_evidence_budget_preserves_whole_current_statement():
    from app.src.agent.memory.personal_memory import _personal_evidence
    current = '我没有' + 'x' * 5990
    payload = _personal_evidence(current, [{'role': 'user', 'content': 'old ' * 100}], 'now')
    assert payload['user_message'] == current
    assert payload['recent_dialogue'] == []
    payload = _personal_evidence('current', [{'role': 'user', 'content': str(i)} for i in range(20)], 'now')
    assert len(payload['recent_dialogue']) == 6
    assert payload['recent_dialogue'][-1]['content'] == '19'

def test_background_dialogue_is_bounded_by_triggering_message(tmp_path, monkeypatch):
    from datetime import datetime
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from app.src.common.config.db_config import Base
    from app.src.domain.entity.chat_entity import ChatMessage
    from app.src.service import chat_service as module
    engine = create_engine(f"sqlite:///{tmp_path / 'dialogue.db'}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    monkeypatch.setattr(module, 'SessionLocal', factory)
    stamp = datetime(2026, 6, 18, 12)
    try:
        with factory() as db:
            prior = ChatMessage(session_id='alice', role='user', content='prior', created_at=stamp)
            db.add(prior)
            db.flush()
            current = ChatMessage(session_id='alice', role='user', content='current', created_at=stamp)
            db.add(current)
            db.flush()
            cutoff = current.id
            db.add_all([ChatMessage(session_id='alice', role='user', content='later', created_at=stamp),
                        ChatMessage(session_id='bob', role='user', content='private', created_at=stamp)])
            db.commit()
        rows = module.ChatService()._load_recent_dialogue('alice', through_message_id=cutoff)
        assert [r['content'] for r in rows] == ['prior', 'current']
        assert rows[-1]['created_at'] == '2026-06-18T12:00:00+00:00'
        assert rows[-1]['message_id'] == cutoff
    finally:
        engine.dispose()


def test_background_worker_uses_cutoff_and_observation_time(monkeypatch):
    from types import SimpleNamespace
    from contextlib import contextmanager
    from app.src.service import chat_service as module
    queued = []
    monkeypatch.setattr(module, '_POST_GRAPH_EXECUTOR', SimpleNamespace(submit=queued.append))
    monkeypatch.setattr(module, 'run_learn', Mock())
    @contextmanager
    def context(*args, **kwargs):
        yield
    monkeypatch.setattr(module, 'background_context', context)
    extract = Mock(side_effect=KeyboardInterrupt())
    monkeypatch.setattr(module, 'extract_profile_facts', extract)
    service = module.ChatService()
    loader = Mock(return_value=[{'role':'user', 'content':'current', 'message_id':42, 'created_at':'2026-06-18T12:00:00+00:00'}])
    monkeypatch.setattr(service, '_load_recent_dialogue', loader)
    service._spawn_post_graph('alice','alice','current',{'recent_dialogue':[{'role':'user','content':'stale'}]},'answer',lambda _:None,message_id=42)
    with pytest.raises(KeyboardInterrupt):
        queued[0]()
    loader.assert_called_once_with('alice', limit=12, through_message_id=42)
    from datetime import datetime
    expected_time = datetime.fromisoformat('2026-06-18T12:00:00+00:00').astimezone().replace(tzinfo=None).isoformat(timespec='seconds')
    assert extract.call_args.kwargs['now_iso'] == expected_time
    assert extract.call_args.args[1][0]['content'] == 'current'
