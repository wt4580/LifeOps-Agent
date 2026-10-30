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
