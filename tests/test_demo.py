from datetime import datetime

import pytest
from pydantic import ValidationError

from app.src.demo.provider import decision, response_for


def test_demo_due_fields_follow_router_tool_contract():
    result = decision("明天下午3点开会", datetime(2026, 10, 27, 12))
    assert result["action"] == "propose_todo"
    assert result["args"]["due_date"] == "2026-10-28"
    assert result["args"]["due_time"] == "15:00:00"


def test_demo_weather_is_explicitly_fictional():
    answer = response_for([{"content": "{}"}], {"scenario": "query_weather_summarize"})
    assert "虚构天气" in answer and "20℃" in answer


def test_live_credentials_required_and_demo_seeding_is_explicit():
    from app.src.common.config.base_config import Settings

    with pytest.raises(ValidationError, match="Live mode requires"):
        Settings(_env_file=None, qwen_api_key="", demo_mode=False, demo_seed=False)
    with pytest.raises(ValidationError, match="DEMO_SEED"):
        Settings(_env_file=None, qwen_api_key="key", demo_mode=False, demo_seed=True)
    assert Settings(_env_file=None, qwen_api_key="", demo_mode=True).demo_mode


def test_normal_chat_loads_scoped_markdown(monkeypatch):
    from app.src.agent import graph_chat

    owners = []
    monkeypatch.setattr(graph_chat, "load_markdown_context", lambda session_id: owners.append(session_id) or "")
    monkeypatch.setattr(graph_chat, "chat_completion", lambda *args, **kwargs: "hello")
    answer, _ = graph_chat._build_normal_chat_answer({"session_id": "alice", "user_message": "hello"}, {})
    assert owners == ["alice"]
    assert answer == "hello"
