from __future__ import annotations

from unittest.mock import patch

import pytest

from app.src.agent.memory.memory_extractor import extract_memory_from_dialogue
from app.src.agent.memory.personal_memory import (
    ProactiveAdviceDecision,
    _normalize_event_time_or_now,
    decide_proactive_advice,
    extract_personal_events,
    extract_profile_facts,
)
from app.src.domain.dto.memory_dto import (
    PersonalEventExtraction,
    ProfileFactPatch,
)


class TestNormalizeEventTime:
    def test_valid_time(self):
        result = _normalize_event_time_or_now("2026-06-18T10:00:00", "2026-06-18T12:00:00")
        assert result == "2026-06-18T10:00:00"

    def test_invalid_time_falls_back(self):
        now = "2026-06-18T12:00:00"
        result = _normalize_event_time_or_now("not-a-time", now)
        assert result == now

    def test_none_falls_back(self):
        now = "2026-06-18T12:00:00"
        result = _normalize_event_time_or_now(None, now)
        assert result == now

    def test_empty_string_falls_back(self):
        now = "2026-06-18T12:00:00"
        result = _normalize_event_time_or_now("", now)
        assert result == now


class TestExtractPersonalEvents:
    def test_returns_none_on_invalid_json(self):
        with patch(
            "app.src.agent.memory.personal_memory.chat_completion",
            return_value="not json",
        ):
            result = extract_personal_events("吃了火锅", [], now_iso="2026-06-18T12:00:00")
            assert result is None

    def test_returns_extraction_on_valid_json(self):
        mock_response = (
            '{"items": [{"category": "diet", "title": "吃了火锅",'
            ' "event_time": "2026-06-18T12:00:00", "tags": ["油腻"]}]}'
        )
        with patch(
            "app.src.agent.memory.personal_memory.chat_completion",
            return_value=mock_response,
        ):
            result = extract_personal_events("吃了火锅", [], now_iso="2026-06-18T12:00:00")
            assert isinstance(result, PersonalEventExtraction)
            assert len(result.items) == 1
            assert result.items[0].category == "diet"

    def test_empty_items(self):
        with patch(
            "app.src.agent.memory.personal_memory.chat_completion",
            return_value='{"items": []}',
        ):
            result = extract_personal_events("你好", [], now_iso="2026-06-18T12:00:00")
            assert isinstance(result, PersonalEventExtraction)
            assert len(result.items) == 0


class TestExtractProfileFacts:
    def test_returns_patch_on_valid_json(self):
        mock_response = (
            '{"height_cm": 175, "weight_kg": 70,'
            ' "preferences": ["爱运动"], "conditions": [], "notes": null}'
        )
        with patch(
            "app.src.agent.memory.personal_memory.chat_completion",
            return_value=mock_response,
        ):
            result = extract_profile_facts("我身高175体重70", [], now_iso="2026-06-18T12:00:00")
            assert isinstance(result, ProfileFactPatch)
            assert result.height_cm == 175

    def test_returns_none_on_invalid_json(self):
        with patch(
            "app.src.agent.memory.personal_memory.chat_completion",
            return_value="bad json",
        ):
            result = extract_profile_facts("你好", [], now_iso="2026-06-18T12:00:00")
            assert result is None


class TestDecideProactiveAdvice:
    def test_no_events_no_todos(self):
        decision = decide_proactive_advice(
            user_message="你好",
            assistant_answer="你好",
            recent_events=[],
            upcoming_todos=[],
            profile_context="",
            now_iso="2026-06-18T12:00:00",
            threshold=0.62,
            session_id="s1",
        )
        assert isinstance(decision, ProactiveAdviceDecision)
        assert not decision.should_add

    def test_with_events_no_advice_below_threshold(self):
        with patch(
            "app.src.agent.memory.personal_memory.chat_completion",
            return_value=(
                '{"score": 0.3, "should_add": false,'
                ' "advice": null, "reasons": []}'
            ),
        ):
            decision = decide_proactive_advice(
                user_message="我最近运动很多",
                assistant_answer="不错",
                recent_events=[{"title": "跑步5公里", "category": "activity"}],
                upcoming_todos=[],
                profile_context="爱运动",
                now_iso="2026-06-18T12:00:00",
                threshold=0.62,
                session_id="s1",
            )
            assert isinstance(decision, ProactiveAdviceDecision)
            assert not decision.should_add

    def test_with_events_advice_above_threshold(self):
        with patch(
            "app.src.agent.memory.personal_memory.chat_completion",
            return_value=(
                '{"score": 0.85, "should_add": true,'
                ' "advice": "建议适当增加蛋白质摄入",'
                ' "reasons": ["近3天无蛋白质记录"]}'
            ),
        ):
            decision = decide_proactive_advice(
                user_message="今天吃了米饭",
                assistant_answer="好的",
                recent_events=[
                    {"title": "吃了米饭", "category": "diet", "event_time": "2026-06-17"}
                ],
                upcoming_todos=[],
                profile_context="",
                now_iso="2026-06-18T12:00:00",
                threshold=0.62,
                session_id="s1",
            )
            assert decision.score == 0.85
            assert decision.should_add
            assert decision.advice

    def test_fallback_on_invalid_llm(self):
        with patch(
            "app.src.agent.memory.personal_memory.chat_completion",
            return_value="not json",
        ):
            decision = decide_proactive_advice(
                user_message="你好",
                assistant_answer="你好",
                recent_events=[],
                upcoming_todos=[],
                profile_context="",
                now_iso="2026-06-18T12:00:00",
                threshold=0.62,
                session_id="s1",
            )
            assert isinstance(decision, ProactiveAdviceDecision)
            assert not decision.should_add


class TestExtractMemoryFromDialogue:
    def test_returns_none_on_invalid_json(self):
        with patch(
            "app.src.agent.memory.memory_extractor.chat_completion",
            return_value="not json",
        ):
            result = extract_memory_from_dialogue(
                [{"role": "user", "content": "我明天开会"}]
            )
            assert result is None

    def test_returns_extraction_on_valid_json(self):
        mock_response = (
            '{"items": [{"kind": "meeting", "title": "明天开会",'
            ' "occurred_at": null, "notes": null,'
            ' "confidence": 1.0, "insight_type": "explicit"}]}'
        )
        with patch(
            "app.src.agent.memory.memory_extractor.chat_completion",
            return_value=mock_response,
        ):
            result = extract_memory_from_dialogue(
                [{"role": "user", "content": "我明天开会"}]
            )
            assert result is not None
            assert len(result.items) == 1
            assert result.items[0].kind == "meeting"
