from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.src.domain.dto.planner_dto import PlanItem, PlanProposal
from app.src.domain.dto.router_dto import RouterDecision, RouterTrace
from app.src.domain.dto.chat_dto import ChatDto
from app.src.domain.dto.memory_dto import (
    MemoryItem,
    PersonalEventItem,
    ProfileFactPatch,
    ProactiveAdviceDecision,
)
from app.src.domain.dto.time_dto import DateHint


class TestPlanItem:
    def test_valid_minimal(self):
        item = PlanItem(title="测试")
        assert item.title == "测试"
        assert item.due_at is None

    def test_valid_full(self):
        item = PlanItem(title="测试", due_at="2026-06-18")
        assert item.due_at == "2026-06-18"

    def test_title_required(self):
        with pytest.raises(ValidationError):
            PlanItem()

    def test_empty_title(self):
        item = PlanItem(title="")
        assert item.title == ""


class TestPlanProposal:
    def test_default_items(self):
        p = PlanProposal()
        assert p.items == []

    def test_with_items(self):
        p = PlanProposal(items=[PlanItem(title="a"), PlanItem(title="b")])
        assert len(p.items) == 2


class TestRouterTrace:
    def test_valid(self):
        t = RouterTrace(intent="test", why="just testing")
        assert t.intent == "test"
        assert t.why == "just testing"
        assert t.signals == []

    def test_with_signals(self):
        t = RouterTrace(intent="test", signals=["a", "b"], why="testing")
        assert t.signals == ["a", "b"]


class TestRouterDecision:
    def test_valid(self):
        d = RouterDecision(
            action="normal_chat",
            args={},
            assistant_message="hello",
            trace=RouterTrace(intent="test", why="testing"),
        )
        assert d.action == "normal_chat"

    def test_default_args(self):
        d = RouterDecision(
            action="query_weather",
            assistant_message="",
            trace=RouterTrace(intent="test", why="testing"),
        )
        assert d.args == {}

    def test_trace_required(self):
        with pytest.raises(ValidationError):
            RouterDecision(action="normal_chat", assistant_message="")


class TestChatDto:
    def test_valid(self):
        dto = ChatDto(answer="你好", session_id="s1")
        assert dto.answer == "你好"
        assert dto.session_id == "s1"

    def test_optionals(self):
        dto = ChatDto(
            answer="a",
            session_id="s1",
            proposal_id="p1",
            proposal={"items": []},
            used_tool="plan_proposal",
            citations=[{"path": "x"}],
            trace={"steps": []},
        )
        assert dto.proposal_id == "p1"


class TestMemoryItem:
    def test_valid(self):
        item = MemoryItem(kind="event", title="跑步")
        assert item.confidence == 1.0
        assert item.insight_type == "explicit"

    def test_confidence_range(self):
        with pytest.raises(ValidationError):
            MemoryItem(kind="event", title="x", confidence=1.5)


class TestPersonalEventItem:
    def test_valid(self):
        item = PersonalEventItem(category="diet", title="吃了火锅")
        assert item.amount is None
        assert item.tags == []

    def test_category_is_free_text(self):
        item = PersonalEventItem(category="随便", title="x")
        assert item.category == "随便"


class TestProfileFactPatch:
    def test_all_optional(self):
        patch = ProfileFactPatch()
        assert patch.height_cm is None
        assert patch.preferences == []

    def test_with_values(self):
        patch = ProfileFactPatch(height_cm=175, preferences=["爱运动"])
        assert patch.height_cm == 175.0


class TestProactiveAdviceDecision:
    def test_defaults(self):
        d = ProactiveAdviceDecision()
        assert d.score == 0.0
        assert d.should_add is False
        assert d.advice is None
        assert d.reasons == []


class TestDateHint:
    def test_valid(self):
        h = DateHint(date="2026-06-18", confidence=0.9)
        assert h.date == "2026-06-18"

    def test_default_confidence(self):
        h = DateHint()
        assert h.date is None
        assert h.confidence == 0.5
