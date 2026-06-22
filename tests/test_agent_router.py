from __future__ import annotations

from unittest.mock import patch

import pytest

from app.src.agent.agent_router import (
    RouteContext,
    build_route_context,
    route_decision,
)
from app.src.domain.dto.router_dto import RouterDecision, RouterTrace


class TestRouteContext:
    def test_build_context_without_topics(self):
        with patch(
            "app.src.agent.agent_router.get_available_knowledge_topics",
            side_effect=Exception("no db"),
        ), patch(
            "app.src.agent.agent_router.get_document_outline",
            side_effect=Exception("no db"),
        ):
            ctx = build_route_context(
                session_id="s1",
                user_message="hello",
                recent_dialogue=[{"role": "user", "content": "hi"}],
                summary=None,
            )
        assert ctx.session_id == "s1"
        assert ctx.user_message == "hello"
        assert ctx.summary is None
        assert ctx.available_topics is None
        assert ctx.available_outline is None

    def test_build_context_with_topics(self):
        with patch(
            "app.src.agent.agent_router.get_available_knowledge_topics",
            return_value=[{"topic": "work"}],
        ), patch(
            "app.src.agent.agent_router.get_document_outline",
            return_value=[{"title": "doc1"}],
        ):
            ctx = build_route_context(
                session_id="s1",
                user_message="hello",
                recent_dialogue=[],
                summary="sum",
            )
            assert ctx.available_topics == [{"topic": "work"}]
            assert ctx.available_outline == [{"title": "doc1"}]

    def test_context_defaults(self):
        with patch(
            "app.src.agent.agent_router.get_available_knowledge_topics",
            side_effect=Exception("no db"),
        ), patch(
            "app.src.agent.agent_router.get_document_outline",
            side_effect=Exception("no db"),
        ):
            ctx = build_route_context(
                session_id="s1",
                user_message="hi",
                recent_dialogue=[],
                summary=None,
            )
        assert ctx.available_topics is None
        assert ctx.available_outline is None


class TestRouteDecision:
    def test_returns_decision_with_mocked_llm(self):
        ctx = RouteContext(
            session_id="s1",
            user_message="今天天气怎么样",
            recent_dialogue=[],
            summary=None,
        )
        mock_response = (
            '{"action": "query_weather", "args": {"city": null},'
            ' "assistant_message": "",'
            ' "trace": {"intent": "weather_query", "why": "用户询问天气"}}'
        )
        with patch("app.src.agent.agent_router.chat_completion", return_value=mock_response):
            decision = route_decision(ctx)
            assert isinstance(decision, RouterDecision)
            assert decision.action == "query_weather"

    def test_invalid_json_raises(self):
        ctx = RouteContext(
            session_id="s1",
            user_message="hello",
            recent_dialogue=[],
            summary=None,
        )
        with patch("app.src.agent.agent_router.chat_completion", return_value="not json"):
            with pytest.raises(Exception):
                route_decision(ctx)

    def test_empty_response_raises(self):
        ctx = RouteContext(
            session_id="s1",
            user_message="hello",
            recent_dialogue=[],
            summary=None,
        )
        with patch("app.src.agent.agent_router.chat_completion", return_value=""):
            with pytest.raises(Exception):
                route_decision(ctx)

    def test_code_block_cleaned(self):
        ctx = RouteContext(
            session_id="s1",
            user_message="hi",
            recent_dialogue=[],
            summary=None,
        )
        mock_response = (
            '```json\n{"action": "normal_chat", "args": {},'
            ' "assistant_message": "你好",'
            ' "trace": {"intent": "greeting", "why": "打招呼"}}\n```'
        )
        with patch("app.src.agent.agent_router.chat_completion", return_value=mock_response):
            decision = route_decision(ctx)
            assert decision.action == "normal_chat"
            assert decision.assistant_message == "你好"
