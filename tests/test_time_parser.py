from __future__ import annotations

from datetime import datetime
from unittest.mock import patch

import pytest

from app.src.agent.time.time_parser import build_date_context, normalize_date_hint


class TestBuildDateContext:
    def test_returns_today_info(self):
        ctx = build_date_context()
        assert "今天日期" in ctx
        assert "今天星期" in ctx
        today_str = datetime.now().strftime("%Y-%m-%d")
        assert ctx["今天日期"] == today_str

    def test_remaining_only_filters(self):
        ctx_all = build_date_context()
        ctx_remaining = build_date_context(remaining_only=True)
        if ctx_all.get("今年节日") and ctx_remaining.get("今年剩余节日"):
            assert len(ctx_remaining["今年剩余节日"]) <= len(ctx_all["今年节日"])

    def test_holidays_not_empty_for_current_year(self):
        ctx = build_date_context()
        all_holidays = ctx.get("今年节日", ctx.get("今年剩余节日", []))
        assert len(all_holidays) > 0


class TestNormalizeDateHint:
    def test_returns_none_on_invalid_llm(self):
        with patch(
            "app.src.agent.time.time_parser.chat_completion",
            return_value="not json",
        ):
            result = normalize_date_hint("明天")
            assert result is None

    def test_returns_none_on_empty_date(self):
        with patch(
            "app.src.agent.time.time_parser.chat_completion",
            return_value='{"date": null, "confidence": 0.0}',
        ):
            result = normalize_date_hint("随便")
            assert result is None

    def test_returns_date_on_valid(self):
        with patch(
            "app.src.agent.time.time_parser.chat_completion",
            return_value='{"date": "2026-06-19", "confidence": 0.9}',
        ):
            result = normalize_date_hint("明天")
            assert result == "2026-06-19"

    def test_low_confidence_returns_none(self):
        with patch(
            "app.src.agent.time.time_parser.chat_completion",
            return_value='{"date": "2026-06-19", "confidence": 0.2}',
        ):
            result = normalize_date_hint("大概某天")
            assert result is None

    def test_invalid_date_format_returns_none(self):
        with patch(
            "app.src.agent.time.time_parser.chat_completion",
            return_value='{"date": "not-a-date", "confidence": 0.9}',
        ):
            result = normalize_date_hint("某天")
            assert result is None
