from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from app.src.agent.graph_chat import (
    _is_bare_confirmation_text,
    _is_prev_todo_query,
    _resolve_relative_dates,
    _should_check_proactive,
)


class TestIsBareConfirmationText:
    def test_affirmation_words(self):
        for word in ["要", "需要", "可以", "好的", "好", "行", "确认"]:
            assert _is_bare_confirmation_text(word)

    def test_rejection_words(self):
        for word in ["不用", "不要", "算了", "取消"]:
            assert _is_bare_confirmation_text(word)

    def test_long_text_not_bare(self):
        assert not _is_bare_confirmation_text("需要处理报销")
        assert not _is_bare_confirmation_text("好的我明白了")

    def test_empty_string(self):
        assert not _is_bare_confirmation_text("")
        assert not _is_bare_confirmation_text("  ")


class TestResolveRelativeDates:
    def test_today(self):
        result = _resolve_relative_dates("今天开会")
        today = datetime.now().strftime("%m月%d日")
        assert result == f"{today}开会"

    def test_tomorrow(self):
        result = _resolve_relative_dates("明天去超市")
        tomorrow = (datetime.now() + timedelta(days=1)).strftime("%m月%d日")
        assert result == f"{tomorrow}去超市"

    def test_day_after_tomorrow(self):
        result = _resolve_relative_dates("后天考试")
        d2 = (datetime.now() + timedelta(days=2)).strftime("%m月%d日")
        assert result == f"{d2}考试"

    def test_no_relative_words(self):
        assert _resolve_relative_dates("你好") == "你好"

    def test_empty_string(self):
        assert _resolve_relative_dates("") == ""


class TestShouldCheckProactive:
    def test_skip_hitl_tools(self):
        for tool in ["plan_proposal", "plan_gate_cancel", "plan_gate_wait", "plan_gate_confirm"]:
            state = {"used_tool": tool}
            assert not _should_check_proactive(state)

    def test_check_normal_tools(self):
        state = {"used_tool": "query_weather"}
        assert _should_check_proactive(state)

    def test_check_none_tool(self):
        state = {"used_tool": None}
        assert _should_check_proactive(state)


class TestIsPrevTodoQuery:
    def test_detects_prev_keywords(self):
        state = {
            "user_message": "上一条",
            "recent_dialogue": [{"role": "assistant", "content": "您有以下待办事项"}],
        }
        assert _is_prev_todo_query(state)

    def test_ignores_without_todo_context(self):
        state = {
            "user_message": "上一条",
            "recent_dialogue": [{"role": "assistant", "content": "今天天气不错"}],
        }
        assert not _is_prev_todo_query(state)

    def test_detects_typo(self):
        state = {
            "user_message": "上一台哦",
            "recent_dialogue": [{"role": "assistant", "content": "您有3个待办事项"}],
        }
        assert _is_prev_todo_query(state)

    def test_no_message(self):
        state = {"user_message": "", "recent_dialogue": []}
        assert not _is_prev_todo_query(state)
