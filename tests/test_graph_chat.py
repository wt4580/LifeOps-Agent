from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from app.src.agent.graph_chat import (
    _business_db_inventory_text,
    _build_proposal_item,
    _infer_duration_days,
    _is_bare_confirmation_text,
    _is_prev_todo_query,
    _resolve_relative_dates,
    _should_check_proactive,
)


class TestInferDurationDays:
    def test_one_week(self):
        assert _infer_duration_days("接下来一周每天做2小时LeetCode") == 7

    def test_two_weeks(self):
        assert _infer_duration_days("两周") == 14

    def test_one_month(self):
        assert _infer_duration_days("一个月") == 30

    def test_chinese_digits(self):
        assert _infer_duration_days("未来三天") == 3

    def test_ascii_digits(self):
        assert _infer_duration_days("后面5天") == 5

    def test_daily_without_duration(self):
        assert _infer_duration_days("每天跑步") is None

    def test_empty(self):
        assert _infer_duration_days("") is None
        assert _infer_duration_days(None) is None


class TestBuildProposalItemDurationFallback:
    def test_week_daily_until_computed(self):
        item = _build_proposal_item(
            {
                "title": "每天练习2小时LeetCode",
                "text": "接下来一周每天做2小时LeetCode",
                "start_date": "2026-08-31",
                "repeat": "daily",
                "duration_min": 120,
            }
        )
        assert item["repeat"] == "daily"
        assert item["repeat_until"] == "2026-09-06"
        assert item["due_at"] == "2026-08-31"

    def test_chinese_digits_until(self):
        item = _build_proposal_item(
            {"title": "晨跑", "text": "未来三天早晨跑步", "start_date": "2026-08-31", "repeat": "daily"}
        )
        assert item["repeat_until"] == "2026-09-02"

    def test_weekly_with_duration(self):
        item = _build_proposal_item(
            {"title": "周会", "text": "接下来一周每周一开会", "start_date": "2026-08-31", "repeat": "weekly"}
        )
        assert item["repeat_until"] == "2026-09-06"

    def test_explicit_until_wins(self):
        item = _build_proposal_item(
            {
                "title": "跑步",
                "text": "未来三天跑步",
                "start_date": "2026-08-31",
                "repeat": "daily",
                "repeat_until": "2026-09-30",
            }
        )
        assert item["repeat_until"] == "2026-09-30"

    def test_no_duration_no_until(self):
        item = _build_proposal_item(
            {"title": "跑步", "text": "每天跑步", "start_date": "2026-08-31", "repeat": "daily"}
        )
        assert "repeat_until" not in item


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


class TestBusinessDbInventoryText:
    def test_contains_all_key_tables(self):
        text = _business_db_inventory_text()
        for table in ("user_profiles", "todo_items", "life_events",
                       "accounts", "transactions", "goal_items",
                       "calendar_events", "chat_messages",
                       "conversation_summaries", "memory_candidates"):
            assert table in text

    def test_mentions_session_id_filter(self):
        text = _business_db_inventory_text()
        assert "session_id" in text

    def test_not_empty(self):
        assert len(_business_db_inventory_text()) > 100
