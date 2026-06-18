from __future__ import annotations

from datetime import date, datetime, timedelta
from unittest.mock import patch

import pytest

from app.src.agent.planner.planner import (
    _apply_fixed_date,
    _dedupe_plan_items,
    _fallback_todo_from_complaint,
    _next_or_same_weekday,
    _next_week_weekday,
    _normalize_title,
    _resolve_relative_day_date,
    _resolve_weekday_date,
    detect_affirmation,
    detect_plan_intent,
    detect_rejection,
    propose_plan,
)
from app.src.domain.dto.planner_dto import PlanItem, PlanProposal


class TestDetectPlanIntent:
    def test_keyword_hits(self):
        for kw in ["明天", "待办", "提醒", "计划", "会议", "deadline"]:
            assert detect_plan_intent(f"有个{kw}")

    def test_no_keyword(self):
        assert not detect_plan_intent("你好呀")
        assert not detect_plan_intent("哈哈")

    def test_empty_string(self):
        assert not detect_plan_intent("")


class TestDetectAffirmation:
    def test_exact_affirmations(self):
        for word in ["要", "需要", "好的", "可以", "好", "行", "没问题", "确认"]:
            assert detect_affirmation(word)

    def test_prefix_affirmations(self):
        assert detect_affirmation("需要处理")
        assert detect_affirmation("可以开始")

    def test_rejections_not_affirmation(self):
        for word in ["不用", "不要", "不需要", "算了", "取消"]:
            assert not detect_affirmation(word)

    def test_empty_string(self):
        assert not detect_affirmation("")
        assert not detect_affirmation("   ")

    def test_rejection_prefix_not_affirmation(self):
        assert not detect_affirmation("不需要了")
        assert not detect_affirmation("不要这个")


class TestDetectRejection:
    def test_exact_rejections(self):
        for word in ["不用", "不要", "不需要", "不行", "算了", "取消"]:
            assert detect_rejection(word)

    def test_affirmations_not_rejection(self):
        assert not detect_rejection("要")
        assert not detect_rejection("好的")

    def test_empty_string(self):
        assert not detect_rejection("")


class TestNextOrSameWeekday:
    def test_same_day(self):
        wed = date(2026, 6, 17)
        assert _next_or_same_weekday(wed, 2) == wed

    def test_next_week(self):
        sat = date(2026, 6, 13)
        next_mon = date(2026, 6, 15)
        assert _next_or_same_weekday(sat, 0) == next_mon

    def test_wrap_around(self):
        sun = date(2026, 6, 14)
        next_mon = date(2026, 6, 15)
        assert _next_or_same_weekday(sun, 0) == next_mon


class TestNextWeekWeekday:
    def test_next_week_monday(self):
        wed = date(2026, 6, 17)
        expected = date(2026, 6, 22)
        assert _next_week_weekday(wed, 0) == expected

    def test_next_week_wednesday(self):
        tue = date(2026, 6, 16)
        expected = date(2026, 6, 24)
        assert _next_week_weekday(tue, 2) == expected


class TestResolveWeekdayDate:
    def test_this_week_wednesday(self):
        with patch("app.src.agent.planner.planner.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 6, 15)
            mock_dt.combine = datetime.combine
            mock_dt.min.time = staticmethod(lambda: datetime.min.time())
            result = _resolve_weekday_date("周三开会")
            assert result == "2026-06-17"

    def test_next_week_wednesday(self):
        with patch("app.src.agent.planner.planner.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 6, 15)
            result = _resolve_weekday_date("下周三开会")
            assert result == "2026-06-24"

    def test_no_weekday(self):
        with patch("app.src.agent.planner.planner.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 6, 15)
            assert _resolve_weekday_date("随便哪天") is None


class TestResolveRelativeDayDate:
    def test_today(self):
        with patch("app.src.agent.planner.planner.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 6, 18)
            result = _resolve_relative_day_date("今天开会")
            assert result == "2026-06-18"

    def test_tomorrow(self):
        with patch("app.src.agent.planner.planner.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 6, 18)
            result = _resolve_relative_day_date("明天去超市")
            assert result == "2026-06-19"

    def test_day_after_tomorrow(self):
        with patch("app.src.agent.planner.planner.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 6, 18)
            result = _resolve_relative_day_date("后天考试")
            assert result == "2026-06-20"

    def test_no_match(self):
        with patch("app.src.agent.planner.planner.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 6, 18)
            assert _resolve_relative_day_date("下周") is None


class TestNormalizeTitle:
    def test_remove_todo_reminder(self):
        assert _normalize_title("待办：开会") == "开会"
        assert _normalize_title("提醒买牛奶") == "买牛奶"

    def test_remove_relative_dates(self):
        assert _normalize_title("明天开会") == "开会"
        assert _normalize_title("今天周三交报告") == "交报告"

    def test_clean_punctuation(self):
        assert _normalize_title("：完成作业，") == "完成作业"

    def test_empty_fallback(self):
        assert _normalize_title("") == "待办事项"
        assert _normalize_title("   ") == "待办事项"

    def test_whitespace_collapse(self):
        result = _normalize_title(" 买  牛奶  ")
        assert "  " not in result
        assert result.strip() == result


class TestDedupePlanItems:
    def test_exact_duplicates(self):
        items = [
            PlanItem(title="买牛奶", due_at="2026-06-18"),
            PlanItem(title="买牛奶", due_at="2026-06-18"),
        ]
        result = _dedupe_plan_items(items)
        assert len(result) == 1

    def test_different_titles(self):
        items = [
            PlanItem(title="买牛奶", due_at="2026-06-18"),
            PlanItem(title="开会", due_at="2026-06-18"),
        ]
        result = _dedupe_plan_items(items)
        assert len(result) == 2

    def test_same_title_different_dates(self):
        items = [
            PlanItem(title="买牛奶", due_at="2026-06-18"),
            PlanItem(title="买牛奶", due_at="2026-06-19"),
        ]
        result = _dedupe_plan_items(items)
        assert len(result) == 2

    def test_empty_list(self):
        assert _dedupe_plan_items([]) == []


class TestApplyFixedDate:
    def test_no_due_at(self):
        assert _apply_fixed_date("2026-06-18", None) == "2026-06-18"

    def test_date_only(self):
        assert _apply_fixed_date("2026-06-18", "2026-06-20") == "2026-06-18"

    def test_with_time(self):
        assert _apply_fixed_date("2026-06-18", "2026-06-20T14:30:00") == "2026-06-18T14:30:00"


class TestFallbackTodoFromComplaint:
    def test_teacher_not_reply(self):
        result = _fallback_todo_from_complaint("导员没有回我关于报销的事", None)
        assert result is not None
        assert "导员" in result.title or "跟进" in result.title

    def test_no_complaint_pattern(self):
        result = _fallback_todo_from_complaint("今天天气不错", None)
        assert result is not None
        assert result.title

    def test_empty_text(self):
        assert _fallback_todo_from_complaint("", None) is None


class TestProposePlan:
    def test_returns_proposal_with_mocked_llm(self):
        with patch(
            "app.src.agent.planner.planner.chat_completion",
            return_value='{"items": [{"title": "跟进报销审批", "due_at": null}]}',
        ):
            proposal_id, proposal = propose_plan("导员没回报销")
            assert proposal_id
            assert len(proposal.items) == 1
            assert proposal.items[0].title

    def test_llm_returns_empty_fallback_to_rule(self):
        with patch(
            "app.src.agent.planner.planner.chat_completion",
            return_value='{"items": []}',
        ):
            proposal_id, proposal = propose_plan("导员没有回我关于报销的事")
            assert len(proposal.items) >= 1

    def test_llm_returns_invalid_json(self):
        with patch(
            "app.src.agent.planner.planner.chat_completion",
            return_value="not json at all",
        ):
            proposal_id, proposal = propose_plan("明天开会")
            assert proposal_id
            assert isinstance(proposal, PlanProposal)
