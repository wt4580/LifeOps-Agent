from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from app.src.agent.planner.planner import (
    _apply_fixed_date,
    _dedupe_plan_items,
    _normalize_title,
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


class TestProposePlan:
    def test_returns_proposal_with_mocked_llm(self):
        with patch(
            "app.src.agent.planner.planner.chat_completion",
            return_value='{"items": [{"title": "跟进审批", "due_at": null}]}',
        ):
            proposal_id, proposal = propose_plan("审批还没过")
            assert proposal_id
            assert len(proposal.items) == 1
            assert proposal.items[0].title

    def test_llm_returns_empty_items(self):
        with patch(
            "app.src.agent.planner.planner.chat_completion",
            return_value='{"items": []}',
        ):
            proposal_id, proposal = propose_plan("审批还没过")
            assert len(proposal.items) == 0

    def test_llm_returns_invalid_json(self):
        with patch(
            "app.src.agent.planner.planner.chat_completion",
            return_value="not json at all",
        ):
            with pytest.raises((json.JSONDecodeError, ValidationError)):
                propose_plan("明天开会")
