from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

from app.src.agent import graph_chat as gc
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


class TestRetryBudget:
    """回归：retry_count 不能被 decompose 重置，否则 verify 的重试上限永远触发不了。"""

    def test_decompose_retry_keeps_retry_count(self):
        state = {
            "session_id": "s",
            "retry_info": {"fix_suggestion": "换个关键词"},
            "retry_count": 1,
            "plan": [],
            "plan_index": 0,
            "decision": {"action": "query_knowledge", "args": {"query": "wsl"}},
        }
        out = gc._node_decompose(state)
        assert out["retry_count"] == 1
        assert out["retry_info"] is None
        assert out["decision"]["action"] == "query_knowledge"

    def test_verify_retries_below_max(self):
        state = {
            "session_id": "s",
            "mode": "normal",
            "decision": {"action": "query_knowledge"},
            "answer": "我不知道",
            "user_message": "wsl是什么",
            "plan": [],
            "plan_index": 0,
            "retry_count": 0,
        }
        failed = '{"passed": false, "issues": ["x"], "fix_suggestion": "y"}'
        with patch.object(gc, "chat_completion", return_value=failed):
            out = gc._node_verify(state)
        assert out["retry_info"] is not None
        assert out["retry_count"] == 1

    def test_verify_stops_after_max_retries(self):
        state = {
            "session_id": "s",
            "mode": "normal",
            "decision": {"action": "query_knowledge"},
            "answer": "我不知道",
            "user_message": "wsl是什么",
            "plan": [],
            "plan_index": 0,
            "retry_count": gc._MAX_RETRIES,
        }
        failed = '{"passed": false, "issues": ["x"], "fix_suggestion": "y"}'
        with patch.object(gc, "chat_completion", return_value=failed):
            out = gc._node_verify(state)
        assert out["retry_info"] is None
        assert out["retry_count"] == gc._MAX_RETRIES + 1


class TestToolBudgetAbort:
    def test_run_tool_aborts_when_budget_exhausted(self):
        from app.src.common.harness.context import clear_turn_context, create_turn_context

        turn = create_turn_context("budget-test")
        turn.budget.max_tool_calls = 0
        try:
            state = {
                "session_id": "budget-test",
                "decision": {"action": "query_todos"},
                "trace": {"steps": []},
            }
            out = gc._node_run_tool(state)
        finally:
            clear_turn_context("budget-test")

        assert out["turn_aborted"] is True
        assert out["answer"] == gc._TURN_LIMIT_FALLBACK
        assert out["used_tool"] == "query_todos"


class _FakeStateSnapshot:
    values: dict = {}
    tasks: list = []


class _TurnLimitGraph:
    async def aget_state(self, config):
        return _FakeStateSnapshot()

    async def ainvoke(self, state, config):
        raise gc.TurnLimitExceeded("forced limit")


class _RecursionGraph(_TurnLimitGraph):
    async def ainvoke(self, state, config):
        from langgraph.errors import GraphRecursionError

        raise GraphRecursionError("recursion limit reached")


class TestTurnAbortBoundary:
    """图入口必须把预算/递归异常收口成兜底回答，而不是抛给前端。"""

    def test_turn_limit_returns_fallback(self):
        out = asyncio.run(
            gc.arun_chat_graph(
                compiled_graph=_TurnLimitGraph(),
                session_id="abort-test",
                user_message="wsl是什么",
            )
        )
        assert out["answer"] == gc._TURN_LIMIT_FALLBACK
        assert out["turn_aborted"] is True
        assert out["mode"] == "quick"

    def test_recursion_limit_returns_fallback(self):
        out = asyncio.run(
            gc.arun_chat_graph(
                compiled_graph=_RecursionGraph(),
                session_id="recursion-test",
                user_message="wsl是什么",
            )
        )
        assert out["answer"] == gc._TURN_LIMIT_FALLBACK
        assert out["turn_aborted"] is True


class TestToolPolicyBlock:
    def test_unknown_action_is_blocked_without_aborting_turn(self):
        state = {
            "session_id": "policy-test",
            "decision": {"action": "rm_rf", "args": {}},
            "trace": {"steps": []},
        }
        out = gc._node_run_tool(state)
        assert out.get("turn_aborted") is not True
        assert "安全" in out["answer"]
        names = [s["name"] for s in out["trace"]["steps"]]
        assert "tool_policy_block" in names

    def test_query_database_without_session_filter_is_blocked(self):
        state = {
            "session_id": "policy-test",
            "decision": {
                "action": "query_database",
                "args": {"sql": "SELECT * FROM todo_items"},
            },
            "trace": {"steps": []},
        }
        out = gc._node_run_tool(state)
        assert "安全" in out["answer"]
        assert out.get("turn_aborted") is not True
        block = next(
            s for s in out["trace"]["steps"] if s["name"] == "tool_policy_block"
        )
        assert "session_id" in block["output"]["reason"]


class TestHarnessSummaryTrace:
    def test_summary_is_attached_only_at_turn_exit(self):
        from app.src.common.harness.context import (
            clear_turn_context,
            create_turn_context,
        )

        turn = create_turn_context("finalize-test")
        turn.add_record({"kind": "llm", "name": "decompose", "elapsed_ms": 12.3})
        try:
            state = {
                "session_id": "finalize-test",
                "answer": "hi",
                "history": [{"role": "user", "content": "hi"}],
                "trace": {"steps": []},
            }
            out = gc._node_finalize(state)
            assert all(step.get("name") != "harness_summary" for step in out["trace"]["steps"])
            out = gc._finish_turn_result(out, turn)
        finally:
            clear_turn_context("finalize-test")

        summary = next(
            s for s in out["trace"]["steps"] if s["name"] == "harness_summary"
        )
        assert summary["output"]["invocations"][0]["name"] == "decompose"
