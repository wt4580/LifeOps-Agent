from __future__ import annotations

import pytest

from app.src.common.harness.context import (
    TurnBudget,
    TurnContext,
    TurnLimitExceeded,
    budget_bypass,
    budget_bypassed,
    clear_turn_context,
    create_turn_context,
    get_turn_context,
    turn_context,
)
from app.src.common.harness.interceptors import (
    BudgetInterceptor,
    CallContext,
    Decision,
    InterceptorChain,
    PolicyViolation,
    SessionScopeInterceptor,
    ToolPolicyInterceptor,
    TraceInterceptor,
)


class TestTurnBudget:
    def test_counts_llm_and_raises_at_limit(self):
        budget = TurnBudget(max_llm_calls=1, max_tool_calls=5, deadline_seconds=30)
        budget.before_call("llm")
        assert budget.llm_calls == 1
        with pytest.raises(TurnLimitExceeded):
            budget.before_call("llm")

    def test_counts_tool_and_raises_at_limit(self):
        budget = TurnBudget(max_llm_calls=5, max_tool_calls=1, deadline_seconds=30)
        budget.before_call("tool")
        assert budget.tool_calls == 1
        with pytest.raises(TurnLimitExceeded):
            budget.before_call("tool")

    def test_deadline_exceeded(self):
        # deadline_seconds=-1 → deadline_at 已过期；避免依赖 Windows 上
        # time.monotonic() 约 15.6ms 的时钟粒度。
        budget = TurnBudget(max_llm_calls=5, max_tool_calls=5, deadline_seconds=-1)
        with pytest.raises(TurnLimitExceeded):
            budget.before_call("llm")

    def test_snapshot(self):
        budget = TurnBudget(max_llm_calls=5, max_tool_calls=5, deadline_seconds=30)
        budget.before_call("llm")
        budget.before_call("tool")
        snap = budget.snapshot()
        assert snap["llm_calls"] == 1
        assert snap["tool_calls"] == 1
        assert snap["remaining_seconds"] > 0


class TestTurnContextRegistry:
    def test_create_get_clear(self):
        create_turn_context("s1")
        assert get_turn_context("s1") is not None
        clear_turn_context("s1")
        assert get_turn_context("s1") is None

    def test_context_manager_cleans_up(self):
        with turn_context("s2") as turn:
            assert get_turn_context("s2") is turn
        assert get_turn_context("s2") is None

    def test_clear_stale_instance_does_not_remove_newer(self):
        old = create_turn_context("s3")
        new = create_turn_context("s3")
        clear_turn_context("s3", old)
        assert get_turn_context("s3") is new
        clear_turn_context("s3", new)

    def test_get_missing_returns_none(self):
        assert get_turn_context("missing") is None
        assert get_turn_context(None) is None


class TestBudgetBypass:
    def test_bypass_flag(self):
        assert budget_bypassed() is False
        with budget_bypass():
            assert budget_bypassed() is True
        assert budget_bypassed() is False


class TestBudgetInterceptor:
    def test_uses_turn_budget(self):
        turn = TurnContext(
            session_id="s",
            budget=TurnBudget(max_llm_calls=1, max_tool_calls=1, deadline_seconds=30),
        )
        ctx = CallContext(kind="llm", name="decompose", session_id="s", turn=turn)
        interceptor = BudgetInterceptor()
        assert interceptor.before(ctx) == Decision.CONTINUE
        with pytest.raises(TurnLimitExceeded):
            interceptor.before(ctx)

    def test_no_turn_is_noop(self):
        ctx = CallContext(kind="llm", name="decompose", session_id="s", turn=None)
        assert BudgetInterceptor().before(ctx) == Decision.CONTINUE


class TestToolPolicyInterceptor:
    def test_unknown_action_is_blocked(self):
        ctx = CallContext(
            kind="tool", name="tool:rm_rf", session_id="s", payload={"action": "rm_rf"}
        )
        assert ToolPolicyInterceptor().before(ctx) == Decision.ABORT
        assert "未知工具动作" in ctx.block_reason

    def test_known_action_passes(self):
        ctx = CallContext(
            kind="tool",
            name="tool:query_todos",
            session_id="s",
            payload={"action": "query_todos"},
        )
        assert ToolPolicyInterceptor().before(ctx) == Decision.CONTINUE

    def test_write_sql_is_blocked(self):
        ctx = CallContext(
            kind="tool",
            name="tool:query_database",
            session_id="s",
            payload={
                "action": "query_database",
                "args": {"sql": "DELETE FROM todo_items"},
            },
        )
        assert ToolPolicyInterceptor().before(ctx) == Decision.ABORT
        assert ctx.block_reason

    def test_llm_calls_are_ignored(self):
        ctx = CallContext(kind="llm", name="decompose", session_id="s")
        assert ToolPolicyInterceptor().before(ctx) == Decision.CONTINUE


class TestSessionScopeInterceptor:
    def test_unscoped_query_is_blocked(self):
        ctx = CallContext(
            kind="tool",
            name="tool:query_database",
            session_id="s",
            payload={
                "action": "query_database",
                "args": {"sql": "SELECT * FROM todo_items"},
            },
        )
        assert SessionScopeInterceptor().before(ctx) == Decision.ABORT
        assert "session_id" in ctx.block_reason

    def test_scoped_query_passes(self):
        ctx = CallContext(
            kind="tool",
            name="tool:query_database",
            session_id="s",
            payload={
                "action": "query_database",
                "args": {"sql": "SELECT * FROM todo_items WHERE session_id='s'"},
            },
        )
        assert SessionScopeInterceptor().before(ctx) == Decision.CONTINUE

    def test_empty_sql_passes_to_tool_layer(self):
        ctx = CallContext(
            kind="tool",
            name="tool:query_database",
            session_id="s",
            payload={"action": "query_database", "args": {}},
        )
        assert SessionScopeInterceptor().before(ctx) == Decision.CONTINUE


class TestInterceptorChain:
    def test_abort_becomes_policy_violation_with_reason(self):
        chain = InterceptorChain([ToolPolicyInterceptor()])
        ctx = CallContext(
            kind="tool", name="tool:bad", session_id="s", payload={"action": "bad"}
        )
        with pytest.raises(PolicyViolation) as info:
            chain.before(ctx)
        assert info.value.reason == ctx.block_reason
        assert info.value.interceptor == "ToolPolicyInterceptor"


class TestTraceInterceptor:
    def test_records_elapsed(self):
        turn = TurnContext(session_id="s", budget=TurnBudget(5, 5, 30))
        ctx = CallContext(kind="llm", name="decompose", session_id="s", turn=turn)
        interceptor = TraceInterceptor()
        interceptor.before(ctx)
        interceptor.after(ctx, "ok")
        assert len(turn.records) == 1
        assert turn.records[0]["name"] == "decompose"
        assert turn.records[0]["elapsed_ms"] >= 0
