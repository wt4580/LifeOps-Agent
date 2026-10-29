import asyncio
import json
import sqlite3
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from langgraph.checkpoint.memory import MemorySaver
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker

from app.src.agent import graph_chat as graph
from app.src.common.config.db_config import Base
from app.src.common.domain.tool_errors import ToolExecutionFailure, failure_result
from app.src.common.harness.context import turn_context
from app.src.domain.entity.chat_entity import GoalItem, Transaction
from app.src.service import bookkeeping_service as bookkeeping
from app.src.service.weather_service import WeatherServiceError


@pytest.fixture
def business(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'business.db'}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    monkeypatch.setattr(graph, "SessionLocal", factory)
    monkeypatch.setattr(bookkeeping, "SessionLocal", factory)
    yield factory
    engine.dispose()


def test_read_failure_stops_dependent_write_and_is_not_a_successful_call(business, monkeypatch, caplog):
    monkeypatch.setattr(graph.weather_service, "query", Mock(side_effect=WeatherServiceError("secret-api-key")))
    write = Mock(side_effect=AssertionError("Must not execute dependent write"))
    monkeypatch.setattr(bookkeeping.bookkeeping_service, "record_transaction", write)
    planner = Mock(return_value=json.dumps({"needs_plan": True, "steps": [
        {"action": "query_weather", "args": {"city": "杭州"}},
        {"action": "record_expense", "args": {"amount": 10}},
    ]}))
    monkeypatch.setattr(graph, "chat_completion", planner)
    result = graph.run_chat_graph(compiled_graph=graph.build_chat_graph(checkpointer=MemorySaver()),
                                  session_id="alice", user_message="先查天气，再记账")
    assert result["tool_error"]["code"] == "dependency_unavailable"
    assert result["tool_error"]["outcome"] == "not_completed"
    assert result["plan_index"] == 0 and "result" not in result["plan"][0]
    summary = next(step["output"] for step in result["trace"]["steps"] if step.get("name") == "harness_summary")
    assert summary["status"] == "tool_failed"
    assert summary["invocations"][-1]["status"] == "error"
    assert planner.call_count == 1
    assert "secret-api-key" not in json.dumps(result) + caplog.text
    write.assert_not_called()


def test_business_rejection_is_not_applied_and_preserves_prior_plan_evidence(business):
    state = {"session_id": "alice", "user_message": "记账10元", "plan_index": 1,
             "decision": {"action": "record_expense", "args": {"amount": 10, "account_name": "missing"}},
             "plan": [{"action": "query_accounts", "result": "已查询"}, {"action": "record_expense"}]}
    with turn_context("alice") as turn:
        result = graph._node_run_tool(state)
        assert result["tool_error"]["code"] == "business_rule_rejected"
        assert result["tool_error"]["outcome"] == "not_applied"
        assert result["tool_error"]["completed_plan_steps"] == 1
        graph._finish_turn_result(result, turn)
        assert turn.status == "tool_rejected"
    with business() as db:
        assert db.scalar(select(func.count()).select_from(Transaction)) == 0


def test_postcommit_failure_is_persisted_and_same_request_retry_does_not_repeat_write(business, monkeypatch):
    from app.src.service import chat_service as service
    monkeypatch.setattr(service, "SessionLocal", business)
    monkeypatch.setattr(service.settings, "background_learning_enabled", True)
    compiled = graph.build_chat_graph(checkpointer=MemorySaver())
    monkeypatch.setattr(service, "get_chat_graph", lambda: compiled)
    monkeypatch.setattr(graph, "chat_completion", Mock(return_value='{"action":"record_expense","args":{"amount":10},"mode":"quick"}'))
    original = bookkeeping.bookkeeping_service.record_transaction
    def commit_then_fail(**kwargs):
        original(**kwargs)
        raise RuntimeError("private postcommit implementation detail")
    mutation = Mock(side_effect=commit_then_fail)
    monkeypatch.setattr(bookkeeping.bookkeeping_service, "record_transaction", mutation)
    chat = service.ChatService()
    learning = Mock()
    monkeypatch.setattr(chat, "_spawn_post_graph", learning)
    async def run():
        first = await chat.chat("记账10元", "alice", request_id="one")
        assert await chat.chat("记账10元", "alice", request_id="one") == first
        return first
    result = asyncio.run(run())
    assert result["tool_error"]["outcome"] == "some_operations_committed"
    assert result["tool_error"]["committed_operations_count"] == 1
    assert result["tool_error"]["committed_operations"][0]["record_id"]
    assert "private postcommit" not in json.dumps(result)
    assert mutation.call_count == 1
    learning.assert_not_called()
    with business() as db:
        assert db.scalar(select(func.count()).select_from(Transaction)) == 1


def test_goal_batch_partial_commit_is_reported_without_claiming_rollback(business, monkeypatch):
    from app.src.agent.memory import personal_memory
    from app.src.service import goal_service
    monkeypatch.setattr(goal_service, "SessionLocal", business)
    goals = [SimpleNamespace(title=title, category=None, target_date=None, sub_goals=[], milestones=[], notes=None)
             for title in ("first", "second")]
    monkeypatch.setattr(personal_memory, "extract_goal", lambda **kwargs: SimpleNamespace(items=goals))
    original = goal_service.goal_service.save_goal
    def save(**kwargs):
        if kwargs["title"] == "second":
            raise sqlite3.OperationalError("private database path")
        return original(**kwargs)
    monkeypatch.setattr(goal_service.goal_service, "save_goal", save)
    with turn_context("alice"):
        result = graph._node_run_tool({"session_id": "alice", "user_message": "拆解目标",
                                      "decision": {"action": "decompose_goal", "args": {}}})
    assert result["tool_error"]["code"] == "storage_unavailable"
    assert result["tool_error"]["outcome"] == "some_operations_committed"
    assert result["tool_error"]["committed_operations_count"] == 1
    with business() as db:
        assert [goal.title for goal in db.scalars(select(GoalItem)).all()] == ["first"]


def test_unknown_write_outcome_is_not_claimed_unapplied(business, monkeypatch):
    monkeypatch.setattr(bookkeeping.bookkeeping_service, "add_account", Mock(side_effect=RuntimeError("private-storage-detail")))
    with turn_context("alice"):
        result = graph._node_run_tool({"session_id": "alice", "user_message": "创建账户",
                                      "decision": {"action": "create_account", "args": {"name": "wallet"}}})
    assert result["tool_error"]["outcome"] == "unknown"
    assert result["tool_error"]["retryable"] is False
    assert result["tool_error"]["recovery"] == "check_records"
    assert "不要重复提交" in result["answer"]


def test_failed_confirmation_invalidates_only_its_own_draft(business, monkeypatch):
    from app.src.service import idempotency
    monkeypatch.setattr(idempotency, "begin_write", Mock(side_effect=sqlite3.OperationalError("secret file path")))
    state = {"session_id": "alice", "user_message": "确认", "interrupt_answer": "确认", "proposal_id": "old-draft",
             "proposal": {"items": [{"title": "meeting"}]}, "decision": {"action": "propose_todo", "args": {}}}
    with turn_context("alice"):
        result = graph._node_run_tool(state)
    assert result["tool_error"]["outcome"] == "unknown"
    assert result["tool_error"]["invalidated_proposal_id"] == "old-draft"
    assert result["proposal_id"] is None and result["proposal"] is None
    assert result["interrupt_message"] is None


@pytest.mark.parametrize("error", [asyncio.CancelledError(), KeyboardInterrupt()])
def test_cancellation_and_process_interrupt_are_not_swallowed(monkeypatch, error):
    monkeypatch.setattr(graph, "_run_tool_impl", Mock(side_effect=error))
    with pytest.raises(type(error)), turn_context("alice"):
        graph._node_run_tool({"session_id": "alice", "decision": {"action": "normal_chat", "args": {}}})


def test_nested_failure_envelopes_preserve_source_class_without_message():
    wrapped = ToolExecutionFailure("query_database", ToolExecutionFailure("query_database", sqlite3.OperationalError("private detail")))
    envelope, answer = failure_result(wrapped, action="query_database", effect="read", committed={})
    assert envelope["code"] == "storage_unavailable"
    assert envelope["error_type"] == "OperationalError"
    assert "private detail" not in json.dumps(envelope) + answer


def test_unhandled_chat_job_errors_do_not_expose_provider_secrets():
    from app.src.service.session_queue import _execute, JOB_END
    async def scenario():
        events = asyncio.Queue()
        async def fail():
            raise RuntimeError("sk-private-provider-secret")
        await _execute(events, fail)
        error = await events.get()
        assert error["code"] == "request_execution_failed"
        assert error["outcome"] == "unconfirmed"
        assert error["retryable"] is False
        assert "sk-private" not in json.dumps(error)
        assert await events.get() is JOB_END
    asyncio.run(scenario())


def test_cached_receipt_is_observed_in_each_scope_even_when_known_to_turn(business):
    from app.src.service.idempotency import write_scope
    with turn_context("alice"):
        with write_scope("alice", "same-operation") as first:
            result = bookkeeping.bookkeeping_service.record_transaction(amount=-10, session_id="alice")
        with write_scope("alice", "same-operation") as replay:
            assert bookkeeping.bookkeeping_service.record_transaction(amount=-10, session_id="alice") == result
        assert len(first.committed) == len(replay.committed) == 1
        assert first.committed == replay.committed


def test_explicit_operation_key_cannot_bypass_write_scope_identity(business):
    from app.src.service.idempotency import write_scope, IdempotencyConflict
    with write_scope("alice", "scope"), pytest.raises(IdempotencyConflict):
        bookkeeping.bookkeeping_service.record_transaction(amount=-10, session_id="bob", operation_key="explicit-key")


@pytest.mark.parametrize("expired", [False, True])
def test_known_request_rejections_are_terminal_and_safe(expired):
    from app.src.common.domain.request_errors import ChatRequestRejected
    from app.src.service.idempotency import IdempotencyConflict
    from app.src.service.session_queue import _execute, JOB_END
    async def scenario():
        events = asyncio.Queue()
        async def fail():
            if expired:
                raise ChatRequestRejected("proposal_expired", "草案已失效", invalidated_proposal_id="old")
            raise IdempotencyConflict("private mismatched input")
        await _execute(events, fail, "original-request")
        error = await events.get()
        assert error["code"] == ("proposal_expired" if expired else "request_id_conflict")
        assert error["retryable"] is False
        assert error["outcome"] == "rejected"
        assert "private mismatched" not in json.dumps(error)
        if expired:
            assert error["invalidated_proposal_id"] == "old"
        assert await events.get() is JOB_END
    asyncio.run(scenario())


@pytest.mark.parametrize("action,code", [("propose_todo", "empty_proposal"), ("revise_proposal", "empty_proposal"),
                                         ("decompose_goal", "goal_not_identified")])
def test_empty_business_results_are_not_reported_as_success(business, monkeypatch, action, code):
    from app.src.domain.dto.planner_dto import PlanProposal
    from app.src.agent.memory import personal_memory
    monkeypatch.setattr(graph, "propose_plan", lambda *args, **kwargs: ("empty", PlanProposal(items=[])))
    monkeypatch.setattr(personal_memory, "extract_goal", lambda **kwargs: None)
    state = {"session_id": "alice", "user_message": "处理目标或草案", "decision": {"action": action, "args": {}},
             "superseded_proposal_id": "old" if action == "revise_proposal" else None,
             "plan": [{"action": action}, {"action": "record_expense", "args": {"amount": 10}}], "plan_index": 0}
    with turn_context("alice"):
        result = graph._node_run_tool(state)
    assert result["tool_error"]["code"] == code
    assert result["tool_error"]["kind"] == "rejection"
    assert graph._route_after_advance(result) == "finalize"
    assert not result.get("interrupt_message")
    assert not result.get("proposal_id")
    if action == "revise_proposal":
        assert result["tool_error"]["invalidated_proposal_id"] == "old"


def test_empty_model_output_does_not_reuse_previous_step_answer(business, monkeypatch):
    monkeypatch.setattr(graph, "chat_completion", Mock(return_value=""))
    with turn_context("alice"):
        result = graph._node_run_tool({"session_id": "alice", "user_message": "解释一下",
                                      "answer": "previous-step-success", "decision": {"action": "normal_chat", "args": {}}})
    assert result["tool_error"]["kind"] == "execution"
    assert "previous-step-success" not in result["answer"]


def test_empty_confirmation_never_creates_success_receipt(business):
    from app.src.domain.entity.chat_entity import WriteOperation, TodoItem
    with turn_context("alice"):
        result = graph._node_run_tool({"session_id": "alice", "user_message": "确认", "interrupt_answer": "确认",
                                      "proposal_id": "empty", "proposal": {"items": []},
                                      "decision": {"action": "propose_todo", "args": {}}})
    assert result["tool_error"]["code"] == "empty_proposal"
    with business() as db:
        assert db.scalar(select(func.count()).select_from(WriteOperation)) == 0
        assert db.scalar(select(func.count()).select_from(TodoItem)) == 0
