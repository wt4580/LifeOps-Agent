import asyncio
import json
import sqlite3
import threading
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from concurrent.futures import ThreadPoolExecutor

import pytest
from langgraph.checkpoint.memory import MemorySaver
from sqlalchemy import create_engine, delete, func, select
from sqlalchemy import update
from sqlalchemy.orm import sessionmaker

from app.src.agent import graph_chat as graph
from app.src.common.config.db_config import Base
from app.src.common.domain.request_errors import ChatRequestRejected
from app.src.domain.entity.chat_entity import ChatExecution, ChatMessage, ChatRequest, Transaction
from app.src.service import chat_service as service, idempotency as idem, request_recovery as recovery
from app.src.service import bookkeeping_service as bookkeeping


@pytest.fixture
def business(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'business.db'}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    for module in (service, graph, bookkeeping):
        monkeypatch.setattr(module, "SessionLocal", factory)
    monkeypatch.setattr(service.settings, "background_learning_enabled", False)
    yield factory
    engine.dispose()


def install_graph(monkeypatch):
    compiled = graph.build_chat_graph(checkpointer=MemorySaver())
    monkeypatch.setattr(service, "get_chat_graph", lambda: compiled)
    plans = []
    def completion(*args, **kwargs):
        scenario = kwargs.get("runtime_context", {}).get("scenario")
        if scenario == "decompose":
            # Replanning would move the write to another position and bypass its old receipt.
            steps = [{"action": "record_expense", "args": {"amount": 10}}, {"action": "normal_chat", "args": {}}]
            if plans:
                steps.reverse()
            plans.append(steps)
            return json.dumps({"needs_plan": True, "steps": steps})
        if scenario in {"verify_step", "quality_gate"}:
            return '{"passed":true}'
        return "已处理"
    monkeypatch.setattr(graph, "chat_completion", completion)
    return compiled, plans


def assert_counts(factory, *, transactions=1, assistants=1):
    with factory() as db:
        assert db.scalar(select(func.count()).select_from(Transaction)) == transactions
        assert db.scalar(select(func.count()).select_from(ChatMessage).where(ChatMessage.role == "assistant")) == assistants


def test_staged_response_survives_response_commit_failure_without_replanning(business, monkeypatch):
    _, plans = install_graph(monkeypatch)
    original = idem.persist_response
    counter = []
    def fail_once(*args):
        counter.append(1)
        if len(counter) == 1:
            raise sqlite3.OperationalError("simulated final response commit failure")
        return original(*args)
    monkeypatch.setattr(idem, "persist_response", fail_once)
    chat = service.ChatService()
    async def scenario():
        with pytest.raises(sqlite3.OperationalError):
            await chat.chat("记账10元后解释", "alice", request_id="one")
        key, message_id, _ = idem.start_request(business, "alice", "one", "记账10元后解释", None)
        staged = recovery.load_outcome(key, "alice", message_id, idem.request_fingerprint("记账10元后解释", None))
        assert staged is not None
        assert_counts(business, assistants=0)
        restored = await chat.chat("记账10元后解释", "alice", request_id="one")
        assert restored == staged
        assert await chat.chat("记账10元后解释", "alice", request_id="one") == restored
        assert recovery.load_outcome(key, "alice", message_id, idem.request_fingerprint("记账10元后解释", None)) is None
    asyncio.run(scenario())
    assert len(plans) == 1
    assert_counts(business)


def test_commit_succeeded_but_acknowledgement_failed_does_not_add_another_assistant(business, monkeypatch):
    _, plans = install_graph(monkeypatch)
    original = idem.persist_response
    responses = []
    def commit_then_fail(*args):
        responses.append(original(*args))
        raise sqlite3.OperationalError("lost commit acknowledgement")
    monkeypatch.setattr(idem, "persist_response", commit_then_fail)
    chat = service.ChatService()
    async def scenario():
        with pytest.raises(sqlite3.OperationalError):
            await chat.chat("记账10元", "alice", request_id="one")
        assert await chat.chat("记账10元", "alice", request_id="one") == responses[0]
    asyncio.run(scenario())
    assert len(plans) == len(responses) == 1
    assert_counts(business)


def test_journal_failure_does_not_break_a_successful_primary_response_commit(business, monkeypatch):
    _, plans = install_graph(monkeypatch)
    monkeypatch.setattr(recovery, "stage_outcome", Mock(side_effect=sqlite3.OperationalError("journal unavailable")))
    result = asyncio.run(service.ChatService().chat("记账10元", "alice", request_id="one"))
    assert result["answer"] and not result["tool_error"]
    assert len(plans) == 1
    assert_counts(business)


def test_completed_checkpoint_can_recover_when_both_journal_and_response_staging_failed(business, monkeypatch):
    _, plans = install_graph(monkeypatch)
    monkeypatch.setattr(recovery, "stage_outcome", Mock(side_effect=sqlite3.OperationalError("journal unavailable")))
    original = idem.persist_response
    count = []
    def fail_once(*args):
        count.append(1)
        if len(count) == 1:
            raise sqlite3.OperationalError("primary unavailable")
        return original(*args)
    monkeypatch.setattr(idem, "persist_response", fail_once)
    chat = service.ChatService()
    async def scenario():
        with pytest.raises(sqlite3.OperationalError):
            await chat.chat("记账10元", "alice", request_id="one")
        result = await chat.chat("记账10元", "alice", request_id="one")
        assert result["recovery"] == {"source": "completed_checkpoint", "execution_metadata_complete": False}
        assert result["trace"]["meta"]["recovered_from_checkpoint"] is True
    asyncio.run(scenario())
    assert len(plans) == 1
    assert_counts(business)


def test_unfinished_execution_cannot_replan_even_after_repeated_retries(business, monkeypatch):
    key, message_id, _ = idem.start_request(business, "alice", "one", "记账10元", None)
    assert idem.claim_request(business, key, "alice", message_id)
    bookkeeping.bookkeeping_service.record_transaction(amount=-10, session_id="alice", operation_key="prior-write")
    checkpoint = SimpleNamespace(values={"session_id": "alice", "operation_id": f"message:{message_id}",
                                         "request_key": key, "answer": "partial"}, tasks=(), next=("decompose",))
    monkeypatch.setattr(service, "get_chat_graph", lambda: SimpleNamespace(aget_state=AsyncMock(return_value=checkpoint)))
    dispatch = AsyncMock(side_effect=AssertionError("Must not restart an unfinished graph"))
    monkeypatch.setattr(service, "arun_chat_graph", dispatch)
    async def scenario():
        for _ in range(2):
            with pytest.raises(ChatRequestRejected) as exc:
                await service.ChatService().chat("记账10元", "alice", request_id="one")
            assert exc.value.code == "request_outcome_unconfirmed"
            assert exc.value.retryable is False
    asyncio.run(scenario())
    dispatch.assert_not_called()
    assert_counts(business, assistants=0)


def test_prepared_but_never_dispatched_request_can_still_run_once(business, monkeypatch):
    idem.start_request(business, "alice", "one", "记账10元", None)
    _, plans = install_graph(monkeypatch)
    result = asyncio.run(service.ChatService().chat("记账10元", "alice", request_id="one"))
    assert result["answer"] and len(plans) == 1
    assert_counts(business)


def test_legacy_unfinished_receipt_without_dispatch_claim_is_not_permission_to_execute(business, monkeypatch):
    key, _, _ = idem.start_request(business, "alice", "legacy", "old", None)
    with business() as db:
        db.execute(delete(ChatExecution).where(ChatExecution.request_key == key))
        db.commit()
    monkeypatch.setattr(service, "get_chat_graph", lambda: None)
    dispatch = AsyncMock(side_effect=AssertionError("Legacy unknown execution must not rerun"))
    monkeypatch.setattr(service, "arun_chat_graph", dispatch)
    with pytest.raises(ChatRequestRejected, match="已阻止重新执行"):
        asyncio.run(service.ChatService().chat("old", "alice", request_id="legacy"))
    dispatch.assert_not_called()


def test_response_commit_is_first_wins_and_assistant_is_saved_once(business):
    key, message_id, _ = idem.start_request(business, "alice", "one", "hello", None)
    first = {"session_id": "alice", "answer": "first"}
    assert idem.persist_response(business, key, "alice", message_id, first) == first
    assert idem.persist_response(business, key, "alice", message_id, {**first, "answer": "second"}) == first
    assert_counts(business, transactions=0)


def test_journal_is_immutable_and_rejects_wrong_identity_or_payload():
    original = {"session_id": "alice", "answer": "first"}
    assert recovery.stage_outcome("key", "alice", 1, "fingerprint", original) == original
    assert recovery.stage_outcome("key", "alice", 1, "fingerprint", {**original, "answer": "different"}) == original
    for args in [("key", "bob", 1, "fingerprint"), ("key", "alice", 2, "fingerprint"), ("key", "alice", 1, "other")]:
        with pytest.raises(ChatRequestRejected):
            recovery.load_outcome(*args)


@pytest.mark.parametrize("change", ["tasks", "next", "session_id", "request_key", "operation_id", "finalize"])
def test_checkpoint_recovery_requires_exact_identity_and_proven_completion(change):
    values = {"session_id": "alice", "request_key": "key", "operation_id": "message:1", "answer": "done",
              "trace": {"steps": [{"name": "finalize"}]}}
    snapshot = SimpleNamespace(values=deepcopy(values), tasks=(), next=())
    if change in {"tasks", "next"}:
        setattr(snapshot, change, ("pending",))
    elif change == "finalize":
        snapshot.values["trace"] = {"steps": []}
    else:
        snapshot.values[change] = "different"
    compiled = SimpleNamespace(aget_state=AsyncMock(return_value=snapshot))
    assert asyncio.run(recovery.completed_checkpoint(compiled, "alice", 1, "key")) is None


def test_metadata_io_is_joined_before_cancellation_returns():
    started, release, finished = threading.Event(), threading.Event(), threading.Event()
    def slow_io():
        started.set()
        assert release.wait(5)
        finished.set()
    async def scenario():
        task = asyncio.create_task(service.ChatService._request_io(slow_io))
        await asyncio.to_thread(started.wait, 5)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert finished.is_set()
    try:
        asyncio.run(scenario())
    finally:
        release.set()


def test_dispatch_claim_is_atomic_and_scoped(business):
    key, message_id, _ = idem.start_request(business, "alice", "one", "hello", None)
    assert idem.claim_request(business, key, "bob", message_id) is False
    with ThreadPoolExecutor(max_workers=2) as pool:
        attempts = [pool.submit(idem.claim_request, business, key, "alice", message_id) for _ in range(2)]
        assert sorted(attempt.result() for attempt in attempts) == [False, True]


def test_recovery_store_outage_before_dispatch_does_not_consume_claim(business, monkeypatch):
    _, plans = install_graph(monkeypatch)
    original = recovery.load_outcome
    monkeypatch.setattr(recovery, "load_outcome", Mock(side_effect=sqlite3.OperationalError("temporarily locked")))
    chat = service.ChatService()
    async def scenario():
        with pytest.raises(ChatRequestRejected) as exc:
            await chat.chat("记账10元", "alice", request_id="one")
        assert exc.value.retryable is True
        assert len(plans) == 0
        monkeypatch.setattr(recovery, "load_outcome", original)
        assert (await chat.chat("记账10元", "alice", request_id="one"))["answer"]
    asyncio.run(scenario())
    assert len(plans) == 1
    assert_counts(business)


def test_legacy_completed_response_is_still_available_without_a_claim(business, monkeypatch):
    key, message_id, _ = idem.start_request(business, "alice", "legacy", "hello", None)
    result = {"answer": "legacy response", "session_id": "alice"}
    with business() as db:
        idem.finish_request(db, key, result)
        db.execute(delete(ChatExecution).where(ChatExecution.request_key == key))
        db.commit()
    dispatch = AsyncMock(side_effect=AssertionError("Cached legacy response must not invoke graph"))
    monkeypatch.setattr(service, "arun_chat_graph", dispatch)
    assert asyncio.run(service.ChatService().chat("hello", "alice", request_id="legacy")) == result
    dispatch.assert_not_called()


@pytest.mark.parametrize("value", ['not-json', '{"session_id":"bob","answer":"private response"}', '[]'])
def test_corrupt_or_wrong_identity_response_never_dispatches_again(business, monkeypatch, value):
    key, _, _ = idem.start_request(business, "alice", "one", "hello", None)
    with business() as db:
        db.execute(update(ChatRequest).where(ChatRequest.id == key).values(result_json=value))
        db.commit()
    dispatch = AsyncMock(side_effect=AssertionError("Invalid cached response must not replan"))
    monkeypatch.setattr(service, "arun_chat_graph", dispatch)
    with pytest.raises(ChatRequestRejected) as exc:
        asyncio.run(service.ChatService().chat("hello", "alice", request_id="one"))
    assert exc.value.code == "request_result_corrupt"
    assert "private response" not in str(exc.value)
    dispatch.assert_not_called()


def test_cancellation_before_graph_dispatch_cannot_later_execute_the_old_intent(business, monkeypatch):
    started, release = threading.Event(), threading.Event()
    def slow_lookup(*args):
        started.set()
        assert release.wait(5)
        return None
    monkeypatch.setattr(recovery, "load_outcome", slow_lookup)
    dispatch = AsyncMock(side_effect=AssertionError("Stopped request must not dispatch"))
    monkeypatch.setattr(service, "arun_chat_graph", dispatch)
    chat = service.ChatService()
    async def scenario():
        task = asyncio.create_task(chat.chat("记账10元", "alice", request_id="stopped"))
        assert await asyncio.to_thread(started.wait, 5)
        task.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        result = await chat.chat("记账10元", "alice", request_id="stopped")
        assert result["request_status"] == "cancelled"
    try:
        asyncio.run(scenario())
    finally:
        release.set()
    dispatch.assert_not_called()
    assert_counts(business, transactions=0)


def test_stop_after_result_staging_preserves_the_first_completed_outcome(business, monkeypatch):
    _, plans = install_graph(monkeypatch)
    started, release = threading.Event(), threading.Event()
    original = recovery.stage_outcome
    saved = []
    def slow_stage(*args):
        value = original(*args)
        if args[-1].get("request_status") != "cancelled":
            saved.append(value)
            started.set()
            assert release.wait(5)
        return value
    monkeypatch.setattr(recovery, "stage_outcome", slow_stage)
    chat = service.ChatService()
    async def scenario():
        task = asyncio.create_task(chat.chat("记账10元", "alice", request_id="one"))
        assert await asyncio.to_thread(started.wait, 5)
        task.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert await chat.chat("记账10元", "alice", request_id="one") == saved[0]
    try:
        asyncio.run(scenario())
    finally:
        release.set()
    assert len(plans) == 1
    assert_counts(business)
