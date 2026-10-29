import asyncio
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from threading import Barrier

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker

from app.src.common.config.db_config import Base
from app.src.domain.entity.chat_entity import Account, ChatMessage, TodoItem, Transaction, WriteOperation
from app.src.service import bookkeeping_service as bookkeeping
from app.src.service import idempotency as idem


@pytest.fixture
def business(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'business.db'}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    monkeypatch.setattr(bookkeeping, "SessionLocal", factory)
    yield factory
    engine.dispose()


def test_transaction_retry_returns_original_result_without_reapplying_balance(business):
    service = bookkeeping.BookkeepingService()
    account = service.add_account("wallet", 100, "alice")
    args = dict(amount=-10, session_id="alice", account_id=account["id"], operation_key="request-1")
    first = service.record_transaction(**args)
    assert service.record_transaction(**args) == first
    assert service.get_account_balance("alice")[0]["balance"] == 90
    with business() as db:
        assert db.scalar(select(func.count()).select_from(Transaction)) == 1
    with pytest.raises(idem.IdempotencyConflict):
        service.record_transaction(**{**args, "amount": -20})
    service.record_transaction(**{**args, "operation_key": "request-2"})
    assert service.get_account_balance("alice")[0]["balance"] == 80


def test_operation_receipt_rolls_back_with_failed_business_write(business, monkeypatch):
    service = bookkeeping.BookkeepingService()
    account = service.add_account("wallet", 100, "alice")
    original = bookkeeping.finish_write
    monkeypatch.setattr(bookkeeping, "finish_write", lambda *args: (_ for _ in ()).throw(RuntimeError("crash")))
    args = dict(amount=-10, session_id="alice", account_id=account["id"], operation_key="r")
    with pytest.raises(RuntimeError):
        service.record_transaction(**args)
    with business() as db:
        assert db.scalar(select(func.count()).select_from(WriteOperation)) == 0
        assert db.scalar(select(func.count()).select_from(Transaction)) == 0
        assert db.get(Account, account["id"]).balance == 100
    monkeypatch.setattr(bookkeeping, "finish_write", original)
    service.record_transaction(**args)
    assert service.get_account_balance("alice")[0]["balance"] == 90


def test_concurrent_receipts_allow_only_one_transaction(business):
    service = bookkeeping.BookkeepingService()
    account = service.add_account("wallet", 100, "alice")
    barrier = Barrier(2)

    def write():
        barrier.wait()
        return service.record_transaction(amount=-10, session_id="alice", account_id=account["id"], operation_key="same")

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: write(), range(2)))
    assert results[0] == results[1]
    assert service.get_account_balance("alice")[0]["balance"] == 90


def test_confirm_replay_after_business_commit_does_not_duplicate_todos(business, monkeypatch):
    from app.src.agent import graph_chat as graph

    monkeypatch.setattr(graph, "SessionLocal", business)
    original = {"session_id": "alice", "proposal_id": "proposal-1", "decision": {"action": "propose_todo"},
                "proposal": {"items": [{"title": "meeting", "due_at": "2026-11-01"}]}, "trace": {"steps": []}}
    graph._handle_interrupt_resume(deepcopy(original), "确认")
    # Reload the OLD checkpoint as if the process died after the business commit.
    graph._handle_interrupt_resume(deepcopy(original), "确认")
    with business() as db:
        assert db.scalar(select(func.count()).select_from(TodoItem)) == 1
    original["proposal_id"] = "proposal-2"
    graph._handle_interrupt_resume(deepcopy(original), "确认")
    with business() as db:
        assert db.scalar(select(func.count()).select_from(TodoItem)) == 2


def test_balance_replay_cannot_revert_a_later_operation(business):
    service = bookkeeping.BookkeepingService()
    account = service.add_account("wallet", 100, "alice", operation_key="create")
    assert service.add_account("wallet", 100, "alice", operation_key="create") == account
    first = service.set_account_balance(account["id"], 200, "alice", operation_key="one")
    service.set_account_balance(account["id"], 300, "alice", operation_key="two")
    assert service.set_account_balance(account["id"], 200, "alice", operation_key="one") == first
    assert service.get_account_balance("alice")[0]["balance"] == 300


def test_operation_keys_are_scoped_to_user(business):
    service = bookkeeping.BookkeepingService()
    alice = service.add_account("wallet", 100, "alice", operation_key="same")
    bob = service.add_account("wallet", 200, "bob", operation_key="same")
    assert alice["id"] != bob["id"]


def test_completed_chat_retry_skips_graph_and_duplicate_messages(business, monkeypatch):
    from app.src.service import chat_service as module
    from unittest.mock import AsyncMock

    monkeypatch.setattr(module, "SessionLocal", business)
    monkeypatch.setattr(module.settings, "background_learning_enabled", False)
    graph = AsyncMock(return_value={"answer": "done", "trace": {"steps": []}})
    monkeypatch.setattr(module, "arun_chat_graph", graph)
    service = module.ChatService()

    async def run():
        first = await service.chat("hello", "alice", request_id="one")
        assert await service.chat("hello", "alice", request_id="one") == first
        with pytest.raises(idem.IdempotencyConflict):
            await service.chat("different", "alice", request_id="one")

    asyncio.run(run())
    assert graph.await_count == 1
    with business() as db:
        assert db.scalar(select(func.count()).select_from(ChatMessage)) == 2


def test_incomplete_request_retains_stable_message_id(business):
    key, first_id, result = idem.start_request(business, "alice", "r", "hello", None)
    assert result is None
    assert idem.start_request(business, "alice", "r", "hello", None) == (key, first_id, None)


def test_http_write_keys_replay_results_and_reject_changed_payload(business, monkeypatch):
    from fastapi.testclient import TestClient
    from app.src.common.config.app_config import create_app
    from app.src.service import auth_service

    monkeypatch.setattr(auth_service, "SessionLocal", business)
    token = auth_service.login_user("alice", "test-password", allow_create=True)
    client = TestClient(create_app())
    client.cookies.set(auth_service.COOKIE_NAME, token)
    try:
        account_body = {"session_id": "alice", "name": "wallet", "balance": 100}
        headers = {"Idempotency-Key": "account-1"}
        first = client.post("/api/accounts", json=account_body, headers=headers)
        assert first.status_code == 200
        assert client.post("/api/accounts", json=account_body, headers=headers).json() == first.json()
        assert client.post("/api/accounts", json={**account_body, "balance": 200}, headers=headers).status_code == 409
        transaction_body = {"session_id": "alice", "amount": -10, "account_id": first.json()["result"]["id"]}
        headers = {"Idempotency-Key": "transaction-1"}
        first = client.post("/api/transactions", json=transaction_body, headers=headers)
        assert first.status_code == 200
        assert client.post("/api/transactions", json=transaction_body, headers=headers).json() == first.json()
        assert client.post("/api/transactions", json={**transaction_body, "amount": -20}, headers=headers).status_code == 409
        assert bookkeeping.BookkeepingService().get_account_balance("alice")[0]["balance"] == 90
    finally:
        client.close()


def test_plan_service_confirmation_survives_cache_consumption(business, monkeypatch):
    from app.src.common.config import db_config
    from app.src.service.plan_service import PlanService

    monkeypatch.setattr(db_config, "SessionLocal", business)
    service = PlanService()
    service.proposal_cache["p"] = [{"title": "meeting", "due_at": "2026-11-01"}]
    service.proposal_owners["p"] = "alice"
    assert service.confirm_proposal("p", "alice")
    assert service.confirm_proposal("p", "alice")
    assert not service.confirm_proposal("p", "bob")
    with business() as db:
        assert db.scalar(select(func.count()).select_from(TodoItem)) == 1
