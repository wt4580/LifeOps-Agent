import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app.src.common.config.db_config import Base
from app.src.common.harness.context import get_turn_context, turn_context
from app.src.domain.entity.chat_entity import InterruptedTurn, TodoItem
from app.src.service.idempotency import begin_write, finish_write, start_request
from app.src.service.interruption_service import save_interruption, take_interruption


@pytest.fixture
def business(tmp_path):
    engine = create_engine(f"sqlite:///{(tmp_path / 'business.db').as_posix()}")
    Base.metadata.create_all(engine)
    yield sessionmaker(bind=engine, expire_on_commit=False)
    engine.dispose()


def test_cancel_preserves_committed_effect_and_next_request_receives_handoff(business, monkeypatch):
    from app.src.agent import graph_chat as graph
    from app.src.service import chat_service as service
    monkeypatch.setattr(graph, "SessionLocal", business)
    monkeypatch.setattr(service, "SessionLocal", business)
    monkeypatch.setattr(service.settings, "background_learning_enabled", False)
    _, message_id, _ = start_request(business, "alice", "old", "创建待办再安排一天", None)

    class StoppedGraph:
        async def aget_state(self, config):
            return SimpleNamespace(tasks=[], values={})

        async def ainvoke(self, state, config):
            turn = get_turn_context("alice")
            with business() as db:
                key, _ = begin_write(db, "alice", "add_todo", {"title": "开会"}, "todo-one")
                todo = TodoItem(session_id="alice", title="开会")
                db.add(todo)
                db.flush()
                finish_write(db, key, {"id": todo.id})
                db.commit()
            turn.last_node = "_node_finalize"
            turn.partial_answer = "待办已经创建，日程建议是"
            turn.last_state = {**state, "plan": [{"action": "add_todo"}, {"action": "normal_chat"}], "plan_index": 1}
            turn.cancelled.set()
            raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(graph.arun_chat_graph(compiled_graph=StoppedGraph(), session_id="alice",
                    user_message="创建待办再安排一天", operation_id=f"message:{message_id}"))
    with business() as db:
        snapshot = json.loads(db.scalar(select(InterruptedTurn)).snapshot_json)
        assert db.scalar(select(TodoItem)).title == "开会"
    assert snapshot["committed_writes"][0]["kind"] == "add_todo"
    assert snapshot["plan_index"] == 1
    assert snapshot["partial_answer_unfinished"].endswith("日程建议是")

    mock_graph = AsyncMock(return_value={"answer": "按新条件继续", "trace": {"steps": []}})
    monkeypatch.setattr(service, "arun_chat_graph", mock_graph)
    chat = service.ChatService()
    old = asyncio.run(chat.chat("创建待办再安排一天", "alice", request_id="old"))
    assert old["request_status"] == "cancelled"
    assert mock_graph.await_count == 0
    asyncio.run(chat.chat("下午要留给运动", "alice", request_id="new"))
    assert mock_graph.call_args.kwargs["interrupted_context"] == snapshot
    assert mock_graph.call_args.kwargs["user_message"] == "下午要留给运动"
    assert take_interruption(business, "bob", message_id + 100) is None
    new_id = int(mock_graph.call_args.kwargs["operation_id"].split(":")[1])
    assert take_interruption(business, "alice", new_id) == snapshot
    assert take_interruption(business, "alice", new_id + 1) is None


def test_rolled_back_operation_is_never_reported_committed(business):
    with turn_context("alice") as turn:
        with business() as db:
            key, _ = begin_write(db, "alice", "add_todo", {"title": "未保存"}, "rollback")
            finish_write(db, key, {"id": 1})
            db.rollback()
        assert turn.committed_writes == {}
        # Cancellation metadata is permitted; cancelled business writes remain forbidden.
        turn.cancelled.set()
        with pytest.raises(asyncio.CancelledError), business() as db:
            db.add(TodoItem(session_id="alice", title="不能保存"))
            db.commit()
        save_interruption(business, turn, {}, "message:1", "原问题")
    assert take_interruption(business, "alice", 2)["committed_writes"] == []


def test_repeated_stops_preserve_original_goal_and_latest_conditions(business):
    with turn_context("alice") as turn:
        save_interruption(business, turn, {}, "message:1", "安排周末")
    context = take_interruption(business, "alice", 2)
    with turn_context("alice") as turn:
        turn.interrupted_context = context
        save_interruption(business, turn, {}, "message:2", "不要户外活动")
    latest = take_interruption(business, "alice", 3)
    assert latest["original_request"] == "安排周末"
    assert latest["user_inputs"] == ["安排周末", "不要户外活动"]
    assert take_interruption(business, "alice", 4) is None
