from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
from datetime import datetime, timedelta
from unittest.mock import Mock
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select, func
from sqlalchemy.orm import sessionmaker

from app.src.common.config.db_config import Base
from app.src.common.config.knowledge_db_config import KnowledgeBase
from app.src.domain.entity.chat_entity import (
    Account, ChatMessage, ConversationSummary, TodoItem, Transaction, NotificationReceipt,
)


@pytest.fixture
def business(tmp_path):
    engine = create_engine(f"sqlite:///{(tmp_path / 'business.db').as_posix()}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    yield factory
    engine.dispose()


@pytest.fixture
def knowledge(tmp_path, monkeypatch):
    from app.src.util.retrieval import retrieval
    engine = create_engine(f"sqlite:///{(tmp_path / 'knowledge.db').as_posix()}")
    KnowledgeBase.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    monkeypatch.setattr(retrieval, "KnowledgeSessionLocal", factory)
    yield factory
    engine.dispose()


@pytest.mark.parametrize("text", ["可以改成后天吗", "确认之前先改时间", "好的，但改成下午五点"])
def test_corrections_do_not_confirm(text):
    from app.src.agent.planner.planner import detect_affirmation
    assert not detect_affirmation(text)


def test_multi_date_plan_preserves_each_date(monkeypatch):
    from app.src.agent.planner import planner
    monkeypatch.setattr(planner, "normalize_date_hint", lambda *a, **k: "2026-10-28")
    monkeypatch.setattr(planner, "chat_completion", lambda *a, **k: json.dumps({"items": [
        {"title": "开会", "due_at": "2026-10-28T15:00:00"},
        {"title": "交报告", "due_at": "2026-10-29T10:00:00"},
    ]}))
    _, result = planner.propose_plan("明天开会，后天交报告")
    assert [item.due_at[:10] for item in result.items] == ["2026-10-28", "2026-10-29"]


def test_learning_appends_and_isolates_sessions(tmp_path, monkeypatch):
    from app.src.agent import post_chat_learn as learn
    from app.src.common import md_loader
    monkeypatch.setattr(learn, "DATA_DIR", tmp_path)
    monkeypatch.setattr(md_loader, "DATA_DIR", tmp_path)
    assert learn._write_markdown("AGENT_MEMO.md", "old memory", session_id="alice")
    assert learn._write_markdown("AGENT_MEMO.md", "new memory", session_id="alice")
    assert learn._write_markdown("AGENT_MEMO.md", "private bob", session_id="bob")
    context = md_loader.load_markdown_context("alice")
    assert "old memory" in context and "new memory" in context
    assert "private bob" not in context


def test_summary_failure_never_deletes_history(business, monkeypatch):
    from app.src.service import chat_service as module
    monkeypatch.setattr(module, "SessionLocal", business)
    with business() as db:
        db.add_all(ChatMessage(session_id="alice", role="user", content=str(i)) for i in range(60))
        db.commit()
    monkeypatch.setattr(module, "chat_completion", Mock(side_effect=RuntimeError("offline")))
    service = module.ChatService()
    asyncio.run(service._maybe_update_summary("alice"))
    service._maybe_prune_chat_history("alice")
    with business() as db:
        assert db.scalar(select(func.count()).select_from(ChatMessage)) == 60
        assert db.scalar(select(func.count()).select_from(ConversationSummary)) == 0


def test_summary_cursor_advances_without_deleting(business, monkeypatch):
    from app.src.service import chat_service as module
    monkeypatch.setattr(module, "SessionLocal", business)
    with business() as db:
        db.add_all(ChatMessage(session_id="alice", role="user", content=str(i)) for i in range(60))
        db.commit()
    monkeypatch.setattr(module, "chat_completion", Mock(return_value="summary"))
    service = module.ChatService()
    asyncio.run(service._maybe_update_summary("alice"))
    assert service._summary_cursor("alice") == 10
    with business() as db:
        assert db.scalar(select(func.count()).select_from(ChatMessage)) == 60


def test_sql_execution_scope_blocks_tautology_and_main_bypass(tmp_path):
    from app.src.service.db_query_service import scoped_connection
    path = tmp_path / "scope.db"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE chat_messages(session_id TEXT, content TEXT)")
        db.executemany("INSERT INTO chat_messages VALUES (?, ?)", [("alice", "A"), ("bob", "B")])
    with scoped_connection(str(path), "alice") as db:
        assert db.execute("SELECT * FROM chat_messages WHERE session_id='alice' OR 1=1").fetchall() == [("alice", "A")]
        assert db.execute("SELECT * FROM chat_messages WHERE 'session_id=alice' != ''").fetchall() == [("alice", "A")]
        assert db.execute("SELECT count(*) FROM chat_messages").fetchone() == (1,)
        with pytest.raises(sqlite3.DatabaseError):
            db.execute("SELECT * FROM main.chat_messages")
        with pytest.raises(sqlite3.DatabaseError):
            db.execute("SELECT * FROM (WITH chat_messages AS (SELECT * FROM main.chat_messages) SELECT * FROM chat_messages)")
        with pytest.raises(sqlite3.DatabaseError):
            db.execute("PRAGMA user_version=8")
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT count(*) FROM chat_messages").fetchone() == (2,)


def test_sql_indirect_messages_are_scoped(tmp_path):
    from app.src.service.db_query_service import scoped_connection
    path = tmp_path / "scope.db"
    with sqlite3.connect(path) as db:
        db.executescript("CREATE TABLE topic_episodes(id INTEGER, session_id TEXT); CREATE TABLE topic_episode_messages(episode_id INTEGER, content TEXT);")
        db.executemany("INSERT INTO topic_episodes VALUES (?,?)", [(1,"alice"),(2,"bob")])
        db.executemany("INSERT INTO topic_episode_messages VALUES (?,?)", [(1,"A"),(2,"B")])
    with scoped_connection(str(path), "alice") as db:
        assert db.execute("SELECT content FROM topic_episode_messages").fetchall() == [("A",)]


def test_document_summary_filter_handles_real_summary(knowledge):
    from app.src.util.retrieval import retrieval
    from app.src.domain.entity.knowledge_entity import Document
    from types import SimpleNamespace
    with knowledge() as db:
        db.add(Document(file_path="a.txt", summary="alpha beta", chunk_count=1))
        db.commit()
    row = SimpleNamespace(file_path="a.txt", chunk_text="alpha beta")
    assert retrieval._apply_document_summary_filter([row], "alpha") == [row]
    assert retrieval._apply_document_summary_filter([], "alpha") == []


def test_index_replaces_rewritten_and_deleted_files(knowledge, tmp_path, monkeypatch):
    from app.src.util.retrieval import retrieval
    from app.src.util.ingest.scanner import Document, TextChunk
    from app.src.domain.entity.knowledge_entity import DocumentChunk
    monkeypatch.setattr(retrieval.settings, "docs_dir", str(tmp_path))
    docs = [Document(path="a.txt", chunks=[TextChunk("old")])]
    monkeypatch.setattr(retrieval, "scan_documents", lambda _: docs)
    retrieval._index_documents_sqlite(rebuild=False)
    docs[:] = [Document(path="a.txt", chunks=[TextChunk("new")])]
    retrieval._index_documents_sqlite(rebuild=False)
    with knowledge() as db:
        assert db.scalars(select(DocumentChunk.chunk_text)).all() == ["new"]
    docs.clear()
    retrieval._index_documents_sqlite(rebuild=False)
    with knowledge() as db:
        assert db.scalar(select(func.count()).select_from(DocumentChunk)) == 0


def test_rerank_disabled_preserves_fusion_scores(monkeypatch):
    from app.src.util.retrieval import retrieval
    monkeypatch.setattr(retrieval.settings, "rag_use_rerank", False)
    hits = [retrieval.Citation(path="a",page=None,snippet="A",score=35),
            retrieval.Citation(path="b",page=None,snippet="B",score=60)]
    result = retrieval._rerank_fused("query", hits, top_k=2)
    assert [(h.path,h.score) for h in result] == [("a",35),("b",60)]


def test_balance_confirmation_and_account_ownership(business, monkeypatch):
    from app.src.service import bookkeeping_service as module
    from app.src.agent import graph_chat
    monkeypatch.setattr(module, "SessionLocal", business)
    service = module.BookkeepingService()
    account = service.add_account("wallet", 100, session_id="alice")
    state = {"session_id":"alice", "decision":{"action":"ask_user_confirm_balance"},
             "pending_balance_update":{"id":account["id"],"new_balance":200,"name":"wallet"}}
    result = graph_chat._handle_interrupt_resume(state, "确认")
    assert "200.00" in result["answer"]
    assert service.list_accounts("alice")[0]["balance"] == 200
    with pytest.raises(module.BookkeepingError):
        service.record_transaction(amount=-10, session_id="bob", account_id=account["id"])
    service.record_transaction(amount=-10, session_id="alice", account_id=account["id"], transaction_at="2026-10-26T08:00:00")
    assert service.list_transactions("alice",start_date="2026-10-26",end_date="2026-10-26")[0]["account_name"] == "wallet"
    assert service.list_accounts("alice")[0]["balance"] == 190


def test_balance_state_survives_graph_boundary():
    from langgraph.graph import StateGraph, START, END
    from app.src.domain.entity.chatstate_entity import ChatState
    graph = StateGraph(ChatState)
    graph.add_node("write", lambda state:{"pending_balance_update":{"id":1,"new_balance":200}})
    graph.add_node("read", lambda state:{"answer":str(state["pending_balance_update"]["new_balance"])})
    graph.add_edge(START,"write"); graph.add_edge("write","read"); graph.add_edge("read",END)
    assert graph.compile().invoke({"session_id":"alice"})["answer"] == "200"


def test_calendar_overlap_and_session_scope(business, monkeypatch):
    from app.src.service import calendar_service as module
    monkeypatch.setattr(module, "SessionLocal", business)
    service = module.CalendarService()
    for sid in ["alice", "bob"]:
        service.add_event(sid, datetime(2026,10,26,23), datetime(2026,10,27,3), session_id=sid)
    results = service.list_events("2026-10-27", "2026-10-28", session_id="alice")
    titles = {row["summary"] for row in results}
    assert "alice" in titles and "bob" not in titles


def test_recurring_task_not_active_before_start():
    from app.src.service.todo_service import TodoService
    todo = TodoItem(title="later",due_at=datetime(2026,11,1,9),repeat_rule="daily")
    assert not TodoService._is_recurring_active(todo, datetime(2026,10,27,9))


def test_due_notifications_retry_then_drain_all_candidates(business, monkeypatch):
    from app.src.service import scheduler, push_service
    from app.src.common.config import db_config
    monkeypatch.setattr(db_config,"SessionLocal",business)
    now = datetime.now().replace(second=0,microsecond=0)
    class Clock(datetime):
        @classmethod
        def now(cls): return now
    monkeypatch.setattr(scheduler,"datetime",Clock)
    with business() as db:
        for i in range(4):
            db.add(TodoItem(session_id="alice",title=f"task{i}",due_at=now-timedelta(minutes=1),is_completed=False))
        db.add(TodoItem(session_id="alice",title="daily",due_at=now-timedelta(days=2,minutes=1),
                        repeat_rule="daily",is_completed=True,completed_at=now-timedelta(days=1)))
        db.commit()
    push = Mock(return_value=False)
    monkeypatch.setattr(push_service,"push_notification",push)
    asyncio.run(scheduler._push_due_todos_job())
    with business() as db:
        assert db.scalar(select(func.count()).select_from(NotificationReceipt)) == 0
    push.return_value=True
    asyncio.run(scheduler._push_due_todos_job())
    asyncio.run(scheduler._push_due_todos_job())
    with business() as db:
        assert db.scalar(select(func.count()).select_from(NotificationReceipt)) == 5


def test_cancelled_sync_tool_cannot_commit_after_stop(business):
    from app.src.agent.graph_chat import _cancellable_node
    from app.src.common.harness.context import turn_context
    entered=threading.Event(); release=threading.Event()
    def node(state):
        with business() as db:
            db.add(ChatMessage(session_id="alice",role="user",content="must not commit"))
            entered.set(); release.wait(2); db.commit()
        return state
    async def scenario():
        with turn_context("alice"):
            task=asyncio.create_task(_cancellable_node(node)({"session_id":"alice"}))
            await asyncio.to_thread(entered.wait,2)
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
            release.set()
            with pytest.raises(asyncio.CancelledError): await task
    asyncio.run(scenario())
    with business() as db:
        assert db.scalar(select(func.count()).select_from(ChatMessage)) == 0


def test_password_login_api_scope_and_revocation():
    from app.src.common.config.app_config import create_app
    client = TestClient(create_app())
    username = "auth-"+uuid4().hex[:12]
    from app.src.service.auth_service import login_user
    login_user(username,"correct-password",allow_create=True)
    assert client.get("/api/accounts",params={"session_id":username}).status_code == 401
    assert client.post("/api/auth/login",json={"username":username,"password":"correct-password"}).status_code == 200
    assert client.get("/api/accounts",params={"session_id":username}).status_code == 200
    assert client.get("/api/accounts",params={"session_id":"somebody-else"}).status_code == 403
    assert client.request("DELETE","/api/accounts/999",params={"session_id":"somebody-else"},json={"session_id":username}).status_code == 403
    assert client.post("/api/accounts",params={"session_id":username},json={"name":"orphan","balance":1}).status_code == 422
    assert client.post("/api/accounts",json={"session_id":username,"name":"wallet","balance":12}).status_code == 200
    assert client.post("/api/auth/logout").status_code == 200
    assert client.get("/api/auth/me").status_code == 401
    assert client.post("/api/auth/login",json={"username":username,"password":"wrong-password"}).status_code == 401
    client.close()


def test_async_graph_hitl_revision_and_confirmation(business, monkeypatch):
    from app.src.agent import graph_chat as graph
    from app.src.common.config import db_config
    from app.src.domain.dto.planner_dto import PlanProposal, PlanItem
    from langgraph.checkpoint.memory import MemorySaver
    monkeypatch.setattr(graph,"SessionLocal",business)
    monkeypatch.setattr(db_config,"SessionLocal",business)
    revised = Mock(return_value=("new-proposal",PlanProposal(items=[PlanItem(title="meeting",due_at="2026-10-29T15:00:00")])))
    monkeypatch.setattr(graph,"propose_plan",revised)
    compiled=graph.build_chat_graph(checkpointer=MemorySaver())
    async def scenario():
        first=await graph.arun_chat_graph(compiled_graph=compiled,session_id="alice",user_message="安排开会",
            forced_action="propose_todo",forced_args={"title":"meeting","due_at":"2026-10-28T15:00:00"})
        assert first["proposal_id"]
        assert (await compiled.aget_state({"configurable":{"thread_id":"alice"}})).tasks
        second=await graph.arun_chat_graph(compiled_graph=compiled,session_id="alice",user_message="可以改成后天吗")
        assert second["used_tool"] == "revise_proposal"
        assert "meeting" in revised.call_args.args[0]
        with business() as db:
            assert db.scalar(select(func.count()).select_from(TodoItem)) == 0
        third=await graph.arun_chat_graph(compiled_graph=compiled,session_id="alice",user_message="确认")
        assert third["used_tool"] == "plan_gate_confirm"
        with business() as db:
            todos=db.scalars(select(TodoItem)).all()
            assert len(todos)==1 and todos[0].due_at == datetime(2026,10,29,15)
    asyncio.run(scenario())


def test_stale_proposal_id_cannot_confirm_new_card(monkeypatch):
    from app.src.service import chat_service
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    graph=SimpleNamespace(aget_state=AsyncMock(return_value=SimpleNamespace(tasks=[object()],values={"proposal_id":"new"})))
    monkeypatch.setattr(chat_service,"get_chat_graph",lambda:graph)
    with pytest.raises(ValueError,match="草案已失效"):
        asyncio.run(chat_service.ChatService().chat("确认","alice",proposal_id="old"))


def test_stop_request_id_cannot_cancel_a_different_job():
    from app.src.service.session_queue import submit_chat_job,cancel_and_wait,close_session
    async def scenario():
        waiting=asyncio.Event()
        async def job():await waiting.wait()
        submit_chat_job("request-id-test",asyncio.Queue(),job,request_id="current")
        await asyncio.sleep(0)
        assert not await cancel_and_wait("request-id-test",request_id="old")
        assert await cancel_and_wait("request-id-test",request_id="current")
        await close_session("request-id-test")
    asyncio.run(scenario())


def test_existing_database_migration_preserves_rows(tmp_path,monkeypatch):
    from app.src.common.config import db_config
    path=tmp_path/"legacy.db"
    with sqlite3.connect(path) as db:
        db.executescript("""
        CREATE TABLE memory_candidates(id INTEGER PRIMARY KEY,kind TEXT,title TEXT,occurred_at TEXT,notes TEXT,confidence FLOAT,insight_type TEXT,created_at DATETIME);
        INSERT INTO memory_candidates(id,kind,title) VALUES(1,'fact','legacy memory');
        CREATE TABLE conversation_summaries(id INTEGER PRIMARY KEY,session_id TEXT,summary_text TEXT,updated_at DATETIME);
        INSERT INTO conversation_summaries(id,session_id,summary_text) VALUES(1,'alice','legacy summary');
        """)
    engine=create_engine(f"sqlite:///{path.as_posix()}")
    monkeypatch.setattr(db_config,"engine",engine)
    monkeypatch.setattr(db_config.settings,"database_url",f"sqlite:///{path.as_posix()}")
    db_config.init_db()
    db_config.init_db()
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT title,session_id FROM memory_candidates").fetchone()==("legacy memory",None)
        assert db.execute("SELECT summary_text,summarized_through_id FROM conversation_summaries").fetchone()==("legacy summary",0)
    engine.dispose()


def test_new_remote_username_cannot_claim_existing_data():
    from app.src.common.config.app_config import create_app
    client=TestClient(create_app())
    response=client.post("/api/auth/login",json={"username":"remote-"+uuid4().hex[:10],"password":"a-long-password"})
    assert response.status_code==401
    assert "localhost" in response.json()["detail"]
    client.close()


def test_application_startup_and_shutdown_without_external_services(monkeypatch):
    from app.src.common.config.app_config import create_app
    from app.src.service import scheduler
    monkeypatch.setattr(scheduler,"start_scheduler",lambda:None)
    monkeypatch.setattr(scheduler,"stop_scheduler",lambda:None)
    with TestClient(create_app()) as client:
        assert client.get("/health").status_code==200


def test_reflection_counter_accumulates_on_shared_service(business,monkeypatch):
    from app.src.service import chat_service as module
    monkeypatch.setattr(module,"SessionLocal",business)
    llm=Mock(return_value="{}")
    monkeypatch.setattr(module,"chat_completion",llm)
    service=module.ChatService()
    async def scenario():
        for _ in range(3): await service._maybe_deep_reflection("alice","alice")
    asyncio.run(scenario())
    assert llm.call_count==1


def test_empty_vector_rebuild_preserves_unrelated_files(tmp_path,monkeypatch):
    from app.src.util.retrieval import rag_langchain
    monkeypatch.setattr(rag_langchain.settings,"rag_vector_dir",str(tmp_path))
    monkeypatch.setattr(rag_langchain,"_load_documents_from_db",lambda:[])
    unrelated=tmp_path/"keep.txt"
    unrelated.write_text("keep",encoding="utf-8")
    rag_langchain.build_vector_index(rebuild=True)
    assert unrelated.read_text(encoding="utf-8")=="keep"


def test_plan_endpoint_keeps_query_session_contract(monkeypatch):
    from app.src.common.config.app_config import create_app
    from app.src.service.auth_service import login_user
    from app.src.service.plan_service import plan_service
    username="plan-"+uuid4().hex[:10]
    login_user(username,"correct-password",allow_create=True)
    monkeypatch.setattr(plan_service,"generate_proposal",lambda **kw:{"proposal_id":"test","proposal":[]})
    client=TestClient(create_app())
    assert client.post("/api/auth/login",json={"username":username,"password":"correct-password"}).status_code==200
    assert client.post("/api/plan/propose",params={"session_id":username},json={"text":"明天开会"}).status_code==200
    client.close()


def test_existing_account_change_interrupts_and_confirms_in_real_graph(business,monkeypatch):
    from app.src.agent import graph_chat as graph
    from app.src.service import bookkeeping_service
    from langgraph.checkpoint.memory import MemorySaver
    monkeypatch.setattr(graph,"SessionLocal",business)
    monkeypatch.setattr(bookkeeping_service,"SessionLocal",business)
    service=bookkeeping_service.BookkeepingService()
    service.add_account("wallet",100,session_id="alice")
    compiled=graph.build_chat_graph(checkpointer=MemorySaver())
    async def scenario():
        first=await graph.arun_chat_graph(compiled_graph=compiled,session_id="alice",user_message="我的账户现在有200元",
            forced_action="create_account",forced_args={"name":"wallet","balance":200})
        assert first["pending_balance_update"]["new_balance"]==200
        assert (await compiled.aget_state({"configurable":{"thread_id":"alice"}})).tasks
        assert service.list_accounts("alice")[0]["balance"]==100
        await graph.arun_chat_graph(compiled_graph=compiled,session_id="alice",user_message="确认")
        assert service.list_accounts("alice")[0]["balance"]==200
    asyncio.run(scenario())


def test_document_summary_budget_keeps_committed_progress(knowledge,monkeypatch):
    from app.src.util.retrieval import retrieval
    from app.src.domain.entity.knowledge_entity import Document,DocumentChunk
    from app.src.common.harness.context import get_turn_context
    from app.src.common.harness.interceptors import CallContext,llm_chain
    monkeypatch.setattr(retrieval.settings,"background_max_llm_calls",1)
    with knowledge() as db:
        for i in range(3):
            db.add(Document(file_path=f"{i}.txt",chunk_count=1))
            db.add(DocumentChunk(file_path=f"{i}.txt",chunk_text="content",chunk_hash=str(i)))
        db.commit()
    def fake_llm(*args,**kwargs):
        turn=get_turn_context(None)
        return llm_chain().invoke(CallContext("llm","document_summary",turn.session_id,turn=turn),lambda:"summary")
    monkeypatch.setattr(retrieval,"chat_completion",fake_llm)
    retrieval._generate_document_summaries()
    with knowledge() as db:
        results=db.scalars(select(Document.summary).order_by(Document.id)).all()
    assert results==["summary",None,None]


def test_partial_summary_coverage_does_not_hide_unprocessed_documents(knowledge,monkeypatch):
    from app.src.util.retrieval import retrieval
    from app.src.domain.entity.knowledge_entity import Document
    from types import SimpleNamespace
    with knowledge() as db:
        db.add(Document(file_path="matched.txt",summary="alpha",chunk_count=1))
        db.add(Document(file_path="other.txt",summary="beta",chunk_count=1))
        db.add(Document(file_path="pending.txt",summary=None,chunk_count=1))
        db.commit()
    class BM25:
        def __init__(self,corpus): pass
        def get_scores(self,tokens): return [1,0]
    monkeypatch.setattr(retrieval,"BM25Okapi",BM25)
    rows=[SimpleNamespace(file_path=path) for path in ["matched.txt","other.txt","pending.txt"]]
    results=retrieval._apply_document_summary_filter(rows,"alpha")
    assert [row.file_path for row in results]==["matched.txt","pending.txt"]
