from __future__ import annotations

import asyncio
import json
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from app.src.common.config import llm_config
from app.src.common.config.base_config import settings
from app.src.common.harness.context import (
    TurnBudget, TurnContext, TurnLimitExceeded, background_context,
    ensure_turn_active, get_turn_context, turn_context,
)
from app.src.common.harness.interceptors import (
    CallContext, Decision, InterceptorChain,
    PolicyViolation, TraceInterceptor, llm_chain, tool_chain,
)


def client_for(monkeypatch, response=None, error=None):
    response = response or SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="hello"), finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=3, total_tokens=13),
    )
    create = Mock(return_value=response, side_effect=error)
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    client.with_options = Mock(return_value=client)
    monkeypatch.setattr(llm_config, "get_client", lambda: client)
    return client, create


@pytest.fixture(autouse=True)
def audit_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "harness_trace_dir", str(tmp_path / "audit"))


@pytest.mark.parametrize("exception, status", [
    (RuntimeError("private error text"), "error"),
    (asyncio.CancelledError(), "cancelled"),
    (TurnLimitExceeded("budget"), "budget_exceeded"),
])
def test_failed_invocation_is_recorded_once(exception, status):
    turn = TurnContext("test", TurnBudget(3, 3, 30))
    call = CallContext("llm", "test", "test", turn=turn)
    def fail():
        raise exception
    with pytest.raises(type(exception)):
        llm_chain().invoke(call, fail)
    assert turn.budget.llm_calls == 1
    assert len(turn.records) == 1
    assert turn.records[0]["status"] == status
    assert turn.records[0]["error_type"] == type(exception).__name__
    assert "private error text" not in json.dumps(turn.records)


def test_rejected_before_hook_and_budget_are_recorded():
    turn = TurnContext("test", TurnBudget(3, 3, 30))
    call = CallContext("tool", "tool:unknown", "test", turn=turn, payload={"action":"unknown"})
    operation = Mock()
    with pytest.raises(PolicyViolation):
        tool_chain().invoke(call, operation)
    operation.assert_not_called()
    assert turn.records[0]["status"] == "blocked"
    turn.budget.max_llm_calls = 0
    with pytest.raises(TurnLimitExceeded):
        llm_chain().invoke(CallContext("llm", "test", "test", turn=turn), operation)
    assert turn.records[-1]["status"] == "budget_exceeded"
    assert turn.budget.llm_calls == 0


def test_after_hook_failure_is_not_reported_as_success():
    class BrokenAfter:
        def before(self, ctx): return Decision.CONTINUE
        def after(self, ctx, result): raise ValueError("after failed")
    turn = TurnContext("test", TurnBudget(3, 3, 30))
    chain = InterceptorChain([BrokenAfter(), TraceInterceptor()])
    with pytest.raises(ValueError):
        chain.invoke(CallContext("tool", "test", "test", turn=turn), lambda: "result")
    assert len(turn.records) == 1 and turn.records[0]["status"] == "error"


def test_nested_llm_record_links_to_tool():
    turn = TurnContext("test", TurnBudget(3, 3, 30))
    parent = CallContext("tool", "tool:normal_chat", "test", turn=turn, payload={"action":"normal_chat"})
    child = CallContext("llm", "normal_chat", "test", turn=turn)
    tool_chain().invoke(parent, lambda: llm_chain().invoke(child, lambda: "hello"))
    assert turn.records[0]["parent_call_id"] == parent.call_id
    assert turn.records[1]["parent_call_id"] is None


def test_reported_usage_is_settled_and_unreported_usage_is_reserved(monkeypatch):
    _, create = client_for(monkeypatch)
    with turn_context("usage") as turn:
        assert llm_config.chat_completion([{"role":"user","content":"hello"}]) == "hello"
        assert turn.budget.llm_calls == 1
        assert turn.budget.charged_tokens == 13 and turn.budget.reserved_tokens == 0
        assert turn.records[0]["usage_source"] == "reported"
        assert create.call_args.kwargs["max_tokens"] == settings.llm_max_output_tokens
        create.return_value.usage = None
        llm_config.chat_completion([{"role":"user","content":"hello"}])
        assert turn.budget.usage_missing_calls == 1
        assert turn.records[-1]["usage_source"] == "reservation"
        assert turn.budget.charged_tokens > 13 and turn.budget.reserved_tokens == 0


def test_token_limit_blocks_before_network(monkeypatch):
    _, create = client_for(monkeypatch)
    with turn_context("tokens") as turn:
        turn.budget.max_total_tokens = 1
        with pytest.raises(TurnLimitExceeded):
            llm_config.chat_completion([{"role":"user","content":"hello"}])
        create.assert_not_called()
        assert turn.budget.charged_tokens == turn.budget.reserved_tokens == 0
        assert turn.records[-1]["status"] == "budget_exceeded"
        assert turn.records[-1]["usage_source"] == "not_sent"


def test_provider_failure_consumes_conservative_reservation(monkeypatch):
    _, create = client_for(monkeypatch, error=RuntimeError("provider failed"))
    with turn_context("provider-error") as turn:
        with pytest.raises(RuntimeError):
            llm_config.chat_completion([{"role":"user","content":"hello"}])
        assert create.call_count == 1
        assert turn.budget.charged_tokens > 0
        assert turn.budget.reserved_tokens == 0
        assert turn.records[-1]["status"] == "error"


def test_client_creation_failure_releases_reservation(monkeypatch):
    monkeypatch.setattr(llm_config, "get_client", Mock(side_effect=RuntimeError("setup failed")))
    with turn_context("client-error") as turn:
        with pytest.raises(RuntimeError):
            llm_config.chat_completion([{"role":"user","content":"hello"}])
        assert turn.budget.charged_tokens == turn.budget.reserved_tokens == 0
        assert turn.records[-1]["usage_source"] == "not_sent"


class Stream:
    def __init__(self, chunks):
        self.chunks = chunks
        self.closed = False
    def __iter__(self): return iter(self.chunks)
    def close(self): self.closed = True


def test_stream_usage_footer_and_cleanup(monkeypatch):
    stream = Stream([
        SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="hello"),finish_reason=None)]),
        SimpleNamespace(choices=[],usage={"prompt_tokens":9,"completion_tokens":3,"total_tokens":12}),
    ])
    _, create = client_for(monkeypatch, response=stream)
    monkeypatch.setattr(settings, "llm_stream_include_usage", True)
    with turn_context("stream") as turn:
        tokens = []
        assert llm_config.chat_completion([{"role":"user","content":"hello"}],on_token=tokens.append) == "hello"
        assert tokens == ["hello"] and stream.closed
        assert turn.budget.charged_tokens == 12
        assert create.call_args.kwargs["stream_options"] == {"include_usage":True}


def test_stream_cancellation_closes_and_records(monkeypatch):
    stream = Stream([SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="hello"),finish_reason=None)])])
    client_for(monkeypatch, response=stream)
    def cancel(_): raise asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        with turn_context("cancel-stream") as turn:
            llm_config.chat_completion([{"role":"user","content":"hello"}],on_token=cancel)
    assert stream.closed
    assert turn.status == turn.records[-1]["status"] == "cancelled"
    assert turn.budget.reserved_tokens == 0
    assert get_turn_context("cancel-stream") is None


def test_background_budget_does_not_replace_foreground(monkeypatch):
    client_for(monkeypatch)
    with turn_context("same") as foreground:
        with background_context("same",parent_turn_id=foreground.turn_id) as background:
            background.budget.max_llm_calls = 1
            llm_config.chat_completion([{"role":"user","content":"hello"}])
            with pytest.raises(TurnLimitExceeded):
                llm_config.chat_completion([{"role":"user","content":"hello"}])
            assert foreground.budget.llm_calls == 0
            assert background.budget.llm_calls == 1
        assert get_turn_context("same") is foreground
        assert background.summary()["parent_turn_id"] == foreground.turn_id


def test_audit_contains_metadata_not_prompt_or_tool_payload(tmp_path):
    with turn_context("private-user-name") as turn:
        turn.add_record({"kind":"tool","name":"normal_chat","status":"success",
                         "prompt":"secret message", "payload":{"api_key":"secret key"}})
    records = [json.loads(line) for path in (tmp_path/"audit").glob("*.jsonl") for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 1
    text = json.dumps(records)
    assert "private-user-name" not in text and "secret message" not in text and "secret key" not in text
    assert records[0]["turn_id"] == turn.turn_id and records[0]["status"] == "success"


def test_final_summary_includes_post_finalize_quality_call(monkeypatch):
    from app.src.agent import graph_chat as graph
    def quality(*args, **kwargs):
        current = get_turn_context(kwargs["runtime_context"]["session_id"])
        return llm_chain().invoke(CallContext("llm","quality_gate",current.session_id,turn=current),lambda:'{"passed":true}')
    monkeypatch.setattr(graph,"chat_completion",quality)
    state={"session_id":"summary","mode":"normal","answer":"hello","user_message":"hello","used_tool":"normal_chat","history":[],"trace":{"steps":[]}}
    with turn_context("summary") as turn:
        graph._node_finalize(state)
        graph._node_quality_gate(state)
        result=graph._finish_turn_result(state,turn)
    summaries=[step["output"] for step in result["trace"]["steps"] if step["name"]=="harness_summary"]
    assert len(summaries)==1 and summaries[0]["llm_calls"]==1
    assert summaries[0]["invocations"][0]["name"]=="quality_gate"


def test_wall_deadline_stops_async_graph_and_cleans_context(monkeypatch):
    from app.src.agent import graph_chat as graph
    monkeypatch.setattr(settings,"chat_turn_deadline_seconds",0.03)
    class SlowGraph:
        async def aget_state(self,config): return SimpleNamespace(values={},tasks=[])
        async def ainvoke(self,state,config): await asyncio.Event().wait()
    result=asyncio.run(graph.arun_chat_graph(compiled_graph=SlowGraph(),session_id="deadline",user_message="hello"))
    assert result["turn_aborted"]
    summary=next(step["output"] for step in result["trace"]["steps"] if step.get("name")=="harness_summary")
    assert summary["status"]=="budget_exceeded"
    assert get_turn_context("deadline") is None


def test_deadline_waits_for_sync_node_and_blocks_late_side_effect(monkeypatch):
    from app.src.agent import graph_chat as graph
    monkeypatch.setattr(settings,"chat_turn_deadline_seconds",0.04)
    release=threading.Event()
    exited=threading.Event()
    side_effects=[]
    def node(state):
        try:
            release.wait(1)
            ensure_turn_active()
            side_effects.append("write")
            return state
        finally: exited.set()
    class SlowGraph:
        async def aget_state(self,config): return SimpleNamespace(values={},tasks=[])
        async def ainvoke(self,state,config): return await graph._cancellable_node(node)(state)
    async def scenario():
        asyncio.get_running_loop().call_later(0.1,release.set)
        return await graph.arun_chat_graph(compiled_graph=SlowGraph(),session_id="sync-deadline",user_message="hello")
    result=asyncio.run(scenario())
    assert result["turn_aborted"] and exited.is_set() and not side_effects


def test_post_graph_worker_stops_at_its_own_budget(monkeypatch):
    from app.src.service import chat_service as service
    client_for(monkeypatch)
    monkeypatch.setattr(settings,"background_max_llm_calls",1)
    monkeypatch.setattr(service,"_POST_GRAPH_EXECUTOR",SimpleNamespace(submit=lambda operation:operation()))
    reached=[]
    def learn(**kwargs):
        llm_config.chat_completion([{"role":"user","content":"first"}])
        llm_config.chat_completion([{"role":"user","content":"must be blocked"}])
        reached.append("after budget")
    monkeypatch.setattr(service,"run_learn",learn)
    instance=service.ChatService()
    with turn_context("worker") as foreground:
        instance._spawn_post_graph("worker","worker","hello",{"mode":"normal","recent_dialogue":[{}]},"hello",lambda _:None)
        assert foreground.budget.llm_calls==0
    assert not reached


def test_audit_io_failure_does_not_break_context_cleanup(tmp_path,monkeypatch):
    blocked=tmp_path/"not-a-directory"
    blocked.write_text("keep",encoding="utf-8")
    monkeypatch.setattr(settings,"harness_trace_dir",str(blocked))
    with turn_context("audit-failure") as turn:
        pass
    assert turn.status=="success" and get_turn_context("audit-failure") is None
    assert blocked.read_text(encoding="utf-8")=="keep"


def test_isolated_api_context_does_not_clear_foreground_registry():
    with turn_context("same") as foreground:
        with turn_context("same",register=False,operation="rag_ask") as standalone:
            assert get_turn_context("same") is standalone
        assert get_turn_context("same") is foreground


def test_usage_above_reservation_is_recorded_and_stops_turn(monkeypatch):
    _, create=client_for(monkeypatch)
    create.return_value.usage=SimpleNamespace(prompt_tokens=200000,completion_tokens=1,total_tokens=200001)
    with turn_context("usage-overflow") as turn:
        with pytest.raises(TurnLimitExceeded):
            llm_config.chat_completion([{"role":"user","content":"hello"}])
        assert turn.budget.charged_tokens==200001
        assert turn.records[-1]["usage_source"]=="reported"
        assert turn.records[-1]["status"]=="budget_exceeded"


def test_standalone_api_budget_is_enforced(monkeypatch):
    from fastapi.testclient import TestClient
    from uuid import uuid4
    from app.src.common.config.app_config import create_app
    from app.src.service.auth_service import login_user
    from app.src.controller import rag_controller, plan_controller
    username="harness-"+uuid4().hex[:8]
    login_user(username,"correct-password",allow_create=True)
    _,create=client_for(monkeypatch)
    def ask(**kwargs):
        llm_config.chat_completion([{"role":"user","content":"hello"}])
        return {"answer":"hello","citations":[]}
    monkeypatch.setattr(rag_controller.rag_service,"ask_question",ask)
    monkeypatch.setattr(plan_controller.plan_service,"generate_proposal",ask)
    client=TestClient(create_app())
    assert client.post("/api/auth/login",json={"username":username,"password":"correct-password"}).status_code==200
    monkeypatch.setattr(settings,"chat_turn_max_llm_calls",0)
    response=client.post("/api/ask",json={"question":"hello"})
    assert response.status_code==200
    summary=response.json()["result"]["trace"]["steps"][0]["output"]
    assert summary["status"]=="budget_exceeded" and summary["operation"]=="rag_ask"
    assert client.post("/api/plan/propose",params={"session_id":username},json={"text":"hello"}).status_code==429
    create.assert_not_called()
    client.close()


def test_already_cancelled_model_attempt_is_recorded(monkeypatch):
    _,create=client_for(monkeypatch)
    with turn_context("already-cancelled") as turn:
        turn.cancelled.set()
        with pytest.raises(asyncio.CancelledError):
            llm_config.chat_completion([{"role":"user","content":"hello"}])
        assert turn.records[-1]["status"]=="cancelled"
        create.assert_not_called()


def test_unrelated_timeout_is_not_misclassified_as_budget_exceeded():
    from app.src.agent import graph_chat as graph
    class BrokenGraph:
        async def aget_state(self,config): return SimpleNamespace(values={},tasks=[])
        async def ainvoke(self,state,config): raise TimeoutError("independent operation failed")
    with pytest.raises(TimeoutError):
        asyncio.run(graph.arun_chat_graph(compiled_graph=BrokenGraph(),session_id="operation-timeout",user_message="hello"))
    assert get_turn_context("operation-timeout") is None


def test_valid_long_plan_is_not_silently_cut_off_at_four_steps():
    from app.src.agent import graph_chat as graph
    plan=[{"action":"query_todos","args":{},"purpose":str(i)} for i in range(6)]
    state={"session_id":"plan","user_message":"hello","cycle_count":99,"plan":plan,"plan_index":4}
    result=graph._node_decompose(state)
    assert result["decision"]["action"]=="query_todos" and len(result["plan"])==6


def test_aborted_plan_exits_without_advancing_or_retrying():
    from app.src.agent import graph_chat as graph
    state={"turn_aborted":True,"retry_info":{"reason":"retry"},"retry_count":0,
           "plan":[{"action":"normal_chat"} for _ in range(100)],"plan_index":0,"answer":"stopped"}
    assert graph._node_advance_plan(state)["plan_index"]==0
    assert graph._route_after_advance(state)=="finalize"
