import asyncio
import json
import logging
import re
from uuid import uuid4
from types import SimpleNamespace
from unittest.mock import AsyncMock
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from app.src.common import telemetry
from app.src.common.harness.context import turn_context


@pytest.fixture
def collector(monkeypatch):
    from app.src.service import session_queue
    value = telemetry.Metrics()
    monkeypatch.setattr(telemetry, "metrics", value)
    monkeypatch.setattr(session_queue, "metrics", value)
    session_queue.start_chat_jobs()
    yield value
    session_queue.start_chat_jobs()


def test_http_trace_measures_full_stream_and_keeps_content_out_of_logs(collector, caplog):
    caplog.set_level(logging.INFO, logger=telemetry.__name__)
    async def scenario():
        started, release = asyncio.Event(), asyncio.Event()
        seen = []
        async def app(scope, receive, send):
            scope["route"] = SimpleNamespace(path="/stream/{id}")
            seen.append(telemetry.current_trace_id())
            await send({"type": "http.response.start", "status": 200,
                        "headers": [(b"x-trace-id", b"not-server-generated")]})
            started.set()
            await release.wait()
            await send({"type": "http.response.body", "body": b"private answer"})
        sent = []
        async def send(message):
            sent.append(message)
        scope = {"type": "http", "method": "GET", "path": "/stream/private-user-name",
                 "query_string": b"secret=private-key", "headers": [(b"x-trace-id", b"forged")]}
        task = asyncio.create_task(telemetry.TelemetryMiddleware(app)(scope, AsyncMock(), send))
        await started.wait()
        during = collector.render()
        assert "lifeops_http_in_progress 1" in during
        assert "lifeops_http_requests_total" not in during
        release.set()
        await task
        assert re.fullmatch("[0-9a-f]{32}", seen[0])
        assert dict(sent[0]["headers"])[b"x-trace-id"].decode() == seen[0]
        assert telemetry.current_trace_id() is None
        assert 'lifeops_http_request_duration_seconds_count{method="GET",route="/stream/{id}"} 1' in collector.render()
        assert "lifeops_http_in_progress 0" in collector.render()
    asyncio.run(scenario())
    emitted = caplog.text + collector.render()
    assert all(secret not in emitted for secret in ("private-user-name", "private-key", "private answer", "forged"))


def test_each_queued_job_keeps_its_own_request_context(collector):
    from app.src.service import session_queue as queue
    async def scenario():
        started, release = asyncio.Event(), asyncio.Event()
        seen = []
        async def first():
            started.set()
            await release.wait()
            seen.append(telemetry.current_trace_id())
            telemetry.bind_request_key("a" * 64)
        async def second():
            seen.append(telemetry.current_trace_id())
            telemetry.bind_request_key("b" * 64)
        first_state, second_state = {"trace_id": "first"}, {"trace_id": "second"}
        token = telemetry._HTTP_STATE.set(first_state)
        queue.submit_chat_job("context", asyncio.Queue(), first)
        telemetry._HTTP_STATE.reset(token)
        await started.wait()
        token = telemetry._HTTP_STATE.set(second_state)
        queue.submit_chat_job("context", asyncio.Queue(), second)
        telemetry._HTTP_STATE.reset(token)
        release.set()
        while queue.is_session_busy("context"):
            await asyncio.sleep(0)
        await queue.close_session("context")
        assert seen == ["first", "second"]
        assert first_state["request_key"] == "a" * 64
        assert second_state["request_key"] == "b" * 64
        assert telemetry.current_trace_id() is None
    asyncio.run(scenario())
    assert 'lifeops_chat_jobs_total{status="completed"} 2' in collector.render()


def test_turn_audit_is_correlated_and_metrics_only_use_bounded_metadata(collector, tmp_path, monkeypatch):
    from app.src.common.config.base_config import settings
    monkeypatch.setattr(settings, "harness_trace_dir", str(tmp_path))
    token = telemetry._HTTP_STATE.set({"trace_id": "c" * 32})
    try:
        with turn_context("private-user-name") as turn:
            turn.request_key = "d" * 64
            turn.add_record({"kind": "tool", "name": "private-tool-input", "status": "blocked", "elapsed_ms": 25})
            turn.budget.prompt_tokens = 2
            turn.budget.completion_tokens = 3
            turn.budget.charged_tokens = 12
    finally:
        telemetry._HTTP_STATE.reset(token)
    record = json.loads(next(tmp_path.glob("*.jsonl")).read_text(encoding="utf-8"))
    assert record["trace_id"] == "c" * 32 and record["request_key"] == "d" * 64
    report = collector.render()
    assert 'lifeops_agent_calls_total{kind="tool",status="blocked"} 1' in report
    assert 'lifeops_agent_charged_tokens_total{source="reported"} 5' in report
    assert 'lifeops_agent_charged_tokens_total{source="reservation"} 7' in report
    assert all(value not in report for value in ("private-user-name", "private-tool-input", "c" * 32, "d" * 64))


def test_background_executor_can_preserve_parent_correlation_without_inheriting_its_budget(collector, tmp_path, monkeypatch):
    from app.src.common.config.base_config import settings
    from app.src.common.harness.context import background_context
    monkeypatch.setattr(settings, "harness_trace_dir", str(tmp_path))
    def background():
        assert telemetry.current_trace_id() is None
        with background_context("worker", parent_turn_id="parent", trace_id="a" * 32, request_key="b" * 64) as turn:
            assert turn.phase == "background" and turn.parent_turn_id == "parent"
            assert turn.budget.llm_calls == 0
    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(background).result(timeout=5)
    record = json.loads(next(tmp_path.glob("*.jsonl")).read_text(encoding="utf-8"))
    assert record["trace_id"] == "a" * 32 and record["request_key"] == "b" * 64
    assert record["parent_turn_id"] == "parent" and record["phase"] == "background"


def test_metrics_are_disabled_by_default_and_require_operator_bearer_not_user_cookie(collector, monkeypatch):
    from app.src.common.config.app_config import create_app
    from app.src.common.config.base_config import settings
    from app.src.service.auth_service import COOKIE_NAME
    monkeypatch.setattr(settings, "ops_metrics_token", "")
    with TestClient(create_app()) as client:
        assert client.get("/metrics").status_code == 404
        monkeypatch.setattr(settings, "ops_metrics_token", "operator-test-token")
        client.cookies.set(COOKIE_NAME, "ordinary-user-cookie")
        assert client.get("/metrics").status_code == 401
        assert client.get("/metrics", headers={"Authorization": "Bearer wrong"}).status_code == 401
        response = client.get("/metrics", headers={"Authorization": "Bearer operator-test-token"})
        assert response.status_code == 200
        assert "version=0.0.4" in response.headers["content-type"]
        assert response.headers["cache-control"] == "no-store"
        assert "lifeops_chat_pending_jobs 0" in response.text
        assert "operator-test-token" not in response.text


def test_unmatched_paths_do_not_create_unbounded_labels_and_500_has_trace(collector):
    from app.src.common.config.app_config import create_app
    app = create_app()
    @app.get("/broken-test")
    async def broken():
        raise RuntimeError("private provider credential")
    with TestClient(app, raise_server_exceptions=False) as client:
        for i in range(20):
            assert client.get(f"/unknown-private-{i}").status_code == 404
        response = client.get("/broken-test")
        assert response.status_code == 500
        assert response.json()["trace_id"] == response.headers["x-trace-id"]
        assert "private provider credential" not in response.text
    report = collector.render()
    assert 'lifeops_http_requests_total{method="GET",route="unmatched",status="404"} 20' in report
    assert "unknown-private" not in report
    assert 'route="/broken-test",status="500"' in report


def test_admission_metrics_distinguish_capacity_and_shutdown_rejections(collector, monkeypatch):
    from app.src.service import session_queue as queue
    from app.src.common.config.base_config import settings
    monkeypatch.setattr(settings, "chat_max_inflight_jobs", 1)
    monkeypatch.setattr(settings, "chat_max_session_jobs", 1)
    async def scenario():
        started = asyncio.Event()
        async def work():
            started.set()
            await asyncio.Event().wait()
        queue.submit_chat_job("saturation", asyncio.Queue(), work)
        with pytest.raises(queue.ChatCapacityExceeded):
            queue.submit_chat_job("other", asyncio.Queue(), work)
        monkeypatch.setattr(settings, "chat_max_inflight_jobs", 10)
        with pytest.raises(queue.ChatCapacityExceeded):
            queue.submit_chat_job("saturation", asyncio.Queue(), work)
        await started.wait()
        assert queue.queue_gauges()["lifeops_chat_active_jobs"] == 1
        await queue.shutdown_chat_jobs(0)
        with pytest.raises(queue.ChatCapacityExceeded):
            queue.submit_chat_job("other", asyncio.Queue(), work)
        assert queue.queue_gauges()["lifeops_chat_active_jobs"] == 0
        assert queue.queue_gauges()["lifeops_chat_pending_jobs"] == 0
    try:
        asyncio.run(scenario())
    finally:
        queue.start_chat_jobs()
    report = collector.render()
    for reason in ("global_limit", "session_limit", "draining"):
        assert f'lifeops_chat_rejections_total{{reason="{reason}"}} 1' in report
    assert 'lifeops_chat_jobs_total{status="cancelled"} 1' in report


def test_http_sse_turn_and_cached_retry_share_receipt_but_have_distinct_http_traces(collector, tmp_path, monkeypatch, caplog):
    from app.src.common.config.app_config import create_app
    from app.src.common.config.base_config import settings
    from app.src.service import chat_service as service
    from app.src.service.auth_service import COOKIE_NAME, login_user
    caplog.set_level(logging.INFO, logger=telemetry.__name__)
    monkeypatch.setattr(settings, "harness_trace_dir", str(tmp_path))
    monkeypatch.setattr(settings, "background_learning_enabled", False)
    async def fake_graph(**kwargs):
        with turn_context(kwargs["session_id"]) as turn:
            turn.request_key = kwargs["request_key"]
            turn.finish("success")
            return {"answer": "ok", "trace": {"steps": [{"name": "harness_summary", "output": turn.summary()}]}}
    graph = AsyncMock(side_effect=fake_graph)
    monkeypatch.setattr(service, "arun_chat_graph", graph)
    username = "telemetry-" + uuid4().hex
    token = login_user(username, "isolated-test-password", allow_create=True)
    payload = {"session_id": username, "message": "private-message-marker", "request_id": "client-selected-private-key"}
    with TestClient(create_app()) as client:
        client.cookies.set(COOKIE_NAME, token)
        first = client.post("/api/chat/stream", json=payload)
        second = client.post("/api/chat/stream", json=payload)
    assert first.status_code == second.status_code == 200
    assert first.headers["x-trace-id"] != second.headers["x-trace-id"]
    def final(response):
        events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
        return next(event["result"] for event in events if event["type"] == "final")
    result = final(first)
    assert final(second) == result
    summary = result["trace"]["steps"][0]["output"]
    assert summary["trace_id"] == first.headers["x-trace-id"]
    assert re.fullmatch("[0-9a-f]{64}", summary["request_key"])
    assert graph.await_count == 1
    record = json.loads(next(tmp_path.glob("*.jsonl")).read_text(encoding="utf-8"))
    assert record["trace_id"] == summary["trace_id"] and record["request_key"] == summary["request_key"]
    logs = "\n".join(entry.getMessage() for entry in caplog.records if entry.name == telemetry.__name__)
    assert first.headers["x-trace-id"] in logs and second.headers["x-trace-id"] in logs
    assert logs.count(summary["request_key"]) == 2
    assert all(value not in logs + collector.render() for value in (username, payload["message"], payload["request_id"]))
