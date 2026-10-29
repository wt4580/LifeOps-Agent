"""Process-local operational metrics and server-generated request correlation.

Prometheus text exposition, bounded metadata labels, no user content/identity.
Counters reset on restart; persistent turn diagnostics remain in harness JSONL.
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import threading
import time
from collections import Counter
from contextvars import ContextVar
from uuid import uuid4

logger = logging.getLogger(__name__)
_HTTP_STATE = ContextVar("lifeops_http_trace_state", default=None)
BUCKETS = (0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300)
STATUSES = {"success", "error", "cancelled", "blocked", "budget_exceeded", "waiting_for_input", "tool_rejected", "tool_failed", "running"}


def current_trace_id():
    state = _HTTP_STATE.get()
    return state.get("trace_id") if state else None


def bind_request_key(key):
    state = _HTTP_STATE.get()
    if state is not None and isinstance(key, str) and re.fullmatch(r"[0-9a-f]{64}", key):
        state["request_key"] = key


def _labels(labels):
    def escape(value):
        return str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    return "{" + ",".join(f'{key}="{escape(value)}"' for key, value in labels) + "}" if labels else ""


class Metrics:
    def __init__(self):
        self._lock = threading.Lock()
        self._counters = Counter()
        self._histograms = {}
        self._http_active = 0

    def increment(self, name, **labels):
        with self._lock:
            self._counters[(name, tuple(sorted(labels.items())))] += 1

    def observe(self, name, seconds, **labels):
        seconds = max(0.0, float(seconds))
        if not math.isfinite(seconds):
            return
        with self._lock:
            key = (name, tuple(sorted(labels.items())))
            counts, total, count = self._histograms.get(key, ([0] * len(BUCKETS), 0.0, 0))
            for index, bound in enumerate(BUCKETS):
                counts[index] += int(seconds <= bound)
            self._histograms[key] = (counts, total + seconds, count + 1)

    def http_active(self, delta):
        with self._lock:
            self._http_active += delta

    def turn(self, summary):
        phase = "background" if summary.get("phase") == "background" else "foreground"
        status = summary.get("status") if summary.get("status") in STATUSES else "error"
        self.increment("lifeops_agent_turns_total", phase=phase, status=status)
        self.observe("lifeops_agent_turn_duration_seconds", summary.get("elapsed_seconds", 0), phase=phase)
        for call in summary.get("invocations", []):
            kind = "tool" if call.get("kind") == "tool" else "llm"
            call_status = call.get("status") if call.get("status") in STATUSES else "error"
            self.increment("lifeops_agent_calls_total", kind=kind, status=call_status)
            self.observe("lifeops_agent_call_duration_seconds", call.get("elapsed_ms", 0) / 1000, kind=kind)
        # Budget accounting distinguishes supplier-reported usage from conservative reservations.
        reported = summary.get("prompt_tokens", 0) + summary.get("completion_tokens", 0)
        reserved = max(0, summary.get("charged_tokens", 0) - reported)
        with self._lock:
            self._counters[("lifeops_agent_charged_tokens_total", (("source", "reported"),))] += reported
            self._counters[("lifeops_agent_charged_tokens_total", (("source", "reservation"),))] += reserved

    def render(self, gauges=None):
        with self._lock:
            counters = list(self._counters.items())
            histograms = [(key, (list(value[0]), value[1], value[2])) for key, value in self._histograms.items()]
            active = self._http_active
        lines, typed = [], set()
        def emit_type(name, kind):
            if name not in typed:
                lines.append(f"# TYPE {name} {kind}")
                typed.add(name)
        for (name, labels), value in sorted(counters):
            emit_type(name, "counter")
            lines.append(f"{name}{_labels(labels)} {value}")
        for (name, labels), (counts, total, count) in sorted(histograms):
            emit_type(name, "histogram")
            for bound, value in zip(BUCKETS, counts):
                lines.append(f'{name}_bucket{_labels((*labels, ("le", str(bound))))} {value}')
            lines.append(f'{name}_bucket{_labels((*labels, ("le", "+Inf")))} {count}')
            lines.append(f"{name}_sum{_labels(labels)} {total}")
            lines.append(f"{name}_count{_labels(labels)} {count}")
        for name, value in sorted({"lifeops_http_in_progress": active, **(gauges or {})}.items()):
            emit_type(name, "gauge")
            lines.append(f"{name} {value}")
        return "\n".join(lines) + "\n"


metrics = Metrics()


class TelemetryMiddleware:
    """Pure ASGI so latency includes the entire SSE stream, not just its headers."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        state = scope.setdefault("state", {})
        state["trace_id"] = uuid4().hex
        token = _HTTP_STATE.set(state)
        method = scope.get("method", "OTHER")
        method = method if method in {"GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"} else "OTHER"
        started, status = time.monotonic(), 500
        response_started = False
        metrics.http_active(1)
        async def traced_send(message):
            nonlocal status, response_started
            if message["type"] == "http.response.start":
                status = message["status"]
                response_started = True
                message = {**message, "headers": [
                    (key, value) for key, value in message.get("headers", []) if key.lower() != b"x-trace-id"
                ] + [(b"x-trace-id", state["trace_id"].encode("ascii"))]}
            await send(message)
        try:
            await self.app(scope, receive, traced_send)
        except asyncio.CancelledError:
            if not response_started:
                status = 499
            raise
        finally:
            elapsed = time.monotonic() - started
            route = getattr(scope.get("route"), "path", None) or ("/static" if scope.get("path", "").startswith("/static/") else "unmatched")
            metrics.increment("lifeops_http_requests_total", method=method, route=route, status=str(status))
            metrics.observe("lifeops_http_request_duration_seconds", elapsed, method=method, route=route)
            metrics.http_active(-1)
            logger.info("http_request %s", json.dumps({"trace_id": state["trace_id"], "request_key": state.get("request_key"),
                        "method": method, "route": route, "status": status, "elapsed_ms": round(elapsed * 1000, 1)}))
            _HTTP_STATE.reset(token)
