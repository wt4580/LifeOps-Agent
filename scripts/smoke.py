"""Smoke test an explicitly labelled demo deployment over real HTTP/SSE."""

from __future__ import annotations

import argparse
import http.cookiejar
import json
import os
import time
import urllib.parse
import urllib.request
from uuid import uuid4


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:5001")
    parser.add_argument("--username", default=os.getenv("DEMO_USERNAME", "demo"))
    parser.add_argument("--password", default=os.getenv("DEMO_PASSWORD", "lifeops-demo-2026"))
    parser.add_argument("--wait", type=int, default=60)
    args = parser.parse_args()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))

    def request(path, body=None, sse=False):
        req = urllib.request.Request(
            args.url.rstrip("/") + path,
            data=json.dumps(body).encode() if body is not None else None,
            headers={"Content-Type": "application/json"} if body is not None else {},
        )
        with opener.open(req, timeout=30) as response:
            data = response.read().decode("utf-8")
        if sse:
            events = [json.loads(line[6:]) for line in data.splitlines() if line.startswith("data: ")]
            errors = [event for event in events if event["type"] == "error"]
            assert not errors, errors
            finals = [event["result"] for event in events if event["type"] == "final"]
            assert finals, "missing SSE final"
            return finals[-1]
        return json.loads(data)

    deadline = time.monotonic() + args.wait
    while True:
        try:
            request("/health/ready")
            break
        except (OSError, ValueError):
            if time.monotonic() >= deadline:
                raise
            time.sleep(1)
    assert request("/health")["result"]["mode"] == "demo", "Smoke writes require demo mode"
    sid = request("/api/auth/login", {"username": args.username, "password": args.password})["session_id"]
    token = "SMOKE" + uuid4().hex[:10]

    def chat(message):
        return request("/api/chat/stream", {"session_id": sid, "message": message}, sse=True)

    query = urllib.parse.urlencode({"session_id": sid})

    def matching():
        payload = request("/api/todos/all?" + query)["result"]
        rows = (
            payload.get("todos", payload.get("pending", []) + payload.get("completed", []))
            if isinstance(payload, dict)
            else payload
        )
        return [row for row in rows if token in row["title"]]

    chat(f"明天下午3点开会{token}")
    assert not matching(), "proposal wrote before confirmation"
    chat("取消")
    assert not matching(), "cancel wrote a todo"
    chat(f"明天下午3点开会{token}")
    chat("确认")
    assert len(matching()) == 1, "confirmation did not write exactly once"
    answer = chat("报销需要哪些材料")
    assert any("reimbursement.txt" in item["path"] for item in answer.get("citations", [])), "missing RAG evidence"
    assert any(step.get("name") == "harness_summary" for step in answer["trace"]["steps"]), "missing harness summary"
    print(
        json.dumps(
            {
                "passed": True,
                "checks": ["readiness", "login", "SSE", "HITL cancel", "HITL confirm", "RAG citation", "harness"],
            }
        )
    )


if __name__ == "__main__":
    main()
