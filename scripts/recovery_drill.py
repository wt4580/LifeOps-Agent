"""Isolated, offline recovery drill against real HTTP/SSE, SQLite and checkpoint.

Runs seed and recovery in separate processes. Never reads configured personal DBs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from .backup import DATABASES, create_snapshot, restore_snapshot

USER = "recovery-demo"
PASSWORD = "recovery-demo-2026"
EXPENSE = {"session_id": USER, "message": "我花了10元买午饭", "request_id": "recovery-expense"}


def application_stage(stage, root, proof):
    repo = Path(__file__).resolve().parents[1]
    os.environ.update({
        "DEMO_MODE": "true", "DEMO_SEED": "false", "QWEN_API_KEY": "unused-offline-drill",
        "BACKGROUND_LEARNING_ENABLED": "false", "SCHEDULER_ENABLED": "false", "RAG_BACKEND": "bm25",
        "DATABASE_URL": f"sqlite:///{(root / DATABASES[0]).as_posix()}",
        "KNOWLEDGE_DATABASE_URL": f"sqlite:///{(root / DATABASES[1]).as_posix()}",
        "CHECKPOINTER_DB_PATH": str(root / DATABASES[2]), "PERSONAL_MEMORY_DIR": str(root / "memory"),
        "HARNESS_TRACE_DIR": str(root / "harness"),
        "DOCS_DIR": str(repo / "examples/knowledge" if stage == "seed" else root / "no-source-documents"),
    })
    logging.disable(logging.CRITICAL)
    from fastapi.testclient import TestClient
    from sqlalchemy import func, select
    from app.src.common.config.app_config import create_app
    from app.src.common.config.chat_graph_holder import set_chat_graph
    from app.src.common.config.db_config import SessionLocal, engine
    from app.src.common.config.knowledge_db_config import knowledge_engine
    from app.src.domain.entity.chat_entity import TodoItem, Transaction
    from app.src.util.retrieval.retrieval import _index_documents_sqlite, _sync_document_records

    def chat(client, payload, *, cached=False):
        response = client.post("/api/chat/stream", json=payload)
        response.raise_for_status()
        events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
        finals = [event["result"] for event in events if event["type"] == "final"]
        assert len(finals) == 1, "Expected a completed SSE response"
        assert not finals[0].get("tool_error"), "Tool execution was rejected"
        summary = next(step["output"] for step in finals[0]["trace"]["steps"] if step.get("name") == "harness_summary")
        if cached:
            assert summary["trace_id"] != response.headers["x-trace-id"], "A retry must have a fresh HTTP trace"
        else:
            assert summary["trace_id"] == response.headers["x-trace-id"], "HTTP and agent traces are disconnected"
            records = [json.loads(line) for path in (root / "harness").glob("*.jsonl")
                       for line in path.read_text(encoding="utf-8").splitlines()]
            assert any(record.get("trace_id") == summary["trace_id"] and record.get("request_key") == summary["request_key"]
                       and record["turn_id"] == summary["turn_id"] for record in records), "Persistent audit correlation is missing"
        return finals[0]

    checks = []
    try:
        with TestClient(create_app(), base_url="http://localhost", client=("127.0.0.1", 50000)) as client:
            login = client.post("/api/auth/login", json={"username": USER, "password": PASSWORD})
            login.raise_for_status()
            checks.append("login_after_fresh_process_start")
            if stage == "seed":
                _index_documents_sqlite(rebuild=False)
                _sync_document_records()
                knowledge = chat(client, {"session_id": USER, "message": "报销需要哪些材料", "request_id": "recovery-rag-seed"})
                assert knowledge["citations"], "Seed knowledge is missing"
                expense = chat(client, EXPENSE)
                draft = chat(client, {"session_id": USER, "message": "明天下午3点开会", "request_id": "recovery-draft"})
                assert draft["proposal_id"], "Pending confirmation checkpoint is missing"
                with SessionLocal() as db:
                    assert db.scalar(select(func.count()).select_from(TodoItem)) == 0
                    assert db.scalar(select(func.count()).select_from(Transaction)) == 1
                (root / "memory").mkdir(exist_ok=True)
                (root / "memory/recovery.md").write_text("虚构演练：偏好室内活动", encoding="utf-8")
                proof.write_text(json.dumps({"proposal_id": draft["proposal_id"], "expense": expense}, ensure_ascii=False), encoding="utf-8")
                checks.extend(["expense_committed_once", "proposal_pending_not_written", "knowledge_index_created"])
            else:
                previous = json.loads(proof.read_text(encoding="utf-8"))
                replay = chat(client, EXPENSE, cached=True)
                assert replay == previous["expense"], "Request receipt did not survive restore"
                with SessionLocal() as db:
                    assert db.scalar(select(func.count()).select_from(Transaction)) == 1, "Request replay duplicated expense"
                checks.append("request_receipt_replay_without_duplicate_write")
                confirmed = chat(client, {"session_id": USER, "message": "确认", "request_id": "recovery-confirm",
                                          "proposal_id": previous["proposal_id"]})
                assert confirmed["used_tool"] == "plan_gate_confirm", "Checkpoint did not resume its pending confirmation"
                with SessionLocal() as db:
                    todos = db.scalars(select(TodoItem)).all()
                    assert len(todos) == 1 and "开会" in todos[0].title
                checks.append("pending_checkpoint_resumed_and_todo_written_once")
                knowledge = chat(client, {"session_id": USER, "message": "报销需要哪些材料", "request_id": "recovery-rag-restored"})
                assert knowledge["citations"], "Persisted knowledge cannot be searched after restore"
                checks.append("persisted_fts_search_without_original_documents")
                assert (root / "memory/recovery.md").read_text(encoding="utf-8") == "虚构演练：偏好室内活动"
                checks.append("personal_memory_restored")
            checks.append("http_agent_and_persistent_audit_correlated")
    finally:
        set_chat_graph(None)
        engine.dispose()
        knowledge_engine.dispose()
    return checks


def run_drill():
    with tempfile.TemporaryDirectory(prefix="lifeops-recovery-") as temporary:
        root = Path(temporary)
        source = root / "source"
        source.mkdir()
        proof = root / "proof.json"
        checks = []
        for stage, data in (("seed", source), ("recover", root / "restored")):
            if stage == "recover":
                snapshot = root / "snapshot"
                create_snapshot(snapshot, [source / name for name in DATABASES], source / "memory", service_stopped=True)
                restore_snapshot(snapshot, data)
                checks.append("offline_snapshot_verified_and_restored_to_new_directory")
            result = subprocess.run([sys.executable, "-m", "scripts.recovery_drill", "--stage", stage,
                                     "--data-root", str(data), "--proof", str(proof)],
                                    capture_output=True, text=True, encoding="utf-8", timeout=120,
                                    env={**os.environ, "PYTHONIOENCODING": "utf-8"})
            if result.returncode:
                raise RuntimeError(f"Recovery drill stage {stage} failed:\n{result.stderr[-4000:]}")
            checks.extend(json.loads(result.stdout.splitlines()[-1])["checks"])
        repo = Path(__file__).resolve().parents[1]
        digest = hashlib.sha256()
        for path in sorted([*repo.glob("app/src/**/*.py"), *repo.glob("scripts/**/*.py")]):
            digest.update(path.relative_to(repo).as_posix().encode())
            digest.update(path.read_bytes())
        return {"status": "passed", "mode": "offline-demo", "fresh_processes": 2,
                "created_at": datetime.now(timezone.utc).isoformat(), "source_sha256": digest.hexdigest(),
                "checks": checks, "n_checks": len(checks)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--stage", choices=("seed", "recover"), help=argparse.SUPPRESS)
    parser.add_argument("--data-root", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--proof", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.stage:
        if args.data_root is None or args.proof is None:
            parser.error("Internal stages require --data-root and --proof")
        report = {"checks": application_stage(args.stage, args.data_root, args.proof)}
    else:
        report = run_drill()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
