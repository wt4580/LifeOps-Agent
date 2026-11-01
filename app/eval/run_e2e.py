"""Isolated HTTP/SSE and database contract evaluation; demo by default."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import statistics
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .e2e_cases import cases


def source_revision(root):
    """Revision metadata must not prevent writing a completed evaluation report."""
    fallback = os.getenv("GIT_REVISION") or "unknown"
    if not (root / ".git").exists():
        return fallback
    try:
        result = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root,
                                capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        return fallback
    return result.stdout.strip() if result.returncode == 0 and result.stdout.strip() else fallback


def percentile(values, q):
    values = sorted(values)
    if not values:
        return 0
    index = (len(values) - 1) * q
    left = int(index)
    return values[left] + (values[min(left + 1, len(values) - 1)] - values[left]) * (index - left)


def case_fingerprint(selected):
    """Bind prompts and expected outcomes, independently of implementation code."""
    payload = json.dumps(sorted(selected, key=lambda case: case["id"]),
                         sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def usage_summary(results, mode):
    """Observed usage is not provider billing, especially on failed requests."""
    missing = sum(row["usage_missing_calls"] for row in results)
    failures = sum(not row["passed"] for row in results)
    return {"llm_calls_observed": sum(row["llm_calls"] for row in results),
            "reported_tokens_observed": sum(row["reported_tokens"] for row in results),
            "budget_charged_tokens": sum(row["charged_tokens"] for row in results),
            "usage_missing_calls": missing,
            "usage_complete": bool(results) and missing == 0 and failures == 0,
            "simulated": mode == "contract", "monetary_cost_measured": False,
            "scope": "harness observations; failures may lack final summaries; budget charges are not provider invoices"}


def stream_events(response):
    response.raise_for_status()
    return [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]


def configure(directory, mode):
    root = Path(__file__).resolve().parents[2]
    os.environ.update(
        {
            "DEMO_MODE": "true" if mode == "contract" else "false",
            "DEMO_SEED": "false",
            "BACKGROUND_LEARNING_ENABLED": "false",
            "SCHEDULER_ENABLED": "false",
            "DATABASE_URL": f"sqlite:///{(directory / 'business.db').as_posix()}",
            "KNOWLEDGE_DATABASE_URL": f"sqlite:///{(directory / 'knowledge.db').as_posix()}",
            "CHECKPOINTER_DB_PATH": str(directory / "checkpointer.db"),
            "PERSONAL_MEMORY_DIR": str(directory / "memory"),
            "HARNESS_TRACE_DIR": str(directory / "harness"),
            "DOCS_DIR": str(root / "examples" / "knowledge"),
            "RAG_BACKEND": "bm25",
        }
    )
    if mode == "contract":
        os.environ["QWEN_API_KEY"] = "unused-offline-demo"
    return root


def evaluate(mode, limit=0):
    with tempfile.TemporaryDirectory(prefix="lifeops-e2e-") as temporary:
        directory = Path(temporary)
        root = configure(directory, mode)
        # All application imports occur after isolation settings are established.
        from fastapi.testclient import TestClient
        from sqlalchemy import select

        from app.src.common.config.app_config import create_app
        from app.src.common.config.base_config import settings
        from app.src.common.config.chat_graph_holder import set_chat_graph
        from app.src.common.config.db_config import SessionLocal, engine
        from app.src.common.config.knowledge_db_config import knowledge_engine
        from app.src.domain.entity.chat_entity import Account, TodoItem, Transaction
        from app.src.service.auth_service import COOKIE_NAME, login_user
        from app.src.util.retrieval.retrieval import _index_documents_sqlite, _sync_document_records

        selected = [case for case in cases() if mode == "contract" or not case.get("contract_only")]
        if limit:
            selected = selected[:limit]
        results = []
        try:
            with TestClient(create_app(), base_url="http://localhost") as client:
                _index_documents_sqlite(rebuild=False)
                _sync_document_records()
                for case in selected:
                    sid = "eval-" + case["id"]
                    token = login_user(sid, "isolated-evaluation-password", allow_create=True)
                    client.cookies.set(COOKIE_NAME, token)
                    original_limit = settings.chat_turn_max_llm_calls
                    if "llm_limit" in case:
                        settings.chat_turn_max_llm_calls = case["llm_limit"]
                    started = time.perf_counter()
                    finals, events, summaries = [], [], []
                    error = None
                    try:
                        for index, message in enumerate(case["messages"]):
                            batch = stream_events(
                                client.post(
                                    "/api/chat/stream",
                                    json={"session_id": sid, "message": message, "request_id": f"{case['id']}-{index}"},
                                )
                            )
                            events.extend(batch)
                            final = next((event["result"] for event in batch if event["type"] == "final"), None)
                            if final:
                                finals.append(final)
                                summaries.extend(
                                    step["output"]
                                    for step in final.get("trace", {}).get("steps", [])
                                    if step.get("name") == "harness_summary"
                                )
                            elif not case.get("provider_error"):
                                raise AssertionError(f"No final response: {batch}")
                        with SessionLocal() as db:
                            todos = db.scalars(select(TodoItem).where(TodoItem.session_id == sid)).all()
                            transactions = db.scalars(select(Transaction).where(Transaction.session_id == sid)).all()
                            accounts = db.scalars(select(Account).where(Account.session_id == sid)).all()
                            assert len(todos) == case["todo_count"], f"todo count {len(todos)}"
                            if case.get("date_offsets"):
                                expected = sorted(
                                    (datetime.now() + timedelta(days=offset)).date().isoformat()
                                    for offset in case["date_offsets"]
                                )
                                assert sorted(todo.due_at.date().isoformat() for todo in todos) == expected, (
                                    "wrong due dates"
                                )
                            if "expense" in case:
                                from app.src.service.bookkeeping_service import _local_transaction_time

                                assert len(transactions) == 1 and transactions[0].amount == case["expense"], (
                                    "wrong transaction"
                                )
                                assert (
                                    _local_transaction_time(transactions[0].transaction_at)[:10]
                                    == (datetime.now() - timedelta(days=1)).date().isoformat()
                                ), "wrong transaction date"
                            if "balance" in case:
                                assert len(accounts) == 1 and accounts[0].balance == case["balance"], (
                                    "wrong account balance"
                                )
                        if case.get("citation_contains"):
                            assert any(
                                case["citation_contains"] in citation["path"]
                                for final in finals
                                for citation in final.get("citations", [])
                            ), "missing expected evidence"
                        if case.get("policy_block"):
                            assert any(
                                step.get("name") == "tool_policy_block"
                                for final in finals
                                for step in final.get("trace", {}).get("steps", [])
                            ), "policy was not exercised"
                        if case.get("budget_abort"):
                            assert any(summary["status"] == "budget_exceeded" for summary in summaries), (
                                "budget was not enforced"
                            )
                        if case.get("provider_error"):
                            assert any(
                                event["type"] == "error" and event.get("code") == "request_execution_failed"
                                and event.get("outcome") == "unconfirmed" and event.get("retryable") is True
                                and isinstance(event.get("trace_id"), str) and len(event["trace_id"]) == 32
                                for event in events
                            ), "expected safe, correlated provider error was not surfaced"
                            assert all("simulated provider failure" not in json.dumps(event) for event in events), (
                                "raw provider error leaked into response"
                            )
                    except Exception as exc:
                        error = f"{type(exc).__name__}: {exc}"
                    finally:
                        settings.chat_turn_max_llm_calls = original_limit
                    results.append(
                        {
                            "id": case["id"],
                            "group": case["group"],
                            "passed": error is None,
                            "error": error,
                            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                            "llm_calls": sum(summary["llm_calls"] for summary in summaries),
                            "reported_tokens": sum(
                                summary["prompt_tokens"] + summary["completion_tokens"] for summary in summaries
                            ),
                            "charged_tokens": sum(summary["charged_tokens"] for summary in summaries),
                            "usage_missing_calls": sum(summary["usage_missing_calls"] for summary in summaries),
                        }
                    )
                    print(
                        f"{'PASS' if error is None else 'FAIL'} {case['id']}" + (f" {error}" if error else ""),
                        flush=True,
                    )
        finally:
            set_chat_graph(None)
            engine.dispose()
            knowledge_engine.dispose()
        groups = defaultdict(list)
        for result in results:
            groups[result["group"]].append(result)
        latencies = [result["latency_ms"] for result in results]
        revision = source_revision(root)
        corpus_hash = hashlib.sha256(
            b"".join(path.read_bytes() for path in sorted((root / "examples" / "knowledge").glob("*.txt")))
        ).hexdigest()
        source_hash = hashlib.sha256()
        for path in sorted((root / "app").rglob("*.py")):
            source_hash.update(path.relative_to(root).as_posix().encode())
            source_hash.update(path.read_bytes())
        return {
            "meta": {
                "mode": mode,
                "simulated_provider": mode == "contract",
                "model_quality_measured": mode != "contract",
                "scope": "in-process HTTP/SSE + real graph + isolated SQLite; not network/load benchmark",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "model": "rule-based-demo" if mode == "contract" else settings.qwen_model,
                "revision": revision,
                "source_sha256": source_hash.hexdigest(),
                "python": sys.version.split()[0],
                "working_tree_may_be_dirty": True,
                "corpus_sha256": corpus_hash,
                "cases_sha256": case_fingerprint(selected),
                "backend": settings.rag_backend,
                "turn_limits": {"max_llm_calls": settings.chat_turn_max_llm_calls,
                                "max_tokens": settings.chat_turn_max_tokens, "monetary_cap_enforced": False},
            },
            "metrics": {
                "n_cases": len(results),
                "passed": sum(r["passed"] for r in results),
                "contract_pass_rate": sum(r["passed"] for r in results) / len(results) if results else 0,
                "latency_p50_ms": round(statistics.median(latencies), 2) if latencies else 0,
                "latency_p95_ms": round(percentile(latencies, 0.95), 2),
            },
            "groups": {
                name: {"n": len(items), "passed": sum(item["passed"] for item in items)}
                for name, items in groups.items()
            },
            "usage": usage_summary(results, mode),
            "cases": results,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["contract", "real"], default="contract")
    parser.add_argument("--allow-paid-calls", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--plan-only", action="store_true", help="Export selected cases without initializing the application or calling a provider")
    parser.add_argument("--turn-max-llm-calls", type=int, help="Override the per-turn logical model-call budget")
    parser.add_argument("--turn-max-tokens", type=int, help="Override the per-turn harness token budget; not a billing cap")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be nonnegative")
    if any(value is not None and value < 1 for value in (args.turn_max_llm_calls, args.turn_max_tokens)):
        parser.error("Explicit per-turn budgets must be positive")
    if args.plan_only:
        selected = [case for case in cases() if args.mode == "contract" or not case.get("contract_only")]
        if args.limit:
            selected = selected[:args.limit]
        groups = defaultdict(int)
        for case in selected:
            groups[case["group"]] += 1
        report = {"plan_only": True, "mode": args.mode, "provider_calls_executed": 0,
                  "model_quality_measured": False, "cost_estimate_available": False,
                  "execution_requires_paid_opt_in": args.mode == "real",
                  "turn_limits": {"max_llm_calls": args.turn_max_llm_calls, "max_tokens": args.turn_max_tokens,
                                  "unspecified_limits_use_runtime_config": True, "monetary_cap_enforced": False},
                  "n_cases": len(selected), "n_messages": sum(len(case["messages"]) for case in selected),
                  "groups": dict(groups), "cases_sha256": case_fingerprint(selected), "cases": selected}
        output = args.output or Path("artifacts/e2e-plan.json")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({key: value for key, value in report.items() if key != "cases"}, ensure_ascii=False))
        return 0
    args.output = args.output or Path("artifacts/e2e.json")
    if args.mode == "real" and not args.allow_paid_calls:
        parser.error("Real mode can incur API charges; explicitly pass --allow-paid-calls")
    if args.turn_max_llm_calls is not None:
        os.environ["CHAT_TURN_MAX_LLM_CALLS"] = str(args.turn_max_llm_calls)
    if args.turn_max_tokens is not None:
        os.environ["CHAT_TURN_MAX_TOKENS"] = str(args.turn_max_tokens)
    logging.basicConfig(level=logging.ERROR)
    report = evaluate(args.mode, args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    metrics = report["metrics"]
    markdown = f"# End-to-end {args.mode} report\n\nSimulated provider: {report['meta']['simulated_provider']}. This is not production model accuracy or a network load benchmark.\n\nPassed: {metrics['passed']}/{metrics['n_cases']}\n\nP50: {metrics['latency_p50_ms']} ms; P95: {metrics['latency_p95_ms']} ms.\n\n| Group | Passed | Total |\n|---|---:|---:|\n"
    markdown += "".join(f"| {name} | {group['passed']} | {group['n']} |\n" for name, group in report["groups"].items())
    args.output.with_suffix(".md").write_text(markdown, encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False))
    return 0 if metrics["passed"] == metrics["n_cases"] and metrics["n_cases"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
