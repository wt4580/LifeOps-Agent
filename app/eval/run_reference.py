"""Offline, isolated BM25 development benchmark over bundled synthetic references.

This measures document retrieval, not answer correctness or real-model quality.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from .metrics import evaluate_retrieval_results
from .run_e2e import configure, source_revision


def load_reference_cases(root):
    cases = json.loads((root / "app/eval/reference_cases.json").read_text(encoding="utf-8"))
    names = {path.name for path in (root / "examples/reference_eval").glob("*.txt")}
    if not cases or len({case["id"] for case in cases}) != len(cases):
        raise ValueError("Reference cases must have unique IDs")
    for case in cases:
        if not case.get("query") or not isinstance(case.get("expected"), list) or not set(case["expected"]).issubset(names):
            raise ValueError("Reference cases must label existing bundled document names")
    return cases


def evaluate():
    root = Path(__file__).resolve().parents[2]
    cases = load_reference_cases(root)
    corpus = root / "examples/reference_eval"
    digest = hashlib.sha256()
    for path in [root / "app/eval/reference_cases.json", *sorted(corpus.glob("*.txt"))]:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    source_digest = hashlib.sha256()
    for path in sorted((root / "app").rglob("*.py")):
        source_digest.update(path.relative_to(root).as_posix().encode())
        source_digest.update(path.read_bytes())
    with tempfile.TemporaryDirectory(prefix="lifeops-reference-eval-") as temporary:
        configure(Path(temporary), "contract")
        os.environ.update({"DOCS_DIR": str(corpus), "RAG_BACKEND": "bm25", "RAG_USE_RERANK": "false"})
        # Import only after isolation settings, never open configured personal databases.
        from app.src.common.config.knowledge_db_config import init_knowledge_db, knowledge_engine
        from app.src.util.retrieval import retrieval
        init_knowledge_db()
        rows = []
        try:
            retrieval._index_documents_sqlite(rebuild=False)
            retrieval._sync_document_records()
            def no_model_calls(*args, **kwargs):
                raise AssertionError("The lexical benchmark must not invoke a model")
            retrieval.chat_completion = no_model_calls
            for case in cases:
                started = time.perf_counter()
                hits = retrieval.search_chunks(case["query"], top_k=5, session_id="reference-eval")
                rows.append({**case, "ranked_paths": [Path(hit.path).name for hit in hits],
                             "latency_ms": round((time.perf_counter() - started) * 1000, 2)})
        finally:
            knowledge_engine.dispose()
    positive = [row for row in rows if row["expected"]]
    negative = [row for row in rows if not row["expected"]]
    metrics = evaluate_retrieval_results(positive)
    metrics.pop("details", None)
    return {"meta": {"dataset_split": "development", "synthetic": True, "model_quality_measured": False,
                     "scope": "document-level retrieval, deduplicated by filename; not answer accuracy",
                     "backend": "bm25", "top_k": 5, "revision": source_revision(root), "corpus_sha256": digest.hexdigest(),
                     "source_sha256": source_digest.hexdigest(), "working_tree_may_be_dirty": True,
                     "labels": "manually authored document relevance; development fixtures, not a held-out set",
                     "timestamp": datetime.now(timezone.utc).isoformat()},
            "metrics": metrics, "negative_queries": {"n": len(negative), "empty_results": sum(not row["ranked_paths"] for row in negative)},
            "cases": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/reference-eval.json"))
    parser.add_argument("--min-recall5", type=float, default=0.8)
    args = parser.parse_args()
    if not 0 <= args.min_recall5 <= 1:
        parser.error("--min-recall5 must be between zero and one")
    report = evaluate()
    negatives = report["negative_queries"]
    report["gate"] = {"minimum_recall@5": args.min_recall5, "negative_queries_must_be_empty": True,
                      "passed": report["metrics"]["recall@5"] >= args.min_recall5 and negatives["empty_results"] == negatives["n"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"metrics": report["metrics"], "negative_queries": report["negative_queries"], "gate": report["gate"]}, ensure_ascii=False))
    return 0 if report["gate"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
