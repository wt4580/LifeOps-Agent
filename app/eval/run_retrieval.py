# -*- coding: utf-8 -*-
"""检索质量评估运行器。

对 app/eval/cases.py 里的每个 retrieval case：
  1) 用生产路径 search_chunks（hybrid：多通道召回 + RRF 融合 + 重排）跑出 top-k 排序；
  2) 用 search_with_rerank（向量 + 重排通道）跑出语义基线，用于计算重排增益与通道对比；
  3) 计算 Recall@1/3/5、MRR、NDCG@5，输出 JSON + 控制台汇总。

用法：
  E:\\Anaconda\\envs\\lifeops\\python.exe -m app.eval.run_retrieval
  可选环境变量 EVAL_SAVE 指定 JSON 输出路径（默认 app/eval/reports/retrieval.json）。
"""

from __future__ import annotations

import json
import os
import sys
import time
import logging
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
logging.getLogger().setLevel(logging.ERROR)
logging.disable(logging.WARNING)

from app.src.util.retrieval.retrieval import search_chunks
from app.src.util.retrieval.rag_langchain import search_with_rerank
from app.eval.cases import load_cases
from app.eval.metrics import evaluate_retrieval_results, recall_at_k

REPORT_DIR = Path(__file__).resolve().parent / "reports"
TOP_K = 5


def _basename(path: str) -> str:
    return path.replace("\\", "/").split("/")[-1]


def _run_case(case: dict) -> dict:
    query = case["query"]
    expected = case["expected"]

    prod = search_chunks(query, top_k=TOP_K, session_id="eval")
    prod_paths = [c.path for c in prod]

    baseline: list = []
    try:
        base_hits = search_with_rerank(query, top_k=TOP_K, candidate_k=24)
        baseline = [h.path for h in base_hits]
    except Exception as exc:
        baseline = []
        print(f"  [warn] search_with_rerank failed for '{query}': {exc!r}")

    return {
        "query": query,
        "expected": expected,
        "ranked_paths": prod_paths,
        "baseline_paths": baseline,
    }


def _channel_overlap(prod_paths: list[str], baseline_paths: list[str]) -> dict:
    """生产路径（多通道）与纯向量路径的重合度，反映各通道的补充价值。"""
    prod_set = {_basename(p) for p in prod_paths}
    base_set = {_basename(p) for p in baseline_paths}
    if not base_set:
        return {"prod_only": list(prod_set), "shared": [], "base_only": []}
    return {
        "prod_only": sorted(prod_set - base_set),
        "shared": sorted(prod_set & base_set),
        "base_only": sorted(base_set - prod_set),
    }


def main() -> int:
    cases = load_cases("retrieval")
    print(f"[eval] running {len(cases)} retrieval cases, top_k={TOP_K}")
    print(f"[eval] using production path: search_chunks (hybrid + rerank) + search_with_rerank\n")

    save_path = os.environ.get("EVAL_SAVE", str(REPORT_DIR / "retrieval.json"))
    results: list[dict] = []
    for idx, case in enumerate(cases, start=1):
        t0 = time.time()
        r = _run_case(case)
        dt = time.time() - t0
        top = [_basename(p) for p in r["ranked_paths"][:1]]
        hit = "Y" if top and top[0] in {_basename(x) for x in case["expected"]} else "n"
        print(
            f"[{idx:02d}/{len(cases)}] recall@1={'Y' if hit=='Y' else 'n'} "
            f"top1={top} ({dt:.1f}s)  query={case['query'][:28]}"
        )
        results.append(r)
        _save_partial(save_path, results, cases)

    return _finalize(results, save_path)


def _save_partial(save_path: str, results: list[dict], cases: list[dict]) -> None:
    """把增量结果落盘，避免长跑被中断丢数据。"""
    try:
        summary = evaluate_retrieval_results(results)
        report = {
            "meta": {"n_cases": len(results), "top_k": TOP_K, "backend": "hybrid",
                     "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")},
            "metrics": _scalar_metrics(summary),
            "rerank_gain_recall1": round(sum(recall_at_k(r["baseline_paths"], r["expected"], 1) for r in results) / len(results), 4) if results else 0.0,
            "details": summary.get("details", []),
        }
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
    except Exception as exc:
        print(f"  [warn] partial save failed: {exc!r}")


def _finalize(results: list[dict], save_path: str) -> int:
    """汇总最终指标并落盘 + 控制台打印。"""
    if not results:
        print("[eval] no results")
        return 1

    summary = evaluate_retrieval_results(results)

    report = {
        "meta": {"n_cases": len(results), "top_k": TOP_K, "backend": "hybrid",
                 "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")},
        "metrics": _scalar_metrics(summary),
        "rerank_gain_recall1": round(sum(recall_at_k(r["baseline_paths"], r["expected"], 1) for r in results) / len(results), 4),
        "details": summary.get("details", []),
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    _print_summary(report)
    print(f"\n[eval] report saved -> {save_path}")
    return 0


def _scalar_metrics(summary: dict) -> dict:
    """只保留 summary 中的标量指标（recall@k / mrr / ndcg@k），丢弃 n/k_values/details。"""
    keep = {}
    for k, v in summary.items():
        if k in ("n", "k_values", "details"):
            continue
        if isinstance(v, (int, float)):
            try:
                keep[k] = round(float(v), 4)
            except (TypeError, ValueError):
                keep[k] = v
    return keep


def _print_summary(report: dict) -> None:
    m = report["metrics"]
    print("\n" + "=" * 56)
    print("检索质量评估汇总（production path: hybrid + rerank）")
    print("=" * 56)
    print(f"  评估条数      : {report['meta']['n_cases']}")
    for k in ("recall@1", "recall@3", "recall@5", "mrr", "ndcg@1", "ndcg@3", "ndcg@5"):
        if k in m:
            print(f"  {k:>12} : {m[k]:.4f}")
    print(f"  {'rerank_gain_recall1':>12} : {report.get('rerank_gain_recall1', 0):.4f}  (纯向量基线)")
    print("=" * 56)


if __name__ == "__main__":
    raise SystemExit(main())
