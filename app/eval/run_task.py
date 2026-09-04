# -*- coding: utf-8 -*-
"""任务/路由评估运行器。

对 app/eval/cases.py 里的每个 route case，调用真实的 LLM 路由 `_node_decompose`
（生产路径，与线上完全一致），读取出它决策的动作（state["decision"]["action"]），
与期望动作做精确匹配，统计：动作准确率、各动作命中明细、以及 top-期望是否落在多步计划里。

用法：
  E:\\Anaconda\\envs\\lifeops\\python.exe -m app.eval.run_task
  可选环境变量 EVAL_SAVE 指定 JSON 输出路径（默认 app/eval/reports/task.json）。
"""

from __future__ import annotations

import json
import os
import sys
import time
import logging
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
logging.disable(logging.WARNING)

from app.src.agent.graph_chat import _node_decompose
from app.eval.cases import load_cases

REPORT_DIR = Path(__file__).resolve().parent / "reports"


def _route_one(query: str, session_id: str) -> dict:
    """驱动真实 LLM 路由，返回 {action, args, mode, plan_actions}。"""
    state: dict = {"user_message": query, "session_id": session_id}
    out = _node_decompose(state)
    decision = out.get("decision") or {}
    plan = out.get("plan") or []
    plan_actions = [s.get("action") for s in plan if isinstance(s, dict)]
    return {
        "action": decision.get("action") or "normal_chat",
        "args": decision.get("args") or {},
        "mode": out.get("mode", ""),
        "plan_actions": plan_actions,
    }


def main() -> int:
    cases = load_cases("route")
    print(f"[eval] running {len(cases)} route cases (real LLM router, `_node_decompose`)\n")

    results: list[dict] = []
    correct = 0
    save_path = os.environ.get("EVAL_SAVE", str(REPORT_DIR / "task.json"))
    for idx, case in enumerate(cases, start=1):
        q = case["query"]
        exp = case["expect_action"]
        t0 = time.time()
        r = _route_one(q, f"eval-route-{idx}")
        dt = time.time() - t0
        hit = r["action"] == exp
        # 若期望动作不在第一步但在多步计划里，算 partial
        partial = (not hit) and (exp in r["plan_actions"])
        if hit:
            correct += 1
        mark = "OK " if hit else ("~" if partial else "FAIL")
        print(
            f"[{idx:02d}/{len(cases)}] {mark} expect={exp:<12} got={r['action']:<22} "
            f"plan={r['plan_actions']} ({dt:.1f}s)"
        )
        results.append({**case, "got_action": r["action"], "mode": r["mode"],
                        "plan_actions": r["plan_actions"], "hit": hit, "partial": partial})
        _save_partial(save_path, results)

    return _finalize(results, save_path)


def _save_partial(save_path: str, results: list[dict]) -> None:
    try:
        n = len(results)
        exact = sum(1 for r in results if r["hit"]) / n if n else 0.0
        partial = sum(1 for r in results if r["hit"] or r["partial"]) / n if n else 0.0
        report = {
            "meta": {"n_cases": n, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")},
            "metrics": {
                "action_exact_accuracy": round(exact, 4),
                "exact_or_in_plan_accuracy": round(partial, 4),
                "correct": sum(1 for r in results if r["hit"]),
                "n": n,
            },
            "details": results,
        }
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
    except Exception as exc:
        print(f"  [warn] partial save failed: {exc!r}")


def _finalize(results: list[dict], save_path: str) -> int:
    n = len(results)
    if n == 0:
        print("[eval] no results")
        return 1
    exact = sum(1 for r in results if r["hit"]) / n
    partial_ok = sum(1 for r in results if r["hit"] or r["partial"]) / n

    per_action: dict[str, dict] = {}
    for r in results:
        a = r["expect_action"]
        d = per_action.setdefault(a, {"n": 0, "hit": 0})
        d["n"] += 1
        if r["hit"]:
            d["hit"] += 1

    report = {
        "meta": {"n_cases": n, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")},
        "metrics": {
            "action_exact_accuracy": round(exact, 4),
            "exact_or_in_plan_accuracy": round(partial_ok, 4),
            "correct": sum(1 for r in results if r["hit"]),
            "n": n,
        },
        "per_action": {a: {"n": d["n"], "hit": d["hit"], "acc": round(d["hit"] / d["n"], 3)} for a, d in per_action.items()},
        "details": results,
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 56)
    print("任务/路由评估汇总（真实 LLM 路由）")
    print("=" * 56)
    print(f"  评估条数            : {n}")
    print(f"  动作精确匹配准确率  : {exact:.3f}")
    print(f"  (精确或落入多步计划): {partial_ok:.3f}")
    print("  按动作：")
    for a, d in per_action.items():
        print(f"    {a:<22} {d['hit']}/{d['n']} = {d['acc']:.2f}")
    print("=" * 56)
    print(f"\n[eval] report saved -> {save_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
