# -*- coding: utf-8 -*-
"""把评估 JSON 报告渲染成 Markdown，方便直接贴给观众/放进文档。

用法：
  E:\\Anaconda\\envs\\lifeops\\python.exe -m app.eval.report  [json路径]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

DEFAULT = Path(__file__).resolve().parent / "reports" / "retrieval.json"


def render_retrieval(report: dict) -> str:
    m = report.get("metrics", {})
    meta = report.get("meta", {})
    lines = []
    lines.append("# 检索质量评估报告")
    lines.append("")
    lines.append(f"- 评估条数：{meta.get('n_cases')}")
    lines.append(f"- 检索后端：{meta.get('backend')}")
    lines.append(f"- 生成时间：{meta.get('timestamp')}")
    lines.append("")
    lines.append("## 核心指标")
    lines.append("")
    rows = [
        ("Recall@1", m.get("recall@1")),
        ("Recall@3", m.get("recall@3")),
        ("Recall@5", m.get("recall@5")),
        ("MRR（首个命中的排名倒数）", m.get("mrr")),
        ("NDCG@1", m.get("ndcg@1")),
        ("NDCG@3", m.get("ndcg@3")),
        ("NDCG@5", m.get("ndcg@5")),
        ("纯向量基线 Recall@1（重排增益对比）", report.get("rerank_gain_recall1")),
    ]
    lines.append("| 指标 | 分值 |")
    lines.append("| --- | --- |")
    for name, val in rows:
        lines.append(f"| {name} | {round(val, 4) if val is not None else '-'} |")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else str(DEFAULT)
    with open(path, "r", encoding="utf-8") as f:
        report = json.load(f)
    md = render_retrieval(report)
    out = Path(path).with_suffix(".md")
    out.write_text(md, encoding="utf-8")
    print(md)
    print(f"\n[report] markdown saved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
