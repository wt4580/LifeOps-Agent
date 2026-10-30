"""Compare matched E2E reports without treating demo contracts as model quality."""
import argparse
import json
from pathlib import Path


def compare(baseline, candidate):
    for key in ("mode", "model", "corpus_sha256", "cases_sha256", "backend", "simulated_provider"):
        before, after = baseline["meta"].get(key), candidate["meta"].get(key)
        if before is None or after is None or before != after:
            raise ValueError(f"Incomparable report metadata: {key}")

    def index(report):
        rows = report["cases"]
        if not rows or any(not isinstance(row.get("passed"), bool) or not row.get("id") for row in rows):
            raise ValueError("Cases require IDs and boolean outcomes")
        result = {row["id"]: row for row in rows}
        if len(result) != len(rows):
            raise ValueError("Duplicate case IDs")
        return result

    before, after = index(baseline), index(candidate)
    if before.keys() != after.keys():
        raise ValueError("Incomparable case sets")
    regressions = sorted(key for key in before if before[key]["passed"] and not after[key]["passed"])
    improvements = sorted(key for key in before if not before[key]["passed"] and after[key]["passed"])
    return {"scope": "matched E2E case outcomes; no statistical quality or latency claim",
            "mode": candidate["meta"]["mode"], "model": candidate["meta"]["model"],
            "baseline_revision": baseline["meta"].get("revision"),
            "candidate_revision": candidate["meta"].get("revision"),
            "n_cases": len(before), "regressions": regressions, "improvements": improvements,
            "remaining_failures": sorted(key for key in after if not after[key]["passed"]),
            "regression_gate_passed": not regressions}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = compare(json.loads(args.baseline.read_text(encoding="utf-8")),
                         json.loads(args.candidate.read_text(encoding="utf-8")))
    except (ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["regression_gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
