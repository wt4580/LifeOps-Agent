from copy import deepcopy

import pytest

from app.eval.compare import compare


def report():
    return {"meta": {"mode": "contract", "model": "demo", "corpus_sha256": "corpus",
                     "backend": "bm25", "simulated_provider": True, "cases_sha256": "cases"},
            "cases": [{"id": "a", "passed": True}, {"id": "b", "passed": False}]}


def test_case_regression_cannot_hide_behind_unchanged_total():
    original = report()
    candidate = deepcopy(original)
    candidate["cases"][0]["passed"] = False
    candidate["cases"][1]["passed"] = True
    result = compare(original, candidate)
    assert result["regressions"] == ["a"]
    assert result["improvements"] == ["b"]
    assert not result["regression_gate_passed"]


@pytest.mark.parametrize("key", ["mode", "model", "corpus_sha256", "cases_sha256", "backend", "simulated_provider"])
def test_mismatched_runs_are_rejected(key):
    candidate = report()
    candidate["meta"][key] = "different"
    with pytest.raises(ValueError, match="Incomparable"):
        compare(report(), candidate)


def test_changed_case_set_is_rejected():
    candidate = report()
    candidate["cases"].pop()
    with pytest.raises(ValueError, match="case sets"):
        compare(report(), candidate)


def test_case_fingerprint_binds_prompts_and_expectations():
    from app.eval.run_e2e import case_fingerprint
    cases = [{"id": "a", "messages": ["明天开会"], "todo_count": 1}, {"id": "b", "todo_count": 0}]
    assert case_fingerprint(cases) == case_fingerprint(list(reversed(cases)))
    changed = deepcopy(cases)
    changed[0]["messages"] = ["后天开会"]
    assert case_fingerprint(cases) != case_fingerprint(changed)
    changed = deepcopy(cases)
    changed[0]["todo_count"] = 2
    assert case_fingerprint(cases) != case_fingerprint(changed)


def test_legacy_report_without_case_fingerprint_is_rejected():
    legacy = report()
    legacy["meta"].pop("cases_sha256")
    with pytest.raises(ValueError, match="cases_sha256"):
        compare(legacy, report())
