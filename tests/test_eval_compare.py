from copy import deepcopy

import pytest

from app.eval.compare import compare


def report():
    return {"meta": {"mode": "contract", "model": "demo", "corpus_sha256": "corpus",
                     "backend": "bm25", "simulated_provider": True},
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


@pytest.mark.parametrize("key", ["mode", "model", "corpus_sha256", "backend", "simulated_provider"])
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
