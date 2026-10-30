import json
import subprocess
from types import SimpleNamespace

from scripts.verify import verify


def test_failed_stage_retains_evidence_and_runs_remaining_checks(tmp_path, monkeypatch):
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        kwargs["stdout"].write("stage evidence\n")
        if len(calls) == 2:
            raise subprocess.TimeoutExpired(command, 600)
        return SimpleNamespace(returncode=1 if len(calls) == 1 else 0)

    monkeypatch.setattr(subprocess, "run", run)
    assert not verify(tmp_path)
    report = json.loads((tmp_path / "verification.json").read_text())
    assert report["complete"] and not report["passed"]
    assert [r["exit_code"] for r in report["stages"]] == [1, 124, 0, 0, 0]
    assert len(calls) == 5
    assert all((tmp_path / f"{r['stage']}.log").is_file() for r in report["stages"])
