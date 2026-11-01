import subprocess
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from app.eval import run_e2e


def test_execution_diagnostics_preserve_budget_and_transport_evidence_only():
    result = run_e2e.execution_diagnostics(
        [{"status": "completed"}, {"status": "budget_exceeded", "answer": "private"}],
        [{"type": "error", "code": "request_execution_failed", "message": "private"},
         {"type": "error", "code": "request_execution_failed"}, {"type": "final"}],
    )
    assert result == {"turn_statuses": ["completed", "budget_exceeded"],
                      "budget_exceeded": True, "stream_error_codes": ["request_execution_failed"]}
    assert run_e2e.execution_diagnostics([], []) == {
        "turn_statuses": [], "budget_exceeded": False, "stream_error_codes": []}


@pytest.mark.parametrize("error", [FileNotFoundError("git"), PermissionError("git"), subprocess.TimeoutExpired("git", 5)])
def test_revision_falls_back_when_git_is_unavailable(tmp_path, monkeypatch, error):
    (tmp_path / ".git").mkdir()
    monkeypatch.setenv("GIT_REVISION", "build-revision")
    monkeypatch.setattr(run_e2e.subprocess, "run", Mock(side_effect=error))
    assert run_e2e.source_revision(tmp_path) == "build-revision"


def test_revision_uses_git_only_when_successful_and_never_returns_empty(tmp_path, monkeypatch):
    (tmp_path / ".git").mkdir()
    monkeypatch.delenv("GIT_REVISION", raising=False)
    command = Mock(return_value=SimpleNamespace(stdout="abc123\n", returncode=0))
    monkeypatch.setattr(run_e2e.subprocess, "run", command)
    assert run_e2e.source_revision(tmp_path) == "abc123"
    command.return_value = SimpleNamespace(stdout="", returncode=128)
    assert run_e2e.source_revision(tmp_path) == "unknown"


def test_unpacked_source_does_not_require_git(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_REVISION", "container-build")
    command = Mock(side_effect=AssertionError("must not invoke git"))
    monkeypatch.setattr(run_e2e.subprocess, "run", command)
    assert run_e2e.source_revision(tmp_path) == "container-build"
