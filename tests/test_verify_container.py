import json
from types import SimpleNamespace

from scripts.verify_container import verify_container


def test_failed_verification_exports_before_removing_owned_volume(tmp_path, monkeypatch):
    import subprocess
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if "inspect" in command:
            return SimpleNamespace(returncode=0, stdout=json.dumps({"state": {"Status": "exited", "ExitCode": 1}, "image_id": "sha256:test"}))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, "run", run)
    assert not verify_container("lifeops-agent:test", tmp_path)
    operations = [command[3] for command in calls]
    assert operations == ["create", "start", "inspect", "cp", "rm"]
    assert "type=volume,destination=/reports" in calls[0]
    assert "-v" in calls[-1]
    assert json.loads((tmp_path / "container-result.json").read_text())["reports_exported"]


import pytest


@pytest.mark.parametrize("evidence, expected", [(None, False), ({"complete": False, "passed": True}, False),
    ({"complete": True, "passed": False}, False), ({"complete": True, "passed": True}, True)])
def test_zero_exit_requires_exported_complete_passing_evidence(tmp_path, monkeypatch, evidence, expected):
    import subprocess
    def run(command, **kwargs):
        if "inspect" in command:
            return SimpleNamespace(returncode=0, stdout=json.dumps({"state": {"Status": "exited", "ExitCode": 0}, "image_id": "sha256:immutable"}))
        if "cp" in command and evidence is not None:
            folder = tmp_path / "verification"
            folder.mkdir()
            (folder / "verification.json").write_text(json.dumps(evidence))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(subprocess, "run", run)
    assert verify_container("mutable:tag", tmp_path) is expected
    report = json.loads((tmp_path / "container-result.json").read_text())
    assert report["image_id"] == "sha256:immutable"
