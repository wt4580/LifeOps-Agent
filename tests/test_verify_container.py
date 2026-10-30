import json
from types import SimpleNamespace

from scripts.verify_container import verify_container


def test_failed_verification_exports_before_removing_owned_volume(tmp_path, monkeypatch):
    import subprocess
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if "inspect" in command:
            return SimpleNamespace(returncode=0, stdout=json.dumps({"Status": "exited", "ExitCode": 1}))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, "run", run)
    assert not verify_container("lifeops-agent:test", tmp_path)
    operations = [command[3] for command in calls]
    assert operations == ["create", "start", "inspect", "cp", "rm"]
    assert "type=volume,destination=/reports" in calls[0]
    assert "-v" in calls[-1]
    assert json.loads((tmp_path / "container-result.json").read_text())["reports_exported"]
