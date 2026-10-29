import json
import hashlib
import os
import sqlite3
import subprocess
import sys

import pytest

from scripts.backup import DATABASES, SnapshotError, create_snapshot, restore_snapshot, verify_snapshot


@pytest.fixture
def source(tmp_path):
    databases = []
    connections = []
    for name in DATABASES:
        path = tmp_path / name
        db = sqlite3.connect(path)
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA wal_autocheckpoint=0")
        db.execute("CREATE TABLE evidence (id INTEGER PRIMARY KEY, value TEXT)")
        db.execute("INSERT INTO evidence VALUES (1, ?)", (name,))
        db.commit()
        databases.append(path)
        connections.append(db)
    memory = tmp_path / "memory"
    (memory / "alice").mkdir(parents=True)
    (memory / "alice" / "memo.md").write_text("用户偏好室内运动", encoding="utf-8")
    # Open idle connections keep WAL sidecars present. There are no active writers.
    yield databases, memory
    for db in connections:
        db.close()


def test_wal_backups_restore_all_databases_and_memory_without_touching_original(tmp_path, source):
    databases, memory = source
    assert all(path.with_name(path.name + "-wal").stat().st_size > 0 for path in databases)
    output = tmp_path / "snapshot"
    manifest = create_snapshot(output, databases, memory, service_stopped=True)
    assert manifest["consistency"] == "offline-service-stopped"
    assert verify_snapshot(output) == manifest
    restored = tmp_path / "restored"
    restore_snapshot(output, restored)
    for name in DATABASES:
        db = sqlite3.connect(restored / name)
        try:
            assert db.execute("SELECT value FROM evidence").fetchone()[0] == name
            db.execute("INSERT INTO evidence VALUES (2, 'new')")
            db.commit()
        finally:
            db.close()
    assert (restored / "memory/alice/memo.md").read_text(encoding="utf-8") == "用户偏好室内运动"
    with sqlite3.connect(databases[0]) as db:
        assert db.execute("SELECT count(*) FROM evidence").fetchone()[0] == 1


def test_requires_stopped_service_and_never_overwrites(tmp_path, source):
    databases, memory = source
    output = tmp_path / "snapshot"
    with pytest.raises(SnapshotError, match="service-stopped"):
        create_snapshot(output, databases, memory)
    assert not output.exists()
    create_snapshot(output, databases, memory, service_stopped=True)
    with pytest.raises(SnapshotError, match="new directory"):
        create_snapshot(output, databases, memory, service_stopped=True)
    with pytest.raises(SnapshotError, match="new directory"):
        restore_snapshot(output, memory)
    assert (memory / "alice/memo.md").exists()


def test_tampered_or_incomplete_snapshot_never_publishes_restore(tmp_path, source):
    databases, memory = source
    output = tmp_path / "snapshot"
    create_snapshot(output, databases, memory, service_stopped=True)
    (output / "memory/alice/memo.md").write_text("modified", encoding="utf-8")
    with pytest.raises(SnapshotError, match="checksum"):
        restore_snapshot(output, tmp_path / "restored")
    assert not (tmp_path / "restored").exists()
    (output / "checkpointer.db").unlink()
    with pytest.raises(SnapshotError, match="missing"):
        verify_snapshot(output)


@pytest.mark.parametrize("path", ["../outside", "/absolute", "memory/../../outside", "memory\\outside", "E:/outside", "unexpected.db"])
def test_untrusted_manifest_paths_are_rejected(tmp_path, source, path):
    databases, memory = source
    output = tmp_path / "snapshot"
    manifest = create_snapshot(output, databases, memory, service_stopped=True)
    manifest["files"][path] = {"sha256": "untrusted", "bytes": 1}
    (output / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(SnapshotError):
        restore_snapshot(output, tmp_path / "restored")
    assert not (tmp_path / "restored").exists()


def test_extra_wal_file_cannot_change_verified_database_contents(tmp_path, source):
    databases, memory = source
    output = tmp_path / "snapshot"
    create_snapshot(output, databases, memory, service_stopped=True)
    (output / "lifeops.db-wal").write_bytes(b"unlisted")
    with pytest.raises(SnapshotError, match="unlisted"):
        verify_snapshot(output)


def test_failed_backup_is_not_published_and_cannot_recurse_into_memory(tmp_path, source):
    databases, memory = source
    with pytest.raises(SnapshotError, match="outside"):
        create_snapshot(memory / "snapshot", databases, memory, service_stopped=True)
    missing = tmp_path / "missing.db"
    with pytest.raises(SnapshotError, match="does not exist"):
        create_snapshot(tmp_path / "snapshot", [databases[0], missing, databases[2]], memory, service_stopped=True)
    assert not (tmp_path / "snapshot").exists()
    assert not missing.exists()


def test_cli_verify_and_restore_need_no_app_credentials(tmp_path, source):
    databases, memory = source
    output = tmp_path / "snapshot"
    create_snapshot(output, databases, memory, service_stopped=True)
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    env.pop("QWEN_API_KEY", None)
    commands = [
        ["verify", str(output)],
        ["restore", str(output), "--output", str(tmp_path / "restored")],
        ["create", "--business-db", str(databases[0]), "--knowledge-db", str(databases[1]),
         "--checkpoint-db", str(databases[2]), "--memory-dir", str(memory),
         "--output", str(tmp_path / "cli-snapshot"), "--service-stopped"],
    ]
    for args in commands:
        result = subprocess.run([sys.executable, "-m", "scripts.backup", *args],
                                check=True, capture_output=True, text=True, encoding="utf-8", env=env)
        assert json.loads(result.stdout)["status"] == "ok"
    assert (tmp_path / "restored/memory/alice/memo.md").is_file()


def test_corrupt_sqlite_is_rejected_even_when_its_hash_matches_manifest(tmp_path, source):
    databases, memory = source
    output = tmp_path / "snapshot"
    manifest = create_snapshot(output, databases, memory, service_stopped=True)
    content = b"not a SQLite database"
    (output / "lifeops.db").write_bytes(content)
    manifest["files"]["lifeops.db"] = {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
    (output / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(SnapshotError, match="SQLite validation"):
        restore_snapshot(output, tmp_path / "restored")
    assert not (tmp_path / "restored").exists()


def test_link_in_memory_tree_cannot_copy_outside_data(tmp_path, source):
    databases, memory = source
    outside = tmp_path / "outside.txt"
    outside.write_text("not part of memory", encoding="utf-8")
    try:
        (memory / "external-link").symlink_to(outside)
    except OSError:
        pytest.skip("Creating symlinks requires an OS privilege unavailable on this host")
    with pytest.raises(SnapshotError, match="links"):
        create_snapshot(tmp_path / "snapshot", databases, memory, service_stopped=True)
    assert not (tmp_path / "snapshot").exists()
