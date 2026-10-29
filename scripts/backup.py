"""Offline SQLite + memory snapshots. Restore only into a new directory.

No application import, API key or model call is required. Stop all application
processes (including schedulers) before creating a cross-database snapshot.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sqlite3
import tempfile
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

DATABASES = ("lifeops.db", "knowledge.db", "checkpointer.db")
FORMAT_VERSION = 1


class SnapshotError(ValueError):
    pass


def _digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_db(path):
    return sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=5)


def _check_db(path):
    db = _read_db(path)
    try:
        if db.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
            raise SnapshotError(f"SQLite integrity check failed: {path.name}")
        if db.execute("PRAGMA foreign_key_check").fetchone() is not None:
            raise SnapshotError(f"SQLite foreign key check failed: {path.name}")
    except sqlite3.Error as exc:
        raise SnapshotError(f"SQLite validation failed: {path.name}") from exc
    finally:
        db.close()


def _new_destination(output):
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise SnapshotError("Destination must be a new directory; existing paths are never overwritten")
    if not output.parent.is_dir():
        raise SnapshotError("Destination parent directory must already exist")
    return output


def _safe_file(root, relative):
    path = root / relative
    if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise SnapshotError(f"Missing or unsafe snapshot file: {relative}")
    return path


def create_snapshot(output, databases, memory_dir, *, service_stopped=False):
    if not service_stopped:
        raise SnapshotError("Stop all service processes and pass --service-stopped before creating a snapshot")
    if len(databases) != len(DATABASES):
        raise SnapshotError("All three database paths are required")
    output = _new_destination(output)
    sources = [Path(path).resolve() for path in databases]
    memory = Path(memory_dir).resolve()
    if len(set(sources)) != len(sources):
        raise SnapshotError("Database paths must be distinct")
    if output.resolve().is_relative_to(memory):
        raise SnapshotError("Snapshot destination must be outside the memory directory")
    for path in sources:
        if not path.is_file():
            raise SnapshotError(f"Database does not exist: {path.name}")
        _check_db(path)
    # A sibling staging directory makes publication atomic on the same filesystem.
    with tempfile.TemporaryDirectory(prefix=".lifeops-snapshot-", dir=output.parent) as temporary:
        stage = Path(temporary) / "snapshot"
        stage.mkdir()
        for name, source in zip(DATABASES, sources):
            with closing(_read_db(source)) as source_db, closing(sqlite3.connect(stage / name)) as target_db:
                source_db.backup(target_db)
                target_db.execute("PRAGMA journal_mode=DELETE")
            _check_db(stage / name)
        (stage / "memory").mkdir()
        if memory.exists():
            if not memory.is_dir():
                raise SnapshotError("Memory path must be a directory")
            for source in memory.rglob("*"):
                if source.is_symlink() or not source.resolve().is_relative_to(memory):
                    raise SnapshotError("Memory tree must not contain links escaping its root")
                if source.is_file():
                    target = stage / "memory" / source.relative_to(memory)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source, target)
        files = {path.relative_to(stage).as_posix(): {"bytes": path.stat().st_size, "sha256": _digest(path)}
                 for path in sorted(stage.rglob("*")) if path.is_file()}
        manifest = {"format_version": FORMAT_VERSION, "created_at": datetime.now(timezone.utc).isoformat(),
                    "consistency": "offline-service-stopped", "sqlite_version": sqlite3.sqlite_version, "files": files}
        (stage / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        verify_snapshot(stage)
        stage.rename(output)
    return manifest


def verify_snapshot(snapshot):
    root = Path(snapshot).resolve()
    try:
        manifest = json.loads(_safe_file(root, "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SnapshotError("Snapshot manifest cannot be read") from exc
    if not isinstance(manifest, dict) or manifest.get("format_version") != FORMAT_VERSION:
        raise SnapshotError("Unsupported snapshot format")
    files = manifest.get("files")
    if not isinstance(files, dict) or not set(DATABASES).issubset(files):
        raise SnapshotError("Snapshot must include business, knowledge and checkpoint databases")
    present = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    if present != set(files) | {"manifest.json"}:
        raise SnapshotError("Snapshot contains unlisted or missing files")
    for relative, entry in files.items():
        # Accept exactly the database names and files under memory/. No absolute,
        # drive, backslash or traversal paths, even when validating on Linux.
        parts = relative.split("/")
        if (not relative or "\\" in relative or ":" in relative or any(part in {"", ".", ".."} for part in parts)
                or (relative not in DATABASES and (len(parts) < 2 or parts[0] != "memory"))):
            raise SnapshotError("Manifest contains an invalid file path")
        path = _safe_file(root, relative)
        if (not isinstance(entry, dict) or path.stat().st_size != entry.get("bytes")
                or _digest(path) != entry.get("sha256")):
            raise SnapshotError(f"Snapshot checksum mismatch: {relative}")
    for name in DATABASES:
        _check_db(root / name)
    return manifest


def restore_snapshot(snapshot, output):
    root = Path(snapshot).resolve()
    output = _new_destination(output)
    manifest = verify_snapshot(root)
    with tempfile.TemporaryDirectory(prefix=".lifeops-restore-", dir=output.parent) as temporary:
        stage = Path(temporary) / "restored"
        stage.mkdir()
        (stage / "memory").mkdir()
        for relative in manifest["files"]:
            target = stage / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(_safe_file(root, relative), target)
            if _digest(target) != manifest["files"][relative]["sha256"]:
                raise SnapshotError("Snapshot changed while restoring")
        for name in DATABASES:
            _check_db(stage / name)
        stage.rename(output)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create", help="Stop all app processes before creating a snapshot")
    for name in ("business-db", "knowledge-db", "checkpoint-db", "memory-dir", "output"):
        create.add_argument(f"--{name}", required=True, type=Path)
    create.add_argument("--service-stopped", action="store_true", help="Assert that all writers are stopped")
    verify = commands.add_parser("verify")
    verify.add_argument("snapshot", type=Path)
    restore = commands.add_parser("restore", help="Restore into a new directory without overwriting existing data")
    restore.add_argument("snapshot", type=Path)
    restore.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        if args.command == "create":
            manifest = create_snapshot(args.output, [args.business_db, args.knowledge_db, args.checkpoint_db],
                                       args.memory_dir, service_stopped=args.service_stopped)
        elif args.command == "verify":
            manifest = verify_snapshot(args.snapshot)
        else:
            manifest = restore_snapshot(args.snapshot, args.output)
    except (SnapshotError, sqlite3.Error, OSError) as exc:
        parser.exit(1, f"Snapshot operation failed: {exc}\n")
    print(json.dumps({"command": args.command, "status": "ok", "files": len(manifest["files"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
