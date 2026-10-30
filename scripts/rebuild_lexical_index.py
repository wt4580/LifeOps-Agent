"""Rebuild a derived FTS index in a new SQLite copy; stop writers first."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3

from sqlalchemy import create_engine


def rebuild(source, output, *, service_stopped=False):
    if not service_stopped:
        raise ValueError("Stop all writers and pass --service-stopped")
    source, output = Path(source).resolve(), Path(output).absolute()
    if not source.is_file() or output.exists():
        raise ValueError("Source must exist and output must be a new file")
    from app.src.util.retrieval.sqlite_fts import install_index, sync_index
    from app.src.util.retrieval.retrieval import _tokenize
    from app.src.common.config.knowledge_db_config import KnowledgeBase
    from app.src.domain.entity import knowledge_entity  # noqa: F401
    from app.src.common.config.schema_migrations import migrate

    # Exclusive creation protects an existing target even under concurrent invocations.
    with output.open("xb"):
        pass
    engine = None
    try:
        with sqlite3.connect(source.as_uri() + "?mode=ro", uri=True) as original:
            with sqlite3.connect(output) as copy:
                original.backup(copy)
        engine = create_engine(f"sqlite:///{output.as_posix()}")

        def upgrade(conn):
            # Unknown versions require matching software, not destructive guessing.
            exists = conn.exec_driver_sql("SELECT 1 FROM sqlite_master WHERE name='lexical_index_state'").first()
            if exists:
                rows = conn.exec_driver_sql("SELECT id,version FROM lexical_index_state").all()
                if rows not in ([], [(1, 1)]):
                    raise ValueError("Unsupported lexical index version")
            for kind in ("chunk", "summary"):
                for action in ("insert", "update", "delete"):
                    conn.exec_driver_sql(f"DROP TRIGGER IF EXISTS lexical_{kind}_{action}")
                conn.exec_driver_sql(f"DROP TABLE IF EXISTS lexical_{kind}")
                conn.exec_driver_sql(f"DROP TABLE IF EXISTS lexical_{kind}_dirty")
            conn.exec_driver_sql("DROP TABLE IF EXISTS lexical_index_state")
            install_index(conn)

        # Match the application's known ledger, without inventing a repair revision.
        migrate(engine, KnowledgeBase.metadata, [("legacy_knowledge_and_fts_v1", lambda conn: None)],
                namespace="knowledge", validate=upgrade)
        sync_index(engine, _tokenize)
        with engine.connect() as conn:
            if conn.exec_driver_sql("PRAGMA integrity_check").scalar_one() != "ok":
                raise ValueError("Rebuilt database failed integrity check")
            return {"status": "ok", "output": str(output),
                    "chunks": conn.exec_driver_sql("SELECT count(*) FROM document_chunks").scalar_one()}
    except BaseException:
        if engine is not None:
            engine.dispose()
        output.unlink(missing_ok=True)
        raise
    finally:
        if engine is not None:
            engine.dispose()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--service-stopped", action="store_true")
    args = parser.parse_args()
    print(json.dumps(rebuild(args.source, args.output, service_stopped=args.service_stopped)))


if __name__ == "__main__":
    main()
