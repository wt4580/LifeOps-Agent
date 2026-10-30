import hashlib

import pytest
from sqlalchemy import create_engine

from scripts.rebuild_lexical_index import rebuild


def test_rebuild_preserves_source_and_recovers_existing_chunks(tmp_path):
    from app.src.common.config.knowledge_db_config import KnowledgeBase, _baseline
    from app.src.common.config.schema_migrations import migrate
    from app.src.domain.entity import knowledge_entity  # noqa: F401
    source, target = tmp_path / "source.db", tmp_path / "repaired.db"
    engine = create_engine(f"sqlite:///{source.as_posix()}")
    migrate(engine, KnowledgeBase.metadata, [("legacy_knowledge_and_fts_v1", _baseline)], namespace="knowledge")
    with engine.begin() as conn:
        conn.exec_driver_sql("INSERT INTO document_chunks(id,file_path,chunk_text,chunk_hash,created_at) VALUES(1,'guide','apple','hash',CURRENT_TIMESTAMP)")
        conn.exec_driver_sql("DROP TABLE lexical_chunk")
    engine.dispose()
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    assert rebuild(source, target, service_stopped=True)["chunks"] == 1
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before
    repaired = create_engine(f"sqlite:///{target.as_posix()}")
    with repaired.connect() as conn:
        assert conn.exec_driver_sql("SELECT rowid FROM lexical_chunk WHERE lexical_chunk MATCH 't6170706c65'").all() == [(1,)]
    repaired.dispose()
    with pytest.raises(ValueError):
        rebuild(source, target, service_stopped=True)


def test_requires_stopped_writers(tmp_path):
    with pytest.raises(ValueError, match="Stop all writers"):
        rebuild(tmp_path / "source.db", tmp_path / "target.db")


def test_rejected_copy_is_removed_and_source_handle_closed(tmp_path):
    import sqlite3
    source, target = tmp_path / "future.db", tmp_path / "rejected.db"
    with sqlite3.connect(source) as db:
        db.execute("CREATE TABLE knowledge_schema_revisions(version INTEGER PRIMARY KEY,name TEXT)")
        db.execute("INSERT INTO knowledge_schema_revisions VALUES(99,'future')")
    db.close()
    from app.src.common.config.schema_migrations import SchemaMigrationError
    with pytest.raises(SchemaMigrationError):
        rebuild(source, target, service_stopped=True)
    assert not target.exists()
    # Windows cannot rename a database if the copy operation left its handle open.
    renamed = source.with_name("retained.db")
    source.rename(renamed)
    assert renamed.is_file()
