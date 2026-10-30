"""Transactional SQLite schema upgrades with separate ordered revision ledgers.

Future schema changes must append a revision; never edit an applied revision.
Downgrades require restoring a verified offline backup with its matching code.
"""
from __future__ import annotations

from sqlalchemy import inspect, text


class SchemaMigrationError(RuntimeError):
    pass


def migrate(engine, metadata, revisions, *, namespace="business", validate=None):
    """Serialize startup upgrades, rolling back both DDL and revision records."""
    if namespace not in ("business", "knowledge"):
        raise ValueError("Unsupported schema namespace")
    ledger = f"{namespace}_schema_revisions"
    if engine.dialect.name != "sqlite":
        raise SchemaMigrationError("Only SQLite schema migrations are supported")
    with engine.connect() as conn:
        # Explicit BEGIN is necessary: sqlite3 legacy mode does not begin on DDL.
        conn.exec_driver_sql("BEGIN IMMEDIATE")
        try:
            conn.exec_driver_sql(f"CREATE TABLE IF NOT EXISTS {ledger} "
                                 "(version INTEGER PRIMARY KEY, name TEXT NOT NULL, "
                                 "applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)")
            applied = conn.execute(text(f"SELECT version,name FROM {ledger} ORDER BY version")).all()
            expected = [(i + 1, name) for i, (name, _) in enumerate(revisions)]
            if applied != expected[:len(applied)]:
                raise SchemaMigrationError(f"Unknown or inconsistent {namespace}-schema revision; use matching code or restore an offline backup")
            for version, (name, upgrade) in enumerate(revisions, 1):
                if version <= len(applied):
                    continue
                upgrade(conn)
                conn.execute(text(f"INSERT INTO {ledger}(version,name) VALUES (:version,:name)"),
                             {"version": version, "name": name})
            # A recorded version alone is not proof of a usable schema.
            inspector = inspect(conn)
            for table in metadata.sorted_tables:
                if not inspector.has_table(table.name):
                    raise SchemaMigrationError(f"Missing {namespace} table: {table.name}")
                columns = {c["name"] for c in inspector.get_columns(table.name)}
                if not set(table.columns.keys()).issubset(columns):
                    raise SchemaMigrationError(f"Missing {namespace} columns: {table.name}")
            if validate is not None:
                validate(conn)
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
