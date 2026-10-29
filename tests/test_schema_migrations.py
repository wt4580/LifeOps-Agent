from concurrent.futures import ThreadPoolExecutor

import pytest
from sqlalchemy import Column, Integer, MetaData, Table, create_engine, inspect, text

from app.src.common.config.schema_migrations import SchemaMigrationError, migrate


@pytest.fixture
def database(tmp_path):
    engine = create_engine(f"sqlite:///{(tmp_path / 'migration.db').as_posix()}")
    yield engine
    engine.dispose()


def schema():
    metadata = MetaData()
    Table("items", metadata, Column("id", Integer, primary_key=True))
    return metadata


def test_fresh_and_repeated_startups_apply_once(database):
    metadata = schema()
    calls = []

    def baseline(conn):
        metadata.create_all(conn)
        conn.execute(text("INSERT INTO items VALUES (7)"))
        calls.append(1)

    for _ in range(2):
        migrate(database, metadata, [("baseline", baseline)])
    with database.connect() as conn:
        assert conn.execute(text("SELECT id FROM items")).all() == [(7,)]
        assert conn.execute(text("SELECT version,name FROM business_schema_revisions")).all() == [(1, "baseline")]
    assert calls == [1]


@pytest.mark.parametrize("records", [[(2, "future")], [(1, "changed")], [(1, "baseline"), (3, "gap")]])
def test_incompatible_ledger_rejected_before_any_upgrade(database, records):
    with database.begin() as conn:
        conn.exec_driver_sql("CREATE TABLE business_schema_revisions(version INTEGER PRIMARY KEY,name TEXT NOT NULL)")
        for version, name in records:
            conn.execute(text("INSERT INTO business_schema_revisions VALUES (:v,:n)"), {"v": version, "n": name})
    calls = []
    with pytest.raises(SchemaMigrationError):
        migrate(database, schema(), [("baseline", lambda conn: calls.append(1))])
    assert not calls
    assert not inspect(database).has_table("items")


def test_failed_upgrade_rolls_back_ddl_data_and_ledger(database):
    metadata = schema()
    migrate(database, metadata, [("baseline", metadata.create_all)])

    def broken(conn):
        conn.exec_driver_sql("ALTER TABLE items ADD COLUMN label TEXT")
        conn.exec_driver_sql("INSERT INTO items(id,label) VALUES (1,'partial')")
        raise RuntimeError("injected failure")

    with pytest.raises(RuntimeError, match="injected"):
        migrate(database, metadata, [("baseline", metadata.create_all), ("broken", broken)])
    assert [c["name"] for c in inspect(database).get_columns("items")] == ["id"]
    with database.connect() as conn:
        assert conn.execute(text("SELECT * FROM items")).all() == []
        assert conn.execute(text("SELECT version FROM business_schema_revisions")).all() == [(1,)]


def test_missing_schema_not_silently_repaired_or_stamped(database):
    metadata = schema()
    with pytest.raises(SchemaMigrationError, match="Missing business table"):
        migrate(database, metadata, [("baseline", lambda conn: None)])
    assert not inspect(database).has_table("business_schema_revisions")


def test_applied_schema_drift_fails_startup(database):
    metadata = schema()
    migrate(database, metadata, [("baseline", metadata.create_all)])
    Table("items", metadata, Column("extra", Integer), extend_existing=True)
    with pytest.raises(SchemaMigrationError, match="Missing business columns"):
        migrate(database, metadata, [("baseline", metadata.create_all)])


def test_concurrent_startups_serialize_upgrade(database):
    metadata = schema()
    calls = []

    def upgrade(conn):
        metadata.create_all(conn)
        conn.exec_driver_sql("INSERT INTO items VALUES (1)")
        calls.append(1)

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda _: migrate(database, metadata, [("baseline", upgrade)]), range(2)))
    assert calls == [1]
