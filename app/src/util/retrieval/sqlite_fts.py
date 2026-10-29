"""Persistent lexical indexes; only dirty batches and matching chunks enter Python.

SQL triggers enqueue changes in the writer's transaction, including raw SQL writes.
The queue is drained at ingestion and before searching. No SQL Python UDF is needed.
"""
from threading import RLock
from weakref import WeakSet

from sqlalchemy import text

_ready = WeakSet()
_lock = RLock()


def ensure_index(engine):
    if engine.dialect.name != "sqlite":
        raise ValueError("Persistent BM25 retrieval requires SQLite with FTS5")
    with _lock:
        if engine in _ready:
            return
        with engine.begin() as conn:
            # Retire the previous full-corpus cache invalidation machinery.
            for table in ("documents", "document_chunks"):
                for action in ("insert", "update", "delete"):
                    conn.execute(text(f"DROP TRIGGER IF EXISTS revision_{table}_{action}"))
            conn.execute(text("CREATE TABLE IF NOT EXISTS lexical_index_state "
                              "(id INTEGER PRIMARY KEY, version INTEGER NOT NULL)"))
            initialized = conn.execute(text("SELECT version FROM lexical_index_state WHERE id=1")).first()
            # A separate index for summaries avoids rebuilding BM25 over all summaries.
            for kind, table, column in (("chunk", "document_chunks", "chunk_text"),
                                        ("summary", "documents", "summary")):
                conn.execute(text(f"CREATE VIRTUAL TABLE IF NOT EXISTS lexical_{kind} "
                                  "USING fts5(tokens, tokenize='unicode61 remove_diacritics 0')"))
                conn.execute(text(f"CREATE TABLE IF NOT EXISTS lexical_{kind}_dirty "
                                  "(id INTEGER PRIMARY KEY)"))
                for action, ref in (("INSERT", "new"), ("UPDATE", "new"), ("DELETE", "old")):
                    # Restrict UPDATE triggers to indexed columns; metadata needs no re-tokenization.
                    event = f"UPDATE OF {column}" if action == "UPDATE" else action
                    conn.execute(text(
                        f"CREATE TRIGGER IF NOT EXISTS lexical_{kind}_{action.lower()} "
                        f"AFTER {event} ON {table} BEGIN "
                        f"INSERT OR IGNORE INTO lexical_{kind}_dirty VALUES ({ref}.id); END"
                    ))
                if not initialized:
                    conn.execute(text(f"INSERT OR IGNORE INTO lexical_{kind}_dirty SELECT id FROM {table}"))
            conn.execute(text("INSERT OR IGNORE INTO lexical_index_state VALUES (1, 1)"))
        _ready.add(engine)


def sync_index(engine, tokenize):
    ensure_index(engine)
    # The initial migration is batched too. At most 256 source texts per batch.
    for kind, table, column in (("chunk", "document_chunks", "chunk_text"),
                                ("summary", "documents", "summary")):
        while True:
            with engine.connect() as conn:
                dirty = conn.execute(text(f"SELECT 1 FROM lexical_{kind}_dirty LIMIT 1")).first()
            if not dirty:
                break
            with engine.begin() as conn:
                # Claim SQLite's writer lock before reading the queue. Other processes
                # cannot change source rows between reading, indexing and acknowledgement.
                conn.execute(text("UPDATE lexical_index_state SET version=version WHERE id=1"))
                rows = conn.execute(text(
                    f"SELECT q.id, s.{column} AS body FROM lexical_{kind}_dirty q "
                    f"LEFT JOIN {table} s ON s.id=q.id ORDER BY q.id LIMIT 256"
                )).all()
                for row in rows:
                    conn.execute(text(f"DELETE FROM lexical_{kind} WHERE rowid=:id"), {"id": row.id})
                    tokens = tokenize(row.body or "")
                    if tokens:
                        conn.execute(text(f"INSERT INTO lexical_{kind}(rowid, tokens) VALUES (:id,:tokens)"),
                                     {"id": row.id, "tokens": " ".join(_term(t) for t in tokens)})
                    conn.execute(text(f"DELETE FROM lexical_{kind}_dirty WHERE id=:id"), {"id": row.id})


def _term(token):
    # Encode each application token as one ASCII term, so unicode61 cannot
    # split identifiers such as "gpt-4.1" differently from our tokenizer.
    return "t" + token.encode("utf-8").hex()


def match_expression(query, tokenize):
    # Treat user input as literal tokens, never as FTS operators or column names.
    tokens = dict.fromkeys(tokenize(query))
    return " OR ".join('"' + _term(token) + '"' for token in tokens)


def search(db, expression, limit, personal=False):
    """Apply the existing source/summary filters in SQL before taking top-k.

    IDF now uses the persistent global corpus rather than a per-query subset.
    FTS5 returns negative BM25 values: lower is better.
    """
    if not expression or limit <= 0:
        return []
    source = ""
    if personal:
        allowed = "COALESCE(doc_topic,'other') NOT IN ('guideline','recipe','manual','paper')"
        source = f" AND ({allowed} OR NOT EXISTS (SELECT 1 FROM document_chunks WHERE {allowed}))"
    summary = """
      AND (
        NOT EXISTS (SELECT 1 FROM lexical_summary WHERE lexical_summary MATCH :query)
        OR NOT EXISTS (SELECT 1 FROM documents d JOIN lexical_summary f ON f.rowid=d.id
                       WHERE d.file_path=c.file_path)
        OR c.file_path IN (SELECT d.file_path FROM documents d JOIN lexical_summary f ON f.rowid=d.id
                          WHERE lexical_summary MATCH :query)
      )
    """
    return db.execute(text(
        "SELECT c.id, bm25(lexical_chunk) AS lexical_score "
        "FROM lexical_chunk JOIN document_chunks c ON c.id=lexical_chunk.rowid "
        "WHERE lexical_chunk MATCH :query" + source + summary +
        " ORDER BY lexical_score ASC, c.id ASC LIMIT :limit"
    ), {"query": expression, "limit": limit}).all()
