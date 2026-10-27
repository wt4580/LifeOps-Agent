from __future__ import annotations

import re
import sqlite3
import time
import secrets
from contextlib import contextmanager
from pathlib import Path

from ..common.config.db_config import SessionLocal
from ..common.config.knowledge_db_config import KnowledgeSessionLocal
from ..common.config.log_config import logger


_SAFE_PREFIXES = ("SELECT", "PRAGMA", "EXPLAIN")
_BLOCKED_KEYWORDS = (
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER",
    "CREATE", "ATTACH", "DETACH", "REINDEX", "VACUUM",
    "EXECUTE", "EXEC",
)


def _validate_read_only(sql: str) -> None:
    cleaned = sql.strip().rstrip(";").strip()
    first_word = cleaned.split(None, 1)[0].upper() if cleaned else ""
    if not any(first_word.startswith(p) for p in _SAFE_PREFIXES):
        raise ValueError(f"只允许 SELECT/PRAGMA/EXPLAIN 查询，收到: {first_word}")
    upper_sql = cleaned.upper()
    for kw in _BLOCKED_KEYWORDS:
        if re.search(rf'\b{kw}\b', upper_sql):
            raise ValueError(f"禁止执行写操作: {kw}")


def validate_read_only(sql: str) -> None:
    """公开的只读校验入口，供 harness 策略层复用。"""
    _validate_read_only(sql)


@contextmanager
def scoped_connection(db_path: str, session_id: str):
    """Private read-only connection: shadow user tables with filtered TEMP views.

    SQLite's authorizer prevents explicit main.table, subqueries or UNION from
    bypassing the views. Nothing is installed on a pooled application connection.
    """
    if not session_id:
        raise ValueError("查询必须提供会话 ID")
    uri = Path(db_path).resolve().as_uri() + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        scoped = set()
        private_views = {}
        literal = "'" + session_id.replace("'", "''") + "'"
        for table in tables:
            quoted = '"' + table.replace('"', '""') + '"'
            columns = {r[1] for r in conn.execute(f"PRAGMA table_info({quoted})")}
            if "session_id" in columns:
                private = "scope_" + secrets.token_hex(16)
                private_views[table] = private
                conn.execute(f'CREATE TEMP VIEW "{private}" AS SELECT * FROM main.{quoted} WHERE session_id={literal}')
                conn.execute(f'CREATE TEMP VIEW {quoted} AS SELECT * FROM "{private}"')
                scoped.add(table)
        if {"topic_episode_messages", "topic_episodes"} <= tables:
            private = "scope_" + secrets.token_hex(16)
            private_views["topic_episode_messages"] = private
            conn.execute(
                f'CREATE TEMP VIEW "{private}" AS SELECT m.* '
                "FROM main.topic_episode_messages m JOIN main.topic_episodes e "
                f"ON e.id=m.episode_id WHERE e.session_id={literal}"
            )
            conn.execute(f'CREATE TEMP VIEW topic_episode_messages AS SELECT * FROM "{private}"')
            scoped.add("topic_episode_messages")
        globals_allowed = {"documents", "document_chunks", "sqlite_master", "sqlite_schema"}

        def authorize(action, table, column, database, source):
            if action == sqlite3.SQLITE_READ:
                if database == "temp" and (table in scoped or table in private_views.values()):
                    return sqlite3.SQLITE_OK
                if table in scoped and (source == private_views.get(table) or (
                    source == private_views.get("topic_episode_messages") and table == "topic_episodes"
                )):
                    return sqlite3.SQLITE_OK
                if table in globals_allowed:
                    return sqlite3.SQLITE_OK
                return sqlite3.SQLITE_DENY
            if action == sqlite3.SQLITE_PRAGMA:
                return sqlite3.SQLITE_OK if table.lower() in {"table_info", "table_xinfo", "index_list", "index_info"} else sqlite3.SQLITE_DENY
            if action == sqlite3.SQLITE_FUNCTION:
                return sqlite3.SQLITE_DENY if (column or "").lower() in {"load_extension", "readfile", "writefile"} else sqlite3.SQLITE_OK
            return sqlite3.SQLITE_OK if action in {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_RECURSIVE} else sqlite3.SQLITE_DENY

        conn.set_authorizer(authorize)
        deadline = time.monotonic() + 5
        conn.set_progress_handler(lambda: time.monotonic() > deadline, 1000)
        yield conn
    finally:
        conn.close()


def execute_read_query(sql: str, max_rows: int = 50, *, session_id: str) -> dict:
    _validate_read_only(sql)
    max_rows = min(200, max(1, int(max_rows)))

    for session_factory, db_name in [(SessionLocal, "business"), (KnowledgeSessionLocal, "knowledge")]:
        try:
            engine = session_factory.kw["bind"]
            if engine.dialect.name != "sqlite":
                raise ValueError("只读作用域查询当前仅支持 SQLite")
            with scoped_connection(engine.url.database, session_id) as db:
                result = db.execute(sql)
                columns = [column[0] for column in result.description]
                all_rows = result.fetchmany(max_rows + 1)
                truncated = len(all_rows) > max_rows
                rows_data = all_rows[:max_rows] if truncated else all_rows
                rows = [list(r) for r in rows_data]
                return {
                    "db": db_name,
                    "columns": columns,
                    "rows": rows,
                    "row_count": len(rows),
                    "truncated": truncated,
                }
        except Exception as exc:
            logger.debug("db_query [%s] failed: %s", db_name, exc)
            continue

    return {
        "db": None,
        "columns": [],
        "rows": [],
        "row_count": 0,
        "error": "查询失败：两个数据库均未找到匹配的表或存在语法错误",
    }


db_query_service = execute_read_query
