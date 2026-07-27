from __future__ import annotations

import re
import logging

from sqlalchemy import text

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


def execute_read_query(sql: str, max_rows: int = 50) -> dict:
    _validate_read_only(sql)
    stmt = text(sql)

    for session_factory, db_name in [(SessionLocal, "business"), (KnowledgeSessionLocal, "knowledge")]:
        try:
            with session_factory() as db:
                result = db.execute(stmt)
                columns = list(result.keys())
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
