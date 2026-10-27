"""SQL session 作用域校验。

``query_database`` 目前完全依赖 LLM 在 prompt 约束下自己写 ``WHERE session_id=...``，
一旦模型漏写，就会跨会话读数据。这里做一层保守但明确的兜底：

- 只读语句（SELECT / PRAGMA / EXPLAIN）才放行，写操作交给 db_query_service 的
  ``validate_read_only`` 处理。
- 只要 SQL 引用了「会话级表」，就必须出现与当前 ``session_id`` 精确匹配的过滤条件。
- ``topic_episode_messages`` 本身没有 session_id，必须 JOIN ``topic_episodes``。
- 知识库表、``memory_candidates`` 等全局表不受此限制。

这不是完整的 SQL 解析器，只作为给模型的预检查反馈。
真正的数据隔离由 db_query_service.scoped_connection 的过滤视图和 SQLite 授权器强制执行，
不能将本模块的正则匹配结果作为访问权限依据。
"""

from __future__ import annotations

import re

# 带 session_id 列、必须按会话隔离的业务表。
SESSION_SCOPED_TABLES: frozenset[str] = frozenset(
    {
        "chat_messages",
        "topic_episodes",
        "todo_items",
        "life_events",
        "user_profiles",
        "conversation_summaries",
        "goal_items",
        "calendar_events",
        "accounts",
        "transactions",
    }
)

# 自身没有 session_id、必须通过父表间接隔离的表。
INDIRECTLY_SCOPED_TABLES: frozenset[str] = frozenset({"topic_episode_messages"})

_IDENT = r"[`\"\[]?[\w]+[`\"\]]?"
_TABLE_RE = re.compile(
    rf"\b(?:from|join)\s+((?:{_IDENT}\.)?{_IDENT})",
    re.IGNORECASE,
)
_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
_LINE_COMMENT_RE = re.compile(r"--[^\n]*")


def _strip_comments(sql: str) -> str:
    return _LINE_COMMENT_RE.sub(" ", _BLOCK_COMMENT_RE.sub(" ", sql))


def extract_tables(sql: str) -> set[str]:
    """提取 FROM / JOIN 后出现的表名（去引号、去 schema 前缀、转小写）。"""
    tables: set[str] = set()
    for match in _TABLE_RE.finditer(_strip_comments(sql)):
        name = match.group(1)
        if "." in name:
            name = name.rsplit(".", 1)[-1]
        name = name.strip("`\"[]")
        if name:
            tables.add(name.lower())
    return tables


def _has_session_filter(sql: str, session_id: str) -> bool:
    if not session_id:
        return False
    sid = re.escape(session_id)
    patterns = (
        rf"\bsession_id\b\s*=\s*['\"]?{sid}['\"]?",
        rf"\bsession_id\b\s+in\s*\(\s*['\"]?{sid}['\"]?\s*\)",
        rf"['\"]?{sid}['\"]?\s*=\s*\bsession_id\b",
    )
    return any(re.search(p, sql, re.IGNORECASE) for p in patterns)


def check_session_scope(sql: str, session_id: str) -> str | None:
    """返回拦截原因；``None`` 表示放行。"""
    cleaned = _strip_comments(sql).strip()
    if not cleaned:
        return None

    first_word = cleaned.split(None, 1)[0].upper()
    if first_word.startswith(("PRAGMA", "EXPLAIN")):
        return None

    tables = extract_tables(cleaned)
    if not tables:
        return None

    scoped = tables & SESSION_SCOPED_TABLES

    if tables & INDIRECTLY_SCOPED_TABLES:
        if "topic_episodes" not in tables:
            return (
                "查询 topic_episode_messages 时必须 JOIN topic_episodes，"
                "并通过它限定 session_id。"
            )
        scoped = scoped | {"topic_episodes"}

    if not scoped:
        return None

    if not _has_session_filter(cleaned, session_id):
        names = ", ".join(sorted(scoped))
        return (
            f"查询用户数据表（{names}）必须带上 session_id = "
            f"'{session_id}' 的过滤条件。"
        )
    return None
