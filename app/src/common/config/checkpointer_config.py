"""LangGraph checkpointer 配置。"""

from __future__ import annotations

from pathlib import Path

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from .base_config import settings


def _resolve_checkpoint_path() -> Path:
    return Path(settings.checkpointer_db_path).expanduser().resolve()


def create_checkpointer():
    """创建基于 SQLite 的异步 LangGraph checkpointer 上下文管理器。"""

    checkpoint_path = _resolve_checkpoint_path()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    return AsyncSqliteSaver.from_conn_string(checkpoint_path.as_posix())
