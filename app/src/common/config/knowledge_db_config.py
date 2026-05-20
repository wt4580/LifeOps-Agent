"""
知识库数据库配置。

职责：
1) 为知识库（document_chunks）提供独立的 SQLite 连接与会话工厂。
2) 在应用启动时初始化知识库表结构。
"""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import create_engine, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from .base_config import settings


class KnowledgeBase(DeclarativeBase):
    """知识库 ORM 基类。"""

    pass


knowledge_engine = create_engine(settings.knowledge_database_url, future=True)
KnowledgeSessionLocal = sessionmaker(bind=knowledge_engine, autoflush=False, autocommit=False, future=True)


def _ensure_sqlite_dir() -> None:
    """确保知识库 SQLite 文件所在目录存在。"""

    url = settings.knowledge_database_url
    if not url.startswith("sqlite:///"):
        return

    db_path = url.replace("sqlite:///", "", 1)
    Path(db_path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


def _ensure_document_chunks_columns() -> None:
    """为已存在知识库做轻量列迁移（不依赖 Alembic）。"""

    url = settings.knowledge_database_url
    if not url.startswith("sqlite:///"):
        return

    with knowledge_engine.begin() as conn:
        table_exists = conn.execute(
            text("SELECT name FROM sqlite_master WHERE type='table' AND name='document_chunks'")
        ).fetchone()
        if not table_exists:
            return

        rows = conn.execute(text("PRAGMA table_info(document_chunks)")).fetchall()
        existing = {r[1] for r in rows}

        if "source_type" not in existing:
            conn.execute(text("ALTER TABLE document_chunks ADD COLUMN source_type VARCHAR(32)"))
        if "doc_topic" not in existing:
            conn.execute(text("ALTER TABLE document_chunks ADD COLUMN doc_topic VARCHAR(64)"))


def init_knowledge_db() -> None:
    """初始化知识库数据库。"""

    _ensure_sqlite_dir()

    from ...domain.entity import knowledge_entity  # noqa: F401

    KnowledgeBase.metadata.create_all(bind=knowledge_engine)
    _ensure_document_chunks_columns()
