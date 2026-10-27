from __future__ import annotations

from datetime import datetime, timedelta

from sqlalchemy import select

from ..common.config.base_config import settings
from ..common.config.db_config import SessionLocal
from ..domain.entity.chat_entity import AuthUser, TodoItem
from ..service.auth_service import login_user
from ..service.bookkeeping_service import bookkeeping_service
from ..util.retrieval.retrieval import _index_documents_sqlite, _sync_document_records


def seed_demo():
    if not settings.demo_mode:
        raise RuntimeError("Demo fixtures are forbidden in live mode")
    if len(settings.demo_password) < 8:
        raise ValueError("DEMO_PASSWORD must contain at least 8 characters")
    with SessionLocal() as db:
        if db.scalar(select(AuthUser).where(AuthUser.username != settings.demo_username).limit(1)):
            raise RuntimeError("Refusing to seed a database containing personal users; use a dedicated demo volume")
    login_user(settings.demo_username, settings.demo_password, allow_create=True)
    if not bookkeeping_service.find_account("演示钱包", session_id=settings.demo_username):
        bookkeeping_service.add_account("演示钱包", 1000, session_id=settings.demo_username)
    with SessionLocal() as db:
        exists = db.scalar(
            select(TodoItem.id).where(
                TodoItem.session_id == settings.demo_username, TodoItem.title == "演示：准备项目介绍"
            )
        )
        if exists is None:
            db.add(
                TodoItem(
                    session_id=settings.demo_username,
                    title="演示：准备项目介绍",
                    due_at=(datetime.now() + timedelta(days=1)).replace(hour=15, minute=0, second=0, microsecond=0),
                    source="demo_seed",
                )
            )
            db.commit()
    _index_documents_sqlite(rebuild=False)
    _sync_document_records()
