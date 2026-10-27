"""Password authentication and revocable, server-side browser sessions."""
from __future__ import annotations

import hashlib
import hmac
import secrets
import time
import threading
import shutil
import logging
from collections import deque
from datetime import datetime, timedelta

from sqlalchemy import select, func, update
from sqlalchemy.exc import IntegrityError

from ..common.config.db_config import SessionLocal
from ..domain.entity.chat_entity import AuthUser, AuthSession, MemoryCandidate

COOKIE_NAME = "lifeops_auth"
logger = logging.getLogger(__name__)
_LOGIN_ATTEMPTS: dict[str, deque] = {}
_ATTEMPT_LOCK = threading.Lock()


def allow_login_attempt(peer: str) -> bool:
    now = time.monotonic()
    with _ATTEMPT_LOCK:
        for key in list(_LOGIN_ATTEMPTS):
            attempts = _LOGIN_ATTEMPTS[key]
            while attempts and attempts[0] <= now - 60:
                attempts.popleft()
            if not attempts:
                del _LOGIN_ATTEMPTS[key]
        attempts = _LOGIN_ATTEMPTS.setdefault(peer, deque())
        if len(attempts) >= 10:
            return False
        attempts.append(now)
        return True


def _derive(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), 600_000).hex()


def login_user(username: str, password: str, *, allow_create: bool = False) -> str:
    with SessionLocal() as db:
        user = db.get(AuthUser, username)
        first_owner = False
        if user is None:
            if not allow_create:
                raise ValueError("首次设置密码请在本机 localhost 页面操作")
            first_owner = db.scalar(select(func.count()).select_from(AuthUser)) == 0
            salt = secrets.token_hex(16)
            user = AuthUser(username=username, salt=salt, password_hash=_derive(password, salt))
            db.add(user)
            try:
                db.flush()
            except IntegrityError as exc:
                db.rollback()
                raise ValueError("账户刚刚创建，请重新登录") from exc
        elif not hmac.compare_digest(user.password_hash, _derive(password, user.salt)):
            raise ValueError("用户名或密码不正确")
        token = secrets.token_urlsafe(32)
        db.add(AuthSession(token_hash=hashlib.sha256(token.encode()).hexdigest(), username=username,
                           expires_at=datetime.utcnow() + timedelta(days=30)))
        if first_owner:
            db.execute(update(MemoryCandidate).where(MemoryCandidate.session_id.is_(None)).values(session_id=username))
        db.commit()
        if first_owner:
            # The first locally enrolled owner inherits existing single-user notes.
            # Leave the original files intact for recovery.
            from ..common.md_loader import memory_directory, MARKDOWN_FILES
            destination = memory_directory(username)
            destination.mkdir(parents=True, exist_ok=True)
            for filename in MARKDOWN_FILES:
                source = memory_directory() / filename
                target = destination / filename
                if source.exists() and not target.exists():
                    try:
                        shutil.copy2(source, target)
                    except OSError:
                        logger.warning("Legacy memory import failed for %s; original retained", filename)
        return token


def authenticate(token: str | None) -> str | None:
    if not token:
        return None
    with SessionLocal() as db:
        row = db.get(AuthSession, hashlib.sha256(token.encode()).hexdigest())
        return row.username if row and row.expires_at > datetime.utcnow() else None


def logout(token: str | None) -> None:
    if not token:
        return
    with SessionLocal() as db:
        row = db.get(AuthSession, hashlib.sha256(token.encode()).hexdigest())
        if row:
            db.delete(row)
            db.commit()
