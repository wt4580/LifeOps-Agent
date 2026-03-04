from __future__ import annotations

import os
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase

from .settings import settings


class Base(DeclarativeBase):
    pass


engine = create_engine(settings.database_url, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


def _ensure_sqlite_dir() -> None:
    """确保 SQLite 文件所在目录存在。

    我们把 DB URL 固定成绝对路径后，不能再简单 os.makedirs('./data')。
    否则会出现“目录创建了，但 DB 文件在另一个位置”的错觉。
    """

    url = settings.database_url
    if not url.startswith("sqlite:///"):
        return

    # url 形如：sqlite:///E:/.../data/lifeops.db
    db_path = url.replace("sqlite:///", "", 1)
    Path(db_path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


def init_db() -> None:
    _ensure_sqlite_dir()
    from . import models  # noqa: F401

    Base.metadata.create_all(bind=engine)
