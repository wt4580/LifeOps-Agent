from __future__ import annotations

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase

from .settings import settings


class Base(DeclarativeBase):
    pass


engine = create_engine(settings.database_url, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


def init_db() -> None:
    os.makedirs("./data", exist_ok=True)
    from . import models  # noqa: F401

    Base.metadata.create_all(bind=engine)

