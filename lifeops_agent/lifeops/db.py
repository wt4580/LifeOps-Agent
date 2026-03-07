"""
lifeops.db

这个模块是项目的“数据库底座”，职责只有两件：
1) 创建数据库连接与会话工厂（engine / SessionLocal）。
2) 在应用启动时初始化表结构（init_db）。

对初学者的心智模型：
- engine：数据库“连接能力”。
- SessionLocal：每次请求拿到的一次“数据库操作上下文”。
- Base：所有 ORM 表模型的共同父类。
"""

from __future__ import annotations

from pathlib import Path
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, DeclarativeBase

from .settings import settings


class Base(DeclarativeBase):
    """ORM 基类。所有模型类都继承它，SQLAlchemy 才能收集元数据。"""

    pass


# 创建数据库引擎：
# - 这里使用 `future=True` 启用 2.x 风格行为，减少兼容层歧义。
engine = create_engine(settings.database_url, future=True)

# 创建会话工厂：
# - autoflush=False：避免查询前自动刷盘带来的“隐式写入”困惑。
# - autocommit=False：显式 commit 更可控。
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


def _ensure_sqlite_dir() -> None:
    """确保 SQLite 文件所在目录存在。

    背景：
    - 当前 database_url 使用绝对路径。
    - 如果目录不存在，SQLite 无法创建 db 文件，会在启动时报错。
    """

    url = settings.database_url
    if not url.startswith("sqlite:///"):
        # 非 sqlite 场景（例如未来切到 MySQL/Postgres）无需该步骤。
        return

    # 例：sqlite:///E:/.../data/lifeops.db -> 提取真实文件路径。
    db_path = url.replace("sqlite:///", "", 1)

    # 只创建父目录，不触碰具体文件。
    Path(db_path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


def _ensure_document_chunks_columns() -> None:
    """为已存在 SQLite 库做轻量列迁移（不依赖 Alembic）。"""

    url = settings.database_url
    if not url.startswith("sqlite:///"):
        return

    with engine.begin() as conn:
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


def init_db() -> None:
    """初始化数据库：
    1) 确保 SQLite 目录存在。
    2) 导入 models 触发表定义注册。
    3) create_all 按模型创建缺失表（不会删除已有表）。
    """

    _ensure_sqlite_dir()

    # 仅为触发 ORM 模型注册到 Base.metadata。
    from . import models  # noqa: F401

    Base.metadata.create_all(bind=engine)
    _ensure_document_chunks_columns()
