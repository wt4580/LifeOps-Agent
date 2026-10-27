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
from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import sessionmaker, DeclarativeBase, Session

from .base_config import settings


class Base(DeclarativeBase):
    """ORM 基类。所有模型类都继承它，SQLAlchemy 才能收集元数据。"""

    pass


# 创建数据库引擎：
# - 这里使用 `future=True` 启用 2.x 风格行为，减少兼容层歧义。
engine = create_engine(settings.database_url, future=True)


def _apply_sqlite_pragmas(dbapi_connection, connection_record) -> None:
    """每个 SQLite 连接建立时统一设置：WAL + NORMAL 同步。

    后台旁路（图后学习/记忆/摘要）与主流程分线程并发访问数据库，
    WAL 让读者不被写者阻塞，NORMAL 同步降低每次提交的 fsync 开销。
    """
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")


if settings.database_url.startswith("sqlite:///"):
    event.listen(engine, "connect", _apply_sqlite_pragmas)

# 创建会话工厂：
# - autoflush=False：避免查询前自动刷盘带来的“隐式写入”困惑。
# - autocommit=False：显式 commit 更可控。
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


@event.listens_for(Session, "before_commit")
def _prevent_cancelled_turn_commit(session):
    from ..harness.context import ensure_turn_active
    ensure_turn_active()


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


def _ensure_todo_columns() -> None:
    """为已存在 todo_items 表补齐完成状态列（轻量迁移）。"""

    url = settings.database_url
    if not url.startswith("sqlite:///"):
        return

    with engine.begin() as conn:
        table_exists = conn.execute(
            text("SELECT name FROM sqlite_master WHERE type='table' AND name='todo_items'")
        ).fetchone()
        if not table_exists:
            return

        rows = conn.execute(text("PRAGMA table_info(todo_items)")).fetchall()
        existing = {r[1] for r in rows}

        if "is_completed" not in existing:
            conn.execute(text("ALTER TABLE todo_items ADD COLUMN is_completed BOOLEAN NOT NULL DEFAULT 0"))
        if "completed_at" not in existing:
            conn.execute(text("ALTER TABLE todo_items ADD COLUMN completed_at DATETIME"))
        if "repeat_rule" not in existing:
            conn.execute(text("ALTER TABLE todo_items ADD COLUMN repeat_rule VARCHAR(16)"))
        if "repeat_until" not in existing:
            conn.execute(text("ALTER TABLE todo_items ADD COLUMN repeat_until DATETIME"))


def _ensure_life_events_columns() -> None:
    """为现有 life_events 表补齐 importance 列。"""
    url = settings.database_url
    if not url.startswith("sqlite:///"):
        return
    with engine.begin() as conn:
        table_exists = conn.execute(
            text("SELECT name FROM sqlite_master WHERE type='table' AND name='life_events'")
        ).fetchone()
        if not table_exists:
            return
        rows = conn.execute(text("PRAGMA table_info(life_events)")).fetchall()
        existing = {r[1] for r in rows}
        if "importance" not in existing:
            conn.execute(text("ALTER TABLE life_events ADD COLUMN importance VARCHAR(8) NOT NULL DEFAULT 'low'"))
        if "event_time" not in existing:
            conn.execute(text("ALTER TABLE life_events ADD COLUMN event_time DATETIME"))


def _ensure_user_profile_columns() -> None:
    """为现有 user_profiles 表补齐新增列。"""
    url = settings.database_url
    if not url.startswith("sqlite:///"):
        return
    with engine.begin() as conn:
        table_exists = conn.execute(
            text("SELECT name FROM sqlite_master WHERE type='table' AND name='user_profiles'")
        ).fetchone()
        if not table_exists:
            return
        rows = conn.execute(text("PRAGMA table_info(user_profiles)")).fetchall()
        existing = {r[1] for r in rows}
        new_columns = {
            "age": "INTEGER",
            "gender": "VARCHAR(16)",
            "occupation": "VARCHAR(128)",
            "city": "VARCHAR(64)",
            "diet_json": "TEXT",
            "allergies_json": "TEXT",
            "sleep_schedule": "VARCHAR(64)",
            "exercise_habits": "VARCHAR(128)",
            "work_hours": "VARCHAR(32)",
            "family_status": "VARCHAR(32)",
            "goals_json": "TEXT",
        }
        for col, col_type in new_columns.items():
            if col not in existing:
                conn.execute(text(f"ALTER TABLE user_profiles ADD COLUMN {col} {col_type}"))


def _ensure_memory_columns() -> None:
    url = settings.database_url
    if not url.startswith("sqlite:///"):
        return
    with engine.begin() as conn:
        table_exists = conn.execute(
            text("SELECT name FROM sqlite_master WHERE type='table' AND name='memory_candidates'")
        ).fetchone()
        if not table_exists:
            return
        rows = conn.execute(text("PRAGMA table_info(memory_candidates)")).fetchall()
        existing = {r[1] for r in rows}
        if "confidence" not in existing:
            conn.execute(text("ALTER TABLE memory_candidates ADD COLUMN confidence FLOAT NOT NULL DEFAULT 1.0"))
        if "insight_type" not in existing:
            conn.execute(text("ALTER TABLE memory_candidates ADD COLUMN insight_type VARCHAR(16) NOT NULL DEFAULT 'explicit'"))


def _ensure_chat_message_columns() -> None:
    """chat_messages 增量列迁移（现有库缺列时自动补齐）。"""
    url = settings.database_url
    if not url.startswith("sqlite:///"):
        return
    with engine.begin() as conn:
        table_exists = conn.execute(
            text("SELECT name FROM sqlite_master WHERE type='table' AND name='chat_messages'")
        ).fetchone()
        if not table_exists:
            return
        rows = conn.execute(text("PRAGMA table_info(chat_messages)")).fetchall()
        existing = {r[1] for r in rows}
        if "topic" not in existing:
            conn.execute(text("ALTER TABLE chat_messages ADD COLUMN topic VARCHAR(64)"))


def init_db() -> None:
    _ensure_sqlite_dir()
    from ...domain.entity import chat_entity  # noqa: F401
    Base.metadata.create_all(bind=engine)
    _ensure_todo_columns()
    _ensure_memory_columns()
    _ensure_user_profile_columns()
    _ensure_life_events_columns()
    _ensure_chat_message_columns()
    if settings.database_url.startswith("sqlite:///"):
        with engine.begin() as conn:
            for table, column, definition in (
                ("memory_candidates", "session_id", "VARCHAR(64)"),
                ("conversation_summaries", "summarized_through_id", "INTEGER NOT NULL DEFAULT 0"),
            ):
                columns = {r[1] for r in conn.execute(text(f"PRAGMA table_info({table})"))}
                if column not in columns:
                    conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {definition}"))
