"""待办事项服务层 - 封装待办查询相关的业务逻辑"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import select

from ..common.config.db_config import SessionLocal
from ..common.config.base_config import settings
from ..common.config.log_config import logger
from ..domain.entity.chat_entity import TodoItem, ReminderItem


class TodoService:
    """待办事项服务
    
    职责：
    1. 查询今日待办
    2. 查询即将到来的待办
    3. 管理待办状态
    """
    
    def _serialize_todo(self, todo: TodoItem) -> dict:
        return {
            "id": todo.id,
            "title": todo.title,
            "due_at": todo.due_at.isoformat() if todo.due_at else None,
            "repeat_rule": todo.repeat_rule,
            "repeat_until": todo.repeat_until.isoformat() if todo.repeat_until else None,
            "source": todo.source,
            "is_completed": bool(todo.is_completed),
            "completed_at": todo.completed_at.isoformat() if todo.completed_at else None,
        }

    @staticmethod
    def _is_recurring_active(todo: TodoItem, d: datetime) -> bool:
        """周期待办在 d 当天是否处于生效期（daily 每日有效，weekly 取锚点星期匹配）。"""
        rule = todo.repeat_rule
        if rule not in ("daily", "weekly"):
            return False
        day = d.replace(hour=0, minute=0, second=0, microsecond=0)
        if todo.repeat_until is not None and todo.repeat_until < day:
            return False
        if rule == "weekly":
            anchor = todo.due_at or day
            if anchor.weekday() != day.weekday():
                return False
        return True

    @staticmethod
    def _done_today(todo: TodoItem, today_start: datetime) -> bool:
        return todo.is_completed and todo.completed_at is not None and todo.completed_at >= today_start

    def _apply_scope_filter(self, stmt, session_id: str):
        session_col = getattr(TodoItem, "session_id", None)
        if session_col is not None:
            return stmt.where(session_col == session_id)
        return stmt

    def get_todos_today(self, session_id: str) -> dict:
        """获取今日待办事项
        
        Args:
            session_id: 会话ID（用于过滤作用域）
            
        Returns:
            今日待办列表
        """
        now = datetime.now()
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        today_end = today_start + timedelta(days=1)

        with SessionLocal() as db:
            stmt = self._apply_scope_filter(
                select(TodoItem).order_by(TodoItem.due_at.asc()), session_id
            )
            rows = db.execute(stmt).scalars().all()

        pending, completed = [], []
        for row in rows:
            due_today = row.due_at is not None and today_start <= row.due_at < today_end
            recurring = self._is_recurring_active(row, now)
            if not (due_today or recurring):
                continue
            serialized = self._serialize_todo(row)
            if self._done_today(row, today_start):
                completed.append(serialized)
            else:
                pending.append(serialized)
        return {
            "pending": pending,
            "completed": completed,
            "todos": [*pending, *completed],
        }
    
    def get_upcoming_todos(self, session_id: str, hours: int = 48) -> list[dict]:
        """获取未来指定小时内的待办事项
        
        Args:
            session_id: 会话ID
            hours: 时间窗口（小时），默认48小时
            
        Returns:
            即将到来的待办列表
        """
        now = datetime.now()
        end = now + timedelta(hours=hours)
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

        with SessionLocal() as db:
            stmt = self._apply_scope_filter(select(TodoItem), session_id)
            rows = db.execute(stmt).scalars().all()

        result: list[dict] = []
        for row in rows:
            recurring = self._is_recurring_active(row, now)
            if row.due_at is not None:
                show = now <= row.due_at < end
            else:
                show = not row.is_completed
            if recurring:
                show = not self._done_today(row, today_start)
            if show:
                result.append(self._serialize_todo(row))
        result.sort(key=lambda r: r["due_at"] or "9999-12-31T00:00:00")
        return result

    def get_all_todos(self, session_id: str) -> dict:
        """获取全部待办事项（只显示未完成且未过期的）。"""
        now = datetime.now()
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        with SessionLocal() as db:
            stmt = self._apply_scope_filter(select(TodoItem), session_id)
            rows = db.execute(stmt).scalars().all()

        result: list[dict] = []
        for row in rows:
            recurring = self._is_recurring_active(row, now)
            if recurring:
                if not self._done_today(row, today_start):
                    result.append(self._serialize_todo(row))
                continue
            if row.is_completed:
                continue
            if row.due_at is not None and row.due_at < now:
                continue
            result.append(self._serialize_todo(row))
        result.sort(key=lambda r: r["due_at"] or "9999-12-31T00:00:00")
        return {"pending": result, "completed": []}

    def update_todo_status(self, todo_id: int, completed: bool, session_id: str | None = None) -> dict | None:
        with SessionLocal() as db:
            stmt = select(TodoItem).where(TodoItem.id == todo_id)
            if session_id:
                stmt = self._apply_scope_filter(stmt, session_id)

            todo = db.execute(stmt).scalars().first()
            if not todo:
                return None

            todo.is_completed = bool(completed)
            todo.completed_at = datetime.now() if completed else None
            db.commit()
            db.refresh(todo)
            return self._serialize_todo(todo)


# 单例模式
todo_service = TodoService()


class ReminderService:
    """提醒事项服务 - 管理 AI 检测到的"在干但没干完"的提醒（非待办）。"""

    BASE_INTERVAL = timedelta(hours=2)
    MAX_INTERVAL = timedelta(days=7)

    def _serialize(self, r: ReminderItem) -> dict:
        return {
            "id": r.id,
            "title": r.title,
            "source": r.source,
            "remind_count": r.remind_count,
            "last_remind_at": r.last_remind_at.isoformat() if r.last_remind_at else None,
            "next_remind_at": r.next_remind_at.isoformat() if r.next_remind_at else None,
            "is_dismissed": bool(r.is_dismissed),
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }

    @staticmethod
    def _compute_next_remind_at(remind_count: int) -> datetime:
        interval = ReminderService.BASE_INTERVAL * (2 ** remind_count)
        if interval > ReminderService.MAX_INTERVAL:
            interval = ReminderService.MAX_INTERVAL
        return datetime.now() + interval

    def upsert_reminder(self, session_id: str, title: str, source: str = "proactive") -> dict:
        """找到相同 title 未 dismiss 的提醒；若有则加计数、更新下次时间；否则新建。"""
        with SessionLocal() as db:
            stmt = (
                select(ReminderItem)
                .where(ReminderItem.session_id == session_id)
                .where(ReminderItem.title == title)
                .where(ReminderItem.is_dismissed == False)
                .order_by(ReminderItem.created_at.desc())
                .limit(1)
            )
            existing = db.execute(stmt).scalars().first()

            if existing:
                existing.remind_count = (existing.remind_count or 0) + 1
                existing.last_remind_at = datetime.now()
                existing.next_remind_at = self._compute_next_remind_at(existing.remind_count)
                db.commit()
                db.refresh(existing)
                return self._serialize(existing)
            else:
                now = datetime.now()
                item = ReminderItem(
                    session_id=session_id,
                    title=title,
                    source=source,
                    remind_count=1,
                    last_remind_at=now,
                    next_remind_at=self._compute_next_remind_at(1),
                )
                db.add(item)
                db.commit()
                db.refresh(item)
                return self._serialize(item)

    def load_pending_reminders(self, session_id: str) -> list[dict]:
        """加载所有提醒时间已到且未 dismiss 的提醒。"""
        now = datetime.now()
        with SessionLocal() as db:
            stmt = (
                select(ReminderItem)
                .where(ReminderItem.session_id == session_id)
                .where(ReminderItem.next_remind_at <= now)
                .where(ReminderItem.is_dismissed == False)
                .order_by(ReminderItem.next_remind_at.asc())
            )
            rows = db.execute(stmt).scalars().all()
        return [self._serialize(r) for r in rows]

    def dismiss_reminder(self, reminder_id: int) -> bool:
        with SessionLocal() as db:
            stmt = select(ReminderItem).where(ReminderItem.id == reminder_id)
            row = db.execute(stmt).scalars().first()
            if not row:
                return False
            row.is_dismissed = True
            db.commit()
            return True


reminder_service = ReminderService()
