"""待办事项服务层 - 封装待办查询相关的业务逻辑"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import select

from ..common.config.db_config import SessionLocal
from ..common.config.base_config import settings
from ..common.config.log_config import logger
from ..domain.entity.chat_entity import TodoItem


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
