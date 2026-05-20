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
            "source": todo.source,
            "is_completed": bool(todo.is_completed),
            "completed_at": todo.completed_at.isoformat() if todo.completed_at else None,
        }

    def _apply_scope_filter(self, stmt, session_id: str):
        scope = (settings.personal_memory_scope or "global").strip().lower()
        session_col = getattr(TodoItem, "session_id", None)
        if scope == "session" and session_col is not None:
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
            stmt = (
                select(TodoItem)
                .where(TodoItem.due_at != None)
                .where(TodoItem.due_at >= today_start)
                .where(TodoItem.due_at < today_end)
                .order_by(TodoItem.due_at.asc())
            )
            stmt = self._apply_scope_filter(stmt, session_id)
            rows = db.execute(stmt).scalars().all()

        pending = [self._serialize_todo(row) for row in rows if not row.is_completed]
        completed = [self._serialize_todo(row) for row in rows if row.is_completed]
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
        
        with SessionLocal() as db:
            stmt = (
                select(TodoItem)
                .where(TodoItem.due_at != None)
                .where(TodoItem.due_at >= now)
                .where(TodoItem.due_at < end)
                .where(TodoItem.is_completed == False)
                .order_by(TodoItem.due_at.asc())
            )
            stmt = self._apply_scope_filter(stmt, session_id)
            rows = db.execute(stmt).scalars().all()

        return [self._serialize_todo(row) for row in rows]

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
