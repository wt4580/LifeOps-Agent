"""待办控制器 - 处理待办事项查询相关的HTTP请求"""

from fastapi import APIRouter, HTTPException

from ..common.domain.response import success_response
from ..service.todo_service import todo_service, reminder_service
from ..domain.dto.todo_dto import TodoStatusUpdateRequest

router = APIRouter()


@router.get("/api/todos/today")
def get_todos_today(session_id: str | None = None):
    """获取今日待办事项
    
    Args:
        session_id: 可选的会话ID，用于过滤作用域
    """
    try:
        result = todo_service.get_todos_today(session_id=session_id or "default")
        return success_response(data=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/todos/upcoming")
def get_upcoming_todos(hours: int | None = None, days: int | None = None, session_id: str | None = None):
    """获取即将到来的待办事项
    
    Args:
        hours: 时间窗口（小时），默认48小时
        session_id: 可选的会话ID，用于过滤作用域
    """
    try:
        effective_hours = hours
        if effective_hours is None:
            effective_hours = (days or 2) * 24
        todos = todo_service.get_upcoming_todos(
            session_id=session_id or "default",
            hours=effective_hours
        )
        return success_response(data={"todos": todos, "hours": effective_hours})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/reminders/pending")
def get_pending_reminders(session_id: str = "default"):
    try:
        result = reminder_service.load_pending_reminders(session_id=session_id)
        return success_response(data={"reminders": result})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/reminders/{reminder_id}/dismiss")
def dismiss_reminder(reminder_id: int):
    try:
        ok = reminder_service.dismiss_reminder(reminder_id)
        if not ok:
            raise HTTPException(status_code=404, detail="提醒不存在")
        return success_response(data={"id": reminder_id})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/todos/{todo_id}/status")
def update_todo_status(todo_id: int, req: TodoStatusUpdateRequest):
    """更新待办完成状态。"""

    try:
        todo = todo_service.update_todo_status(
            todo_id=todo_id,
            completed=req.completed,
            session_id=req.session_id,
        )
        if not todo:
            raise HTTPException(status_code=404, detail="待办不存在")
        return success_response(data={"todo": todo})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
