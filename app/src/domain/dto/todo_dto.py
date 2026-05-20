from pydantic import BaseModel, Field


class TodoStatusUpdateRequest(BaseModel):
    """待办完成状态更新请求。"""

    completed: bool = Field(..., description="是否完成")
    session_id: str | None = Field(default=None, description="会话ID（session 范围下用于权限过滤）")
