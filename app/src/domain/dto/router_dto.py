from typing import Any

from pydantic import BaseModel, Field


class RouterTrace(BaseModel):
    """给前端/日志看的决策轨迹。"""

    intent: str = Field(..., description="例如 schedule_query / todo_create / chitchat")
    signals: list[str] = Field(default_factory=list, description="命中的关键词/句式信号")
    why: str = Field(..., description="为什么选择该 action（简短可读）")


class RouterDecision(BaseModel):
    action: str
    args: dict[str, Any] = Field(default_factory=dict)
    assistant_message: str = Field(..., description="给用户的第一句回应（可为空字符串）")
    trace: RouterTrace
