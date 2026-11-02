from pydantic import BaseModel, Field
from typing import Literal


class PlanItem(BaseModel):
    """单个待办草案项。"""

    title: str
    due_at: str | None = None
    repeat: Literal['once', 'daily', 'weekly'] | None = None
    repeat_until: str | None = None
    duration_min: int | None = Field(default=None, ge=1, le=1440)
    description: str | None = None


class PlanProposal(BaseModel):
    """草案容器：一个 proposal 可包含多条待办。"""

    items: list[PlanItem] = Field(default_factory=list)
