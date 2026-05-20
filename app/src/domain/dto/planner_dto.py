from pydantic import BaseModel, Field


class PlanItem(BaseModel):
    """单个待办草案项。"""

    title: str
    due_at: str | None = None


class PlanProposal(BaseModel):
    """草案容器：一个 proposal 可包含多条待办。"""

    items: list[PlanItem] = Field(default_factory=list)
