from pydantic import BaseModel,Field,field_validator
from typing import Optional


class ChatDto(BaseModel):
    """聊天出参。

    answer:
      - 助手回复。

    proposal_id / proposal:
      - 如果本轮触发了“待办草案”，会一并返回。
      - 前端可展示草案，并提供确认按钮。

    used_tool:
      - 便于调试：告诉你这次走了哪个分支（schedule/plan_gate/plan_proposal/None）。

    citations:
      - 当本轮触发知识库工具时，返回引用列表（path/page/snippet/score/reason）。
      - 前端可直接展示“答案来自哪里”。

    trace:
      - 可审计决策轨迹，包含 router 决策、工具输入输出、关键中间结果。
    """

    answer: str=Field(..., description="助手回复的文本内容")
    session_id: str=Field(..., description="会话 ID")
    proposal_id: Optional[str]=Field(None, description="待办草案 ID")
    proposal: Optional[dict] =Field(None, description="待办草案内容")
    used_tool: Optional[str] =Field(None, description="使用的工具")
    citations: Optional[list[dict]] =Field(None, description="引用列表")
    trace: Optional[dict] =Field(None, description="审计轨迹")