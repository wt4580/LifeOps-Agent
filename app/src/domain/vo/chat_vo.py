from pydantic import BaseModel,Field,field_validator
from typing import Optional

class ChatVo(BaseModel):
    """聊天入参。

    session_id:
      - 前端保存在 localStorage 的会话 ID（UUID）。
      - 首次没有则由后端生成。

    message:
      - 用户输入的自然语言文本。
    """

    message: str=Field(..., description="用户输入的自然语言文本")
    session_id: Optional[str]=Field(None, description="会话 ID")

    @field_validator("message",mode="before")
    @classmethod
    def validate_message(cls, value: str) -> str:
        if not value:
            raise ValueError("输入的文本不能为空")
        if len(value)>1000:
            raise ValueError("输入的文本长度不能超过1000")

        if isinstance(value, str):
           value = value.strip()

        return value
