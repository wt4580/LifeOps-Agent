from pydantic import BaseModel


class DateHint(BaseModel):
    """日期解析结果。"""

    date: str | None = None
    confidence: float = 0.5
