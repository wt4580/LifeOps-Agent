from pydantic import BaseModel, Field


class MemoryItem(BaseModel):
    """单条记忆候选。"""

    kind: str = Field(..., description="meeting|deadline|birthday|todo|event|preference|pattern")
    title: str
    occurred_at: str | None = None
    notes: str | None = None
    confidence: float = Field(default=1.0, ge=0, le=1, description="确信度 0-1")
    insight_type: str = Field(default="explicit", description="explicit|inferred")


class MemoryExtraction(BaseModel):
    """记忆抽取结果容器。"""

    items: list[MemoryItem]


class PersonalEventItem(BaseModel):
    """单条个人事件。"""

    category: str = Field(..., description="diet|activity|research|finance|schedule|health|other")
    title: str = Field(..., description="事件标题，简短清晰")
    event_time: str | None = Field(default=None, description="ISO 时间字符串，可为空")
    amount: float | None = Field(default=None, description="金额或数量")
    amount_unit: str | None = Field(default=None, description="单位，如 元/次/公里")
    tags: list[str] = Field(default_factory=list, description="标签，如 油腻/高盐/通勤/检索")
    notes: str | None = Field(default=None, description="补充说明")


class PersonalEventExtraction(BaseModel):
    """事件抽取容器。"""

    items: list[PersonalEventItem] = Field(default_factory=list)


class ProfileFactPatch(BaseModel):
    """身份画像增量更新结构。"""

    height_cm: float | None = None
    weight_kg: float | None = None
    preferences: list[str] = Field(default_factory=list)
    conditions: list[str] = Field(default_factory=list)
    notes: str | None = None


class ProactiveAdviceDecision(BaseModel):
    """是否追加建议的决策结果。"""

    score: float = Field(default=0.0, description="0-1 风险/提醒必要性分数")
    should_add: bool = Field(default=False, description="是否建议追加")
    advice: str | None = Field(default=None, description="建议文本；无需追加时可为空")
    reasons: list[str] = Field(default_factory=list, description="触发原因（可审计）")
