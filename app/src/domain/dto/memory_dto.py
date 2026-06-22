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

    # 新增字段
    age: int | None = None
    gender: str | None = None
    occupation: str | None = None
    city: str | None = None
    diet: list[str] = Field(default_factory=list, description="饮食偏好，如 素食/低糖/低碳水")
    allergies: list[str] = Field(default_factory=list, description="过敏/忌口，如 海鲜过敏/不吃辣")
    sleep_schedule: str | None = Field(default=None, description="作息规律，如 早睡早起/夜猫子/7点起12点睡")
    exercise_habits: str | None = Field(default=None, description="运动习惯，如 每周跑步3次")
    work_hours: str | None = Field(default=None, description="工作时间段，如 9-18")
    family_status: str | None = Field(default=None, description="家庭状况，如 独居/已婚有娃/和父母住")
    goals: list[str] = Field(default_factory=list, description="短期目标/优先事项，如 减肥/攒钱/学英语")


class SubGoalItem(BaseModel):
    """目标拆解的子步骤。"""

    title: str = Field(..., description="子目标标题")
    target_date: str | None = Field(default=None, description="预计完成时间，ISO 日期或人性化描述")


class MilestoneItem(BaseModel):
    """里程碑节点。"""

    title: str = Field(..., description="里程碑描述")
    target_date: str | None = Field(default=None, description="预计达成时间")
    is_completed: bool = False


class GoalPatch(BaseModel):
    """从对话中提取的目标信息。"""

    title: str = Field(..., description="目标标题，如'今年减肥20斤'")
    category: str | None = Field(default=None, description="health|career|finance|learning|lifestyle|other")
    target_date: str | None = Field(default=None, description="目标截止日期（ISO）")
    sub_goals: list[SubGoalItem] = Field(default_factory=list, description="拆解出的子步骤")
    milestones: list[MilestoneItem] = Field(default_factory=list, description="里程碑节点")
    notes: str | None = Field(default=None, description="补充说明")


class GoalExtraction(BaseModel):
    """目标抽取结果容器。"""

    items: list[GoalPatch] = Field(default_factory=list)


class ProactiveAdviceDecision(BaseModel):
    """是否追加建议的决策结果。"""

    score: float = Field(default=0.0, description="0-1 风险/提醒必要性分数")
    should_add: bool = Field(default=False, description="是否建议追加")
    advice: str | None = Field(default=None, description="建议文本；无需追加时可为空")
    reasons: list[str] = Field(default_factory=list, description="触发原因（可审计）")
