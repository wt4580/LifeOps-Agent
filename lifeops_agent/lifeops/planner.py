from __future__ import annotations

import json
import logging
from uuid import uuid4
from datetime import datetime, timedelta
from pydantic import BaseModel, Field, ValidationError

from .llm_qwen import chat_completion


logger = logging.getLogger(__name__)

# Minimal keyword intent check to avoid extra LLM calls.
_INTENT_KEYWORDS = [
    "todo",
    "task",
    "remind",
    "schedule",
    "plan",
    "deadline",
    "meeting",
    "记住",
    "提醒",
    "待办",
    "安排",
    "计划",
    "会议",
    "开会",
    "截止",
    "下周",
    "周一",
    "周二",
    "周三",
    "周四",
    "周五",
    "周六",
    "周日",
    "周天",
    "明天",
    "今天",
]

_AFFIRM_KEYWORDS = [
    "需要",
    "要",
    "可以",
    "好的",
    "好",
    "行",
    "没问题",
    "可以的",
]


class PlanItem(BaseModel):
    title: str
    due_at: str | None = None


class PlanProposal(BaseModel):
    items: list[PlanItem] = Field(default_factory=list)


def detect_plan_intent(user_input: str) -> bool:
    text = user_input.lower()
    return any(keyword in text for keyword in _INTENT_KEYWORDS)


def detect_affirmation(user_input: str) -> bool:
    text = user_input.strip().lower()
    return any(keyword == text or keyword in text for keyword in _AFFIRM_KEYWORDS)


_WEEKDAY_MAP = {
    "周一": 0,
    "周二": 1,
    "周三": 2,
    "周四": 3,
    "周五": 4,
    "周六": 5,
    "周日": 6,
    "周天": 6,
}


def _resolve_weekday_date(text: str) -> str | None:
    for key, target in _WEEKDAY_MAP.items():
        if key in text:
            today = datetime.now().date()
            base_weekday = today.weekday()
            days_ahead = (target - base_weekday + 7) % 7
            if "下周" in text:
                days_ahead = days_ahead + 7 if days_ahead > 0 else 7
            target_date = today + timedelta(days=days_ahead)
            return target_date.isoformat()
    return None


def _apply_fixed_date(fixed_date: str, due_at: str | None) -> str:
    if not due_at:
        return fixed_date
    if "T" in due_at:
        _, time_part = due_at.split("T", 1)
        return f"{fixed_date}T{time_part}"
    return fixed_date


def propose_plan(user_input: str) -> tuple[str, PlanProposal]:
    today = datetime.now().date().isoformat()
    fixed_date = _resolve_weekday_date(user_input)
    system_prompt = (
        "You turn the user intent into a concise plan JSON. "
        "Return ONLY valid JSON: {items:[{title,due_at}]} . "
        "Use ISO8601 for due_at (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS). "
        "If no clear due date, set due_at to null. "
        f"Today is {today}. "
        + (f"If the user mentions a weekday, use date {fixed_date}. " if fixed_date else "")
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input},
    ]
    content = chat_completion(messages, temperature=0.2)
    try:
        data = json.loads(content)
        proposal = PlanProposal.model_validate(data)
    except (json.JSONDecodeError, ValidationError) as exc:
        logger.warning("Plan proposal invalid: %s", exc)
        proposal = PlanProposal(items=[])
    if fixed_date:
        proposal = PlanProposal(items=[
            PlanItem(title=item.title, due_at=_apply_fixed_date(fixed_date, item.due_at))
            for item in proposal.items
        ])
    return str(uuid4()), proposal
