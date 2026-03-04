from __future__ import annotations

import json
import logging
from uuid import uuid4
from datetime import datetime, timedelta, date
from pydantic import BaseModel, Field, ValidationError

from .llm_qwen import chat_completion
from .time_parser import normalize_date_hint


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


def _next_or_same_weekday(current: date, target_weekday: int) -> date:
    """返回从 current 起（含当天）到下一个 target_weekday 的日期。"""

    delta = (target_weekday - current.weekday()) % 7
    return current + timedelta(days=delta)


def _next_week_weekday(current: date, target_weekday: int) -> date:
    """返回“下周X”的日期：下一个自然周的 target_weekday。

    规则：
    - 先跳到“下周的周一”作为基点，然后找该周的 target_weekday。
    - 这样可以避免今天是周三时，“下周三”被算成本周三或 +1 混乱。
    """

    # 本周周一
    this_monday = current - timedelta(days=current.weekday())
    next_monday = this_monday + timedelta(days=7)
    return next_monday + timedelta(days=target_weekday)


def _resolve_weekday_date(text: str) -> str | None:
    """把“周三/下周三”解析成一个确定日期（YYYY-MM-DD）。"""

    today = datetime.now().date()

    for key, target in _WEEKDAY_MAP.items():
        if key not in text:
            continue

        # “下周三”等：下一个自然周的周三
        if "下周" in text:
            return _next_week_weekday(today, target).isoformat()

        # “周三”等：从今天起（含今天）找最近的周三
        return _next_or_same_weekday(today, target).isoformat()

    return None


def _resolve_relative_day_date(text: str) -> str | None:
    """解析“今天/明天/后天”为确定日期（YYYY-MM-DD）。"""

    today = datetime.now().date()
    if "今天" in text:
        return today.isoformat()
    if "明天" in text:
        return (today + timedelta(days=1)).isoformat()
    if "后天" in text:
        return (today + timedelta(days=2)).isoformat()
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

    # 让大模型解析“相对时间/自然语言日期”（大后天/下下周/这周三...）。
    # 如果解析失败，再退回到我们自己的周几解析（兜底）。
    fixed_date = normalize_date_hint(user_input) or _resolve_weekday_date(user_input)

    system_prompt = (
        "You turn the user intent into a concise plan JSON. "
        "Return ONLY valid JSON: {items:[{title,due_at}]} . "
        "Use ISO8601 for due_at (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS). "
        "If no clear due date, set due_at to null. "
        f"Today is {today}. "
        + (f"If the user mentions a date, use date {fixed_date}. " if fixed_date else "")
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

    # 强制把解析出的日期落到 due_at：
    # - 模型可能只返回 title，但 due_at=null
    # - 如果用户说了“明天/周三”，我们就用 fixed_date 作为兜底日期
    if fixed_date:
        proposal = PlanProposal(
            items=[
                PlanItem(
                    title=item.title,
                    due_at=_apply_fixed_date(fixed_date, item.due_at),
                )
                for item in proposal.items
            ]
        )

    return str(uuid4()), proposal
