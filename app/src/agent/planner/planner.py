from __future__ import annotations

"""lifeops.planner

这个模块负责"把自然语言转成待办草案（PlanProposal）"。

它的角色在 HITL 流程里是：
1) 用户输入可能是待办意图。
2) 路由层先询问"是否生成草案"。
3) 用户确认后，planner 负责生成结构化 JSON 草案。
4) 真正写库由 `/api/plan/confirm` 完成（planner 不直接写数据库）。
"""

import json
import re
from uuid import uuid4
from datetime import datetime
from pydantic import ValidationError

from ...common.config.log_config import logger
from ...common.config.llm_config import chat_completion
from ..time.time_parser import normalize_date_hint
from ...domain.dto.planner_dto import PlanItem, PlanProposal

# 轻量意图关键词：用于便宜且可控的"候选检测"，避免每次都调用 LLM。
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

# 用户"确认"常见表达。
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

# 用户"拒绝/取消"常见表达（优先级高于确认）。
_REJECT_PATTERNS = [
    "不好",
    "不需要",
    "不用",
    "不要",
    "不行",
    "先不用",
    "先不要",
    "算了",
    "取消",
]

# 精准确认词集合：用于严格匹配，降低误判。
_AFFIRM_EXACT = {"需要", "要", "可以", "好的", "好", "行", "可以的", "没问题", "确认"}


def detect_plan_intent(user_input: str) -> bool:
    """轻量检测：判断输入是否"像"待办描述。"""
    text = user_input.lower()
    return any(keyword in text for keyword in _INTENT_KEYWORDS)


def detect_affirmation(user_input: str) -> bool:
    """确认识别：先排除否定，再匹配确认表达。"""
    text = user_input.strip().lower()
    if not text:
        return False
    if any(p in text for p in _REJECT_PATTERNS):
        return False
    if text in _AFFIRM_EXACT:
        return True
    return text.startswith(("需要", "可以", "好的", "行", "确认"))


def detect_rejection(user_input: str) -> bool:
    """拒绝识别：用于 HITL 中止分支。"""
    text = user_input.strip().lower()
    if not text:
        return False
    return any(p in text for p in _REJECT_PATTERNS)


def _apply_fixed_date(fixed_date: str, due_at: str | None) -> str:
    """将解析出的日期"强制合并"到 due_at。

    - 若模型没给时间：直接返回 YYYY-MM-DD。
    - 若模型给了时间：保留时间部分，只替换日期。
    """
    if not due_at:
        return fixed_date
    if "T" in due_at:
        _, time_part = due_at.split("T", 1)
        return "{year}T{time}".format(year=fixed_date, time=time_part)
    return fixed_date


def _normalize_title(title: str) -> str:
    cleaned = " ".join((title or "").strip().split())
    cleaned = cleaned.replace("待办", "").replace("提醒", "").strip("：:，,。 ")
    cleaned = re.sub(r"(今天|明天|后天|大后天|下周|本周|这周|周[一二三四五六日天])", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip("：:，,。 ")
    return cleaned or "待办事项"


def _dedupe_plan_items(items: list[PlanItem]) -> list[PlanItem]:
    seen: set[tuple[str, str | None]] = set()
    deduped: list[PlanItem] = []
    for item in items:
        normalized_title = _normalize_title(item.title)
        due = item.due_at or None
        key = (normalized_title.lower(), due)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(PlanItem(title=normalized_title, due_at=due))
    return deduped


def propose_plan(user_input: str) -> tuple[str, PlanProposal]:
    """根据用户输入生成结构化待办草案。

    返回：
    - proposal_id: 草案 ID（前端确认时会用到）
    - proposal: 结构化草案对象

    错误策略：
    - 如果 LLM 返回非法 JSON，抛出异常，由上层重试。
    """
    today = datetime.now().date().isoformat()
    fixed_date = normalize_date_hint(user_input)

    system_prompt = (
        "你是待办规划器。把用户输入改写成可执行待办清单。"
        "只输出 JSON，不要解释。"
        'schema: {"items":[{"title":"...","due_at":"ISO或null"}]}.\n\n'
        "## 格式要求\n"
        "1) title 是动作导向短句（5-22字），不要复述原话，不含'待办/提醒'字样；\n"
        "2) 用户可能在一条消息里提到多件事，每件事输出一条 items；\n"
        "3) due_at 用 ISO8601（YYYY-MM-DD 或 YYYY-MM-DDTHH:MM:SS），无法确定则为 null；\n"
        "4) 今天是 " + today + "."
        + (" 用户时间表达已解析为 " + fixed_date + "，请优先使用该日期。" if fixed_date else "")
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
        logger.error("Plan proposal LLM returned invalid JSON: %s; raw=%r", exc, content[:200])
        raise

    if fixed_date:
        proposal = PlanProposal(
            items=[
                PlanItem(
                    title=_normalize_title(item.title),
                    due_at=_apply_fixed_date(fixed_date, item.due_at),
                )
                for item in proposal.items
            ]
        )
    else:
        proposal = PlanProposal(
            items=[
                PlanItem(
                    title=_normalize_title(item.title),
                    due_at=item.due_at,
                )
                for item in proposal.items
            ]
        )

    proposal = PlanProposal(items=_dedupe_plan_items(proposal.items))
    return str(uuid4()), proposal
