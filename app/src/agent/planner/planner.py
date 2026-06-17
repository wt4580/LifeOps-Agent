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
from datetime import datetime, timedelta, date
from pydantic import ValidationError

from ...common.config.log_config import logger
from ...common.config.llm_config import chat_completion
from ..time.time_parser import normalize_date_hint
from ...domain.dto.planner_dto import PlanItem, PlanProposal

# 轻量意图关键词：用于便宜且可控的“候选检测”，避免每次都调用 LLM。
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

# 用户“确认”常见表达。
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

# 用户“拒绝/取消”常见表达（优先级高于确认）。
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
    """轻量检测：判断输入是否“像”待办描述。"""

    text = user_input.lower()
    return any(keyword in text for keyword in _INTENT_KEYWORDS)


def detect_affirmation(user_input: str) -> bool:
    """确认识别：先排除否定，再匹配确认表达。

    更严格的确认识别：先排除否定，再做短语确认。
    """

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
    """将解析出的日期“强制合并”到 due_at。

    - 若模型没给时间：直接返回 YYYY-MM-DD。
    - 若模型给了时间：保留时间部分，只替换日期。
    """

    if not due_at:
        return fixed_date
    if "T" in due_at:
        _, time_part = due_at.split("T", 1)
        return f"{fixed_date}T{time_part}"
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


def _fallback_todo_from_complaint(text: str, fixed_date: str | None) -> PlanItem | None:
    cleaned = text.strip().rstrip("。，！？,.!?")
    if not cleaned:
        return None
    patterns = [
        (r"(?:导员|老师|导师|领导|经理|主管)[^，。]*没[有]?回", "跟进{target}回复"),
        (r"[^，。]*还没[有]?(?:审批|审核|批准|通过)", "跟进{target}审批"),
        (r"[^，。]*还没[有]?(?:回复|回|处理)", "跟进{target}处理"),
        (r"[^，。]*?没[有]?(?:到货|发货|来)", "跟进{target}到货"),
        (r"[^，。]*?(?:未完成|还没做|没做)", "完成{target}"),
        (r"[^，。]*?(?:待办|任务|工作)[^，。]*", "处理{target}"),
    ]
    for pat, tmpl in patterns:
        m = re.search(pat, cleaned)
        if m:
            noun = m.group(0)[:20]
            title = tmpl.format(target=noun)
            title = _normalize_title(title)
            return PlanItem(title=title, due_at=fixed_date)
    # generic fallback
    title = _normalize_title(cleaned[:20])
    return PlanItem(title=title, due_at=fixed_date)


def propose_plan(user_input: str) -> tuple[str, PlanProposal]:
    """根据用户输入生成结构化待办草案。

    返回：
    - proposal_id: 草案 ID（前端确认时会用到）
    - proposal: 结构化草案对象

    容错策略：
    - 如果 LLM 返回非法 JSON，返回空草案而不是抛异常中断主流程。
    - 若能确定日期（规则或 LLM），会把日期兜底写入 due_at。
    """

    today = datetime.now().date().isoformat()

    # 让大模型解析“相对时间/自然语言日期”（大后天/下下周/这周三...）。
    # 如果解析失败，再退回到我们自己的周几解析（兜底）。
    fixed_date = normalize_date_hint(user_input) or _resolve_weekday_date(user_input)

    system_prompt = (
        "你是待办规划器。请把用户输入改写成可执行待办清单，并返回严格 JSON。"
        "只输出 JSON，不要解释。"
        "schema: {\"items\":[{\"title\":\"...\",\"due_at\":\"ISO或null\"}]}."
        "规则："
        "1) title 必须是动作导向短句（5-22字），不要复述原话，不要包含‘待办/提醒’字样；"
        "2) 如果用户是在表达困扰/抱怨/未完成的事（如'导员没回报销'、'快递没到'、'作业没交'），推导出对应的跟进动作作为 todo（如'跟进导员报销审批'）；"
        "3) 如果内容是同一件事，只输出一条；"
        "4) due_at 用 ISO8601（YYYY-MM-DD 或 YYYY-MM-DDTHH:MM:SS），无法确定则为 null；"
        f"5) 今天是 {today}。"
        + (f"用户时间表达已解析为 {fixed_date}，请优先使用该日期。" if fixed_date else "")
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
        logger.warning("Plan proposal invalid: %s; raw=%r", exc, content[:200])
        proposal = PlanProposal(items=[])

    if not proposal.items:
        fallback = _fallback_todo_from_complaint(user_input, fixed_date)
        if fallback:
            proposal = PlanProposal(items=[fallback])

    # 强制把解析出的日期落到 due_at：
    # - 模型可能只返回 title，但 due_at=null
    # - 如果用户说了“明天/周三”，我们就用 fixed_date 作为兜底日期
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
