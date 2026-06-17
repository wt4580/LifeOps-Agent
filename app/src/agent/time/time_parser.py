from __future__ import annotations

import json
from datetime import datetime, timedelta

from pydantic import ValidationError

from ...common.config.log_config import logger
from ...common.config.llm_config import chat_completion
from ...domain.dto.time_dto import DateHint
from ...service.chinese_holidays import get_holidays_in_range


def build_date_context(*, remaining_only: bool = False) -> dict:
    now = datetime.now()
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    week_map = {0: "周一", 1: "周二", 2: "周三", 3: "周四", 4: "周五", 5: "周六", 6: "周日"}
    weekday_cn = week_map[now.weekday()]

    year_start = today.replace(month=1, day=1)
    year_end = today.replace(month=12, day=31, hour=23, minute=59, second=59)
    holidays = get_holidays_in_range(year_start, year_end)
    holiday_items = [
        {"节日": h["summary"], "日期": h["start"][:10]}
        for h in holidays
        if not remaining_only or h["start"][:10] >= today.strftime("%Y-%m-%d")
    ]

    return {
        "今天日期": today.strftime("%Y-%m-%d"),
        "今天星期": weekday_cn,
        ("今年剩余节日" if remaining_only else "今年节日"): holiday_items,
    }


def normalize_date_hint(user_text: str) -> str | None:
    ctx = build_date_context()

    system = (
        "你是一个时间解析器。请把用户文本中的时间表达解析为一个具体的公历日期。\n\n"
        "参考信息（据此计算日期）：\n"
        f"{json.dumps(ctx, ensure_ascii=False, indent=2)}\n\n"
        "输出必须是严格 JSON：{\"date\": \"YYYY-MM-DD\" 或 null, \"confidence\": 0到1之间的小数}\n"
        "规则：\n"
        "- 用户可能说：明天、后天、大后天、下周三、这周五、月底、两周后、端午节、春节、中秋、下个月5号等\n"
        "- 你可以参考上面的节日列表来识别传统节日名称对应的日期\n"
        "- 只在用户明确表达了时间/日期/节日时给出 date\n"
        "- 如果无法确定，date=null，confidence 低一些\n"
        "- 只输出 JSON，不要解释文字"
    )

    raw = chat_completion(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps({"text": user_text}, ensure_ascii=False)},
        ],
        temperature=0.0,
    )

    try:
        data = json.loads(raw)
        hint = DateHint.model_validate(data)
    except (json.JSONDecodeError, ValidationError) as exc:
        logger.warning("normalize_date_hint parse failed: %s; raw=%r", exc, raw)
        return None

    if not hint.date:
        return None

    try:
        datetime.fromisoformat(hint.date)
    except ValueError:
        logger.warning("normalize_date_hint invalid date: %r", hint.date)
        return None

    if hint.confidence < 0.4:
        return None

    return hint.date
