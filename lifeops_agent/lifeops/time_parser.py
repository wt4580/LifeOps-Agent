from __future__ import annotations

"""lifeops.time_parser

时间解析：优先让大模型理解自然语言时间，但对中文相对日（今天/明天/后天/大后天）提供确定性规则兜底，避免模型偶发偏移。

设计原则：
- 你希望“时间由大模型处理”，但生产系统里仍需要一些 deterministic guardrail。
- 对“今天/明天/后天/大后天”这种语义非常确定的表达，我们可以直接按规则算，
  既省 token，又避免模型把“大后天”误当成“后天”或算错一天。
- 对更复杂的表达（下下周三、月底、两周后等）再交给 LLM。
"""

import json
import logging
from datetime import datetime, timedelta

from pydantic import BaseModel, ValidationError

from .llm_qwen import chat_completion

logger = logging.getLogger(__name__)


class DateHint(BaseModel):
    date: str | None = None  # YYYY-MM-DD
    confidence: float = 0.5


def _rule_based_relative_day(user_text: str) -> str | None:
    """对中文相对日做确定性解析。

    注意优先级：
    - 必须先判断“大后天”，否则包含“后天”子串会误匹配。
    """

    today = datetime.now().date()

    if "大后天" in user_text:
        return (today + timedelta(days=3)).isoformat()
    if "后天" in user_text:
        return (today + timedelta(days=2)).isoformat()
    if "明天" in user_text:
        return (today + timedelta(days=1)).isoformat()
    if "今天" in user_text:
        return today.isoformat()
    return None


def normalize_date_hint(user_text: str) -> str | None:
    """用 LLM 把相对时间/自然语言日期解析成 YYYY-MM-DD。

    - 若命中确定性的相对日（今天/明天/后天/大后天），直接返回规则结果。
    - 否则让 LLM 处理更复杂的表达。
    """

    ruled = _rule_based_relative_day(user_text)
    if ruled:
        return ruled

    today = datetime.now().date().isoformat()

    system = (
        "你是一个时间解析器。请把用户文本中的日期信息解析为一个具体的公历日期（仅日期，不要时间）。\n"
        "输出必须是严格 JSON：{\"date\": \"YYYY-MM-DD\" 或 null, \"confidence\": 0到1之间的小数}。\n"
        f"今天是 {today}。\n"
        "规则：\n"
        "- 只在用户明确表达了日期/相对日期/周几时给出 date。\n"
        "- 支持：下周X/周X/这周X/本周X/下下周X、两周后、月底 等常见说法。\n"
        "- 如果无法确定，date=null，confidence 低一些。\n"
        "- 只输出 JSON，不要输出解释文字。"
    )

    # 把 today 明确作为单独字段传给模型，减少模型自行假设日期导致的漂移
    raw = chat_completion(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps({"text": user_text, "today": today}, ensure_ascii=False)},
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

