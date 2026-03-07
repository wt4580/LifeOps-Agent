from __future__ import annotations

"""lifeops.personal_memory

个人画像记忆模块：
1) 从用户输入中抽取结构化生活事件（饮食/活动/研究/消费等）。
2) 基于近期事件 + 待办，生成主动建议文案。

设计原则：
- 抽取失败不影响主聊天流程。
- 输出严格 JSON schema，便于稳定入库。
- 建议是“增益项”，不替代主回答。
"""

import json
import logging
from datetime import datetime

from pydantic import BaseModel, Field, ValidationError

from .llm_qwen import chat_completion


logger = logging.getLogger(__name__)


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


def _normalize_event_time_or_now(value: str | None, now_iso: str) -> str:
    """确保事件时间可用：无时间或非法时间时回退为 now_iso。"""

    if value:
        try:
            return datetime.fromisoformat(value).isoformat(timespec="seconds")
        except ValueError:
            pass
    return now_iso


def extract_personal_events(
    user_message: str,
    recent_dialogue: list[dict[str, str]],
    *,
    now_iso: str,
    session_id: str | None = None,
) -> PersonalEventExtraction | None:
    """从用户输入（可参考最近对话）提取结构化个人事件，并补齐事件时间。"""

    system_prompt = (
        "你是个人生活事件抽取器。请从用户消息中抽取可长期记忆的事实，并返回严格 JSON。\n"
        "只输出 JSON，不要解释。\n"
        "schema: {\"items\":[{\"category\":\"diet|activity|research|finance|schedule|health|other\","
        "\"title\":\"...\",\"event_time\":\"ISO或null\",\"amount\":数字或null,"
        "\"amount_unit\":\"...或null\",\"tags\":[\"...\"],\"notes\":\"...或null\"}]}\n"
        "规则：\n"
        "- 只抽取用户明确表达或高置信推断的事实；不确定就不抽取。\n"
        "- event_time 必须尽量填写；若只知道日期则用 YYYY-MM-DDT00:00:00。\n"
        "- 若用户未给具体时间，允许输出 null，服务端会回填当前时间。\n"
        "- 饮食类尽量标注 tags（如 油腻/高糖/高盐/蛋白质）。\n"
        "- 若本句无可记忆事件，返回 {\"items\":[]}。"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": json.dumps(
                {
                    "now_iso": now_iso,
                    "user_message": user_message,
                    "recent_dialogue": recent_dialogue[-6:],
                },
                ensure_ascii=False,
            ),
        },
    ]

    try:
        content = chat_completion(
            messages,
            temperature=0.0,
            runtime_context={"scenario": "personal_event_extraction", "session_id": session_id},
        )
        data = json.loads(content)
        extraction = PersonalEventExtraction.model_validate(data)
        fixed_items = [
            item.model_copy(update={"event_time": _normalize_event_time_or_now(item.event_time, now_iso)})
            for item in extraction.items
        ]
        return PersonalEventExtraction(items=fixed_items)
    except (json.JSONDecodeError, ValidationError) as exc:
        logger.warning("personal event extraction failed: %s", exc)
        return None


def extract_profile_facts(
    user_message: str,
    recent_dialogue: list[dict[str, str]],
    *,
    now_iso: str,
    session_id: str | None = None,
) -> ProfileFactPatch | None:
    """从用户消息中提取“稳定身份画像”信息（身高/体重/喜好/疾病等）。"""

    system_prompt = (
        "你是用户画像抽取器。请从输入中提取稳定且高价值的身份信息，返回严格 JSON。\n"
        "只输出 JSON，不要解释。\n"
        "schema: {\"height_cm\":number|null,\"weight_kg\":number|null,\"preferences\":[string],\"conditions\":[string],\"notes\":string|null}\n"
        "规则：\n"
        "- 只提取用户明确表达的信息；不确定则置空或不填。\n"
        "- 身高统一厘米，体重统一千克。\n"
        "- preferences 适合放饮食偏好、运动偏好、作息偏好。\n"
        "- conditions 适合放疾病/慢性病/禁忌。\n"
        "- 如果本句没有画像信息，返回空补丁：{\"height_cm\":null,\"weight_kg\":null,\"preferences\":[],\"conditions\":[],\"notes\":null}。"
    )

    payload = {
        "now_iso": now_iso,
        "user_message": user_message,
        "recent_dialogue": recent_dialogue[-4:],
    }

    try:
        content = chat_completion(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            temperature=0.0,
            runtime_context={"scenario": "profile_fact_extraction", "session_id": session_id},
        )
        data = json.loads(content)
        return ProfileFactPatch.model_validate(data)
    except (json.JSONDecodeError, ValidationError) as exc:
        logger.warning("profile fact extraction failed: %s", exc)
        return None


def parse_event_time(value: str | None) -> datetime | None:
    """把抽取结果中的时间字符串安全转换成 datetime。"""

    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def build_profile_context(profile: dict | None) -> str:
    """把画像字典压缩成可注入 graph/llm 的短文本上下文。"""

    if not profile:
        return ""

    parts: list[str] = []
    if profile.get("height_cm"):
        parts.append(f"身高{profile['height_cm']}cm")
    if profile.get("weight_kg"):
        parts.append(f"体重{profile['weight_kg']}kg")

    prefs = profile.get("preferences") or []
    conds = profile.get("conditions") or []
    if prefs:
        parts.append("喜好:" + "、".join(prefs[:6]))
    if conds:
        parts.append("健康情况:" + "、".join(conds[:6]))
    if profile.get("notes"):
        parts.append(f"备注:{profile['notes']}")

    return "；".join(parts)


def generate_proactive_advice(
    *,
    user_message: str,
    recent_events: list[dict],
    upcoming_todos: list[dict],
    now_iso: str,
    session_id: str | None = None,
) -> str | None:
    """基于近期事件 + 日程生成主动建议。

    返回：
    - None 或空字符串：不追加建议。
    - 非空文本：作为“补充建议”拼接到主回答后。
    """

    if not recent_events and not upcoming_todos:
        return None

    system_prompt = (
        "你是个人生活管理建议助手。请基于提供的结构化事件和待办，给出1-2句可执行建议。\n"
        "要求：\n"
        "- 只依据输入，不编造。\n"
        "- 优先提示：饮食结构、时间冲突、出行前后注意事项。\n"
        "- 若没有明确建议，输出 NONE。\n"
        "- 输出纯文本，不要 Markdown。"
    )

    payload = {
        "now": now_iso,
        "user_message": user_message,
        "recent_events": recent_events,
        "upcoming_todos": upcoming_todos,
    }

    try:
        text = chat_completion(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            temperature=0.2,
            runtime_context={"scenario": "generate_proactive_advice", "session_id": session_id},
        ).strip()
        if not text or text.upper() == "NONE":
            return None
        return text
    except Exception as exc:
        logger.warning("proactive advice generation failed: %s", exc)
        return None


def generate_profile_pre_advice(
    *,
    user_message: str,
    profile_context: str,
    recent_events: list[dict] | None = None,
    now_iso: str,
    session_id: str | None = None,
) -> str | None:
    """基于身份画像与近期个人事件生成前置提示，注入状态机。"""

    if not profile_context and not (recent_events or []):
        return None

    system_prompt = (
        "你是生活助理的前置分析器。请基于用户画像与近期个人事件，提炼与当前问题最相关的先验知识。\n"
        "要求：\n"
        "- 输出 1-2 句中文短提示，供后续主状态机参考；\n"
        "- 结合当前时间与事件时间判断是否临近冲突；\n"
        "- 只用输入信息，不编造；\n"
        "- 若关联弱，输出 NONE；\n"
        "- 输出纯文本，不要 Markdown。"
    )

    payload = {
        "now": now_iso,
        "user_message": user_message,
        "profile_context": profile_context,
        "recent_events": recent_events or [],
    }

    try:
        text = chat_completion(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            temperature=0.2,
            runtime_context={"scenario": "profile_pre_advice", "session_id": session_id},
        ).strip()
        if not text or text.upper() == "NONE":
            return None
        return text
    except Exception as exc:
        logger.warning("profile pre-advice generation failed: %s", exc)
        return None


def _fallback_proactive_decision(
    *,
    user_message: str,
    recent_events: list[dict],
    upcoming_todos: list[dict],
    threshold: float,
) -> ProactiveAdviceDecision:
    """当 LLM 决策失败时，使用规则兜底，保证主流程稳定。"""

    score = 0.0
    reasons: list[str] = []
    msg = user_message.strip()

    if any(k in msg for k in ["出门", "外出", "开会", "赶路", "饿", "吃", "运动"]):
        score += 0.25
        reasons.append("用户输入包含行动/饮食场景")

    if upcoming_todos:
        score += min(0.35, 0.1 * len(upcoming_todos) + 0.15)
        reasons.append("未来事项存在提醒价值")

    oil_tags = 0
    for item in recent_events:
        tags = item.get("tags") or []
        if any(t in tags for t in ["油腻", "高糖", "高盐"]):
            oil_tags += 1
    if oil_tags > 0:
        score += min(0.30, 0.08 * oil_tags)
        reasons.append("近期饮食记录提示需健康提醒")

    score = max(0.0, min(1.0, score))
    should_add = score >= threshold
    advice = None
    if should_add:
        advice = "你接下来有安排，且近期饮食偏油腻；出门时尽量选择清淡食物并预留行程时间。"

    return ProactiveAdviceDecision(score=score, should_add=should_add, advice=advice, reasons=reasons)


def decide_proactive_advice(
    *,
    user_message: str,
    assistant_answer: str,
    recent_events: list[dict],
    upcoming_todos: list[dict],
    profile_context: str,
    now_iso: str,
    threshold: float,
    session_id: str | None = None,
) -> ProactiveAdviceDecision:
    """综合用户画像/事件/待办，输出“是否追加建议”的阈值决策。"""

    if not recent_events and not upcoming_todos and not profile_context:
        return ProactiveAdviceDecision(score=0.0, should_add=False, advice=None, reasons=["无可用上下文"])

    system_prompt = (
        "你是生活助手的提醒决策器。请根据输入判断是否需要在主回复后追加`补充建议`。\n"
        "输出必须是严格 JSON，不要解释，不要 Markdown。\n"
        "schema: {\"score\":0-1数字,\"should_add\":true/false,\"advice\":\"字符串或null\",\"reasons\":[\"...\"]}\n"
        "规则：\n"
        "- score 表示提醒必要性；越高越应追加建议。\n"
        "- 当用户仅致谢/寒暄/结束语且无紧迫风险时，should_add=false。\n"
        "- should_add=true 时，advice 给 1-2 句可执行建议；否则 advice=null。\n"
        "- 只基于输入，不编造。"
    )

    payload = {
        "now": now_iso,
        "threshold": threshold,
        "user_message": user_message,
        "assistant_answer": assistant_answer,
        "profile_context": profile_context,
        "recent_events": recent_events,
        "upcoming_todos": upcoming_todos,
    }

    try:
        text = chat_completion(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            temperature=0.0,
            runtime_context={"scenario": "proactive_advice_decision", "session_id": session_id},
        ).strip()
        data = json.loads(text)
        decision = ProactiveAdviceDecision.model_validate(data)
        # 阈值最终由服务端硬控制，防止模型漂移导致越权追加。
        decision.should_add = bool(decision.advice) and decision.score >= threshold
        if not decision.should_add:
            decision.advice = None
        return decision
    except Exception as exc:
        logger.warning("proactive advice decision failed, fallback to rules: %s", exc)
        return _fallback_proactive_decision(
            user_message=user_message,
            recent_events=recent_events,
            upcoming_todos=upcoming_todos,
            threshold=threshold,
        )
