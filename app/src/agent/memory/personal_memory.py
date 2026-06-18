from __future__ import annotations

"""lifeops.personal_memory

个人画像记忆模块：
1) 从用户输入中抽取结构化生活事件（饮食/活动/研究/消费等）。
2) 基于近期事件 + 待办，生成主动建议文案。

设计原则：
- 抽取失败不影响主聊天流程。
- 输出严格 JSON schema，便于稳定入库。
- 建议是"增益项"，不替代主回答。
"""

import json
from datetime import datetime

from pydantic import ValidationError

from ...common.config.log_config import logger
from ...common.config.llm_config import chat_completion
from ...domain.dto.memory_dto import (
    PersonalEventItem,
    PersonalEventExtraction,
    ProfileFactPatch,
    ProactiveAdviceDecision,
)


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
        "你是个人生活事件抽取器。请从用户消息中抽取已发生的事实事件，返回严格 JSON。\n"
        "只输出 JSON，不要解释。\n"
        "schema: {\"items\":[{\"category\":\"diet|activity|research|finance|schedule|health|other\","
        "\"title\":\"...\",\"event_time\":\"ISO或null\",\"amount\":数字或null,"
        "\"amount_unit\":\"...或null\",\"tags\":[\"...\"],\"notes\":\"...或null\"}]}\n"
        "规则：\n"
        "- 只抽取已发生的事件（过去时/完成时），不要抽取未来的计划或承诺。\n"
        "- 明确的否定事实也不必抽取（如'没吃饭'不是事件）。\n"
        "- event_time 必须尽量填写；若只知道日期则用 YYYY-MM-DDT00:00:00。\n"
        "- 若用户未给具体时间，允许输出 null，服务端会回填当前时间。\n"
        "- 饮食类尽量标注 tags（如 油腻/高糖/高盐/蛋白质）。\n"
        "- 未来计划、待办、提醒等请留空，不要在 life event 中记录。\n"
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
        logger.warning("personal event extraction failed session=%s msg=%s err=%s", session_id, user_message[:40], exc)
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
        "- preferences 包含喜好和反感（如'喜欢跑步'、'不吃辣'、'不爱运动'、'讨厌油腻'）。\n"
        "- conditions 适合放疾病/慢性病/禁忌。\n"
        "- 用户明确表达否定（不喜欢/讨厌/不吃等）一定要提取，不要忽略。\n"
        "- 如果本句没有画像信息，返回空补丁：{\"height_cm\":null,\"weight_kg\":null,\"preferences\":[],\"conditions\":[],\"notes\":null}。\n"
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
        logger.warning("profile fact extraction failed session=%s msg=%s err=%s", session_id, user_message[:40], exc)
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

    prefs = list(profile.get("preferences") or [])
    conds = profile.get("conditions") or []

    # 特殊偏好条目：单独挑出来展示，避免混入"喜好"列表。
    default_city = ""
    normal_prefs: list[str] = []
    for p in prefs:
        if not isinstance(p, str):
            normal_prefs.append(str(p))
            continue
        if p.startswith("默认城市:") or p.startswith("所在城市:"):
            default_city = p.split(":", 1)[-1].strip()
        else:
            normal_prefs.append(p)

    if normal_prefs:
        parts.append("喜好:" + "、".join(normal_prefs[:6]))
    if conds:
        parts.append("健康情况:" + "、".join(conds[:6]))
    if default_city:
        parts.append(f"所在城市:{default_city}")
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
        logger.warning("proactive advice generation failed session=%s err=%s", session_id, exc)
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
        logger.warning("profile pre-advice generation failed session=%s err=%s", session_id, exc)
        return None


def _fallback_proactive_decision(
    *,
    threshold: float,
) -> ProactiveAdviceDecision:
    return ProactiveAdviceDecision(score=0.0, should_add=False, advice=None, reasons=["fallback: 无有效建议"])


def decide_proactive_advice(
    *,
    user_message: str,
    assistant_answer: str,
    recent_events: list[dict],
    upcoming_todos: list[dict],
    pending_reminders: list[dict] | None = None,
    profile_context: str,
    now_iso: str,
    threshold: float,
    session_id: str | None = None,
) -> ProactiveAdviceDecision:
    if not recent_events and not upcoming_todos and not profile_context and not pending_reminders:
        return ProactiveAdviceDecision(score=0.0, should_add=False, advice=None, reasons=["无可用上下文"])

    system_prompt = (
        "你是一个严谨的提醒筛选器。判断是否需要追加建议，输出严格 JSON。\n"
        'schema: {"score":0-1,"should_add":bool,"advice":"str或null","reasons":["str"]}\n'
        "核心原则：默认 should_add=false，除非：\n"
        "1. 事件/待办之间存在真实关联（时间冲突、因果关系、健康风险）\n"
        "2. 能给出明确、可执行的操作建议\n"
        "3. 不是把两个无关事件强行关联\n\n"
        "反面案例（should_add=false）：\n"
        "- 明天预订肯德基 + 后天端午节 -> 无关，不应建议\n"
        "- 明天去超市 + 下周有会议 -> 无关\n\n"
        "正面案例（should_add=true）：\n"
        "- 明天下午3点开会 + 明天下午2点体检 -> 时间冲突，建议调整\n"
        "- 明天跑10公里 + 膝盖不舒服 -> 健康风险，建议减量\n\n"
        "输出要求：\n"
        "- advice 一句话以内，只说直接相关的操作\n"
        "- 没有真实关联时 should_add=false, advice=null"
    )

    payload = {
        "now": now_iso,
        "user_message": user_message,
        "assistant_answer": assistant_answer,
        "profile_context": profile_context,
        "recent_events": recent_events,
        "upcoming_todos": upcoming_todos,
        "pending_reminders": pending_reminders or [],
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
        decision.should_add = bool(decision.advice) and decision.score >= threshold
        if not decision.should_add:
            decision.advice = None
        return decision
    except Exception as exc:
        logger.warning("proactive advice decision failed session=%s threshold=%s payload=%s err=%s", session_id, threshold, json.dumps(payload, ensure_ascii=False)[:200], exc)
        return _fallback_proactive_decision(threshold=threshold)


