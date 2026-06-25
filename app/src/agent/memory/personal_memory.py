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
from typing import Any, Callable

from pydantic import ValidationError

from ...common.config.log_config import logger
from ...common.config.llm_config import chat_completion
from ...domain.dto.memory_dto import (
    PersonalEventItem,
    PersonalEventExtraction,
    ProfileFactPatch,
    ProactiveAdviceDecision,
    GoalExtraction,
)


_MAX_EXTRACT_RETRIES = 1


def _llm_extract(
    *,
    system_prompt: str,
    payload: dict,
    parse: Callable[[str], Any],
    diagnose_context: str,
    session_id: str | None = None,
) -> Any | None:
    """带 LLM 自诊重试的提取函数。

    1. 调用 LLM 提取
    2. 文本为空或解析失败 → 让 LLM 诊断：数据不足 vs 临时错误
    3. 数据不足 → 返回 None
    4. 临时错误 → 重试
    5. 仍失败 → 返回 None
    """
    payload = dict(payload)
    for attempt in range(_MAX_EXTRACT_RETRIES + 1):
        text = chat_completion(
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
            temperature=0.0,
            runtime_context={"scenario": diagnose_context, "session_id": session_id},
        ).strip()

        if text and text.upper() != "NONE":
            try:
                return parse(text)
            except (json.JSONDecodeError, ValidationError) as exc:
                logger.warning(
                    "FAIL %s attempt=%d/%d err=%s",
                    diagnose_context, attempt + 1, _MAX_EXTRACT_RETRIES + 1, exc,
                )
                if attempt == _MAX_EXTRACT_RETRIES:
                    return None
                diag = _diagnose_failure(diagnose_context, str(exc), text, payload, session_id)
                if diag == "insufficient_data":
                    return None
                payload["_retry_hint"] = f"上次输出解析失败（{exc}），请重新生成有效 JSON。"
                continue

        if attempt == _MAX_EXTRACT_RETRIES:
            logger.warning("EMPTY %s attempts exhausted", diagnose_context)
            return None

        diag = _diagnose_failure(diagnose_context, "returned empty", text, payload, session_id)
        if diag == "insufficient_data":
            return None
        payload["_retry_hint"] = "上次未能提取有效信息，请重新提取。"

    return None


def _diagnose_failure(context: str, error: str, llm_output: str, payload: dict, session_id: str | None) -> str:
    """让 LLM 诊断提取失败原因。返回 'insufficient_data' 或 'transient_error'。"""
    system_prompt = (
        "判断 LLM 提取失败的原因，输出严格 JSON：{\"reason\":\"insufficient_data\"|\"transient_error\"}\n\n"
        "## insufficient_data —— 输入中确实没有需要提取的内容\n"
        "### 常见场景\n"
        "- personal_event_extraction: 用户只说'你好'/'今天天气好'，没有发生过任何事件\n"
        "- profile_fact_extraction: 用户没说个人身份信息，只是闲聊\n"
        "- extract_goal: 用户没有表达任何目标意愿\n"
        "- proactive_advice_decision: 事件/待办/目标之间确实没有真实关联\n\n"
        "### 特征\n"
        "- 输入不含目标信息的可能性远大于 LLM 输出失误的可能性\n"
        "- 即使重试 10 次，仍然会得到同样空/无效的结果\n"
        "- 输出为 NONE、空字符串、或正确的空 JSON 结构（如 {\"items\":[]}）\n\n"
        "## transient_error —— 输入包含目标信息，但 LLM 输出格式不对\n"
        "### 常见场景\n"
        "- 用户说了具体事件，但 LLM 输出了乱码或残缺 JSON\n"
        "- 输出包含多余解释文字而非纯 JSON\n"
        "- 输出内容与输入相关但嵌套了 markdown 代码块\n\n"
        "### 特征\n"
        "- 输入显然包含可提取信息\n"
        "- 重试有较高概率得到有效输出\n"
        "- 错误信息通常是 JSON 解析错误（Expecting value / Extra data 等）\n\n"
        "## 判断原则\n"
        "- 严格区分：不确定时优先选 insufficient_data（安全跳过）\n"
        "- 不要仅仅因为 LLM 输出了无效 JSON 就认定为 transient_error\n"
        "- 先看 payload 是否真的包含可提取的信息"
    )
    diag = chat_completion(
        [{"role": "system", "content": system_prompt},
         {"role": "user", "content": json.dumps({
             "task": context,
             "error": error,
             "llm_output": llm_output[:300],
             "payload": {k: v for k, v in payload.items() if not k.startswith("_")},
         }, ensure_ascii=False)}],
        temperature=0.0,
        runtime_context={"scenario": "diagnose_extract_failure", "session_id": session_id},
    ).strip()
    result: str
    try:
        result = json.loads(diag).get("reason", "insufficient_data")
        return result
    except (json.JSONDecodeError, ValidationError):
        diag_preview = (diag[:120] + "...") if len(diag) > 120 else diag
        logger.warning("diagnose unparseable ctx=%s raw=%s", context, diag_preview)
        return "insufficient_data"


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
        "\"amount_unit\":\"...或null\",\"tags\":[\"...\"],\"notes\":\"...或null\",\"importance\":\"low|high\"}]}\n"
        "## 规则\n"
        "- 只抽取已发生的事件（过去时/完成时），不要抽取未来的计划或承诺。\n"
        "- 明确的否定事实也不必抽取（如'没吃饭'不是事件）。\n"
        "- event_time 必须尽量填写；若只知道日期则用 YYYY-MM-DDT00:00:00。\n"
        "- 若用户未给具体时间，允许输出 null，服务端会回填当前时间。\n"
        "- 饮食类尽量标注 tags（如 油腻/高糖/高盐/蛋白质）。\n"
        "- 未来计划、待办、提醒等请留空，不要在 life event 中记录。\n"
        "- importance: 对用户生活有重大影响（手术/入职/结婚/大病/搬新居）标 high，日常事件标 low。\n"
        "- 若本句无可记忆事件，返回 {\"items\":[]}。"
    )

    payload = {
        "now_iso": now_iso,
        "user_message": user_message,
        "recent_dialogue": recent_dialogue[-6:],
    }

    def _parse(text: str) -> PersonalEventExtraction:
        data = json.loads(text)
        extraction = PersonalEventExtraction.model_validate(data)
        fixed_items = [
            item.model_copy(update={"event_time": _normalize_event_time_or_now(item.event_time, now_iso)})
            for item in extraction.items
        ]
        return PersonalEventExtraction(items=fixed_items)

    return _llm_extract(
        system_prompt=system_prompt,
        payload=payload,
        parse=_parse,
        diagnose_context="personal_event_extraction",
        session_id=session_id,
    )


def extract_profile_facts(
    user_message: str,
    recent_dialogue: list[dict[str, str]],
    *,
    now_iso: str,
    session_id: str | None = None,
) -> ProfileFactPatch | None:
    """从用户消息中提取"稳定身份画像"信息（身高/体重/喜好/疾病等）。"""

    system_prompt = (
        "你是用户画像抽取器。请从输入中提取稳定且高价值的身份信息，返回严格 JSON。\n"
        "只输出 JSON，不要解释。\n"
        "schema: {\n"
        '  "height_cm":number|null, "weight_kg":number|null,\n'
        '  "preferences":[string], "conditions":[string], "notes":string|null,\n'
        '  "age":int|null, "gender":string|null, "occupation":string|null, "city":string|null,\n'
        '  "diet":[string], "allergies":[string],\n'
        '  "sleep_schedule":string|null, "exercise_habits":string|null, "work_hours":string|null,\n'
        '  "family_status":string|null, "goals":[string]\n'
        "}\n"
        "## 字段说明\n"
        "- height_cm/weight_kg: 身高厘米/体重千克\n"
        "- preferences: 喜好和反感，如'喜欢跑步'/'不吃辣'/'不爱运动'\n"
        "- conditions: 疾病/慢性病/身体禁忌\n"
        "- age: 年龄（整数）\n"
        "- gender: 性别\n"
        "- occupation: 职业/工作\n"
        "- city: 所在城市\n"
        "- diet: 饮食模式偏好，如'素食'/'低糖'/'低碳水'/'生酮'\n"
        "- allergies: 过敏/忌口，如'海鲜过敏'/'不吃辣'/'乳糖不耐'\n"
        "- sleep_schedule: 作息规律，如'早睡早起'/'夜猫子'/'7点起12点睡'\n"
        "- exercise_habits: 运动习惯，如'每周跑步3次'/'健身房'\n"
        "- work_hours: 工作时间段，如'9-18'/'朝九晚六'\n"
        "- family_status: 家庭状况，如'独居'/'已婚有娃'/'和父母住'\n"
        "- goals: 短期目标/优先事项，如'减肥'/'攒钱'/'学英语'\n"
        "## 规则\n"
        "- 只提取用户明确表达的信息；不确定则置空或不填。\n"
        "- preferences 包含喜好和反感。\n"
        "- 用户明确表达否定（不喜欢/讨厌/不吃等）一定要提取到 preferences 或 allergies，不要忽略。\n"
        "- 如果本句没有画像信息，返回空补丁：{\"height_cm\":null,\"weight_kg\":null,\"preferences\":[],\"conditions\":[],\"notes\":null,\"age\":null,\"gender\":null,\"occupation\":null,\"city\":null,\"diet\":[],\"allergies\":[],\"sleep_schedule\":null,\"exercise_habits\":null,\"work_hours\":null,\"family_status\":null,\"goals\":[]}。\n"
    )

    payload = {
        "now_iso": now_iso,
        "user_message": user_message,
        "recent_dialogue": recent_dialogue[-4:],
    }

    def _parse(text: str) -> ProfileFactPatch:
        return ProfileFactPatch.model_validate(json.loads(text))

    return _llm_extract(
        system_prompt=system_prompt,
        payload=payload,
        parse=_parse,
        diagnose_context="profile_fact_extraction",
        session_id=session_id,
    )


def extract_goal(
    user_message: str,
    recent_dialogue: list[dict],
    now_iso: str,
    session_id: str | None = None,
) -> GoalExtraction | None:
    """从对话中提取用户目标并进行拆解。

    返回 GoalExtraction 或 None（抽取失败时）。
    """
    system = (
        "你是目标拆解助手。从用户对话中识别明确的目标陈述，并将其拆解为可执行的子步骤。\n\n"
        "## 判断标准\n"
        "- 包含意愿动词（想/要/打算/计划/准备/决定/希望）\n"
        "- 有明确的愿景或结果描述（如'减肥20斤'/'学会Python'/'攒10万'）\n"
        "- 是长期/中期目标而非一次性待办（'周五交报表'不是目标）\n\n"
        "## 拆解原则\n"
        "- 每个大目标拆成 3-7 个具体子步骤\n"
        "- 子步骤应是可执行的行动（如'每周跑步3次'而非'多运动'）\n"
        "- 如有明确的时间节点，设为 milestone\n"
        "- category: health|career|finance|learning|lifestyle|other\n\n"
        "## 输出格式\n"
        "输出严格 JSON，不要 Markdown：\n"
        '{"items": [{"title": "...", "category": "...", "target_date": "2026-12-31", '
        '"sub_goals": [{"title": "每月减2斤", "target_date": "2026-06"}], '
        '"milestones": [{"title": "减到70kg", "target_date": "2026-06", "is_completed": false}], '
        '"notes": "..."}]}'
    )
    recent = recent_dialogue[-8:] if recent_dialogue else []
    payload = {
        "user_message": user_message,
        "recent_dialogue": [
            {"role": m.get("role"), "content": m.get("content")} for m in recent
        ],
        "now_iso": now_iso,
    }

    def _parse(text: str) -> GoalExtraction:
        return GoalExtraction.model_validate(json.loads(text))

    return _llm_extract(
        system_prompt=system,
        payload=payload,
        parse=_parse,
        diagnose_context="extract_goal",
        session_id=session_id,
    )


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
    if profile.get("age"):
        parts.append(f"年龄{profile['age']}岁")
    if profile.get("gender"):
        parts.append(f"{profile['gender']}")
    if profile.get("occupation"):
        parts.append(f"职业:{profile['occupation']}")
    if profile.get("city"):
        parts.append(f"所在城市:{profile['city']}")
    if profile.get("sleep_schedule"):
        parts.append(f"作息:{profile['sleep_schedule']}")
    if profile.get("exercise_habits"):
        parts.append(f"运动:{profile['exercise_habits']}")
    if profile.get("work_hours"):
        parts.append(f"工作时间:{profile['work_hours']}")
    if profile.get("family_status"):
        parts.append(f"家庭:{profile['family_status']}")

    prefs = list(profile.get("preferences") or [])
    # 抽取旧数据中的城市 hack 兼容
    normal_prefs: list[str] = []
    for p in prefs:
        if not isinstance(p, str):
            normal_prefs.append(str(p))
            continue
        if (p.startswith("默认城市:") or p.startswith("所在城市:")) and not profile.get("city"):
            parts.append(f"所在城市:{p.split(':', 1)[-1].strip()}")
        else:
            normal_prefs.append(p)
    if normal_prefs:
        parts.append("喜好:" + "、".join(normal_prefs[:6]))

    if profile.get("diet"):
        parts.append("饮食偏好:" + "、".join(profile["diet"][:4]))
    if profile.get("allergies"):
        parts.append("忌口:" + "、".join(profile["allergies"][:4]))
    conds = profile.get("conditions") or []
    if conds:
        parts.append("健康情况:" + "、".join(conds[:6]))
    if profile.get("goals"):
        parts.append("目标:" + "、".join(profile["goals"][:4]))
    if profile.get("notes"):
        parts.append(f"备注:{profile['notes']}")

    return "；".join(parts)


def generate_proactive_advice(
    *,
    user_message: str,
    recent_events: list[dict],
    upcoming_todos: list[dict],
    now_iso: str,
    goals: list[dict] | None = None,
    session_id: str | None = None,
) -> str | None:
    """基于近期事件 + 日程 + 目标生成主动建议。

    此函数会在回答之外追加一段主动建议，例如：
    - 如果用户刚吃过油腻的，建议清淡饮食
    - 如果即将开会但还有冲突事件，建议调整
    - 如果用户有目标但近期行为与目标冲突，提醒注意
    """

    if not recent_events and not upcoming_todos and not goals:
        return None

    system_prompt = (
        "你是个人生活管理建议助手。读近期事件+待办+目标，判断是否需要主动提醒。\n"
        "## 要求\n"
        "- 只说最相关的一条；没有确定关联则输出 NONE。\n"
        "- 以第二人称「你」开头，简洁口语化。\n"
        "- 结合当前时间与事件时间判断是否冲突。\n"
        "- 如果用户有活跃目标但近期行为与目标方向不一致，可以温和提醒。\n"
        "- 输出纯文本，无 Markdown。"
    )

    payload = {
        "now": now_iso,
        "user_message": user_message,
        "recent_events": recent_events,
        "upcoming_todos": upcoming_todos,
        "goals": goals or [],
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
        "## 要求\n"
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


def decide_proactive_advice(
    *,
    user_message: str,
    assistant_answer: str,
    recent_events: list[dict],
    upcoming_todos: list[dict],
    pending_reminders: list[dict] | None = None,
    profile_context: str,
    goals: list[dict] | None = None,
    now_iso: str,
    threshold: float,
    session_id: str | None = None,
) -> ProactiveAdviceDecision:
    if not recent_events and not upcoming_todos and not profile_context and not pending_reminders and not goals:
        return ProactiveAdviceDecision(score=0.0, should_add=False, advice=None, reasons=["无可用上下文"])

    system_prompt = (
        "你是一个严谨的提醒筛选器。判断是否需要追加建议，输出严格 JSON。\n"
        'schema: {"score":0-1,"should_add":bool,"advice":"str或null","reasons":["str"]}\n'
        "## 核心原则\n"
        "默认 should_add=false，除非：\n"
        "1. 事件/待办/目标之间存在真实关联（时间冲突、因果关系、健康风险）\n"
        "2. 能给出明确、可执行的操作建议\n"
        "3. 不是把两个无关事件强行关联\n\n"
        "## 反面案例（should_add=false）\n"
        "- 明天预订肯德基 + 后天端午节 -> 无关，不应建议\n"
        "- 明天去超市 + 下周有会议 -> 无关\n\n"
        "## 正面案例（should_add=true）\n"
        "- 明天下午3点开会 + 明天下午2点体检 -> 时间冲突，建议调整\n"
        "- 明天跑10公里 + 膝盖不舒服 -> 健康风险，建议减量\n"
        "- 目标'减重20斤' + 晚上约了火锅 -> 与目标冲突，提醒注意饮食\n\n"
        "## 输出要求\n"
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
        "goals": goals or [],
    }

    result = _llm_extract(
        system_prompt=system_prompt,
        payload=payload,
        parse=lambda text: json.loads(text),
        diagnose_context="proactive_advice_decision",
        session_id=session_id,
    )
    if result is None:
        return ProactiveAdviceDecision(score=0.0, should_add=False, advice=None, reasons=["提取失败，跳过"])

    try:
        decision = ProactiveAdviceDecision.model_validate(result)
    except ValidationError:
        return ProactiveAdviceDecision(score=0.0, should_add=False, advice=None, reasons=["解析失败，跳过"])

    decision.should_add = bool(decision.advice) and decision.score >= threshold
    if not decision.should_add:
        decision.advice = None
    return decision


