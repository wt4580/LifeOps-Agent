"""基于 LangGraph 的聊天状态机编排。

Chat -> (LLM decide) -> Tool -> Update state -> (maybe HITL) -> Final answer -> (proactive advice)
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
from sqlalchemy import select

from .agent_router import build_route_context, route_decision
from ..common.config.db_config import SessionLocal
from ..common.config.base_config import settings
from ..common.config.log_config import logger
from ..common.config.llm_config import chat_completion
from ..domain.entity.chat_entity import ConversationSummary, TodoItem
from .planner.planner import detect_affirmation, detect_rejection, propose_plan
from ..util.retrieval.retrieval import rag_answer, search_chunks
from .time.time_parser import normalize_date_hint
from .memory.personal_memory import decide_proactive_advice
from ..service.calendar_service import CalendarService, CalendarServiceError
from ..service.weather_service import weather_service, WeatherServiceError
from ..domain.entity.chatstate_entity import ChatState

# 范围查询待办：当用户问“未来几天安排”时使用。
def _query_upcoming_todos(range_days: int) -> list[dict[str, Any]]:
    now = datetime.now()
    end = now + timedelta(days=range_days)
    with SessionLocal() as db:
        rows = (
            db.execute(
                select(TodoItem)
                .where(TodoItem.due_at != None)
                .where(TodoItem.due_at >= now)
                .where(TodoItem.due_at < end)
                .order_by(TodoItem.due_at.asc())
            )
            .scalars()
            .all()
        )
    return [{"title": r.title, "due_at": r.due_at.isoformat() if r.due_at else None} for r in rows]


# 单天查询待办：当用户消息可解析为具体日期（如“后天”）时优先使用。
def _query_todos_for_date(target_date_iso: str) -> list[dict[str, Any]]:
    target = datetime.fromisoformat(target_date_iso).date()
    start = datetime.combine(target, datetime.min.time())
    end = start + timedelta(days=1)
    with SessionLocal() as db:
        rows = (
            db.execute(
                select(TodoItem)
                .where(TodoItem.due_at != None)
                .where(TodoItem.due_at >= start)
                .where(TodoItem.due_at < end)
                .order_by(TodoItem.due_at.asc())
            )
            .scalars()
            .all()
        )
    return [{"title": r.title, "due_at": r.due_at.isoformat() if r.due_at else None} for r in rows]


def _get_summary(session_id: str) -> str | None:
    with SessionLocal() as db:
        row = (
            db.execute(
                select(ConversationSummary)
                .where(ConversationSummary.session_id == session_id)
                .order_by(ConversationSummary.updated_at.desc())
                .limit(1)
            )
            .scalars()
            .first()
        )
    return row.summary_text if row else None





def _should_check_proactive(state: ChatState) -> bool:
    """判断是否需要执行主动决策。HITL 流程跳过，普通聊天才检查。"""
    used_tool = state.get("used_tool")
    return used_tool not in {"plan_proposal", "plan_gate_cancel", "plan_gate_wait", "plan_gate_confirm"}


# trace 是可审计轨迹，这个函数统一追加步骤，前端右侧面板依赖它渲染。
def _append_trace(state: ChatState, step: dict[str, Any]) -> None:
    trace = state.setdefault("trace", {"steps": []})
    trace.setdefault("steps", []).append(step)


# 统一构造 trace step，保证每一步都有 type/name/timestamp。
def _trace_step(*, step_type: str, name: str, **kwargs: Any) -> dict[str, Any]:
    return {
        "type": step_type,
        "name": name,
        "ts": datetime.now().isoformat(timespec="seconds"),
        **kwargs,
    }





def _resolve_relative_dates(message: str) -> str:
    today = datetime.now().date()
    for word, offset in [("大后天", 3), ("后天", 2), ("明天", 1), ("今天", 0)]:
        if word in message:
            resolved = (today + timedelta(days=offset)).strftime("%m月%d日")
            message = message.replace(word, resolved)
    return message


def _build_effective_message(user_message: str, profile_context: str | None, pre_advice: str | None) -> str:
    parts = [f"用户输入: {user_message}"]
    if profile_context:
        parts.append(f"用户画像: {profile_context}")
    if pre_advice:
        parts.append(f"前置建议: {pre_advice}")
    return "\n".join(parts)


def _is_bare_confirmation_text(text: str) -> bool:
    """识别单独的确认/取消短句，避免把它们误当成新的待办意图。"""

    stripped = (text or "").strip()
    return stripped in {"需要", "要", "可以", "好的", "好", "行", "确认", "不用", "不要", "算了", "取消"}


def _node_handle_pending_confirmation(state: ChatState) -> ChatState:
    """如果上一轮留下了待确认草案，这里优先消化用户的确认/取消回复。"""

    pending_text = state.get("pending_text")
    if not pending_text:
        return state

    user_message = (state.get("user_message") or "").strip()
    if detect_rejection(user_message):
        state["pending_text"] = None
        state["proposal_id"] = None
        state["proposal"] = None
        state["used_tool"] = "plan_gate_cancel"
        state["answer"] = "好的，这次先不生成待办草案。你之后说‘需要’我再帮你建。"
        _append_trace(
            state,
            _trace_step(
                step_type="graph_node",
                name="hitl_resume",
                input={"pending_text": pending_text, "user_message": user_message},
                output={"confirmed": False, "cancelled": True},
            ),
        )
        return state

    if not detect_affirmation(user_message):
        state["used_tool"] = "plan_gate_wait"
        state["answer"] = "我可以为你生成待办草案，回复‘需要’确认，或回复‘不要’取消。"
        _append_trace(
            state,
            _trace_step(
                step_type="graph_node",
                name="hitl_resume",
                input={"pending_text": pending_text, "user_message": user_message},
                output={"confirmed": False, "waiting": True},
            ),
        )
        return state

    proposal_id, proposal = propose_plan(pending_text)
    proposal_payload = proposal.model_dump()

    state["pending_text"] = None
    state["proposal_id"] = proposal_id
    state["proposal"] = proposal_payload
    state["used_tool"] = "plan_proposal"
    state["answer"] = "已生成待办草案，请确认是否加入待办。"
    _append_trace(
        state,
        _trace_step(
            step_type="graph_node",
            name="hitl_resume",
            input={"pending_text": pending_text, "user_message": user_message},
            output={"confirmed": True, "proposal_id": proposal_id},
        ),
    )
    _append_trace(
        state,
        _trace_step(
            step_type="tool",
            name="propose_from_pending",
            input={"text": pending_text},
            output=proposal_payload,
        ),
    )
    return state


# -----------------------------
# LangGraph 节点
# -----------------------------

# 第一个节点：加载上下文。
# 顺序上它永远先于 hitl_check/decide，因为后续节点都可能依赖 checkpoint 里恢复出来的 history/summary。
def _node_load_context(state: ChatState) -> ChatState:
    history = list(state.get("history") or [])
    state["history"] = history
    state["recent_dialogue"] = history[-16:]
    state["summary"] = _get_summary(state["session_id"])
    state["trace"] = {
        "meta": {"graph": "chat_v2", "started_at": datetime.now().isoformat(timespec="seconds")},
        "steps": [],
    }
    state["answer"] = ""
    state["used_tool"] = None
    state["citations"] = []
    state["decision"] = {}
    state["proposal_id"] = None
    state["proposal"] = None
    state["hitl_cancelled"] = False
    state["plan"] = []
    state["plan_index"] = 0
    _append_trace(
        state,
        _trace_step(
            step_type="graph_node",
            name="load_context",
            output={"history_len": len(history), "recent_dialogue_len": len(state["recent_dialogue"]), "has_summary": bool(state.get("summary"))},
        ),
    )
    return state


def _is_simple_query(text: str) -> bool:
    t = text.strip()
    if len(t) < 8:
        return True
    multi_intent = re.search(r"(并且|还有|同时|再加|再|还|顺便|也)", t)
    if not multi_intent:
        return True
    return False


def _node_decompose(state: ChatState) -> ChatState:
    user_msg = state.get("user_message", "")

    if _is_simple_query(user_msg):
        state["plan"] = []
        state["plan_index"] = 0
        _append_trace(state, _trace_step(step_type="graph_node", name="decompose",
            output={"plan": [], "fast_path": True}))
        return state

    system = (
        "你是一个任务分解器。判断用户请求是否需要拆成多个子步骤执行。\n"
        "如果一句话就能回答（如问候、问日期、问单个事件），不需要分解。\n"
        "如果需要查多个不同信息（天气+日程、日程+待办）或多个步骤才能给出完整回答，才需要分解。\n\n"
        "可用工具：\n"
        "- query_calendar: 查日历事件（节日、日程）\n"
        "- query_todos: 查待办事项\n"
        "- query_weather: 查天气\n"
        "- query_knowledge: 查知识库文档\n"
        "- normal_chat: 普通对话 / 综合回答\n"
        "- propose_todo: 生成待办草案\n\n"
        "输出严格 JSON，不要 Markdown：\n"
        '{"needs_plan": false}\n'
        '或 {"needs_plan": true, "steps": [{"action":"...","args":{...},"purpose":"..."}]}'
    )
    raw = chat_completion(
        [{"role": "system", "content": system}, {"role": "user", "content": user_msg}],
        temperature=0.0,
        runtime_context={"scenario": "decompose", "session_id": state.get("session_id")},
    )
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    try:
        data = json.loads(cleaned)
        if data.get("needs_plan") and data.get("steps"):
            steps = data["steps"][:10]
            state["plan"] = steps
            state["plan_index"] = 0
            _append_trace(state, _trace_step(step_type="graph_node", name="decompose",
                output={"plan": steps}))
        else:
            state["plan"] = []
            state["plan_index"] = 0
    except Exception as exc:
        logger.warning("decompose failed session=%s input=%s raw=%s err=%s", state.get("session_id"), user_msg[:40], cleaned, exc)
        state["plan"] = []
        state["plan_index"] = 0
    return state


def _is_prev_todo_query(state: ChatState) -> bool:
    msg = ((state.get("user_message") or "") + " " + (state.get("effective_user_message") or "")).strip().lower()
    if not any(k in msg for k in ["上一条", "上一步", "上一个", "前一个", "上一台", "1条", "上1"]):
        return False
    recent = state.get("recent_dialogue") or []
    for turn in reversed(recent[-6:]):
        role = turn.get("role", "")
        content = (turn.get("content") or "").lower()
        if role == "assistant" and any(k in content for k in ["待办", "事项", "安排", "日程", "todo", "您有"]):
            return True
    return False


def _node_decide(state: ChatState) -> ChatState:
    msg = (state.get("user_message") or "").strip()

    if state.get("pending_add_confirmation"):
        if detect_affirmation(msg):
            decision = {
                "action": "confirm_proposal",
                "args": {},
                "assistant_message": "",
                "trace": {"intent": "confirm_proposal_yes", "why": "用户确认待办加入"},
            }
            state["decision"] = decision
            return state
        if detect_rejection(msg):
            decision = {
                "action": "confirm_proposal",
                "args": {},
                "assistant_message": "",
                "trace": {"intent": "confirm_proposal_no", "why": "用户拒绝待办加入"},
            }
            state["decision"] = decision
            return state

    if _is_prev_todo_query(state):
        decision = {
            "action": "query_todos",
            "args": {"range_days": 7},
            "assistant_message": "",
            "trace": {"intent": "prev_todos", "why": "用户模糊短语查待办（如'上一条'/'上一个'），结合上下文推断"},
        }
        state["decision"] = decision
        return state

    if not state.get("pending_text") and _is_bare_confirmation_text(msg):
        decision = {
            "action": "normal_chat",
            "args": {},
            "assistant_message": "你是说刚才提到的'跟进导员报销审批'这个事吗？如果是，请说'是'或'要'；如果不是，请告诉我具体想记什么。",
            "trace": {"intent": "confirmation_clarify", "why": "短句但无待确认草案，先澄清"},
        }
        state["decision"] = decision
        return state

    plan = state.get("plan") or []
    plan_index = state.get("plan_index", 0)
    if plan and plan_index < len(plan):
        step = plan[plan_index]
        decision = {
            "action": step["action"],
            "args": step.get("args") or {},
            "assistant_message": "",
            "trace": {"intent": "plan_step", "why": step.get("purpose", "多步分解子任务")},
        }
        state["decision"] = decision
        _append_trace(state, _trace_step(step_type="llm", name="plan_step",
            input={"plan_index": plan_index, "step": step},
            output={"action": step["action"]}))
        return state

    forced_action = state.get("forced_action")
    if forced_action:
        decision = {
            "action": forced_action,
            "args": state.get("forced_args") or {},
            "assistant_message": "",
            "trace": {"intent": "forced", "why": "forced by API wrapper"},
        }
        state["decision"] = decision
        return state

    ctx = build_route_context(
        session_id=state["session_id"],
        user_message=state.get("effective_user_message") or state["user_message"],
        recent_dialogue=state.get("recent_dialogue", []),
        summary=state.get("summary"),
    )
    decision = route_decision(ctx)
    state["decision"] = decision.model_dump()

    decision_detail = state["decision"]
    router_trace = decision_detail.get("trace", {})

    _append_trace(
        state,
        _trace_step(
            step_type="llm",
            name="router_decision",
            input={
                "user_message": state["user_message"],
                "summary_present": bool(state.get("summary")),
                "recent_dialogue_len": len(state.get("recent_dialogue", [])),
            },
            output={
                "action": decision_detail.get("action"),
                "args": decision_detail.get("args", {}),
                "decision_reasoning": {"intent": router_trace.get("intent"), "why": router_trace.get("why")},
            },
        ),
    )
    return state


def _build_normal_chat_answer(
    state: ChatState,
    decision: dict[str, Any],
    *,
    allow_assistant_message: bool = True,
    plan_context: str = "",
) -> tuple[str, bool]:
    assistant_message = (decision.get("assistant_message") or "").strip() if allow_assistant_message else ""
    if assistant_message:
        return assistant_message, True

    messages: list[dict[str, str]] = [{"role": "system", "content": "你是一个中文生活助手。"}]
    if plan_context:
        messages.append({"role": "system", "content": f"前面步骤查询到的信息（请基于这些信息回答）：\n{plan_context}"})

    context_parts = []
    if state.get("profile_context"):
        context_parts.append(f"用户画像：{state['profile_context']}")
    if state.get("pre_advice"):
        context_parts.append(f"前置建议：{state['pre_advice']}")
    if state.get("summary"):
        context_parts.append(f"对话摘要：{state['summary']}")

    recent_events = state.get("recent_events") or []
    if recent_events:
        event_lines = []
        for ev in recent_events[:8]:
            title = ev.get("title", "")
            t = ev.get("event_time", "")
            cat = ev.get("category", "")
            if t:
                t_short = t[:10]
                event_lines.append(f"  [{cat}] {title}（{t_short}）")
            else:
                event_lines.append(f"  [{cat}] {title}")
        context_parts.append("近期生活事件：\n" + "\n".join(event_lines))

    upcoming_todos = state.get("upcoming_todos") or []
    if upcoming_todos:
        todo_lines = []
        for td in upcoming_todos[:6]:
            title = td.get("title", "")
            due = td.get("due_at", "")
            if due:
                due_short = due[:10]
                todo_lines.append(f"  {title}（截止{due_short}）")
            else:
                todo_lines.append(f"  {title}")
        context_parts.append("待办事项：\n" + "\n".join(todo_lines))

    if context_parts:
        messages.append({"role": "system", "content": "用户相关背景：\n" + "\n\n".join(context_parts)})
    messages.extend(state.get("recent_dialogue", []))
    messages.append({"role": "user", "content": state.get("effective_user_message") or state["user_message"]})
    return chat_completion(messages), False


def _node_run_tool(state: ChatState) -> ChatState:
    decision = state.get("decision") or {"action": "normal_chat", "args": {}, "assistant_message": ""}
    action = decision.get("action", "normal_chat")
    args = decision.get("args") or {}

    if action == "query_todos":
        # 先尝试解析精确日期（如“后天/周三”）。
        # 能解析到单日就查单日；否则按 range_days 查区间。
        date_hint = normalize_date_hint(state["user_message"])

        if date_hint:
            items = _query_todos_for_date(date_hint)
            query_input = {"target_date": date_hint}
        else:
            range_days = int(args.get("range_days", 7) or 7)
            items = _query_upcoming_todos(range_days)
            query_input = {"range_days": range_days}

        state["used_tool"] = "query_todos"

        if not items:
            state["answer"] = "我查了一下，这个时间点没有已安排的待办。"
        else:
            # LLM 调用点：只负责“自然语言总结”，不改动数据库原始查询结果。
            prompt = (
                "你是生活助理。用户在询问他的安排。\n"
                "下面是从数据库查到的待办列表（真实数据），请用中文做简短总结。\n"
                "要求：不要编造列表中没有的事项；如果时间为空就不要猜。"
            )
            state["answer"] = chat_completion(
                [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": json.dumps(items, ensure_ascii=False)},
                ],
                temperature=0.2,
            )
            _append_trace(
                state,
                _trace_step(
                    step_type="llm",
                    name="schedule_summarize",
                    input={"items_count": len(items), **query_input},
                    output=state["answer"],
                ),
            )

        _append_trace(state, _trace_step(step_type="tool", name="query_todos", input=query_input, output=items))
        return state

    if action == "query_knowledge":
        user_q = (state.get("user_message") or "").strip()
        router_q = (args.get("question") or "").strip()
        eff = (state.get("effective_user_message") or "").strip()
        # 检索只用「真实用户句」：规则/路由常把 effective_user_message（含画像前缀）塞进 question，会严重干扰 BM25/向量匹配。
        search_q = router_q if router_q and router_q != eff else user_q
        if not search_q:
            search_q = user_q
        top_k = int(args.get("top_k", 5) or 5)
        citations = search_chunks(search_q, top_k=top_k)
        state["used_tool"] = "query_knowledge"
        state["citations"] = [
            {
                "path": c.path,
                "page": c.page,
                "snippet": c.snippet,
                "score": c.score,
                "reason": c.reason,
                "source_type": c.source_type,
                "doc_topic": c.doc_topic,
            }
            for c in citations
        ]

        if not citations:
            # RAG miss 回退：进入 normal_chat，而不是直接回答“找不到”。
            # query_knowledge miss 时，不能把 router 的“正在为您查询...”当成最终答案。
            # 此时需要真正走一遍 normal_chat 生成可回答内容。
            state["answer"], used_assistant = _build_normal_chat_answer(
                state,
                decision,
                allow_assistant_message=False,
            )
            state["used_tool"] = "normal_chat_fallback"
            _append_trace(
                state,
                _trace_step(
                    step_type="tool",
                    name="query_knowledge",
                    input={"question": search_q, "top_k": top_k},
                    output={"hits": 0, "fallback": "normal_chat"},
                ),
            )
            _append_trace(
                state,
                _trace_step(
                    step_type="llm",
                    name="normal_chat_fallback",
                    output={"used_assistant_message": used_assistant, "answer": state["answer"]},
                ),
            )
            return state

        state["answer"] = rag_answer(
            user_q,
            citations,
            profile_context=state.get("profile_context"),
            pre_advice=state.get("pre_advice"),
        )

        _append_trace(
            state,
            _trace_step(
                step_type="tool",
                name="query_knowledge",
                input={"question": search_q, "top_k": top_k},
                output={"hits": len(citations)},
            ),
        )
        return state

    if action == "query_weather":
        city_arg = (args.get("city") or "").strip() if isinstance(args, dict) else ""
        profile_ctx = state.get("profile_context") or ""
        query_input: dict[str, Any] = {"city": city_arg or None, "has_profile_city": bool(profile_ctx)}
        try:
            result = weather_service.query(city_arg, profile_context=profile_ctx)
            query_input["resolved_city"] = result.get("city")
            query_input["cached"] = bool(result.get("cached"))
            state["used_tool"] = "query_weather"

            prompt = (
                "你是生活助理。用户在询问天气。下面是一段结构化天气数据（实况 + 未来几天预报，真实数据）。\n"
                "请用中文生成简短、自然的回复：\n"
                "- 先说实况（当前温度/天气/湿度/风）\n"
                "- 再提示未来几天的趋势与出行建议（带伞/穿衣）\n"
                "- 不要编造数据里没有的温度或天气现象；如果字段缺失就跳过。\n"
            )
            state["answer"] = chat_completion(
                [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": json.dumps(result, ensure_ascii=False)},
                ],
                temperature=0.2,
            )
            _append_trace(
                state,
                _trace_step(
                    step_type="llm",
                    name="weather_summarize",
                    input={"city": result.get("city"), "has_live": bool(result.get("live")), "forecast_days": len(result.get("forecast") or [])},
                    output=state["answer"],
                ),
            )
            _append_trace(state, _trace_step(step_type="tool", name="query_weather", input=query_input, output={"city": result.get("city"), "adcode": result.get("adcode"), "cached": bool(result.get("cached"))}))
        except WeatherServiceError as exc:
            logger.warning("query_weather failed: %s", exc)
            state["used_tool"] = "query_weather"
            state["answer"] = str(exc)
            _append_trace(state, _trace_step(step_type="tool", name="query_weather", input=query_input, error=str(exc)))
        return state

    if action == "query_calendar":
        now = datetime.now()
        max_days = 90

        start_date = (args.get("start_date") or "").strip() if isinstance(args, dict) else ""
        end_date = (args.get("end_date") or "").strip() if isinstance(args, dict) else ""

        try:
            if start_date and end_date:
                start_dt = datetime.fromisoformat(start_date)
                end_dt = datetime.fromisoformat(end_date) + timedelta(days=1)
            else:
                range_days = int(args.get("range_days", 7) or 7) if isinstance(args, dict) else 7
                range_days = min(max(1, range_days), max_days)
                start_dt = now.replace(hour=0, minute=0, second=0, microsecond=0)
                end_dt = start_dt + timedelta(days=range_days)
        except ValueError as exc:
            state["used_tool"] = "query_calendar"
            state["answer"] = f"日期格式解析失败: {exc}"
            _append_trace(state, _trace_step(step_type="tool", name="query_calendar", input={"args": args}, error=str(exc)))
            return state

        start_iso = start_dt.isoformat() + ("Z" if not start_dt.tzinfo else "")
        end_iso = end_dt.isoformat() + ("Z" if not end_dt.tzinfo else "")
        query_input = {
            "start": start_dt.isoformat(),
            "end": end_dt.isoformat(),
        }

        try:
            events = CalendarService().list_events(start_iso, end_iso)
            state["used_tool"] = "query_calendar"

            if not events:
                state["answer"] = "我查了一下日历，这段时间没有安排的事件。"
            else:
                prompt = (
                    "你是生活助理。用户在询问他的日程/会议/事件安排。下面是从本地日历查到的真实事件列表。\n"
                    "请用中文做简短总结：\n"
                    "- 按时间顺序列出\n"
                    "- 不要编造列表中没有的事项\n"
                    "- 全天事件单独说明\n"
                    "- 给出 1 句时间冲突或出行提醒（如果明显有）"
                )
                state["answer"] = chat_completion(
                    [
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": json.dumps(events, ensure_ascii=False)},
                    ],
                    temperature=0.2,
                )
                _append_trace(
                    state,
                    _trace_step(
                        step_type="llm",
                        name="calendar_summarize",
                        input={"events_count": len(events)},
                        output=state["answer"],
                    ),
                )
            _append_trace(state, _trace_step(step_type="tool", name="query_calendar", input=query_input, output={"events_count": len(events)}))
        except CalendarServiceError as exc:
            logger.warning("query_calendar failed: %s", exc)
            state["used_tool"] = "query_calendar"
            state["answer"] = str(exc)
            _append_trace(state, _trace_step(step_type="tool", name="query_calendar", input=query_input, error=str(exc)))
        return state

    if action in ("ask_user_confirm_proposal", "propose_todo"):
        pending_text = args.get("text") or state["user_message"]
        proposal_id, proposal = propose_plan(pending_text)
        proposal_payload = proposal.model_dump()

        state["pending_text"] = None
        state["pending_add_confirmation"] = True
        state["proposal_id"] = proposal_id
        state["proposal"] = proposal_payload
        state["used_tool"] = "plan_proposal"
        state["answer"] = "已生成待办草案，请确认是否加入待办。"
        _append_trace(
            state,
            _trace_step(
                step_type="tool",
                name="propose_todo",
                input={"text": pending_text},
                output={"proposal_id": proposal_id},
            ),
        )
        _append_trace(
            state,
            _trace_step(
                step_type="tool",
                name="propose_from_text",
                input={"text": pending_text},
                output=proposal_payload,
            ),
        )
        return state

    if action == "confirm_proposal":
        user_msg = (state.get("user_message") or "").strip()
        if detect_affirmation(user_msg):
            state["proposal_confirmed"] = True
            state["pending_add_confirmation"] = False
            state["proposal_id"] = None
            state["proposal"] = None
            state["used_tool"] = "plan_gate_confirm"
            state["answer"] = "好的，已添加到待办。"
        else:
            state["proposal_id"] = None
            state["proposal"] = None
            state["pending_add_confirmation"] = False
            state["used_tool"] = "plan_gate_cancel"
            state["answer"] = "好的，已取消。"
        _append_trace(
            state,
            _trace_step(
                step_type="tool",
                name="confirm_proposal",
                input={"user_message": user_msg, "proposal_id": state.get("proposal_id")},
                output={"confirmed": state.get("proposal_confirmed", False)},
            ),
        )
        return state

    # normal_chat 分支：
    state["used_tool"] = None
    plan = state.get("plan") or []
    plan_context = ""
    if plan:
        parts = []
        for i, step in enumerate(plan):
            if step.get("result"):
                parts.append(f"【步骤{i+1}：{step.get('purpose', step['action'])}】\n{step['result']}")
        if parts:
            plan_context = "\n\n".join(parts)

    assistant_message = (decision.get("assistant_message") or "").strip()
    if assistant_message:
        state["answer"] = assistant_message
        used_assistant = True
    else:
        state["answer"], used_assistant = _build_normal_chat_answer(state, decision, plan_context=plan_context)

    _append_trace(state, _trace_step(step_type="tool", name="normal_chat", output={"used_assistant_message": used_assistant}))
    _append_trace(state, _trace_step(step_type="llm", name="normal_chat", output=state.get("answer") or ""))
    return state


# 主动决策节点：基于近期事件/待办/画像，决定是否追加主动建议。
def _node_proactive_decision(state: ChatState) -> ChatState:
    recent_events = state.get("recent_events", [])
    upcoming_todos = state.get("upcoming_todos", [])
    profile_context = state.get("profile_context") or ""

    if not recent_events and not upcoming_todos:
        return state

    now_iso = datetime.now().isoformat(timespec="seconds")
    decision = decide_proactive_advice(
        user_message=state["user_message"],
        assistant_answer=state.get("answer", ""),
        recent_events=recent_events,
        upcoming_todos=upcoming_todos,
        profile_context=profile_context,
        now_iso=now_iso,
        threshold=settings.proactive_advice_threshold,
        session_id=state["session_id"],
    )
    
    # 3. 如果需要追加建议，修改 answer
    advice_added = False
    if decision.should_add and decision.advice:
        current_answer = state.get("answer", "")
        if decision.advice not in current_answer:
            state["answer"] = f"{current_answer}\n\n补充建议：{decision.advice}".strip()
            advice_added = True
    
    # 4. 记录 trace（自动进入 checkpoint）
    _append_trace(
        state,
        _trace_step(
            step_type="postprocess",
            name="proactive_advice_decision",
            output={
                "score": decision.score,
                "threshold": settings.proactive_advice_threshold,
                "added": advice_added,
                "reasons": decision.reasons,
                "events_used": len(recent_events),
                "upcoming_todos_used": len(upcoming_todos),
                "now_iso": now_iso,
                "event_window_start": state.get("event_window_start"),
                "event_window_end": state.get("event_window_end"),
            }
        )
    )
    
    logger.info(
        "proactive advice decision score=%.3f threshold=%.3f added=%s events=%s todos=%s",
        decision.score,
        settings.proactive_advice_threshold,
        advice_added,
        len(recent_events),
        len(upcoming_todos),
    )
    
    return state


# 收口节点：保证每条路径都有 answer，并记录最终 used_tool / has_proposal。
def _node_finalize(state: ChatState) -> ChatState:
    # 兜底，避免异常路径没有 answer
    if not state.get("answer"):
        state["answer"] = "抱歉，我刚刚没有处理好。你可以再说一次。"

    history = list(state.get("history") or [])
    if not history or history[-1].get("role") != "assistant" or history[-1].get("content") != state["answer"]:
        history.append({"role": "assistant", "content": state["answer"]})
    state["history"] = history
    state["recent_dialogue"] = history[-16:]

    _append_trace(
        state,
        _trace_step(
            step_type="graph_node",
            name="finalize",
            output={"used_tool": state.get("used_tool"), "has_proposal": bool(state.get("proposal_id"))},
        ),
    )
    return state


# -----------------------------
# 图构建与执行
# -----------------------------

def _node_advance_plan(state: ChatState) -> ChatState:
    plan = state.get("plan") or []
    plan_index = state.get("plan_index", 0)
    if plan and plan_index < len(plan):
        plan[plan_index]["result"] = state.get("answer", "") or ""
        plan[plan_index]["used_tool"] = state.get("used_tool")
        return {**state, "plan_index": plan_index + 1}
    return state


def _route_after_advance(state: ChatState) -> str:
    plan = state.get("plan") or []
    plan_index = state.get("plan_index", 0)
    if plan and plan_index < len(plan):
        return "decide"
    return "finalize"


# 图连线就是"先走谁再走谁"的唯一真相：
# START -> load_context -> hitl_resume? -> decompose -> decide -> run_tool -> advance_plan -> (loop? -> decide | finalize) -> proactive_decision? -> END
def build_chat_graph(
    *,
    checkpointer: Any | None = None,
):
    graph = StateGraph(ChatState)

    graph.add_node("load_context", _node_load_context)
    graph.add_node("hitl_resume", _node_handle_pending_confirmation)
    graph.add_node("decompose", _node_decompose)
    graph.add_node("decide", _node_decide)
    graph.add_node("run_tool", _node_run_tool)
    graph.add_node("advance_plan", _node_advance_plan)
    graph.add_node("finalize", _node_finalize)
    graph.add_node("proactive_decision", _node_proactive_decision)

    graph.add_edge(START, "load_context")
    graph.add_conditional_edges(
        "load_context",
        lambda s: "hitl_resume" if s.get("pending_text") else "decompose",
        {"hitl_resume": "hitl_resume", "decompose": "decompose"},
    )
    graph.add_edge("hitl_resume", "finalize")
    graph.add_edge("decompose", "decide")
    graph.add_edge("decide", "run_tool")
    graph.add_edge("run_tool", "advance_plan")
    graph.add_conditional_edges(
        "advance_plan",
        _route_after_advance,
        {"decide": "decide", "finalize": "finalize"},
    )
    
    # 在 finalize 之后增加条件分支：根据 used_tool 决定是否执行主动决策
    graph.add_conditional_edges(
        "finalize",
        lambda s: "proactive_decision" if _should_check_proactive(s) else END,
        {
            "proactive_decision": "proactive_decision",
            END: END
        }
    )
    
    graph.add_edge("proactive_decision", END)

    return graph.compile(checkpointer=checkpointer)


def _build_checkpoint_context(values: dict) -> dict:
    history = list(values.get("history") or [])
    return {
        "history": history,
        "recent_dialogue": history[-16:],
        "summary": values.get("summary"),
        "pending_text": values.get("pending_text"),
        "proposal_id": values.get("proposal_id"),
        "proposal": values.get("proposal"),
        "pending_add_confirmation": values.get("pending_add_confirmation"),
        "proposal_confirmed": values.get("proposal_confirmed"),
    }


def _build_initial_state(
    *,
    session_id: str,
    user_message: str,
    checkpoint_context: dict,
    profile_context: str | None = None,
    pre_advice: str | None = None,
    pre_steps: list[dict] | None = None,
    forced_action: str | None = None,
    forced_args: dict | None = None,
    recent_events: list[dict] | None = None,
    upcoming_todos: list[dict] | None = None,
    event_window_start: str | None = None,
    event_window_end: str | None = None,
) -> ChatState:
    history = list(checkpoint_context.get("history") or [])
    history.append({"role": "user", "content": _resolve_relative_dates(user_message)})

    steps = [{"type": "input", "user_message": user_message, "session_id": session_id}]
    if pre_steps:
        steps.extend(pre_steps)

    return {
        "session_id": session_id,
        "user_message": user_message,
        "history": history,
        "recent_dialogue": list(checkpoint_context.get("recent_dialogue") or history[-16:]),
        "summary": checkpoint_context.get("summary"),
        "pending_text": checkpoint_context.get("pending_text"),
        "pending_add_confirmation": checkpoint_context.get("pending_add_confirmation"),
        "proposal_id": checkpoint_context.get("proposal_id"),
        "proposal": checkpoint_context.get("proposal"),
        "proposal_confirmed": False,
        "profile_context": profile_context,
        "pre_advice": pre_advice,
        "effective_user_message": _build_effective_message(user_message, profile_context, pre_advice),
        "trace": {
            "meta": {"graph": "chat_v2", "started_at": datetime.now().isoformat(timespec="seconds")},
            "steps": steps,
        },
        "used_tool": None,
        "citations": [],
        "hitl_cancelled": False,
        "forced_action": forced_action,
        "forced_args": forced_args or {},
        "plan": [],
        "plan_index": 0,
        "recent_events": recent_events or [],
        "upcoming_todos": upcoming_todos or [],
        "event_window_start": event_window_start,
        "event_window_end": event_window_end,
    }


def load_chat_checkpoint_context(compiled_graph: Any, session_id: str) -> dict:
    values = getattr(compiled_graph.get_state({"configurable": {"thread_id": session_id}}), "values", None) or {}
    return _build_checkpoint_context(values) if isinstance(values, dict) else {}


async def aload_chat_checkpoint_context(compiled_graph: Any, session_id: str) -> dict:
    values = getattr((await compiled_graph.aget_state({"configurable": {"thread_id": session_id}})), "values", None) or {}
    return _build_checkpoint_context(values) if isinstance(values, dict) else {}


def run_chat_graph(
    *,
    compiled_graph: Any,
    session_id: str,
    user_message: str,
    **kwargs,
) -> ChatState:
    ctx = load_chat_checkpoint_context(compiled_graph, session_id)
    state = _build_initial_state(session_id=session_id, user_message=user_message, checkpoint_context=ctx, **kwargs)
    return compiled_graph.invoke(state, config={"configurable": {"thread_id": session_id}, "recursion_limit": 100})


async def arun_chat_graph(
    *,
    compiled_graph: Any,
    session_id: str,
    user_message: str,
    **kwargs,
) -> ChatState:
    config = {"configurable": {"thread_id": session_id}, "recursion_limit": 100}
    current_state = await compiled_graph.aget_state(config)
    if len(getattr(current_state, "tasks", []) or []) > 0:
        return await compiled_graph.ainvoke(Command(resume=user_message), config=config)

    ctx = await aload_chat_checkpoint_context(compiled_graph, session_id)
    state = _build_initial_state(session_id=session_id, user_message=user_message, checkpoint_context=ctx, **kwargs)
    return await compiled_graph.ainvoke(state, config=config)

