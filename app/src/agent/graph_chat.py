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

from ..common.config.db_config import SessionLocal
from ..common.config.base_config import settings
from ..common.config.log_config import logger
from ..common.config.llm_config import chat_completion
from ..domain.entity.chat_entity import ConversationSummary, TodoItem
from .planner.planner import detect_affirmation, detect_rejection, propose_plan
from ..util.retrieval.retrieval import rag_answer, search_chunks
from .time.time_parser import normalize_date_hint
from .memory.personal_memory import decide_proactive_advice
from ..domain.dto.memory_dto import ProactiveAdviceDecision
from ..common.md_loader import load_markdown_context
from .post_chat_learn import run_learn

# 流式推送：session_id → asyncio.Queue，图节点执行时实时推送，不入 checkpoint
_progress_queues: dict[str, Any] = {}
from ..service.calendar_service import CalendarService, CalendarServiceError
from ..service.weather_service import weather_service, WeatherServiceError
from ..util.retrieval.retrieval import get_available_knowledge_topics
from ..domain.entity.chatstate_entity import ChatState

# 范围查询待办：当用户问“未来几天安排”时使用。
def _query_todos(*, start: datetime, end: datetime) -> list[dict[str, Any]]:
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


def _resolve_memory_owner_id(session_id: str) -> str:
    scope = (settings.personal_memory_scope or "global").strip().lower()
    if scope == "session":
        return session_id
    return "global"


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
    """判断是否需要执行主动决策。HITL 流程和 quick 模式跳过。"""
    if state.get("mode") == "quick":
        return False
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


def _build_effective_message(user_message: str, profile_context: str | None) -> str:
    parts = [f"用户输入: {user_message}"]
    if profile_context:
        parts.append(f"用户画像: {profile_context}")
    return "\n".join(parts)


def _push_progress(state: ChatState, msg: str, detail: dict | None = None):
    q = _progress_queues.get(state.get("session_id", ""))
    if q is not None:
        try:
            q.put_nowait({"type": "step", "message": msg, "detail": detail or {}})
        except Exception:
            pass


def _set_progress_queue(session_id: str, q: Any):
    if q is not None:
        _progress_queues[session_id] = q


def _clear_progress_queue(session_id: str):
    _progress_queues.pop(session_id, None)


def _is_bare_confirmation_text(text: str) -> bool:
    """识别单独的确认/取消短句，避免把它们误当成新的待办意图。"""

    stripped = (text or "").strip()
    return stripped in {"需要", "要", "可以", "好的", "好", "行", "确认", "不用", "不要", "算了", "取消"}


def _node_handle_pending_confirmation(state: ChatState) -> ChatState:
    """如果上一轮留下了待确认草案，这里优先消化用户的确认/取消回复。"""
    _push_progress(state, "正在检查待确认项...")

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
    _push_progress(state, "正在加载上下文...")
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
    try:
        from ..service.goal_service import goal_service
        session_id = state["session_id"]
        memory_owner = _resolve_memory_owner_id(session_id)
        state["goals"] = goal_service.get_active_goals(memory_owner)
    except Exception as exc:
        logger.warning("load goals failed: %s", exc)
        state["goals"] = []
    _append_trace(
        state,
        _trace_step(
            step_type="graph_node",
            name="load_context",
            output={"history_len": len(history), "recent_dialogue_len": len(state["recent_dialogue"]), "has_summary": bool(state.get("summary"))},
        ),
    )
    return state


def _node_decompose(state: ChatState) -> ChatState:
    msg = (state.get("user_message") or "").strip()

    # 待办确认尚未完成，跳过 re-decompose
    if state.get("plan") and state.get("pending_add_confirmation"):
        _push_progress(state, "等待确认中，跳过任务分解...")
        return state

    # ---- 内部路由逻辑（非 LLM，快速分支） ----

    # 1) 用户正在回应待确认草案
    if state.get("pending_add_confirmation"):
        if detect_affirmation(msg):
            state["decision"] = {
                "action": "confirm_proposal", "args": {},
                "assistant_message": "",
                "trace": {"intent": "confirm_proposal_yes", "why": "用户确认待办加入"},
            }
            return state
        if detect_rejection(msg):
            state["decision"] = {
                "action": "confirm_proposal", "args": {},
                "assistant_message": "",
                "trace": {"intent": "confirm_proposal_no", "why": "用户拒绝待办加入"},
            }
            return state

    # 2) 重试：上一步 verify 不通过
    retry_info = state.get("retry_info")
    if retry_info:
        plan = state.get("plan") or []
        plan_index = state.get("plan_index", 0)
        if plan and plan_index < len(plan):
            step = plan[plan_index]
            fix_suggestion = retry_info.get("fix_suggestion", "")
            if step["action"] == "query_knowledge":
                args = step.get("args") or {}
                args["question"] = f"{args.get('question', '')}（修正：{fix_suggestion}）"
                step["args"] = args
            elif step["action"] == "query_weather":
                args = step.get("args") or {}
                if fix_suggestion and not args.get("city"):
                    args["city"] = fix_suggestion
                    step["args"] = args
            state["retry_info"] = None
            state["retry_count"] = state.get("retry_count", 0)

    # 3) 多步计划：还有未执行的步骤
    plan = state.get("plan") or []
    plan_index = state.get("plan_index", 0)
    if plan and plan_index < len(plan):
        step = plan[plan_index]
        state["decision"] = {
            "action": step["action"],
            "args": step.get("args") or {},
            "assistant_message": "",
            "trace": {"intent": "plan_step", "why": step.get("purpose", "多步分解子任务")},
        }
        _append_trace(state, _trace_step(step_type="llm", name="plan_step",
            input={"plan_index": plan_index, "step": step},
            output={"action": step["action"]}))
        return state

    # 4) 强制动作（API 指定）
    forced_action = state.get("forced_action")
    if forced_action:
        state["decision"] = {
            "action": forced_action,
            "args": state.get("forced_args") or {},
            "assistant_message": "",
            "trace": {"intent": "forced", "why": "forced by API wrapper"},
        }
        return state

    # 5) 纯确认短句但无待确认草案 → 澄清
    if not state.get("pending_text") and _is_bare_confirmation_text(msg):
        state["decision"] = {
            "action": "normal_chat",
            "args": {},
            "assistant_message": "你是说刚才提到的待办事项吗？如果是，请说'是'或'要'；如果不是，请告诉我具体想记什么。",
            "trace": {"intent": "confirmation_clarify", "why": "短句但无待确认草案，先澄清"},
        }
        return state

    # 6) 模糊查询上一条待办
    if _is_prev_todo_query(state):
        state["decision"] = {
            "action": "query_todos",
            "args": {"range_days": 7},
            "assistant_message": "",
            "trace": {"intent": "prev_todos", "why": "用户模糊短语查待办（如'上一条'/'上一个'），结合上下文推断"},
        }
        return state

    # ---- LLM 路由：集成分解 + 决策 ----
    pending_cl = state.get("pending_clarification")
    if pending_cl:
        original = pending_cl.get("original_message", "")
        user_msg = f"{original}（上一轮补充信息：{state.get('user_message', '')}）"
        state["pending_clarification"] = None
    else:
        user_msg = msg
    recent_dialogue = state.get("recent_dialogue") or []
    _push_progress(state, "正在分析意图...")

    try:
        available_topics = get_available_knowledge_topics()
    except Exception:
        available_topics = []

    system = (
        "你是一个智能决策引擎。理解用户意图，决定使用哪个工具，以及是否需要拆解成多步执行。\n\n"
        "## 需要拆解的场景\n"
        '- 用户确认/要求同时执行多个操作（如"都干""全做"）\n'
        "- 需要查询多个不同信息才能回答（天气+日程、日程+待办等）\n"
        "- 需要查信息→生成待办的多步流程\n"
        "- 缺少关键信息需反问（如查天气没提城市）\n\n"
        "## 不需要拆解的场景\n"
        "- 一个工具就能完成的请求\n"
        "- 纯聊天的请求\n\n"
        "## 工具说明\n"
        "- query_weather: 查天气\n"
        "- query_todos: 查待办事项\n"
        "- query_calendar: 查日历日程\n"
        "- query_knowledge: 查知识库\n"
        "- propose_todo: 创建待办\n"
        "- decompose_goal: 拆解长期目标\n"
        "- clarify: 反问用户补充信息\n"
        "- revise_proposal: 修改待办草案\n"
        "- normal_chat: 普通对话\n\n"
        "## 纯闲聊判断\n"
        "如果用户只是打招呼、寒暄、说心情、纯聊天，没有任何查询/操作意图：\n"
        '- action 设为 "normal_chat"\n'
        '- mode 设为 "quick"\n\n'
        "## 输出格式\n"
        "输出严格 JSON，不要 Markdown：\n"
        '{"action":"...","args":{...},"mode":"normal|quick","assistant_message":"","needs_plan":false}\n'
        '或 {"action":"...","args":{...},"mode":"normal","needs_plan":true,"steps":[{"action":"...","args":{...},"purpose":"..."}]}'
    )
    payload = {
        "user_message": user_msg,
        "recent_dialogue": recent_dialogue[-6:],
        "available_topics": available_topics or [],
    }
    raw = chat_completion(
        [{"role": "system", "content": system}, {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
        temperature=0.0,
        runtime_context={"scenario": "decompose", "session_id": state.get("session_id")},
    )
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    try:
        data = json.loads(cleaned)
        state["mode"] = data.get("mode", "normal")
        if data.get("needs_plan") and data.get("steps"):
            steps = data["steps"][:10]
            state["plan"] = steps
            state["plan_index"] = 0
            # 设置当前步骤为第一个动作
            first = steps[0]
            state["decision"] = {
                "action": first["action"],
                "args": first.get("args") or {},
                "assistant_message": data.get("assistant_message", ""),
                "trace": {"intent": "plan_step", "why": first.get("purpose", "多步分解子任务")},
            }
        else:
            state["plan"] = []
            state["plan_index"] = 0
            state["decision"] = {
                "action": data.get("action", "normal_chat"),
                "args": data.get("args") or {},
                "assistant_message": data.get("assistant_message", ""),
                "trace": {"intent": "llm_decide", "why": data.get("reason", "LLM 决策")},
            }
    except Exception as exc:
        logger.warning("decompose failed session=%s input=%s raw=%s err=%s", state.get("session_id"), user_msg[:40], cleaned, exc)
        state["plan"] = []
        state["plan_index"] = 0
        state["mode"] = "normal"
        state["decision"] = {
            "action": "normal_chat",
            "args": {},
            "assistant_message": "",
            "trace": {"intent": "fallback", "why": "decompose LLM 失败，降级为普通对话"},
        }
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
    md_context = load_markdown_context()
    if md_context:
        messages.append({"role": "system", "content": f"助手备忘录与技能：\n{md_context}"})
    if plan_context:
        messages.append({"role": "system", "content": f"前面步骤查询到的信息（请基于这些信息回答）：\n{plan_context}"})

    context_parts = []
    if state.get("profile_context"):
        context_parts.append(f"## 用户画像\n{state['profile_context']}")
    if state.get("summary"):
        summary = state["summary"]
        if not summary.startswith("##"):
            summary = f"## 对话摘要\n{summary}"
        context_parts.append(summary)

    important_events = state.get("important_events") or []
    if important_events:
        ie_lines = []
        for ev in important_events:
            title = ev.get("title", "")
            t = ev.get("event_time", "")
            cat = ev.get("category", "")
            t_str = f"（{t[:10]}）" if t else ""
            ie_lines.append(f"  [{cat}] {title}{t_str}")
        context_parts.append("## 重大事件\n" + "\n".join(ie_lines))

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
        context_parts.append("## 近期生活事件\n" + "\n".join(event_lines))

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
        context_parts.append("## 待办事项\n" + "\n".join(todo_lines))

    pending_reminders = state.get("pending_reminders") or []
    if pending_reminders:
        reminder_lines = []
        for rem in pending_reminders[:4]:
            title = rem.get("title", "")
            count = rem.get("remind_count", 0)
            reminder_lines.append(f"  {title}（已提醒{count}次）")
        context_parts.append("## 待提醒事项（AI 检测到需关注但未确认）\n" + "\n".join(reminder_lines))

    goals = state.get("goals") or []
    if goals:
        goal_lines = []
        for gd in goals[:5]:
            line = f"  {gd['title']}"
            if gd.get("sub_goals"):
                subs = [s.get("title", "") for s in gd["sub_goals"][:3]]
                line += " → " + "、".join(subs)
            if gd.get("progress_pct", 0) > 0:
                line += f" ({gd['progress_pct']:.0f}%)"
            goal_lines.append(line)
        context_parts.append("## 活跃目标\n" + "\n".join(goal_lines))

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
        _push_progress(state, "正在查询待办事项...")
        # 先尝试解析精确日期（如"后天/周三"）。
        # 能解析到单日就查单日；否则按 range_days 查区间。
        date_hint = normalize_date_hint(state["user_message"])

        if date_hint:
            target = datetime.fromisoformat(date_hint).date()
            day_start = datetime.combine(target, datetime.min.time())
            items = _query_todos(start=day_start, end=day_start + timedelta(days=1))
            query_input = {"target_date": date_hint}
        else:
            range_days = int(args.get("range_days", 7) or 7)
            now = datetime.now()
            items = _query_todos(start=now, end=now + timedelta(days=range_days))
            query_input = {"range_days": range_days}

        state["used_tool"] = "query_todos"

        if not items:
            state["answer"] = "我查了一下，这个时间点没有已安排的待办。"
        else:
            # LLM 调用点：只负责“自然语言总结”，不改动数据库原始查询结果。
            prompt = (
        "你是生活助理。用户在询问他的安排。\n"
        "下面是从数据库查到的待办列表（真实数据），请用中文做简短总结。\n"
        "## 要求\n"
        "不要编造列表中没有的事项；如果时间为空就不要猜。"
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
        _push_progress(state, "正在搜索知识库...")
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
        city_arg = (args.get("city") or args.get("location") or "").strip() if isinstance(args, dict) else ""
        _push_progress(state, f"正在查询{city_arg or '指定城市'}天气...")
        profile_ctx = state.get("profile_context") or ""
        query_input: dict[str, Any] = {"city": city_arg or None, "has_profile_city": bool(profile_ctx)}
        try:
            result = weather_service.query(city_arg, profile_context=profile_ctx)
            query_input["resolved_city"] = result.get("city")
            query_input["cached"] = bool(result.get("cached"))
            state["used_tool"] = "query_weather"

            forecast_dates = [d.get("date", "") for d in (result.get("forecast") or [])]
            user_question = state.get("user_message", "")
            prompt = (
                "你是生活助理。用户在询问天气。\n"
                f"用户的问题是：{user_question}\n"
                f"以下是该城市的天气数据，数据覆盖日期：{', '.join(forecast_dates) or '无预报数据'}。\n\n"
                "## 回复要求\n"
                "请用中文生成简短、自然的回复：\n"
                "- 先说实况（当前温度/天气/湿度/风）\n"
                "- 再提示未来几天的趋势与出行建议（带伞/穿衣）\n"
                "- 如果用户问的日期不在数据覆盖范围内，请明确说明该日期暂时无法预报。\n"
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
        _push_progress(state, "正在查询日程安排...")
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
                    "## 总结要求\n"
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
        items = proposal_payload.get("items") or []
        if len(items) > 1:
            lines = ["已生成以下待办草案，请确认是否全部加入："]
            for i, it in enumerate(items, 1):
                title = it.get("title", "")
                due = it.get("due_at", "")
                due_str = f"（{due[:10]}）" if due else ""
                lines.append(f"{i}. {title}{due_str}")
            state["answer"] = "\n".join(lines)
        else:
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

    if action == "decompose_goal":
        text = args.get("text") or state["user_message"]
        _push_progress(state, "正在拆解目标...")
        try:
            from ..agent.memory.personal_memory import extract_goal
            from ..service.goal_service import goal_service
            session_id = state["session_id"]
            memory_owner_id = _resolve_memory_owner_id(session_id)
            extraction = extract_goal(
                user_message=text,
                recent_dialogue=state.get("recent_dialogue") or [],
                now_iso=datetime.now().isoformat(timespec="seconds"),
                session_id=session_id,
            )
            if extraction and extraction.items:
                saved = []
                for g in extraction.items:
                    goal_dict = goal_service.save_goal(
                        session_id=memory_owner_id,
                        title=g.title,
                        category=g.category,
                        target_date=g.target_date,
                        sub_goals=[s.model_dump() for s in g.sub_goals] if g.sub_goals else None,
                        milestones=[m.model_dump() for m in g.milestones] if g.milestones else None,
                        notes=g.notes,
                    )
                    saved.append(goal_dict)
                state["used_tool"] = "decompose_goal"
                summary = "已为你拆解了以下目标：\n"
                for gd in saved:
                    summary += f"\n• {gd['title']}"
                    if gd.get("sub_goals"):
                        for s in gd["sub_goals"][:5]:
                            summary += f"\n  - {s.get('title', '')}"
                state["answer"] = summary
            else:
                state["used_tool"] = "decompose_goal"
                state["answer"] = "我暂时没有从你的话中识别出明确的目标。你可以更具体地描述你希望达成什么。"
        except Exception as exc:
            logger.warning("decompose_goal failed: %s", exc)
            state["used_tool"] = "decompose_goal"
            state["answer"] = "目标拆解时出了点问题，请稍后再试试。"
        _append_trace(state, _trace_step(step_type="tool", name="decompose_goal", input={"text": text},
            output={"answer": state.get("answer", "")}))
        return state

    if action == "clarify":
        assistant_message = (decision.get("assistant_message") or "").strip()
        if not assistant_message:
            state["answer"] = "请提供更多信息，我才能帮你处理。"
        else:
            state["answer"] = assistant_message
        state["used_tool"] = "clarify"
        state["pending_clarification"] = {
            "original_message": state.get("user_message", ""),
            "asked_question": state["answer"],
        }
        _append_trace(state, _trace_step(step_type="tool", name="clarify",
            input={"question": assistant_message},
            output={"answer": state["answer"]}))
        return state

    if action == "revise_proposal":
        correction = args.get("correction") or state["user_message"]
        _push_progress(state, "正在根据修正重新生成...")
        try:
            proposal_id, proposal = propose_plan(correction)
            proposal_payload = proposal.model_dump()
            items = proposal_payload.get("items") or []
            if items:
                state["pending_add_confirmation"] = True
                state["proposal_id"] = proposal_id
                state["proposal"] = proposal_payload
                state["used_tool"] = "revise_proposal"
                state["answer"] = "已根据你的修正重新生成待办草案：\n" + "\n".join(
                    f"  {i+1}. {it.get('title', '')}" + (f"（{it.get('due_at','')[:10]}）" if it.get('due_at') else "")
                    for i, it in enumerate(items)
                ) + "\n请确认是否加入待办。"
            else:
                state["used_tool"] = "revise_proposal"
                state["answer"] = "已按你的修正更新。"
        except Exception as exc:
            logger.warning("revise_proposal failed: %s", exc)
            state["used_tool"] = "revise_proposal"
            state["answer"] = "修改时出了点问题，请再描述一次。"
        _append_trace(state, _trace_step(step_type="tool", name="revise_proposal",
            input={"correction": correction},
            output={"answer": state.get("answer", "")}))
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
    _push_progress(state, "正在生成回答...")
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


def _push_upcoming_todos(state: ChatState):
    """待办即将到期（1小时内）时推送通知。"""
    upcoming = state.get("upcoming_todos") or []
    now = datetime.now()
    threshold = now + timedelta(hours=1)
    imminent = [
        t for t in upcoming
        if t.get("due_at") and now <= datetime.fromisoformat(t["due_at"].replace("Z", "+00:00").split("+")[0]) <= threshold
    ]
    if not imminent:
        return
    try:
        from ..service.push_service import push_notification
        for todo in imminent[:3]:
            title = todo.get("title", "待办提醒")
            due = todo.get("due_at", "")[11:16] if todo.get("due_at") else ""
            push_notification(f"⏰ {title}", f"到期时间：{due}")
    except Exception as exc:
        logger.warning("push_service._push_upcoming_todos failed: %s", exc)


def _push_pending_reminders(state: ChatState):
    """DB 中到期的提醒也推送通知。"""
    reminders = state.get("pending_reminders") or []
    if not reminders:
        return
    try:
        from ..service.push_service import push_reminder_due
        for r in reminders[:3]:
            push_reminder_due(r.get("content", ""), r.get("remind_count", 0))
    except Exception as exc:
        logger.warning("push_service._push_pending_reminders failed: %s", exc)


# 主动决策节点：基于近期事件/待办/画像，决定是否追加主动建议。
def _node_proactive_decision(state: ChatState) -> ChatState:
    _push_progress(state, "正在分析是否需要主动建议...")
    recent_events = state.get("recent_events", [])
    upcoming_todos = state.get("upcoming_todos", [])
    pending_reminders = state.get("pending_reminders", [])
    profile_context = state.get("profile_context") or ""

    if not recent_events and not upcoming_todos:
        return state

    now_iso = datetime.now().isoformat(timespec="seconds")
    try:
        decision = decide_proactive_advice(
            user_message=state["user_message"],
            assistant_answer=state.get("answer", ""),
            recent_events=recent_events,
            upcoming_todos=upcoming_todos,
            pending_reminders=pending_reminders,
            profile_context=profile_context,
            goals=state.get("goals"),
            now_iso=now_iso,
            threshold=settings.proactive_advice_threshold,
            session_id=state["session_id"],
        )
    except Exception as exc:
        logger.warning("proactive advice decision LLM failed, skipping: %s", exc)
        decision = ProactiveAdviceDecision(score=0.0, should_add=False, advice=None, reasons=["LLM调用失败，跳过"])

    advice_added = False
    if decision.should_add and decision.advice:
        current_answer = state.get("answer", "")
        if decision.advice not in current_answer:
            state["answer"] = f"{current_answer}\n\n补充建议：{decision.advice}".strip()
            advice_added = True

    # 主动推送：有高价值建议且非对话中已追加时，微信推送
    if decision.should_add and decision.advice and not advice_added and decision.score >= settings.proactive_advice_threshold + 0.2:
        try:
            from ..service.push_service import push_proactive_advice
            push_proactive_advice(decision.advice, decision.reasons)
        except Exception as exc:
            logger.warning("push_service.push_proactive_advice failed: %s", exc)

    # 主动推送：待办即将到期（1小时内）
    _push_upcoming_todos(state)

    # 主动推送：提醒到期
    _push_pending_reminders(state)

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
                "upcoming_todos_total": len(upcoming_todos),
                "pending_reminders": len(pending_reminders),
                "now_iso": now_iso,
                "event_window_start": state.get("event_window_start"),
                "event_window_end": state.get("event_window_end"),
            }
        )
    )

    logger.info(
        "proactive advice decision score=%.3f threshold=%.3f added=%s",
        decision.score,
        settings.proactive_advice_threshold,
        advice_added,
    )

    return state


_VERIFYABLE_ACTIONS = {"query_todos", "query_knowledge", "query_weather", "query_calendar", "normal_chat"}

_MAX_RETRIES = 2


def _node_verify(state: ChatState) -> ChatState:
    """审查单步工具执行结果，不通过时标记 retry。"""
    action = (state.get("decision") or {}).get("action", "")
    if action not in _VERIFYABLE_ACTIONS:
        return state

    answer = (state.get("answer") or "").strip()
    if not answer:
        return state

    _push_progress(state, "正在审查执行结果...")
    user_msg = state.get("user_message", "")
    step_purpose = ""
    plan = state.get("plan") or []
    plan_index = state.get("plan_index", 0)
    if plan and plan_index < len(plan):
        step_purpose = plan[plan_index].get("purpose", action)

    system = (
        "你是执行质量审查员。判断工具执行结果是否满足用户需求。\n"
        "## 输出格式\n"
        "输出严格 JSON：\n"
        '{"passed": true|false, "issues": ["问题描述"], "fix_suggestion": "如何修复"}'
    )
    payload = {
        "user_request": user_msg,
        "step_action": action,
        "step_purpose": step_purpose,
        "step_result": answer[:800],
        "rules": [
            "结果不能为空或仅含错误信息",
            "结果应该与用户请求和步骤目的相关",
        ],
    }
    try:
        text = chat_completion(
            [{"role": "system", "content": system}, {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
            temperature=0.0,
            runtime_context={"scenario": "verify_step", "session_id": state["session_id"]},
        ).strip()
        data = json.loads(text)
        passed = data.get("passed", True)
        if not passed:
            issues = data.get("issues", [])
            suggestion = data.get("fix_suggestion", "")
            retry_count = state.get("retry_count", 0)
            state["retry_count"] = retry_count + 1
            if retry_count < _MAX_RETRIES:
                state["retry_info"] = {
                    "issues": issues,
                    "fix_suggestion": suggestion,
                    "step_index": plan_index,
                }
                logger.info("verify failed for %s (retry %d/%d): %s", action, retry_count + 1, _MAX_RETRIES, issues)
            else:
                logger.warning("verify failed for %s after %d retries, skipping", action, _MAX_RETRIES)
                state["retry_info"] = None
                state["answer"] = answer + "\n\n（此步骤执行未达预期，已跳过）"
    except Exception as exc:
        logger.debug("verify check skipped for %s: %s", action, exc)
    return state


def _node_quality_gate(state: ChatState) -> ChatState:
    """最终回答质量审查：quick 模式跳过。"""
    if state.get("mode") == "quick":
        return state
    _push_progress(state, "正在检查最终回答质量...")
    answer = (state.get("answer") or "").strip()
    if not answer:
        return state

    user_msg = state.get("user_message", "")
    system = (
        "你是回答质量评审员。检查最终回答是否完整、准确地回应了用户。\n"
        "## 输出格式\n"
        "输出严格 JSON：\n"
        '{"passed": true|false, "missing": ["遗漏点"], "fix": "修正后的完整回答"}'
    )
    payload = {
        "user_request": user_msg,
        "assistant_answer": answer,
    }
    try:
        text = chat_completion(
            [{"role": "system", "content": system}, {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
            temperature=0.0,
            runtime_context={"scenario": "quality_gate", "session_id": state["session_id"]},
        ).strip()
        data = json.loads(text)
        passed = data.get("passed", True)
        if not passed:
            fix = data.get("fix", "").strip()
            if fix:
                state["answer"] = fix
                logger.info("quality gate fixed answer (was: %s...)", answer[:60])
    except Exception as exc:
        logger.debug("quality gate skipped: %s", exc)
    return state


# 收口节点：保证每条路径都有 answer，并记录最终 used_tool / has_proposal。
def _node_finalize(state: ChatState) -> ChatState:
    _push_progress(state, "正在汇总最终回答...")
    # 多步计划：把各 step 结果汇总成一个连贯回答。
    plan = state.get("plan") or []
    completed_steps = [s for s in plan if s.get("result")]
    if len(completed_steps) >= 2:
        parts = []
        for i, step in enumerate(completed_steps):
            result = (step.get("result") or "").strip()
            if result:
                parts.append(f"【{step.get('purpose', f'步骤{i+1}')}】\n{result}")
        if parts:
            combined = "\n\n".join(parts)
            try:
                summary = chat_completion(
                    [
                        {"role": "system", "content": "你是生活助理。请把下面几个步骤的结果合并成一段连贯、自然的中文回复。\n## 要求\n按主次顺序排列信息，不要编造。输出纯文本。"},
                        {"role": "user", "content": combined},
                    ],
                    temperature=0.2,
                )
                state["answer"] = summary.strip()
            except Exception:
                state["answer"] = combined

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

    # 对话后学习：quick 模式跳过
    if state.get("mode") != "quick":
        try:
            run_learn(
                user_message=state.get("user_message", ""),
                assistant_answer=state.get("answer", ""),
                session_id=state.get("session_id"),
            )
        except Exception as exc:
            logger.debug("post_chat_learn failed: %s", exc)

    return state


# -----------------------------
# 图构建与执行
# -----------------------------

def _node_advance_plan(state: ChatState) -> ChatState:
    # HITL 绕路（confirm_proposal）不消耗计划步骤
    action = (state.get("decision") or {}).get("action", "")
    if action == "confirm_proposal":
        return state

    plan = state.get("plan") or []
    plan_index = state.get("plan_index", 0)
    if plan and plan_index < len(plan):
        plan[plan_index]["result"] = state.get("answer", "") or ""
        plan[plan_index]["used_tool"] = state.get("used_tool")
        return {**state, "plan_index": plan_index + 1}
    return state


def _route_after_advance(state: ChatState) -> str:
    if state.get("pending_add_confirmation"):
        return "finalize"
    if state.get("retry_info") and state.get("retry_count", 0) <= _MAX_RETRIES:
        return "decompose"
    plan = state.get("plan") or []
    plan_index = state.get("plan_index", 0)
    if plan and plan_index < len(plan):
        return "decompose"
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
    graph.add_node("run_tool", _node_run_tool)
    graph.add_node("verify", _node_verify)
    graph.add_node("advance_plan", _node_advance_plan)
    graph.add_node("finalize", _node_finalize)
    graph.add_node("quality_gate", _node_quality_gate)
    graph.add_node("proactive_decision", _node_proactive_decision)

    graph.add_edge(START, "load_context")
    graph.add_conditional_edges(
        "load_context",
        lambda s: "hitl_resume" if s.get("pending_text") else "decompose",
        {"hitl_resume": "hitl_resume", "decompose": "decompose"},
    )
    graph.add_edge("hitl_resume", "finalize")
    graph.add_edge("decompose", "run_tool")
    graph.add_edge("run_tool", "verify")
    graph.add_conditional_edges(
        "verify",
        lambda s: "advance_plan" if not s.get("retry_info") else "decompose",
        {"advance_plan": "advance_plan", "decompose": "decompose"},
    )
    graph.add_conditional_edges(
        "advance_plan",
        _route_after_advance,
        {"decompose": "decompose", "finalize": "finalize"},
    )
    
    # finalize → quality_gate → proactive_decision → END
    graph.add_edge("finalize", "quality_gate")
    graph.add_conditional_edges(
        "quality_gate",
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
        "pending_reminders": values.get("pending_reminders", []),
        "pending_clarification": values.get("pending_clarification"),
    }


def _build_initial_state(
    *,
    session_id: str,
    user_message: str,
    checkpoint_context: dict,
    profile_context: str | None = None,
    pre_steps: list[dict] | None = None,
    forced_action: str | None = None,
    forced_args: dict | None = None,
    recent_events: list[dict] | None = None,
    important_events: list[dict] | None = None,
    upcoming_todos: list[dict] | None = None,
    pending_reminders: list[dict] | None = None,
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
        "pending_clarification": checkpoint_context.get("pending_clarification"),
        "proposal_id": checkpoint_context.get("proposal_id"),
        "proposal": checkpoint_context.get("proposal"),
        "proposal_confirmed": False,
        "profile_context": profile_context,
        "effective_user_message": _build_effective_message(user_message, profile_context),
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
        "retry_info": None,
        "retry_count": 0,
        "mode": "normal",
        "recent_events": recent_events or [],
        "important_events": important_events or [],
        "upcoming_todos": upcoming_todos or [],
        "pending_reminders": pending_reminders or [],
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
    progress_queue: Any = None,
    **kwargs,
) -> ChatState:
    _set_progress_queue(session_id, progress_queue)
    try:
        ctx = load_chat_checkpoint_context(compiled_graph, session_id)
        state = _build_initial_state(session_id=session_id, user_message=user_message, checkpoint_context=ctx, **kwargs)
        return compiled_graph.invoke(state, config={"configurable": {"thread_id": session_id}, "recursion_limit": 100})
    finally:
        _clear_progress_queue(session_id)


async def arun_chat_graph(
    *,
    compiled_graph: Any,
    session_id: str,
    user_message: str,
    progress_queue: Any = None,
    **kwargs,
) -> ChatState:
    _set_progress_queue(session_id, progress_queue)
    try:
        config = {"configurable": {"thread_id": session_id}, "recursion_limit": 100}
        current_state = await compiled_graph.aget_state(config)
        if len(getattr(current_state, "tasks", []) or []) > 0:
            return await compiled_graph.ainvoke(Command(resume=user_message), config=config)

        ctx = await aload_chat_checkpoint_context(compiled_graph, session_id)
        state = _build_initial_state(session_id=session_id, user_message=user_message, checkpoint_context=ctx, **kwargs)
        return await compiled_graph.ainvoke(state, config=config)
    finally:
        _clear_progress_queue(session_id)

