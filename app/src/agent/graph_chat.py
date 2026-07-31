"""基于 LangGraph 的聊天状态机编排。

Chat -> (LLM decide) -> Tool -> Update state -> (maybe HITL) -> Final answer -> (proactive advice)
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from sqlalchemy import select

from ..common.config.db_config import SessionLocal
from ..common.config.base_config import settings
from ..common.config.log_config import logger

# LLM 每次可见的最近对话条数，必须与 chat_prune_target 一致以避免盲区
RECENT_DIALOGUE_LIMIT = 50
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
    return session_id


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


def _push_delta(state: ChatState, text: str):
    q = _progress_queues.get(state.get("session_id", ""))
    if q is not None:
        try:
            q.put_nowait({"type": "delta", "delta": text})
        except Exception:
            pass


def _make_on_token(state: ChatState):
    q = _progress_queues.get(state.get("session_id", ""))
    if q is None:
        return None
    def on_token(token: str):
        try:
            q.put_nowait({"type": "delta", "delta": token})
        except Exception:
            pass
    return on_token


def _is_bare_confirmation_text(text: str) -> bool:
    stripped = (text or "").strip()
    return stripped in {"需要", "要", "可以", "好的", "好", "行", "确认", "不用", "不要", "算了", "取消"}


# -----------------------------
# LangGraph 节点
# -----------------------------

# 第一个节点：加载上下文。
# 顺序上它永远先于 hitl_check/decide，因为后续节点都可能依赖 checkpoint 里恢复出来的 history/summary。
def _node_load_context(state: ChatState) -> ChatState:
    _push_progress(state, "正在加载上下文...")
    history = list(state.get("history") or [])
    state["history"] = history
    state["recent_dialogue"] = history[-RECENT_DIALOGUE_LIMIT:]
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

    # ---- 内部路由逻辑（非 LLM，快速分支） ----

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
            state["retry_count"] = 0  # 重置，让同一步骤可以重新尝试
            state["decision"] = {
                "action": step["action"],
                "args": step.get("args") or {},
                "assistant_message": "",
                "trace": {"intent": "plan_retry", "why": f"重试第{plan_index+1}步：{step.get('purpose', step['action'])}"},
            }
            return state
        # quick 模式无 plan：直接沿用上一次 decision
        prev = state.get("decision") or {}
        state["retry_info"] = None
        state["retry_count"] = 0
        state["decision"] = {
            "action": prev.get("action", "normal_chat"),
            "args": prev.get("args") or {},
            "assistant_message": "",
            "trace": {"intent": "quick_retry", "why": "快速模式重试"},
        }
        return state

    # 3) 全局循环上限检查
    cycle_count = state.setdefault("cycle_count", 0) + 1
    state["cycle_count"] = cycle_count
    if cycle_count > _MAX_CYCLES:
        logger.warning("cycle_count %d exceeded _MAX_CYCLES, forcing normal_chat", cycle_count)
        state["plan"] = []
        state["plan_index"] = 0
        state["mode"] = "normal"
        state["decision"] = {
            "action": "normal_chat",
            "args": {},
            "assistant_message": "我已经查了几次，暂时无法获取到所需信息。请你再具体说说？",
            "trace": {"intent": "cycle_guard", "why": "超过最大循环次数，降级为普通对话"},
        }
        return state

    # 4) 多步计划：还有未执行的步骤
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
        "- 缺少关键信息需反问（如查天气没提城市）\n"
        "- 用户身份/角色信息不明确时反问（如不确定是实习生还是正式员工）\n\n"
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
        "- clarify: 反问用户补充信息（身份/角色/缺少关键参数时使用）\n"
        "- revise_proposal: 修改待办草案\n"
        "- record_expense: 记录收支\n"
        "- query_expenses: 查询账目\n"
        "- query_accounts: 查询账户余额\n"
        "- create_account: 创建账户\n"
        "- ask_user_confirm_balance: 让用户确认更新余额\n"
        "- query_database: 执行只读 SQL 查询\n"
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
        on_token=_make_on_token(state),
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

    messages: list[dict[str, str]] = [{"role": "system", "content": "你是一个中文生活助手。请在回答时严格区分实时查询结果与历史记录。"}]
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
            summary = f"## 对话摘要（历史记录，仅供了解用户背景，不代表当前状态）\n{summary}"
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

    session_id = state.get("session_id", "")
    q = _progress_queues.get(session_id)

    def on_token(token: str):
        if q is not None:
            try:
                q.put_nowait({"type": "delta", "delta": token})
            except Exception:
                pass

    result = chat_completion(
        messages,
        on_token=on_token if q else None,
        runtime_context={"scenario": "normal_chat", "session_id": session_id},
    )
    return result, False


def _handle_interrupt_resume(state: ChatState, user_answer: str) -> ChatState:
    action = (state.get("decision") or {}).get("action", "")
    if action in ("propose_todo", "ask_user_confirm_proposal", "revise_proposal"):
        if detect_affirmation(user_answer):
            proposal = state.get("proposal") or {}
            items = proposal.get("items") or []
            if items:
                from ..domain.entity.chat_entity import TodoItem
                from ..common.config.db_config import SessionLocal
                session_id = state.get("session_id", "")
                with SessionLocal() as db:
                    for it in items:
                        raw_due = it.get("due_at") or it.get("due_date")
                        due_at = None
                        if raw_due:
                            try:
                                due_at = datetime.fromisoformat(raw_due)
                            except (ValueError, TypeError):
                                due_at = None
                        db.add(TodoItem(
                            session_id=session_id,
                            title=it.get("title", ""),
                            due_at=due_at,
                        ))
                    db.commit()
            state["proposal_confirmed"] = True
            state["proposal_id"] = None
            state["proposal"] = None
            state["answer"] = "好的，已添加到待办。"
            _push_delta(state, state["answer"])
            state["used_tool"] = "plan_gate_confirm"
        elif detect_rejection(user_answer):
            state["proposal_id"] = None
            state["proposal"] = None
            state["answer"] = "好的，已取消。"
            _push_delta(state, state["answer"])
            state["used_tool"] = "plan_gate_cancel"
        else:
            state["proposal_id"] = None
            state["proposal"] = None
            state["answer"] = "请回复'确认'或'取消'。"
            _push_delta(state, state["answer"])
            state["used_tool"] = "plan_gate_wait"
        _append_trace(
            state,
            _trace_step(
                step_type="tool",
                name="confirm_proposal",
                input={"user_answer": user_answer},
                output={"confirmed": state.get("proposal_confirmed", False)},
            ),
        )
        return state

    if action == "ask_user_confirm_balance":
        pending = state.get("pending_balance_update")
        if detect_affirmation(user_answer) and pending:
            try:
                from ..service.bookkeeping_service import bookkeeping_service
                result = bookkeeping_service.set_account_balance(
                    account_id=pending["id"], balance=pending["new_balance"], session_id=state["session_id"]
                )
                state["used_tool"] = "ask_user_confirm_balance"
                state["answer"] = f"已更新账户「{result['name']}」余额为 {result['balance']:.2f}元。"
                _push_delta(state, state["answer"])
            except Exception as exc:
                logger.warning("confirm_balance failed: %s", exc)
                state["used_tool"] = "ask_user_confirm_balance"
                state["answer"] = f"更新余额时出了点问题：{exc}"
                _push_delta(state, state["answer"])
        else:
            state["used_tool"] = "ask_user_confirm_balance"
            name = pending.get("name", "") if pending else ""
            current = pending.get("current_balance", 0) if pending else 0
            state["answer"] = f"已保留账户「{name}」当前余额 {current:.2f}元。"
            _push_delta(state, state["answer"])
        state["pending_balance_update"] = None
        return state

    state["answer"] = user_answer
    state["used_tool"] = "interrupt_resume"
    _push_delta(state, state["answer"])
    return state


def _node_run_tool(state: ChatState) -> ChatState:
    # ---- 打断恢复：处理用户回答 ----
    interrupt_answer = state.get("interrupt_answer")
    if interrupt_answer is not None:
        state["interrupt_answer"] = None
        return _handle_interrupt_resume(state, interrupt_answer)

    decision = state.get("decision") or {"action": "normal_chat", "args": {}, "assistant_message": ""}
    action = decision.get("action", "normal_chat")
    args = decision.get("args") or {}

    if action == "clarify":
        question = (decision.get("assistant_message") or "").strip()
        if not question:
            question = "请提供更多信息，我才能帮你处理。"
        _push_delta(state, question)
        state["interrupt_message"] = question
        state["used_tool"] = "clarify"
        state["plan"] = []
        state["plan_index"] = 0
        return state

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
            _push_delta(state, state["answer"])
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
        citations = search_chunks(search_q, top_k=top_k, session_id=state.get("session_id", ""))
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
            on_token=_make_on_token(state),
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
                on_token=_make_on_token(state),
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
            _push_delta(state, state["answer"])
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
            _push_delta(state, state["answer"])
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
                _push_delta(state, state["answer"])
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
                    on_token=_make_on_token(state),
                    runtime_context={"scenario": "calendar_summarize", "session_id": state.get("session_id", "")},
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
            _push_delta(state, state["answer"])
            _append_trace(state, _trace_step(step_type="tool", name="query_calendar", input=query_input, error=str(exc)))
        return state

    if action in ("ask_user_confirm_proposal", "propose_todo"):
        title = args.get("title")
        pending_text = args.get("text") or state["user_message"]
        if title:
            from uuid import uuid4
            proposal_id = str(uuid4())
            due_date = args.get("due_date") or args.get("date")
            proposal_payload = {"items": [{"title": title, "due_at": due_date}]}
        else:
            proposal_id, proposal = propose_plan(pending_text)
            proposal_payload = proposal.model_dump()

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
            confirm_text = "\n".join(lines)
        else:
            confirm_text = "已生成待办草案，请确认是否加入待办。"

        _push_delta(state, confirm_text)
        _append_trace(
            state,
            _trace_step(
                step_type="tool",
                name="propose_todo",
                input={"text": pending_text},
                output={"proposal_id": proposal_id},
            ),
        )

        state["interrupt_message"] = confirm_text
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
                _push_delta(state, state["answer"])
            else:
                state["used_tool"] = "decompose_goal"
                state["answer"] = "我暂时没有从你的话中识别出明确的目标。你可以更具体地描述你希望达成什么。"
                _push_delta(state, state["answer"])
        except Exception as exc:
            logger.warning("decompose_goal failed: %s", exc)
            state["used_tool"] = "decompose_goal"
            state["answer"] = "目标拆解时出了点问题，请稍后再试试。"
            _push_delta(state, state["answer"])
        _append_trace(state, _trace_step(step_type="tool", name="decompose_goal", input={"text": text},
            output={"answer": state.get("answer", "")}))
        return state

    if action == "revise_proposal":
        correction = args.get("correction") or state["user_message"]
        _push_progress(state, "正在根据修正重新生成...")
        try:
            proposal_id, proposal = propose_plan(correction)
            proposal_payload = proposal.model_dump()
            items = proposal_payload.get("items") or []
            if items:
                state["proposal_id"] = proposal_id
                state["proposal"] = proposal_payload
                state["used_tool"] = "revise_proposal"
                confirm_text = "已根据你的修正重新生成待办草案：\n" + "\n".join(
                    f"  {i+1}. {it.get('title', '')}" + (f"（{it.get('due_at','')[:10]}）" if it.get('due_at') else "")
                    for i, it in enumerate(items)
                ) + "\n请确认是否加入待办。"
                _push_delta(state, confirm_text)

                state["interrupt_message"] = confirm_text
                return state
            else:
                state["used_tool"] = "revise_proposal"
                state["answer"] = "已按你的修正更新。"
                _push_delta(state, state["answer"])
        except Exception as exc:
            logger.warning("revise_proposal failed: %s", exc)
            state["used_tool"] = "revise_proposal"
            state["answer"] = "修改时出了点问题，请再描述一次。"
            _push_delta(state, state["answer"])
        _append_trace(state, _trace_step(step_type="tool", name="revise_proposal",
            input={"correction": correction},
            output={"answer": state.get("answer", "")}))
        return state

    if action == "record_expense":
        _push_progress(state, "正在记录账目...")
        try:
            from ..service.bookkeeping_service import bookkeeping_service
            session_id = state["session_id"]
            amount = float(args.get("amount", 0) or 0)
            tx_type = (args.get("transaction_type") or "expense").strip().lower()
            if tx_type == "expense":
                amount = -abs(amount)
            else:
                amount = abs(amount)
            category = (args.get("category") or "").strip() or None
            desc = (args.get("description") or "").strip() or None
            acct_name = (args.get("account_name") or "").strip() or None

            result = bookkeeping_service.record_transaction(
                amount=amount,
                session_id=session_id,
                account_name=acct_name,
                category=category,
                description=desc,
            )
            state["used_tool"] = "record_expense"
            type_label = "支出" if amount < 0 else "收入"
            answer_parts = [f"已记录{type_label}：{abs(amount):.2f}元"]
            if category:
                answer_parts.append(f"分类：{category}")
            if desc:
                answer_parts.append(f"用途：{desc}")
            if result.get("account_name"):
                answer_parts.append(f"账户：{result['account_name']}")
            state["answer"] = "，".join(answer_parts)
            _push_delta(state, state["answer"])
            _append_trace(state, _trace_step(step_type="tool", name="record_expense",
                input={"amount": amount, "category": category, "description": desc, "account_name": acct_name},
                output=result))
        except Exception as exc:
            logger.warning("record_expense failed: %s", exc)
            state["used_tool"] = "record_expense"
            state["answer"] = f"记录时出了点问题：{exc}"
            _push_delta(state, state["answer"])
        return state

    if action == "query_expenses":
        _push_progress(state, "正在查询账目...")
        try:
            from ..service.bookkeeping_service import bookkeeping_service
            session_id = state["session_id"]
            start = (args.get("start_date") or "").strip() or None
            end = (args.get("end_date") or "").strip() or None
            category = (args.get("category") or "").strip() or None
            group_by = (args.get("group_by") or "category").strip()

            analysis = bookkeeping_service.analyze(
                session_id=session_id,
                start_date=start,
                end_date=end,
                group_by=group_by,
            )
            state["used_tool"] = "query_expenses"
            lines = [f"共 {analysis['transaction_count']} 笔交易"]
            lines.append(f"总支出：{analysis['total_expense']:.2f}元")
            lines.append(f"总收入：{analysis['total_income']:.2f}元")
            lines.append(f"结余：{analysis['net']:.2f}元")
            if analysis["by_category"]:
                lines.append("\n按分类：")
                for c in analysis["by_category"]:
                    label = "支出" if c["total"] < 0 else "收入"
                    lines.append(f"  {c['category']}：{abs(c['total']):.2f}元（{c['count']}笔）")
            state["answer"] = "\n".join(lines)
            _push_delta(state, state["answer"])
            _append_trace(state, _trace_step(step_type="tool", name="query_expenses",
                input={"start_date": start, "end_date": end, "category": category, "group_by": group_by},
                output=analysis))
        except Exception as exc:
            logger.warning("query_expenses failed: %s", exc)
            state["used_tool"] = "query_expenses"
            state["answer"] = f"查询账目时出了点问题：{exc}"
            _push_delta(state, state["answer"])
        return state

    if action == "query_accounts":
        _push_progress(state, "正在查询账户余额...")
        try:
            from ..service.bookkeeping_service import bookkeeping_service
            session_id = state["session_id"]
            accounts = bookkeeping_service.list_accounts(session_id=session_id)
            state["used_tool"] = "query_accounts"
            if not accounts:
                state["answer"] = "你还没有创建任何账户。可以告诉我新建一个，比如'我的微信账户有500元'。"
            else:
                lines = ["你的账户余额如下："]
                total = 0.0
                for a in accounts:
                    lines.append(f"  {a['name']}：{a['balance']:.2f}元")
                    total += a["balance"]
                lines.append(f"合计：{total:.2f}元")
                state["answer"] = "\n".join(lines)
            _push_delta(state, state["answer"])
            _append_trace(state, _trace_step(step_type="tool", name="query_accounts",
                input={}, output={"count": len(accounts)}))
        except Exception as exc:
            logger.warning("query_accounts failed: %s", exc)
            state["used_tool"] = "query_accounts"
            state["answer"] = f"查询账户时出了点问题：{exc}"
            _push_delta(state, state["answer"])
        return state

    if action == "create_account":
        _push_progress(state, "正在处理账户...")
        try:
            from ..service.bookkeeping_service import bookkeeping_service
            session_id = state["session_id"]
            name = (args.get("name") or "").strip()
            balance = float(args.get("balance", 0) or 0)
            if not name:
                state["used_tool"] = "create_account"
                state["answer"] = "请告诉我账户名称，比如'微信''支付宝'。"
                _push_delta(state, state["answer"])
                return state

            existing = bookkeeping_service.find_account(name=name, session_id=session_id)
            if existing is None:
                result = bookkeeping_service.add_account(name=name, balance=balance, session_id=session_id)
                state["used_tool"] = "create_account"
                state["answer"] = f"已创建账户「{result['name']}」，初始余额 {result['balance']:.2f}元。"
                _push_delta(state, state["answer"])
                _append_trace(state, _trace_step(step_type="tool", name="create_account",
                    input={"name": name, "balance": balance}, output=result))
                return state

            current_balance = existing["balance"]
            if abs(current_balance - balance) < 0.01:
                state["used_tool"] = "create_account"
                state["answer"] = f"账户「{name}」的余额已经是 {current_balance:.2f}元，无需更新。"
                _push_delta(state, state["answer"])
                _append_trace(state, _trace_step(step_type="tool", name="create_account",
                    input={"name": name, "balance": balance}, output={"existing": existing}))
                return state

            state["pending_balance_update"] = {
                "id": existing["id"], "name": name,
                "current_balance": current_balance, "new_balance": balance
            }
            state["used_tool"] = "create_account"
            state["answer"] = f"账户「{name}」当前余额是 {current_balance:.2f}元，用户想要 {balance:.2f}元。请使用 ask_user_confirm_balance 工具让用户确认是否更新。"
            _push_delta(state, state["answer"])
        except Exception as exc:
            logger.warning("create_account failed: %s", exc)
            state["used_tool"] = "create_account"
            state["answer"] = f"创建账户时出了点问题：{exc}"
            _push_delta(state, state["answer"])
        return state

    if action == "ask_user_confirm_balance":
        _push_progress(state, "正在等待用户确认...")
        pending = state.get("pending_balance_update")
        if not pending:
            state["used_tool"] = "ask_user_confirm_balance"
            state["answer"] = "没有待确认的余额更新请求。"
            _push_delta(state, state["answer"])
            return state
        try:
            name = pending["name"]
            current = pending["current_balance"]
            new_bal = pending["new_balance"]
            confirm_text = f"你的「{name}」账户当前余额为 {current:.2f}元，你说有 {new_bal:.2f}元，是否要更新为 {new_bal:.2f}元？（回复'确认'或'取消'）"
            _push_delta(state, confirm_text)
            state["interrupt_message"] = confirm_text
        except Exception as exc:
            logger.warning("ask_user_confirm_balance failed: %s", exc)
            state["used_tool"] = "ask_user_confirm_balance"
            state["answer"] = f"确认时出了点问题：{exc}"
            _push_delta(state, state["answer"])
        return state

    if action == "query_database":
        _push_progress(state, "正在查询数据库...")
        try:
            from ..service.db_query_service import execute_read_query
            sql = (args.get("sql") or "").strip()
            max_rows = int(args.get("max_rows", 50) or 50)
            if not sql:
                state["used_tool"] = "query_database"
                state["answer"] = "请提供要执行的 SQL 语句。"
                _push_delta(state, state["answer"])
                return state
            result = execute_read_query(sql, max_rows=max_rows)
            if result.get("error"):
                state["used_tool"] = "query_database"
                state["answer"] = f"查询出错：{result['error']}"
                _push_delta(state, state["answer"])
                return state
            lines = [f"数据库：{result['db']}，共 {result['row_count']} 条记录"]
            if result["columns"]:
                lines.append(" | ".join(str(c) for c in result["columns"]))
                lines.append("-" * 40)
            for r in result["rows"]:
                lines.append(" | ".join(str(v) if v is not None else "NULL" for v in r))
            if result.get("truncated"):
                lines.append(f"...（仅显示前 {max_rows} 条）")
            answer = "\n".join(lines)
            state["used_tool"] = "query_database"
            state["answer"] = answer
            _push_delta(state, state["answer"])
            _append_trace(state, _trace_step(step_type="tool", name="query_database",
                input={"sql": sql, "max_rows": max_rows}, output={"db": result["db"], "row_count": result["row_count"]}))
        except Exception as exc:
            logger.warning("query_database failed: %s", exc)
            state["used_tool"] = "query_database"
            state["answer"] = f"数据库查询出错：{exc}"
            _push_delta(state, state["answer"])
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
        _push_delta(state, state["answer"])
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

    # 推送由独立调度器处理，聊天图中不推送（用户已在对话框看到）
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
_MAX_CYCLES = 4  # 全局最大 decompose→run_tool→verify 循环次数，防止死循环


def _node_verify(state: ChatState) -> ChatState:
    """审查单步工具执行结果，不通过时标记 retry。"""
    action = (state.get("decision") or {}).get("action", "")
    if action not in _VERIFYABLE_ACTIONS:
        return state

    answer = (state.get("answer") or "").strip()
    if not answer:
        return state

    # 空结果自动通过（查询工具返回"没有待办/日程/天气"是有效结果，无需验证）
    emptiness = {"没有", "暂无", "无待办", "未安排", "没有找到", "没有记录", "没有数据", "empty", "none"}
    if action in {"query_todos", "query_calendar"} and any(k in answer.lower() for k in emptiness):
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
            "不要对用户身份（实习生/正式员工/其他）做假设——只检查结果本身是否有效",
        ],
    }
    try:
        text = chat_completion(
            [{"role": "system", "content": system}, {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
            temperature=0.0,
            runtime_context={"scenario": "verify_step", "session_id": state["session_id"]},
            on_token=_make_on_token(state),
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
        "注意：如果回答正确地说「没有找到事件」或类似结论，不要生成包含新信息的 fix。\n"
        "fix 只能基于 assistant_answer 已有的结论补充措辞，绝不能添加原文没有的事实。\n"
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
            on_token=_make_on_token(state),
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
                    on_token=_make_on_token(state),
                )
                state["answer"] = summary.strip()
            except Exception:
                state["answer"] = combined

    # 兜底，避免异常路径没有 answer
    if not state.get("answer"):
        state["answer"] = "抱歉，我刚刚没有处理好。你可以再说一次。"

    # 清理临时字段，防止跨轮次污染
    state["cycle_count"] = 0

    history = list(state.get("history") or [])
    if not history or history[-1].get("role") != "assistant" or history[-1].get("content") != state["answer"]:
        history.append({"role": "assistant", "content": state["answer"]})
    state["history"] = history
    state["recent_dialogue"] = history[-RECENT_DIALOGUE_LIMIT:]

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
    plan = state.get("plan") or []
    plan_index = state.get("plan_index", 0)
    if plan and plan_index < len(plan):
        plan[plan_index]["result"] = state.get("answer", "") or ""
        plan[plan_index]["used_tool"] = state.get("used_tool")
        return {**state, "plan_index": plan_index + 1}
    return state


def _route_after_advance(state: ChatState) -> str:
    if state.get("retry_info") and state.get("retry_count", 0) <= _MAX_RETRIES:
        return "decompose"
    plan = state.get("plan") or []
    plan_index = state.get("plan_index", 0)
    if plan and plan_index < len(plan):
        return "decompose"
    return "finalize"


# -----------------------------
# 通用打断节点：所有需要用户确认/反问的地方都用这一个节点
# 上游节点（run_tool / decompose）负责设 interrupt_message，本节点只负责 interrupt() 和存回答
# -----------------------------
def _node_interrupter(state: ChatState) -> ChatState:
    msg = state.get("interrupt_message")
    if not msg:
        return state

    action = (state.get("decision") or {}).get("action", "")
    _append_trace(
        state,
        _trace_step(
            step_type="tool",
            name=f"interrupt_{action}" if action else "interrupt",
            input={"question": msg},
        ),
    )

    user_answer = interrupt(msg)

    state["interrupt_answer"] = user_answer
    state["interrupt_message"] = None

    history = list(state.get("history") or [])
    history.append({"role": "assistant", "content": msg})
    state["history"] = history
    state["recent_dialogue"] = history[-RECENT_DIALOGUE_LIMIT:]

    _append_trace(
        state,
        _trace_step(
            step_type="tool",
            name=f"interrupt_{action}_resume" if action else "interrupt_resume",
            input={"answer": user_answer},
            output={"answer": user_answer},
        ),
    )
    return state


def _route_after_interrupt(state: ChatState) -> str:
    action = (state.get("decision") or {}).get("action", "")
    if action == "clarify":
        return "decompose"
    return "run_tool"


# 图连线：
# START -> load_context -> decompose -> run_tool -> [interrupt_message?] -> interrupter -> [clarify?] -> decompose | run_tool -> verify -> advance_plan -> [loop?] -> decompose | finalize -> quality_gate -> [proactive?] -> proactive_decision -> END
def build_chat_graph(
    *,
    checkpointer: Any | None = None,
):
    graph = StateGraph(ChatState)

    graph.add_node("load_context", _node_load_context)
    graph.add_node("decompose", _node_decompose)
    graph.add_node("run_tool", _node_run_tool)
    graph.add_node("verify", _node_verify)
    graph.add_node("advance_plan", _node_advance_plan)
    graph.add_node("finalize", _node_finalize)
    graph.add_node("quality_gate", _node_quality_gate)
    graph.add_node("proactive_decision", _node_proactive_decision)
    graph.add_node("interrupter", _node_interrupter)

    graph.add_edge(START, "load_context")
    graph.add_edge("load_context", "decompose")
    graph.add_edge("decompose", "run_tool")
    graph.add_conditional_edges(
        "run_tool",
        lambda s: "interrupter" if s.get("interrupt_message") else "verify",
        {"interrupter": "interrupter", "verify": "verify"},
    )
    graph.add_conditional_edges(
        "interrupter",
        _route_after_interrupt,
        {"decompose": "decompose", "run_tool": "run_tool"},
    )
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
        "proposal_id": values.get("proposal_id"),
        "proposal": values.get("proposal"),
        "proposal_confirmed": values.get("proposal_confirmed"),
        "interrupt_message": None,
        "interrupt_answer": None,
        "pending_reminders": values.get("pending_reminders", []),
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

        "proposal_id": checkpoint_context.get("proposal_id"),
        "proposal": checkpoint_context.get("proposal"),
        "proposal_confirmed": False,
        "interrupt_message": None,
        "interrupt_answer": None,
        "profile_context": profile_context,
        "effective_user_message": _build_effective_message(user_message, profile_context),
        "trace": {
            "meta": {"graph": "chat_v2", "started_at": datetime.now().isoformat(timespec="seconds")},
            "steps": steps,
        },
        "used_tool": None,
        "citations": [],
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
        tasks = getattr(current_state, "tasks", []) or []
        if tasks and (detect_affirmation(user_message) or detect_rejection(user_message)):
            return await compiled_graph.ainvoke(Command(resume=user_message), config=config)
        if tasks:
            await compiled_graph.aupdate_state(config, {"interrupt_message": None, "interrupt_answer": None})

        ctx = await aload_chat_checkpoint_context(compiled_graph, session_id)
        state = _build_initial_state(session_id=session_id, user_message=user_message, checkpoint_context=ctx, **kwargs)
        return await compiled_graph.ainvoke(state, config=config)
    finally:
        _clear_progress_queue(session_id)

