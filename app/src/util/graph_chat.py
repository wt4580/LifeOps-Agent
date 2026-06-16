from __future__ import annotations

"""lifeops.graph_chat

基于 LangGraph 的聊天状态机编排。

目标流程：
Chat -> (LLM decide) -> Tool -> Update state -> (maybe HITL) -> Final answer -> (proactive advice)
"""
import json
import re
from datetime import datetime, timedelta
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
from sqlalchemy import select, func

from .agent_router import build_route_context, route_decision
from ..common.config.db_config import SessionLocal
from ..common.config.base_config import settings
from ..common.config.log_config import logger
from .llm.llm_qwen import chat_completion
from ..domain.entity.chat_entity import ConversationSummary, TodoItem, LifeEvent
from .planner.planner import detect_affirmation, detect_rejection, propose_plan
from .retrieval.retrieval import rag_answer, search_chunks
from .time.time_parser import normalize_date_hint
from .memory.personal_memory import decide_proactive_advice
from app.src.domain.entity.chatstate_entity import ChatState

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


def _load_life_events_in_window(session_id: str, hours: int, now_dt: datetime) -> tuple[list[dict], str, str]:
    """按时间窗口读取个人事件，避免只看固定条数导致时间上下文丢失。"""
    window_start = now_dt - timedelta(hours=hours)
    with SessionLocal() as db:
        rows = (
            db.execute(
                select(LifeEvent)
                .where(LifeEvent.session_id == session_id)
                .where(func.coalesce(LifeEvent.event_time, LifeEvent.created_at) >= window_start)
                .where(func.coalesce(LifeEvent.event_time, LifeEvent.created_at) <= now_dt)
                .order_by(func.coalesce(LifeEvent.event_time, LifeEvent.created_at).asc())
            )
            .scalars()
            .all()
        )
    
    out: list[dict] = []
    for r in rows:
        try:
            tags = json.loads(r.tags_json) if r.tags_json else []
        except json.JSONDecodeError:
            tags = []
        out.append(
            {
                "category": r.category,
                "title": r.title,
                "event_time": r.event_time.isoformat() if r.event_time else None,
                "recorded_at": r.created_at.isoformat() if r.created_at else None,
                "effective_time": (r.event_time or r.created_at).isoformat() if (r.event_time or r.created_at) else None,
                "amount": r.amount,
                "amount_unit": r.amount_unit,
                "tags": tags,
                "notes": r.notes,
            }
        )
    
    return (
        out,
        window_start.isoformat(timespec="seconds"),
        now_dt.isoformat(timespec="seconds"),
    )


def _load_upcoming_todos_for_advice(session_id: str, hours: int = 24) -> list[dict]:
    """读取未来待办事项，用于判断是否要主动提醒用户。"""
    now = datetime.now()
    end = now + timedelta(hours=hours)
    with SessionLocal() as db:
        stmt = (
            select(TodoItem)
            .where(TodoItem.due_at != None)
            .where(TodoItem.due_at >= now)
            .where(TodoItem.due_at < end)
            .order_by(TodoItem.due_at.asc())
        )
        scope = (settings.personal_memory_scope or "global").strip().lower()
        session_col = getattr(TodoItem, "session_id", None)
        if scope == "session" and session_col is not None:
            stmt = stmt.where(session_col == session_id)
        
        rows = db.execute(stmt).scalars().all()
    
    return [{"title": r.title, "due_at": r.due_at.isoformat() if r.due_at else None} for r in rows]


def _resolve_memory_owner_id(session_id: str) -> str:
    """解析记忆所有者 ID。

    `session` 模式按会话隔离；`global` 模式则共享到统一 owner。
    """
    scope = (settings.personal_memory_scope or "global").strip().lower()
    if scope == "session":
        return session_id
    return "global"


def _should_check_proactive(state: ChatState) -> bool:
    """判断是否需要执行主动决策。HITL 流程跳过，普通聊天才检查。"""
    used_tool = state.get("used_tool")
    return used_tool not in {"plan_proposal", "plan_gate_cancel", "plan_gate_wait"}


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


# 规则兜底：明显“文档/文件查询”句式直接路由到 query_knowledge，
# 目的是防止路由模型误判成 normal_chat（例如把“work.png里有什么”当寒暄）。
def _should_force_knowledge_query(user_message: str) -> bool:
    """当问题明显是在问文档/文件内容时，用确定性规则兜底为 query_knowledge。"""

    text = (user_message or "").strip().lower()
    if not text:
        return False

    has_file_hint = bool(re.search(r"[a-z0-9_.-]+\.(pdf|png|txt)", text))
    has_kb_hint = any(k in text for k in ["知识库", "文档", "文件", "论文", "ocr"])
    has_query_intent = any(k in text for k in ["有什么", "是什么", "内容", "讲了什么", "里有什么", "里面有什么", "提到"])

    return (has_file_hint or has_kb_hint) and has_query_intent


def _is_fact_question(user_message: str) -> bool:
    """识别事实型问句：优先走知识检索，再决定是否回退普通聊天。"""

    text = (user_message or "").strip().lower()
    if not text:
        return False

    fact_markers = [
        "叫什么",
        "名字",
        "是谁",
        "哪位",
        "哪里",
        "哪个",
        "哪家",
        "实验室",
        "导师",
        "单位",
        "学校",
        "公司",
    ]
    has_fact = any(m in text for m in fact_markers)
    has_question = any(q in text for q in ["?", "？", "什么", "哪", "谁"])

    # 排除明显待办/日程创建语义，避免误走 RAG。
    has_todo_create = any(k in text for k in ["提醒", "记住", "待办", "安排", "我要", "明天", "后天", "下周"])
    return has_fact and has_question and not has_todo_create


def _is_personal_state_question(user_message: str) -> bool:
    """识别“我的近况/个人状态”类问题，这类问题应优先走个人记忆上下文而不是知识库。"""

    text = (user_message or "").strip().lower()
    if not text:
        return False

    has_self_ref = any(k in text for k in ["我", "我的", "最近", "这段时间"])
    state_markers = [
        "饮食怎么样",
        "吃得怎么样",
        "运动怎么样",
        "状态怎么样",
        "我最近",
        "我今天",
        "我这周",
    ]
    has_state_marker = any(k in text for k in state_markers)

    # 避免把“知识型饮食问题”（如肥胖症饮食指南是什么）拦截掉。
    has_knowledge_target = any(k in text for k in ["指南", "文档", "论文", "知识库", "pdf", "png", "txt"])
    return has_self_ref and has_state_marker and not has_knowledge_target


def _should_try_rag_first(user_message: str) -> bool:
    """统一判定：文档问答或事实问句先尝试 RAG；个人近况问题不强制走 RAG。"""

    if _is_personal_state_question(user_message):
        return False
    return _should_force_knowledge_query(user_message) or _is_fact_question(user_message)


def _build_effective_message(user_message: str, profile_context: str | None, pre_advice: str | None) -> str:
    """拼装给 graph 使用的增强输入。"""

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
    _append_trace(
        state,
        _trace_step(
            step_type="graph_node",
            name="load_context",
            output={"history_len": len(history), "recent_dialogue_len": len(state["recent_dialogue"]), "has_summary": bool(state.get("summary"))},
        ),
    )
    return state


# 决策节点：决定下一步 action。
# 触发顺序：forced_action > 规则兜底 > LLM 路由。
# 其中 LLM 路由通过 route_decision 发生（内部会调用 LLM）。
def _node_decide(state: ChatState) -> ChatState:
    # 如果上一轮并没有待确认草案，而用户只回了“需要/不要”这类短句，先不要让 Router 猜成新待办。
    if not state.get("pending_text") and _is_bare_confirmation_text(state.get("user_message") or ""):
        decision = {
            "action": "normal_chat",
            "args": {},
            "assistant_message": "你是想确认上一条待办，还是要新建一个待办？如果要新建，请直接说具体计划内容。",
            "trace": {
                "intent": "confirmation_clarify",
                "signals": ["bare_confirmation"],
                "why": "用户只回复了确认/取消短句，但当前没有待确认草案，先澄清而不是新建待办",
            },
        }
        state["decision"] = decision
        _append_trace(
            state,
            _trace_step(
                step_type="router_guard",
                name="bare_confirmation_guard",
                input={"user_message": state.get("user_message")},
                output=decision,
            ),
        )
        return state

    forced_action = state.get("forced_action")
    if forced_action:
        # 外部强制路由（例如某 API 包装层指定动作）时，不再调路由 LLM。
        decision = {
            "action": forced_action,
            "args": state.get("forced_args") or {},
            "assistant_message": "",
            "trace": {"intent": "forced", "signals": [], "why": "forced by API wrapper"},
        }
        state["decision"] = decision
        _append_trace(
            state,
            _trace_step(
                step_type="llm",
                name="router",
                input={"forced_action": forced_action, "forced_args": state.get("forced_args") or {}},
                output=decision,
            ),
        )
        return state

    # 规则兜底：文档/知识库问答与事实型问句都先走 query_knowledge。
    if _should_try_rag_first(state["user_message"]):
        decision = {
            "action": "query_knowledge",
            "args": {"question": state["user_message"], "top_k": 5},
            "assistant_message": "",
            "trace": {
                "intent": "knowledge_query",
                "signals": ["rule_guard:rag_first"],
                "why": "规则兜底命中：文档问答或事实型问题，先尝试 query_knowledge",
            },
        }
        state["decision"] = decision
        _append_trace(
            state,
            _trace_step(
                step_type="router_guard",
                name="knowledge_query_guard",
                input={"user_message": state["user_message"]},
                output=decision,
            ),
        )
        return state

    # 到这里才真正调用“路由 LLM”。
    # 输入包括：用户消息 + 信号词 + 最近对话长度 + 摘要。
    ctx = build_route_context(
        session_id=state["session_id"],
        user_message=state.get("effective_user_message") or state["user_message"],
        recent_dialogue=state.get("recent_dialogue", []),
        summary=state.get("summary"),
    )
    decision = route_decision(ctx)
    state["decision"] = decision.model_dump()

    # 二次兜底：LLM 若误判为 normal_chat，且文本明显是知识库查询，则强制改写为 query_knowledge。
    if state["decision"].get("action") == "normal_chat" and _should_try_rag_first(state["user_message"]):
        state["decision"] = {
            "action": "query_knowledge",
            "args": {"question": state["user_message"], "top_k": 5},
            "assistant_message": "",
            "trace": {
                "intent": "knowledge_query",
                "signals": ctx.signals,
                "why": "LLM 误判 normal_chat，命中 RAG 优先兜底改写",
            },
        }

      # 增强的 trace 输出：包含完整的决策理由和上下文
    decision_detail = state["decision"]
    router_trace = decision_detail.get("trace", {})
    
    _append_trace(
        state,
        _trace_step(
            step_type="llm",
            name="router_decision",
            input={
                "user_message": state["user_message"],
                "effective_user_message": state.get("effective_user_message") or state["user_message"],
                "signals": ctx.signals,
                "signal_count": len(ctx.signals),
                "summary_present": bool(state.get("summary")),
                "summary_preview": (state.get("summary") or "")[:100] if state.get("summary") else None,
                "recent_dialogue_len": len(state.get("recent_dialogue", [])),
            },
            output={
                "action": decision_detail.get("action"),
                "args": decision_detail.get("args", {}),
                "assistant_message_preview": (decision_detail.get("assistant_message") or "")[:100],
                "decision_reasoning": {
                    "intent": router_trace.get("intent"),
                    "why": router_trace.get("why"),
                    "signals_used": router_trace.get("signals", []),
                },
            },
            metadata={
                "session_id": state["session_id"],
                "has_summary": bool(state.get("summary")),
                "dialogue_context_length": len(state.get("recent_dialogue", [])),
            }
        ),
    )
    
    # 如果触发了 RAG 优先兜底，额外记录一条 trace
    if state["decision"].get("action") == "query_knowledge" and _should_try_rag_first(state["user_message"]):
        _append_trace(
            state,
            _trace_step(
                step_type="router_guard",
                name="rag_priority_override",
                input={"original_action": "normal_chat", "triggered_by_rule": True},
                output={
                    "overridden_action": "query_knowledge",
                    "reason": "LLM 误判 normal_chat，但命中 RAG 优先规则，强制改写为 query_knowledge",
                    "user_message_keywords": state["user_message"][:50],
                },
            ),
        )
    
    return state


# 工具执行节点：根据 decision.action 执行具体分支。
# 这里是“真正做事”的地方，同时也是 LLM 调用最密集的节点之一。
def _build_normal_chat_answer(
    state: ChatState,
    decision: dict[str, Any],
    *,
    allow_assistant_message: bool = True,
) -> tuple[str, bool]:
    """统一 normal_chat 文本生成逻辑，供常规分支与 RAG miss 回退复用。"""

    assistant_message = (decision.get("assistant_message") or "").strip() if allow_assistant_message else ""
    if assistant_message:
        return assistant_message, True

    messages: list[dict[str, str]] = [{"role": "system", "content": "你是一个中文生活助手。"}]
    if state.get("profile_context"):
        messages.append({"role": "system", "content": f"用户画像：{state['profile_context']}"})
    if state.get("pre_advice"):
        messages.append({"role": "system", "content": f"前置建议：{state['pre_advice']}"})
    if state.get("summary"):
        messages.append({"role": "system", "content": f"Conversation summary: {state['summary']}"})
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

    if action in ("ask_user_confirm_proposal", "propose_todo"):
        # 直接生成待办草案，减少“先问需要、再确认”的多轮打断。
        pending_text = args.get("text") or state["user_message"]
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

    # normal_chat 分支：
    state["used_tool"] = None
    assistant_message = (decision.get("assistant_message") or "").strip()
    if assistant_message:
        state["answer"] = assistant_message
        used_assistant = True
    else:
        state["answer"], used_assistant = _build_normal_chat_answer(state, decision)

    _append_trace(state, _trace_step(step_type="tool", name="normal_chat", output={"used_assistant_message": used_assistant}))
    _append_trace(state, _trace_step(step_type="llm", name="normal_chat", output=state.get("answer")))
    return state


# 主动决策节点：基于近期事件/待办/画像，决定是否追加主动建议。
def _node_proactive_decision(state: ChatState) -> ChatState:
    """执行主动建议决策逻辑。
    
    这个节点负责：
    1. 加载近期事件和未来待办（如果还未加载）
    2. 调用 decide_proactive_advice 进行智能决策
    3. 如果需要追加建议，修改 answer
    4. 记录 trace 到状态中
    """
    
    # 1. 确保上下文已加载（从 Service 传入或在此加载）
    recent_events = state.get("recent_events", [])
    upcoming_todos = state.get("upcoming_todos", [])
    profile_context = state.get("profile_context") or ""
    
    # 如果上下文中没有，尝试从数据库加载
    if not recent_events and not upcoming_todos:
        now_dt = datetime.now()
        memory_owner_id = _resolve_memory_owner_id(state["session_id"])
        
        if not recent_events:
            recent_events, event_window_start, event_window_end = _load_life_events_in_window(
                session_id=memory_owner_id,
                hours=settings.proactive_event_window_hours,
                now_dt=now_dt,
            )
            state["recent_events"] = recent_events
            state["event_window_start"] = event_window_start
            state["event_window_end"] = event_window_end
        
        if not upcoming_todos:
            upcoming_todos = _load_upcoming_todos_for_advice(
                session_id=memory_owner_id,
                hours=settings.proactive_todo_hours,
            )
            state["upcoming_todos"] = upcoming_todos
    
    # 2. 调用决策函数
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

# 图连线就是"先走谁再走谁"的唯一真相：
# START -> load_context -> hitl_resume? -> decide -> run_tool -> finalize -> (proactive_decision?) -> END
def build_chat_graph(
    *,
    checkpointer: Any | None = None,
):
    graph = StateGraph(ChatState)

    graph.add_node("load_context", _node_load_context)
    graph.add_node("hitl_resume", _node_handle_pending_confirmation)
    graph.add_node("decide", _node_decide)
    graph.add_node("run_tool", _node_run_tool)
    graph.add_node("finalize", _node_finalize)
    graph.add_node("proactive_decision", _node_proactive_decision)

    graph.add_edge(START, "load_context")
    graph.add_conditional_edges(
        "load_context",
        lambda s: "hitl_resume" if s.get("pending_text") else "decide",
        {"hitl_resume": "hitl_resume", "decide": "decide"},
    )
    graph.add_edge("hitl_resume", "finalize")
    graph.add_edge("decide", "run_tool")
    graph.add_edge("run_tool", "finalize")
    
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


def load_chat_checkpoint_context(compiled_graph: Any, session_id: str) -> dict[str, Any]:
    """从 checkpointer 恢复最近一次聊天状态，用于本轮上下文构造。"""

    snapshot = compiled_graph.get_state({"configurable": {"thread_id": session_id}})
    values = getattr(snapshot, "values", None) or {}
    if not isinstance(values, dict):
        return {}

    history = list(values.get("history") or [])
    recent_dialogue = history[-16:]
    return {
        "history": history,
        "recent_dialogue": recent_dialogue,
        "summary": values.get("summary"),
        "pending_text": values.get("pending_text"),
        "proposal_id": values.get("proposal_id"),
        "proposal": values.get("proposal"),
    }


async def aload_chat_checkpoint_context(compiled_graph: Any, session_id: str) -> dict[str, Any]:
    """异步版本：从 checkpointer 恢复最近一次聊天状态。"""

    snapshot = await compiled_graph.aget_state({"configurable": {"thread_id": session_id}})
    values = getattr(snapshot, "values", None) or {}
    if not isinstance(values, dict):
        return {}

    history = list(values.get("history") or [])
    recent_dialogue = history[-16:]
    return {
        "history": history,
        "recent_dialogue": recent_dialogue,
        "summary": values.get("summary"),
        "pending_text": values.get("pending_text"),
        "proposal_id": values.get("proposal_id"),
        "proposal": values.get("proposal"),
    }


# 状态机入口：
# - 创建初始状态（包括 input trace）
# - 调 compiled graph 执行
# - 返回最终状态（answer/proposal/citations/trace 都在里面）
def run_chat_graph(
    *,
    compiled_graph: Any,
    session_id: str,
    user_message: str,
    forced_action: str | None = None,
    forced_args: dict[str, Any] | None = None,
    profile_context: str | None = None,
    pre_advice: str | None = None,
    pre_steps: list[dict[str, Any]] | None = None,
    recent_events: list[dict[str, Any]] | None = None,
    upcoming_todos: list[dict[str, Any]] | None = None,
    event_window_start: str | None = None,
    event_window_end: str | None = None,
) -> ChatState:
    checkpoint_context = load_chat_checkpoint_context(compiled_graph, session_id)

    history = list(checkpoint_context.get("history") or [])
    history.append({"role": "user", "content": user_message})

    steps = [{"type": "input", "user_message": user_message, "session_id": session_id}]
    if pre_steps:
        steps.extend(pre_steps)

    initial_state: ChatState = {
        "session_id": session_id,
        "user_message": user_message,
        "history": history,
        "recent_dialogue": list(checkpoint_context.get("recent_dialogue") or history[-16:]),
        "summary": checkpoint_context.get("summary"),
        "pending_text": checkpoint_context.get("pending_text"),
        "proposal_id": checkpoint_context.get("proposal_id"),
        "proposal": checkpoint_context.get("proposal"),
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
        # 主动决策上下文（由 Service 层传入，Graph 节点使用）
        "recent_events": recent_events or [],
        "upcoming_todos": upcoming_todos or [],
        "event_window_start": event_window_start,
        "event_window_end": event_window_end,
    }
    return compiled_graph.invoke(
        initial_state,
        config={"configurable": {"thread_id": session_id}},
    )


async def arun_chat_graph(
    *,
    compiled_graph: Any,
    session_id: str,
    user_message: str,
    forced_action: str | None = None,
    forced_args: dict[str, Any] | None = None,
    profile_context: str | None = None,
    pre_advice: str | None = None,
    pre_steps: list[dict[str, Any]] | None = None,
    recent_events: list[dict[str, Any]] | None = None,
    upcoming_todos: list[dict[str, Any]] | None = None,
    event_window_start: str | None = None,
    event_window_end: str | None = None,
) -> ChatState:
    config = {"configurable": {"thread_id": session_id}}
    current_state = await compiled_graph.aget_state(config)

    if len(getattr(current_state, "tasks", []) or []) > 0:
        resumed = await compiled_graph.ainvoke(Command(resume=user_message), config=config)
        return resumed

    checkpoint_context = await aload_chat_checkpoint_context(compiled_graph, session_id)

    history = list(checkpoint_context.get("history") or [])
    history.append({"role": "user", "content": user_message})

    steps = [{"type": "input", "user_message": user_message, "session_id": session_id}]
    if pre_steps:
        steps.extend(pre_steps)

    initial_state: ChatState = {
        "session_id": session_id,
        "user_message": user_message,
        "history": history,
        "recent_dialogue": list(checkpoint_context.get("recent_dialogue") or history[-16:]),
        "summary": checkpoint_context.get("summary"),
        "pending_text": checkpoint_context.get("pending_text"),
        "proposal_id": checkpoint_context.get("proposal_id"),
        "proposal": checkpoint_context.get("proposal"),
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
        # 主动决策上下文（由 Service 层传入，Graph 节点使用）
        "recent_events": recent_events or [],
        "upcoming_todos": upcoming_todos or [],
        "event_window_start": event_window_start,
        "event_window_end": event_window_end,
    }
    return await compiled_graph.ainvoke(initial_state, config=config)

