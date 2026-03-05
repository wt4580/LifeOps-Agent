from __future__ import annotations

"""lifeops.graph_chat

基于 LangGraph 的聊天状态机编排。

目标流程：
Chat -> (LLM decide) -> Tool -> Update state -> (maybe HITL) -> Final answer

建议阅读顺序（初学者）：
1) 先看 `ChatState`（状态里到底存了哪些数据）。
2) 再看 `_node_*` 函数（每个节点做什么变换）。
3) 最后看 `build_chat_graph`（节点如何连成状态机）。

说明：
- 这个模块只负责“编排”聊天主流程，不改动原有工具实现（planner/retrieval/db）。
- 写入 todo 仍然保持 Human-in-the-loop：聊天阶段只生成 proposal，真正入库仍走 /api/plan/confirm。
- 为了兼容现有前端 trace 面板，每个节点都会写入 trace.steps。
"""

import json
import re
from datetime import datetime, timedelta
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from sqlalchemy import select

from .agent_router import build_route_context, route_decision
from .db import SessionLocal
from .llm_qwen import chat_completion
from .models import ChatMessage, ConversationSummary, TodoItem
from .planner import detect_affirmation, detect_rejection, propose_plan
from .retrieval import rag_answer, search_chunks
from .time_parser import normalize_date_hint


class ChatState(TypedDict, total=False):
    # 输入
    session_id: str
    user_message: str

    # 运行时上下文
    recent_dialogue: list[dict[str, str]]
    summary: str | None

    # 路由/执行
    decision: dict[str, Any]
    forced_action: str | None
    forced_args: dict[str, Any] | None
    answer: str
    used_tool: str | None

    # 提案/HITL
    pending_text: str | None
    hitl_cancelled: bool
    proposal_id: str | None
    proposal: dict[str, Any] | None

    # Trace
    citations: list[dict[str, Any]]
    trace: dict[str, Any]


# -----------------------------
# DB/上下文工具（节点内部复用）
# -----------------------------

def _load_recent_dialogue(session_id: str, limit: int = 16) -> list[dict[str, str]]:
    with SessionLocal() as db:
        rows = (
            db.execute(
                select(ChatMessage)
                .where(ChatMessage.session_id == session_id)
                .order_by(ChatMessage.created_at.desc())
                .limit(limit)
            )
            .scalars()
            .all()
        )
    rows.reverse()
    return [{"role": r.role, "content": r.content} for r in rows]


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


def _append_trace(state: ChatState, step: dict[str, Any]) -> None:
    trace = state.setdefault("trace", {"steps": []})
    trace.setdefault("steps", []).append(step)


def _trace_step(*, step_type: str, name: str, **kwargs: Any) -> dict[str, Any]:
    return {
        "type": step_type,
        "name": name,
        "ts": datetime.now().isoformat(timespec="seconds"),
        **kwargs,
    }


def _should_force_knowledge_query(user_message: str) -> bool:
    """当问题明显是在问文档/文件内容时，用确定性规则兜底为 query_knowledge。"""

    text = (user_message or "").strip().lower()
    if not text:
        return False

    has_file_hint = bool(re.search(r"[a-z0-9_.-]+\.(pdf|png|txt)", text))
    has_kb_hint = any(k in text for k in ["知识库", "文档", "文件", "论文", "ocr"])
    has_query_intent = any(k in text for k in ["有什么", "是什么", "内容", "讲了什么", "里有什么", "里面有什么", "提到"])

    return (has_file_hint or has_kb_hint) and has_query_intent


# -----------------------------
# LangGraph 节点
# -----------------------------

def _node_load_context(state: ChatState) -> ChatState:
    session_id = state["session_id"]
    recent = _load_recent_dialogue(session_id, limit=16)
    summary = _get_summary(session_id)
    state["recent_dialogue"] = recent
    state["summary"] = summary
    _append_trace(
        state,
        _trace_step(
            step_type="graph_node",
            name="load_context",
            output={"recent_dialogue_len": len(recent), "has_summary": bool(summary)},
        ),
    )
    return state


def _node_hitl_check(state: ChatState, pending_intents: dict[str, str]) -> ChatState:
    session_id = state["session_id"]
    user_message = state["user_message"]

    if detect_rejection(user_message):
        removed = pending_intents.pop(session_id, None)
        if removed:
            state["pending_text"] = None
            state["hitl_cancelled"] = True
            state["used_tool"] = "plan_gate_cancel"
            state["answer"] = "好的，这次先不生成待办草案。你之后说“需要”我再帮你建。"
            _append_trace(
                state,
                _trace_step(
                    step_type="graph_node",
                    name="hitl_check",
                    output={"waiting_proposal_confirm": False, "cancelled": True},
                ),
            )
            return state

    if detect_affirmation(user_message):
        pending_text = pending_intents.pop(session_id, None)
        if pending_text:
            state["pending_text"] = pending_text
            _append_trace(
                state,
                _trace_step(
                    step_type="graph_node",
                    name="hitl_check",
                    output={"waiting_proposal_confirm": True, "pending_text": pending_text},
                ),
            )
            return state

    state["pending_text"] = None
    state["hitl_cancelled"] = False
    _append_trace(
        state,
        _trace_step(
            step_type="graph_node",
            name="hitl_check",
            output={"waiting_proposal_confirm": False, "cancelled": False},
        ),
    )
    return state


def _route_after_hitl(state: ChatState) -> str:
    if state.get("hitl_cancelled"):
        return "finalize"
    return "propose_from_pending" if state.get("pending_text") else "decide"


def _node_propose_from_pending(state: ChatState, proposal_cache: dict[str, list[dict[str, Any]]]) -> ChatState:
    pending_text = state.get("pending_text") or ""
    proposal_id, proposal = propose_plan(pending_text)
    proposal_payload = proposal.model_dump()
    proposal_cache[proposal_id] = [item.model_dump() for item in proposal.items]

    state["proposal_id"] = proposal_id
    state["proposal"] = proposal_payload
    state["used_tool"] = "plan_proposal"
    state["answer"] = "已生成待办草案，请确认是否加入待办。"

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


def _node_decide(state: ChatState) -> ChatState:
    forced_action = state.get("forced_action")
    if forced_action:
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

    # 规则兜底：避免 LLM 把明显的文件内容查询误判成寒暄 normal_chat。
    if _should_force_knowledge_query(state["user_message"]):
        decision = {
            "action": "query_knowledge",
            "args": {"question": state["user_message"], "top_k": 5},
            "assistant_message": "",
            "trace": {
                "intent": "knowledge_query",
                "signals": ["rule_guard:file_or_kb_query"],
                "why": "规则兜底命中：文件/知识库查询句式，直接走 query_knowledge",
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

    ctx = build_route_context(
        session_id=state["session_id"],
        user_message=state["user_message"],
        recent_dialogue=state.get("recent_dialogue", []),
        summary=state.get("summary"),
    )
    decision = route_decision(ctx)
    state["decision"] = decision.model_dump()

    # 二次兜底：LLM 若误判为 normal_chat，且文本明显是知识库查询，则强制改写为 query_knowledge。
    if state["decision"].get("action") == "normal_chat" and _should_force_knowledge_query(state["user_message"]):
        state["decision"] = {
            "action": "query_knowledge",
            "args": {"question": state["user_message"], "top_k": 5},
            "assistant_message": "",
            "trace": {
                "intent": "knowledge_query",
                "signals": ctx.signals,
                "why": "LLM 误判 normal_chat，命中知识库查询兜底改写",
            },
        }

    _append_trace(
        state,
        _trace_step(
            step_type="llm",
            name="router",
            input={
                "user_message": state["user_message"],
                "signals": ctx.signals,
                "summary": state.get("summary"),
                "recent_dialogue_len": len(state.get("recent_dialogue", [])),
            },
            output=state["decision"],
        ),
    )
    return state


def _node_run_tool(state: ChatState, pending_intents: dict[str, str]) -> ChatState:
    decision = state.get("decision") or {"action": "normal_chat", "args": {}, "assistant_message": ""}
    action = decision.get("action", "normal_chat")
    args = decision.get("args") or {}

    if action == "query_todos":
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
        question = (args.get("question") or state["user_message"]).strip()
        top_k = int(args.get("top_k", 5) or 5)
        citations = search_chunks(question, top_k=top_k)
        state["used_tool"] = "query_knowledge"
        state["citations"] = [
            {
                "path": c.path,
                "page": c.page,
                "snippet": c.snippet,
                "score": c.score,
                "reason": c.reason,
            }
            for c in citations
        ]

        if not citations:
            state["answer"] = "No relevant context found."
        else:
            state["answer"] = rag_answer(question, citations)

        _append_trace(
            state,
            _trace_step(
                step_type="tool",
                name="query_knowledge",
                input={"question": question, "top_k": top_k},
                output={"hits": len(citations)},
            ),
        )
        return state

    if action in ("ask_user_confirm_proposal", "propose_todo"):
        pending_text = args.get("text") or state["user_message"]
        pending_intents[state["session_id"]] = pending_text
        state["used_tool"] = "plan_gate"
        state["answer"] = "我可以为你设置提醒。需要我为你生成待办草案吗？回复“需要/好/可以”即可。"
        _append_trace(
            state,
            _trace_step(
                step_type="tool",
                name="ask_user_confirm_proposal",
                input={"text": pending_text},
                output={"pending_saved": True},
            ),
        )
        return state

    # normal_chat
    state["used_tool"] = None
    assistant_message = (decision.get("assistant_message") or "").strip()
    if assistant_message:
        state["answer"] = assistant_message
    else:
        messages: list[dict[str, str]] = [{"role": "system", "content": "你是一个中文生活助手。"}]
        if state.get("summary"):
            messages.append({"role": "system", "content": f"Conversation summary: {state['summary']}"})
        messages.extend(state.get("recent_dialogue", []))
        state["answer"] = chat_completion(messages)

    _append_trace(state, _trace_step(step_type="tool", name="normal_chat", output={"used_assistant_message": bool(assistant_message)}))
    _append_trace(state, _trace_step(step_type="llm", name="normal_chat", output=state.get("answer")))
    return state


def _node_finalize(state: ChatState) -> ChatState:
    # 兜底，避免异常路径没有 answer
    if not state.get("answer"):
        state["answer"] = "抱歉，我刚刚没有处理好。你可以再说一次。"
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

def build_chat_graph(pending_intents: dict[str, str], proposal_cache: dict[str, list[dict[str, Any]]]):
    graph = StateGraph(ChatState)

    graph.add_node("load_context", _node_load_context)
    graph.add_node("hitl_check", lambda s: _node_hitl_check(s, pending_intents))
    graph.add_node("propose_from_pending", lambda s: _node_propose_from_pending(s, proposal_cache))
    graph.add_node("decide", _node_decide)
    graph.add_node("run_tool", lambda s: _node_run_tool(s, pending_intents))
    graph.add_node("finalize", _node_finalize)

    graph.add_edge(START, "load_context")
    graph.add_edge("load_context", "hitl_check")
    graph.add_conditional_edges(
        "hitl_check",
        _route_after_hitl,
        {"propose_from_pending": "propose_from_pending", "decide": "decide", "finalize": "finalize"},
    )
    graph.add_edge("propose_from_pending", "finalize")
    graph.add_edge("decide", "run_tool")
    graph.add_edge("run_tool", "finalize")
    graph.add_edge("finalize", END)

    return graph.compile()


def run_chat_graph(
    *,
    compiled_graph: Any,
    session_id: str,
    user_message: str,
    forced_action: str | None = None,
    forced_args: dict[str, Any] | None = None,
) -> ChatState:
    initial_state: ChatState = {
        "session_id": session_id,
        "user_message": user_message,
        "trace": {
            "meta": {"graph": "chat_v2", "started_at": datetime.now().isoformat(timespec="seconds")},
            "steps": [{"type": "input", "user_message": user_message, "session_id": session_id}],
        },
        "used_tool": None,
        "proposal_id": None,
        "proposal": None,
        "citations": [],
        "hitl_cancelled": False,
        "forced_action": forced_action,
        "forced_args": forced_args or {},
    }
    return compiled_graph.invoke(initial_state)
