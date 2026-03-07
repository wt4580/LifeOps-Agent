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

执行时序总览（最重要）：
1) `run_chat_graph` 创建初始状态并写入 trace 的 input 步。
2) 状态机固定先走 `load_context`（读最近对话+摘要）。
3) 再走 `hitl_check`（判断本轮是否是“确认/取消草案”回复）。
4) 命中确认则走 `propose_from_pending` 并结束；未命中则进入 `decide`。
5) `decide` 决定 action（规则兜底或 LLM 路由），然后进入 `run_tool`。
6) `run_tool` 按 action 执行工具；部分分支会调用 LLM 生成自然语言。
7) 最后统一 `finalize` 收口，确保 answer 和 trace 完整。
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


# ChatState 是“图上流动的数据包”：
# 每个 node 读取/修改其中的字段，再交给下一个 node。
# 理解这个结构，就能理解整个状态机的数据依赖。
class ChatState(TypedDict, total=False):
    # 输入
    session_id: str
    user_message: str
    # 前置画像信息：由 main 在进 graph 前准备。
    profile_context: str | None
    pre_advice: str | None
    # 供路由使用的增强输入（用户原文 + 画像/建议）。
    effective_user_message: str

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

# 读取最近 N 轮消息，供后续路由和普通聊天分支作为上下文输入。
# 注意：这里只是“短期记忆”，不是摘要。
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


# 读取会话摘要（中期记忆），用于对话很长或重启后的补偿上下文。
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


# -----------------------------
# LangGraph 节点
# -----------------------------

# 第一个节点：加载上下文。
# 顺序上它永远先于 hitl_check/decide，因为后续节点都可能依赖 recent_dialogue/summary。
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


# 第二个节点：HITL 闸门。
# - 若用户说“不要/算了”且确有 pending，直接取消并可提前收口。
# - 若用户说“需要/好/可以”且确有 pending，进入 propose_from_pending。
# - 其他情况走普通决策 decide。
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


# hitl_check 的条件分流器：
# cancelled -> finalize（短路结束）
# pending_text 存在 -> propose_from_pending
# 否则 -> decide
def _route_after_hitl(state: ChatState) -> str:
    if state.get("hitl_cancelled"):
        return "finalize"
    return "propose_from_pending" if state.get("pending_text") else "decide"


# HITL 确认后的草案生成节点。
# 这里会调用 propose_plan（其内部调用 LLM）把自然语言转结构化草案。
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


# 决策节点：决定下一步 action。
# 触发顺序：forced_action > 规则兜底 > LLM 路由。
# 其中 LLM 路由通过 route_decision 发生（内部会调用 LLM）。
def _node_decide(state: ChatState) -> ChatState:
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
            "args": {"question": state.get("effective_user_message") or state["user_message"], "top_k": 5},
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
            "args": {"question": state.get("effective_user_message") or state["user_message"], "top_k": 5},
            "assistant_message": "",
            "trace": {
                "intent": "knowledge_query",
                "signals": ctx.signals,
                "why": "LLM 误判 normal_chat，命中 RAG 优先兜底改写",
            },
        }

    _append_trace(
        state,
        _trace_step(
            step_type="llm",
            name="router",
            input={
                "user_message": state["user_message"],
                "effective_user_message": state.get("effective_user_message") or state["user_message"],
                "signals": ctx.signals,
                "summary": state.get("summary"),
                "recent_dialogue_len": len(state.get("recent_dialogue", [])),
            },
            output=state["decision"],
        ),
    )
    return state


# 工具执行节点：根据 decision.action 执行具体分支。
# 这里是“真正做事”的地方，同时也是 LLM 调用最密集的节点之一。
def _build_normal_chat_answer(state: ChatState, decision: dict[str, Any]) -> tuple[str, bool]:
    """统一 normal_chat 文本生成逻辑，供常规分支与 RAG miss 回退复用。"""

    assistant_message = (decision.get("assistant_message") or "").strip()
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


def _node_run_tool(state: ChatState, pending_intents: dict[str, str]) -> ChatState:
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
        question = (args.get("question") or state.get("effective_user_message") or state["user_message"]).strip()
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
                "source_type": c.source_type,
                "doc_topic": c.doc_topic,
            }
            for c in citations
        ]

        if not citations:
            # RAG miss 回退：进入 normal_chat，而不是直接回答“找不到”。
            assistant_message = (decision.get("assistant_message") or "").strip()
            if assistant_message:
                state["answer"] = assistant_message
                used_assistant = True
            else:
                messages: list[dict[str, str]] = [{"role": "system", "content": "你是一个中文生活助手。"}]
                if state.get("profile_context"):
                    messages.append({"role": "system", "content": f"用户画像：{state['profile_context']}"})
                if state.get("pre_advice"):
                    messages.append({"role": "system", "content": f"前置建议：{state['pre_advice']}"})
                if state.get("summary"):
                    messages.append({"role": "system", "content": f"Conversation summary: {state['summary']}"})
                messages.extend(state.get("recent_dialogue", []))
                messages.append({"role": "user", "content": state.get("effective_user_message") or state["user_message"]})
                state["answer"] = chat_completion(messages)
                used_assistant = False
            state["used_tool"] = "normal_chat_fallback"
            _append_trace(
                state,
                _trace_step(
                    step_type="tool",
                    name="query_knowledge",
                    input={"question": question, "top_k": top_k},
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
            question,
            citations,
            profile_context=state.get("profile_context"),
            pre_advice=state.get("pre_advice"),
        )

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
        # 第一阶段 HITL：只缓存意图 + 询问是否生成草案，不直接写入
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

    # normal_chat 分支：
    state["used_tool"] = None
    assistant_message = (decision.get("assistant_message") or "").strip()
    if assistant_message:
        state["answer"] = assistant_message
        used_assistant = True
    else:
        messages: list[dict[str, str]] = [{"role": "system", "content": "你是一个中文生活助手。"}]
        if state.get("profile_context"):
            messages.append({"role": "system", "content": f"用户画像：{state['profile_context']}"})
        if state.get("pre_advice"):
            messages.append({"role": "system", "content": f"前置建议：{state['pre_advice']}"})
        if state.get("summary"):
            messages.append({"role": "system", "content": f"Conversation summary: {state['summary']}"})
        messages.extend(state.get("recent_dialogue", []))
        messages.append({"role": "user", "content": state.get("effective_user_message") or state["user_message"]})
        state["answer"] = chat_completion(messages)
        used_assistant = False

    _append_trace(state, _trace_step(step_type="tool", name="normal_chat", output={"used_assistant_message": used_assistant}))
    _append_trace(state, _trace_step(step_type="llm", name="normal_chat", output=state.get("answer")))
    return state


# 收口节点：保证每条路径都有 answer，并记录最终 used_tool / has_proposal。
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

# 图连线就是“先走谁再走谁”的唯一真相：
# START -> load_context -> hitl_check -> (propose_from_pending | decide | finalize)
# propose_from_pending -> finalize
# decide -> run_tool -> finalize
# finalize -> END
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
) -> ChatState:
    steps = [{"type": "input", "user_message": user_message, "session_id": session_id}]
    if pre_steps:
        steps.extend(pre_steps)

    initial_state: ChatState = {
        "session_id": session_id,
        "user_message": user_message,
        "profile_context": profile_context,
        "pre_advice": pre_advice,
        "effective_user_message": _build_effective_message(user_message, profile_context, pre_advice),
        "trace": {
            "meta": {"graph": "chat_v2", "started_at": datetime.now().isoformat(timespec="seconds")},
            "steps": steps,
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

