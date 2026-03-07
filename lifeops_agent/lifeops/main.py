from __future__ import annotations

"""lifeops.main

这是整个 LifeOps-Agent Web 服务的入口文件（FastAPI app + 路由）。

你可以把它理解为“智能体的编排器（orchestrator）”：
- 负责把 HTTP 请求转成内部的动作（保存消息 / 检索 / 建议待办 / 查询日程等）
- 负责决定什么时候调用大模型（LLM），什么时候调用“工具函数”（数据库查询/索引/检索）
- 负责把结果序列化成前端需要的 JSON

建议阅读顺序（初学者）：
1) 先看 Pydantic 请求/响应模型（知道接口吃什么、吐什么）。
2) 再看 `/api/chat`（理解主链路与状态机调用）。
3) 再看 `/api/plan/confirm`（理解 HITL 最终写库点）。
4) 最后看 `/api/index` + `/api/ask`（理解 RAG 外围链路）。

当前架构的核心思想：
1) 所有对话都落库（chat_messages），这是“短期记忆”的来源。
2) 提供一个轻量的“会话摘要”机制（conversation_summaries），用于跨重启的“中期记忆”。
3) 对于明确可工具化的问题（查询待办/日程），优先走工具查询，再用 LLM 做自然语言总结。
4) 对于“可能要写入待办”的意图，严格 Human-in-the-loop：
   - 第一步：检测到意图 -> 只给出固定引导语（不写入）
   - 第二步：用户确认（需要/好/可以）-> 才生成待办草案 proposal
   - 第三步：用户点击确认按钮 -> 才真正写入 todo_items

注意：本文件中有一些设计上的权衡：
- 我们用了关键词/规则做意图检测，这是为了减少 LLM 调用成本、提高可控性。
- 同时在检索部分引入了 LLM query rewrite，让检索更“灵活”。

"""

import json
import logging
from datetime import datetime, timedelta
from uuid import uuid4

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import select, func, delete

from .db import SessionLocal, init_db
from .models import ChatMessage, TodoItem, MemoryCandidate, ConversationSummary, LifeEvent, UserProfile
from .settings import settings
from .llm_qwen import chat_completion
from .memory_extractor import extract_memory_from_dialogue
from .personal_memory import (
    build_profile_context,
    extract_personal_events,
    extract_profile_facts,
    generate_profile_pre_advice,
    parse_event_time,
    decide_proactive_advice,
)
from .planner import propose_plan
from .graph_chat import build_chat_graph, run_chat_graph
from .retrieval import index_documents


# --------------------------------------------------------------------------------------
# 日志与 FastAPI 基础配置
# --------------------------------------------------------------------------------------

# 配置根日志级别（由 .env 中 LOG_LEVEL 控制）
logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)

# 创建 FastAPI 应用实例
app = FastAPI()

# CORS：这里为了演示简单，放开所有来源。
# 若你要上线，建议只允许自己的域名/端口。
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 模板和静态文件目录：
# - / 走 Jinja2 渲染 index.html
# - /static 挂载静态资源（app.js / style.css）
app.mount("/static", StaticFiles(directory="lifeops/static"), name="static")
templates = Jinja2Templates(directory="lifeops/templates")


# --------------------------------------------------------------------------------------
# Pydantic 数据结构（HTTP 入参/出参）
# --------------------------------------------------------------------------------------

class ChatRequest(BaseModel):
    """聊天入参。

    session_id:
      - 前端保存在 localStorage 的会话 ID（UUID）。
      - 首次没有则由后端生成。

    message:
      - 用户输入的自然语言文本。
    """

    message: str
    session_id: str | None = None


class PlanProposeRequest(BaseModel):
    """保留接口：显式让用户生成计划草案（目前 UI 已经不强依赖它）。"""

    text: str


class PlanConfirmRequest(BaseModel):
    """用户确认把草案写入 todo_items 表。"""

    proposal_id: str


class AskRequest(BaseModel):
    """知识库问答入参。"""

    question: str


class ChatResponse(BaseModel):
    """聊天出参。

    answer:
      - 助手回复。

    proposal_id / proposal:
      - 如果本轮触发了“待办草案”，会一并返回。
      - 前端可展示草案，并提供确认按钮。

    used_tool:
      - 便于调试：告诉你这次走了哪个分支（schedule/plan_gate/plan_proposal/None）。

    citations:
      - 当本轮触发知识库工具时，返回引用列表（path/page/snippet/score/reason）。
      - 前端可直接展示“答案来自哪里”。

    trace:
      - 可审计决策轨迹，包含 router 决策、工具输入输出、关键中间结果。
    """

    answer: str
    session_id: str
    proposal_id: str | None = None
    proposal: dict | None = None
    used_tool: str | None = None
    citations: list[dict] | None = None
    trace: dict | None = None


# --------------------------------------------------------------------------------------
# 服务运行时的内存状态（非持久化）
# --------------------------------------------------------------------------------------

# proposal_cache: proposal_id -> items
# 说明：这是为了让“确认写入”时能找到草案内容。
# 注意：它是进程内内存，服务重启会丢失。
proposal_cache: dict[str, list[dict]] = {}

# pending_intents: session_id -> 待办意图的原始文本
# 说明：用于两阶段确认：
# - 第一阶段：用户说了一句像待办的内容，我们先问“要不要生成草案”
# - 第二阶段：用户说“需要/好/可以”，我们才把 pending_text 拿出来生成草案
pending_intents: dict[str, str] = {}

# LangGraph 编排实例：启动时初始化。
chat_graph = None

# 每多少条消息刷新一次摘要（越小越频繁，也越费 token/cost）
SUMMARY_REFRESH_TURNS = 10


def _resolve_memory_owner_id(session_id: str) -> str:
    """根据配置决定个人记忆写入/读取使用的 owner id。"""

    scope = (settings.personal_memory_scope or "global").strip().lower()
    if scope == "session":
        return session_id
    return (settings.personal_memory_global_id or "default_user").strip() or "default_user"


# --------------------------------------------------------------------------------------
# 生命周期：启动时初始化数据库表
# --------------------------------------------------------------------------------------

@app.on_event("startup")
def on_startup() -> None:
    """FastAPI 启动时回调：创建 SQLite 表并初始化 LangGraph。"""

    global chat_graph
    init_db()
    chat_graph = build_chat_graph(pending_intents, proposal_cache)


# --------------------------------------------------------------------------------------
# Web 页面：渲染首页
# --------------------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    """渲染单页 UI。"""

    return templates.TemplateResponse("index.html", {"request": request})


# --------------------------------------------------------------------------------------
# 健康检查：用于部署或本地排查
# --------------------------------------------------------------------------------------

@app.get("/health")
def health_check():
    return {"status": "ok"}


# --------------------------------------------------------------------------------------
# 工具函数：从数据库加载短期对话、摘要、以及生成摘要
# --------------------------------------------------------------------------------------

def _load_recent_dialogue(session_id: str, limit: int = 12) -> list[dict]:
    """读取最近 N 条对话消息，用于构建 LLM 上下文。

    为什么需要 limit：
    - LLM 上下文有 token 限制
    - 全量对话可能很长，成本高

    返回格式：[{role: "user"|"assistant", content: "..."}, ...]
    """

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

    # DB 查询是倒序，为了符合 chat.completions 的顺序，需要反转
    rows.reverse()
    return [{"role": r.role, "content": r.content} for r in rows]


def _get_summary(session_id: str) -> str | None:
    """取最新一条会话摘要（如果存在）。

    摘要最大的价值：
    - 当服务重启 / 对话很长时，仍能给 LLM 一个“压缩后的记忆”。
    """

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


def _save_summary(session_id: str, summary_text: str) -> None:
    """保存一条摘要。这里用“追加写”，而不是覆盖更新，方便回溯历史。"""

    with SessionLocal() as db:
        db.add(ConversationSummary(session_id=session_id, summary_text=summary_text))
        db.commit()


def _maybe_update_summary(session_id: str) -> None:
    """按频率刷新摘要。

    触发条件：当前 session 的消息数能被 SUMMARY_REFRESH_TURNS 整除。

    注意：
    - 这是一个非常简化的实现，真实产品可以：
      - 用后台任务异步做
      - 做增量摘要（summary-of-summaries）
      - 用更强的结构化 schema
    """

    with SessionLocal() as db:
        count = db.execute(
            select(func.count()).select_from(ChatMessage).where(ChatMessage.session_id == session_id)
        ).scalar_one()

    if count > 0 and count % SUMMARY_REFRESH_TURNS == 0:
        recent = _load_recent_dialogue(session_id, limit=SUMMARY_REFRESH_TURNS * 2)
        if not recent:
            return

        prompt = (
            "Summarize the conversation for later retrieval. "
            "Focus on commitments, plans, preferences, and open questions. "
            "Return a concise paragraph."
        )

        # 用 LLM 把最近一段对话压缩成摘要
        summary_text = chat_completion(
            [
                {"role": "system", "content": prompt},
                {"role": "user", "content": json.dumps(recent, ensure_ascii=False)},
            ],
            temperature=0.2,
        )
        _save_summary(session_id, summary_text)


def _maybe_prune_chat_history(session_id: str) -> None:
    """当聊天记录过长时执行“摘要 + 删除最旧消息”的滚动压缩。"""

    with SessionLocal() as db:
        rows = (
            db.execute(
                select(ChatMessage)
                .where(ChatMessage.session_id == session_id)
                .order_by(ChatMessage.created_at.asc())
            )
            .scalars()
            .all()
        )

    total = len(rows)
    if total <= settings.chat_retention_limit:
        return

    target = settings.chat_prune_target
    if target >= settings.chat_retention_limit:
        target = max(100, settings.chat_retention_limit - 100)

    prune_count = max(0, total - target)
    if prune_count <= 0:
        return

    old_rows = rows[:prune_count]
    old_dialogue = [{"role": r.role, "content": r.content} for r in old_rows]

    summary_text = ""
    try:
        summary_text = chat_completion(
            [
                {
                    "role": "system",
                    "content": (
                        "请把这段历史对话压缩成长期记忆摘要。"
                        "重点保留：个人偏好、长期目标、反复出现的任务、已确认事实。"
                        "输出一段简洁中文。"
                    ),
                },
                {"role": "user", "content": json.dumps(old_dialogue, ensure_ascii=False)},
            ],
            temperature=0.2,
        )
    except Exception as exc:
        logger.warning("Chat prune summary failed: %s", exc)

    if summary_text.strip():
        _save_summary(session_id, f"[rolling-prune] {summary_text.strip()}")

    ids = [r.id for r in old_rows]
    with SessionLocal() as db:
        db.execute(delete(ChatMessage).where(ChatMessage.id.in_(ids)))
        db.commit()

    logger.info(
        "Pruned chat history session_id=%s total=%s pruned=%s kept=%s",
        session_id,
        total,
        prune_count,
        total - prune_count,
    )


def _store_personal_events(session_id: str, source_text: str, events: list[dict]) -> int:
    """把抽取到的个人事件落库到 life_events。"""

    if not events:
        return 0

    inserted = 0
    with SessionLocal() as db:
        for item in events:
            db.add(
                LifeEvent(
                    # 这里的 session_id 实际上作为“记忆 owner id”使用，可按配置跨会话共享。
                    session_id=session_id,
                    category=item.get("category", "other"),
                    title=item.get("title", ""),
                    event_time=parse_event_time(item.get("event_time")),
                    amount=item.get("amount"),
                    amount_unit=item.get("amount_unit"),
                    tags_json=json.dumps(item.get("tags", []), ensure_ascii=False),
                    notes=item.get("notes"),
                    source_text=source_text,
                )
            )
            inserted += 1
        db.commit()
    return inserted


def _load_recent_life_events(session_id: str, limit: int = 30) -> list[dict]:
    """读取某个记忆 owner 的近期个人事件，供主动建议使用。

    已改为 `_load_life_events_in_window` 按时间窗口读取，不再保留固定条数版 `_load_recent_life_events`。
    """

    with SessionLocal() as db:
        rows = (
            db.execute(
                select(LifeEvent)
                .where(LifeEvent.session_id == session_id)
                .order_by(LifeEvent.created_at.desc())
                .limit(limit)
            )
            .scalars()
            .all()
        )

    rows.reverse()
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
                "amount": r.amount,
                "amount_unit": r.amount_unit,
                "tags": tags,
                "notes": r.notes,
            }
        )
    return out


def _load_upcoming_todos_for_advice(session_id: str, hours: int = 24) -> list[dict]:
    """读取未来一段时间的待办，供冲突/提醒建议。"""

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


def _load_user_profile(session_id: str) -> dict | None:
    """读取某个记忆 owner 的身份画像。"""

    with SessionLocal() as db:
        row = (
            db.execute(select(UserProfile).where(UserProfile.session_id == session_id).limit(1))
            .scalars()
            .first()
        )

    if not row:
        return None

    try:
        preferences = json.loads(row.preferences_json) if row.preferences_json else []
    except json.JSONDecodeError:
        preferences = []
    try:
        conditions = json.loads(row.conditions_json) if row.conditions_json else []
    except json.JSONDecodeError:
        conditions = []

    return {
        "height_cm": row.height_cm,
        "weight_kg": row.weight_kg,
        "preferences": preferences,
        "conditions": conditions,
        "notes": row.notes,
    }


def _merge_unique(base: list[str], patch: list[str]) -> list[str]:
    seen = {x.strip() for x in base if x and x.strip()}
    out = [x for x in base if x and x.strip()]
    for item in patch:
        t = (item or "").strip()
        if not t or t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


def _upsert_user_profile(session_id: str, patch: dict | None) -> bool:
    """把画像补丁写回 user_profiles（增量更新，按记忆 owner 聚合）。"""

    if not patch:
        return False

    with SessionLocal() as db:
        row = (
            db.execute(select(UserProfile).where(UserProfile.session_id == session_id).limit(1))
            .scalars()
            .first()
        )
        if not row:
            row = UserProfile(session_id=session_id)
            db.add(row)
            db.flush()

        changed = False

        h = patch.get("height_cm")
        if h is not None and h > 0:
            row.height_cm = float(h)
            changed = True

        w = patch.get("weight_kg")
        if w is not None and w > 0:
            row.weight_kg = float(w)
            changed = True

        try:
            cur_prefs = json.loads(row.preferences_json) if row.preferences_json else []
        except json.JSONDecodeError:
            cur_prefs = []
        merged_prefs = _merge_unique(cur_prefs, patch.get("preferences") or [])
        if merged_prefs != cur_prefs:
            row.preferences_json = json.dumps(merged_prefs, ensure_ascii=False)
            changed = True

        try:
            cur_conds = json.loads(row.conditions_json) if row.conditions_json else []
        except json.JSONDecodeError:
            cur_conds = []
        merged_conds = _merge_unique(cur_conds, patch.get("conditions") or [])
        if merged_conds != cur_conds:
            row.conditions_json = json.dumps(merged_conds, ensure_ascii=False)
            changed = True

        notes = (patch.get("notes") or "").strip()
        if notes and notes != (row.notes or ""):
            row.notes = notes
            changed = True

        if changed:
            db.commit()
        else:
            db.rollback()

    return changed


def _load_life_events_in_window(session_id: str, hours: int, now_dt: datetime) -> tuple[list[dict], str, str]:
    """按时间窗口读取个人事件（不再按固定条数），并返回窗口起止时间。"""

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


# --------------------------------------------------------------------------------------
# API：聊天（智能体编排核心）
# --------------------------------------------------------------------------------------

@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """聊天入口（LangGraph 状态机版）。

    主流程：
    Chat -> LLM decide -> Tool -> Update state -> (maybe HITL) -> Final answer

    关键顺序（为什么这样做）：
    1) 先保存用户消息：保证即使后续调用失败，也能保留用户输入审计记录。
    2) 再执行状态机：统一路由、工具调用与回复生成。
    3) 再保存助手消息：形成完整对话对（user/assistant）。
    4) 再做摘要与记忆抽取：这些属于增强能力，不阻塞主回复链路。
    """

    session_id = req.session_id or str(uuid4())

    # A) 保存用户消息（审计优先）。
    with SessionLocal() as db:
        db.add(ChatMessage(session_id=session_id, role="user", content=req.message))
        db.commit()

    if chat_graph is None:
        raise HTTPException(status_code=500, detail="chat graph not initialized")

    # A1) 前置执行：画像更新 + 个人事件抽取落库 + 前置知识提示生成。
    recent_for_profile = _load_recent_dialogue(session_id, limit=12)
    memory_owner_id = _resolve_memory_owner_id(session_id)
    now_dt = datetime.now()
    now_iso = now_dt.isoformat(timespec="seconds")

    profile_patch = extract_profile_facts(
        req.message,
        recent_for_profile,
        now_iso=now_iso,
        session_id=session_id,
    )
    profile_changed = _upsert_user_profile(memory_owner_id, profile_patch.model_dump() if profile_patch else None)
    profile = _load_user_profile(memory_owner_id)
    profile_context = build_profile_context(profile)

    personal = extract_personal_events(
        req.message,
        recent_for_profile,
        now_iso=now_iso,
        session_id=session_id,
    )
    personal_inserted = 0
    if personal and personal.items:
        personal_inserted = _store_personal_events(
            memory_owner_id,
            req.message,
            [item.model_dump() for item in personal.items],
        )

    recent_events, event_window_start, event_window_end = _load_life_events_in_window(
        session_id=memory_owner_id,
        hours=settings.proactive_event_window_hours,
        now_dt=now_dt,
    )
    pre_advice = generate_profile_pre_advice(
        user_message=req.message,
        profile_context=profile_context,
        recent_events=recent_events,
        now_iso=now_iso,
        session_id=session_id,
    )

    pre_steps = [
        {
            "type": "preprocess",
            "name": "profile_context",
            "ts": datetime.now().isoformat(timespec="seconds"),
            "output": {
                "profile_changed": profile_changed,
                "has_profile": bool(profile_context),
                "profile_context": profile_context or "",
                "memory_scope": settings.personal_memory_scope,
                "memory_owner_id": memory_owner_id,
            },
        },
        {
            "type": "preprocess",
            "name": "personal_events",
            "ts": datetime.now().isoformat(timespec="seconds"),
            "output": {
                "inserted": personal_inserted,
                "recent_events_count": len(recent_events),
                "event_window_hours": settings.proactive_event_window_hours,
                "event_window_start": event_window_start,
                "event_window_end": event_window_end,
            },
        },
        {
            "type": "preprocess",
            "name": "pre_graph_hint",
            "ts": datetime.now().isoformat(timespec="seconds"),
            "output": {
                "pre_advice": pre_advice or "NONE",
                "source": "profile+life_events",
            },
        },
    ]

    # B) 交给状态机执行：graph 从一开始就能使用画像与前置知识提示。
    state = run_chat_graph(
        compiled_graph=chat_graph,
        session_id=session_id,
        user_message=req.message,
        profile_context=profile_context or None,
        pre_advice=pre_advice,
        pre_steps=pre_steps,
    )
    answer = state.get("answer") or ""
    proposal_id = state.get("proposal_id")
    proposal_payload = state.get("proposal")
    used_tool = state.get("used_tool")
    citations = state.get("citations") or []
    trace = state.get("trace")

    # B2) 阈值门控：只有提醒必要性分数超过阈值，才追加“补充建议”。
    upcoming_todos = _load_upcoming_todos_for_advice(
        session_id=memory_owner_id,
        hours=settings.proactive_todo_hours,
    )
    decision = decide_proactive_advice(
        user_message=req.message,
        assistant_answer=answer,
        recent_events=recent_events,
        upcoming_todos=upcoming_todos,
        profile_context=profile_context,
        now_iso=now_iso,
        threshold=settings.proactive_advice_threshold,
        session_id=session_id,
    )

    advice_added = False
    if decision.should_add and decision.advice and decision.advice not in answer:
        answer = f"{answer}\n\n补充建议：{decision.advice}".strip()
        advice_added = True

    if isinstance(trace, dict):
        steps = trace.get("steps")
        if isinstance(steps, list):
            steps.append(
                {
                    "type": "postprocess",
                    "name": "proactive_advice_threshold",
                    "ts": datetime.now().isoformat(timespec="seconds"),
                    "output": {
                        "score": decision.score,
                        "threshold": settings.proactive_advice_threshold,
                        "added": advice_added,
                        "reasons": decision.reasons,
                        "events_used": len(recent_events),
                        "upcoming_todos_used": len(upcoming_todos),
                        "now_iso": now_iso,
                        "event_window_start": event_window_start,
                        "event_window_end": event_window_end,
                    },
                }
            )

    logger.info(
        "proactive advice decision score=%.3f threshold=%.3f added=%s events=%s todos=%s",
        decision.score,
        settings.proactive_advice_threshold,
        advice_added,
        len(recent_events),
        len(upcoming_todos),
    )

    # C) 保存助手消息，形成完整对话闭环。
    with SessionLocal() as db:
        db.add(ChatMessage(session_id=session_id, role="assistant", content=answer))
        db.commit()

    # D) 尝试刷新摘要（中期记忆）。
    _maybe_update_summary(session_id)
    # E) 超过上限后滚动压缩聊天记录（摘要+删旧）。
    _maybe_prune_chat_history(session_id)

    # F) 旁路增强：结构化记忆抽取（失败不影响主流程）。
    dialogue = _load_recent_dialogue(session_id, limit=8)
    extraction = extract_memory_from_dialogue(dialogue)
    if extraction:
        with SessionLocal() as db:
            for item in extraction.items:
                db.add(
                    MemoryCandidate(
                        kind=item.kind,
                        title=item.title,
                        occurred_at=item.occurred_at,
                        notes=item.notes,
                    )
                )
            db.commit()

    # G) 统一返回给前端：答案 + 可选草案 + 可选引用 + trace。
    return {
        "answer": answer,
        "session_id": session_id,
        "proposal_id": proposal_id,
        "proposal": proposal_payload,
        "used_tool": used_tool,
        "citations": citations,
        "trace": trace,
    }


# --------------------------------------------------------------------------------------
# API：计划草案与确认（Human-in-the-loop）
# --------------------------------------------------------------------------------------

@app.post("/api/plan/propose")
def plan_propose(req: PlanProposeRequest):
    """显式生成草案（保留接口）。"""

    proposal_id, proposal = propose_plan(req.text)
    proposal_cache[proposal_id] = [item.model_dump() for item in proposal.items]
    return {"proposal_id": proposal_id, "proposal": proposal.model_dump()}


@app.post("/api/plan/confirm")
def plan_confirm(req: PlanConfirmRequest):
    """用户确认写入 todo_items。

    这里从 proposal_cache 取出 items，写入数据库。
    注意：proposal_cache 是内存态，服务重启后 proposal_id 会失效（MVP 的取舍）。
    """

    items = proposal_cache.get(req.proposal_id, [])
    if not items:
        raise HTTPException(status_code=404, detail="proposal_id not found (maybe server restarted)")

    inserted = 0
    with SessionLocal() as db:
        for item in items:
            due_at = None
            if item.get("due_at"):
                try:
                    due_at = datetime.fromisoformat(item["due_at"])
                except ValueError:
                    due_at = None
            db.add(TodoItem(title=item["title"], due_at=due_at, source="proposal"))
            inserted += 1
        db.commit()

    return {"inserted": inserted}


# --------------------------------------------------------------------------------------
# API：待办查询
# --------------------------------------------------------------------------------------

@app.get("/api/todos/today")
def todos_today():
    """查询今天的待办（仅 due_at 不为空的）。"""

    today = datetime.now().date()
    tomorrow = today + timedelta(days=1)

    with SessionLocal() as db:
        rows = (
            db.execute(
                select(TodoItem)
                .where(TodoItem.due_at != None)
                .where(TodoItem.due_at >= datetime.combine(today, datetime.min.time()))
                .where(TodoItem.due_at < datetime.combine(tomorrow, datetime.min.time()))
                .order_by(TodoItem.due_at.asc())
            )
            .scalars()
            .all()
        )

    items = [
        {"id": r.id, "title": r.title, "due_at": r.due_at.isoformat() if r.due_at else None}
        for r in rows
    ]

    # 可选：让 LLM 对清单做一句话总结
    summary = None
    if items:
        prompt = "Summarize today's todo list in one short sentence."
        summary = chat_completion(
            [
                {"role": "system", "content": prompt},
                {"role": "user", "content": json.dumps(items, ensure_ascii=False)},
            ],
            temperature=0.2,
        )

    return {"items": items, "summary": summary}


@app.get("/api/todos/upcoming")
def todos_upcoming(days: int = 7):
    """查询未来 N 天待办（默认 7）。"""

    start = datetime.now()
    end = start + timedelta(days=days)

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

    return {
        "items": [
            {"id": r.id, "title": r.title, "due_at": r.due_at.isoformat() if r.due_at else None}
            for r in rows
        ]
    }


# --------------------------------------------------------------------------------------
# API：知识库索引与问答（RAG）
# --------------------------------------------------------------------------------------

@app.post("/api/index")
def index_docs(rebuild: int = 0):
    """扫描本地 docs_dir 并把文本 chunks 写入 document_chunks。

    文件本体不移动、不复制：
    - 只存 file_path + page + chunk_text
    - 通过 hash 去重

    参数：
    - rebuild=1：先清空 document_chunks 再重建索引（用于 OCR/分块策略改变后的刷新）

    返回：
    - 至少包含 inserted / skipped。
    - 在 langchain/hybrid 后端下，可能额外返回 vector_chunks / vector_dir。
    """

    return index_documents(rebuild=bool(rebuild))


@app.post("/api/ask")
def ask(req: AskRequest):
    """知识库问答入口。

    这里复用同一套 LangGraph，而不是单独再写一套 RAG 流程：
    - 通过 forced_action="query_knowledge" 强制走知识库工具节点；
    - 保证 /api/chat 与 /api/ask 的可观测性和工具行为一致。
    """

    if chat_graph is None:
        raise HTTPException(status_code=500, detail="chat graph not initialized")

    state = run_chat_graph(
        compiled_graph=chat_graph,
        session_id=f"ask-{uuid4()}",
        user_message=req.question,
        forced_action="query_knowledge",
        forced_args={"question": req.question, "top_k": 5},
    )

    return {
        "answer": state.get("answer") or "No relevant context found.",
        "citations": state.get("citations") or [],
        "trace": state.get("trace"),
    }
