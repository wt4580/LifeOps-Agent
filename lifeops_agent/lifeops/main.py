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
from sqlalchemy import select, func

from .db import SessionLocal, init_db
from .models import ChatMessage, TodoItem, MemoryCandidate, ConversationSummary
from .settings import settings
from .llm_qwen import chat_completion
from .memory_extractor import extract_memory_from_dialogue
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


# --------------------------------------------------------------------------------------
# 工具函数：日程/待办查询（tool），避免 LLM 胡编
# --------------------------------------------------------------------------------------

def _is_schedule_query(text: str) -> bool:
    """识别用户是否在“查询”日程/待办，而不是“新增”日程/待办。

    你之前的版本把“周/今天/明天/安排...”都当成查询信号，会导致严重误判：
    - "我明天要去理发" 实际是新增待办，但因为包含"明天"被当成查询

    这里做一个 MVP 但更稳的区分：
    1) 只把“查询句式”当成 query（有什么/有哪些/列出/查看/查一下/我...有什么）
    2) 一旦出现明显的“新增/记录”动词，就判定不是 query（交给待办流程）

    注意：更高级/更鲁棒的方式是引入 LLM router，这里先把闭环做正确。
    """

    text = text.strip()

    # 明显“新增/记录”动词：出现这些就不要走查询
    create_markers = [
        "记住",
        "记录",
        "帮我",
        "提醒我",
        "设个提醒",
        "设置提醒",
        "加入",
        "添加",
        "安排一下",
        "我要",
        "我明天要",
        "我后天要",
        "我下周",
    ]
    if any(m in text for m in create_markers):
        return False

    # 明显的“查询句式/疑问句”：只在这些情况下走工具查询
    query_markers = [
        "有什么",
        "有哪些",
        "有啥",
        "有没有",
        "查看",
        "查一下",
        "查询",
        "列出",
        "我今天有什么",
        "我明天有什么",
        "我这周有什么",
        "我下周有什么",
        "我周",
        "我星期",
    ]
    if any(m in text for m in query_markers):
        return True

    # 兜底：含“？”通常是查询，但仍然要避免被“我要...”误判
    if "?" in text or "？" in text:
        return True

    return False


def _answer_schedule(session_id: str) -> str:
    """查询未来 7 天待办，并让 LLM 做一句话总结。

    为什么“查询 + 总结”的组合很重要：
    - 查询保证事实正确
    - 总结让输出更自然

    注意：这里默认查未来 7 天，你也可以根据用户问题解析时间范围。
    """

    now = datetime.now()
    end = now + timedelta(days=7)

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

    if not rows:
        return "未来7天没有已安排的待办。"

    items = [
        {"title": r.title, "due_at": r.due_at.isoformat() if r.due_at else None}
        for r in rows
    ]

    prompt = "Summarize the upcoming schedule in one or two short sentences."
    return chat_completion(
        [
            {"role": "system", "content": prompt},
            {"role": "user", "content": json.dumps(items, ensure_ascii=False)},
        ],
        temperature=0.2,
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

    # B) 交给状态机执行：内部会完成 Router 决策、工具调用、answer 生成。
    state = run_chat_graph(compiled_graph=chat_graph, session_id=session_id, user_message=req.message)
    answer = state.get("answer") or ""
    proposal_id = state.get("proposal_id")
    proposal_payload = state.get("proposal")
    used_tool = state.get("used_tool")
    citations = state.get("citations") or []
    trace = state.get("trace")

    # C) 保存助手消息，形成完整对话闭环。
    with SessionLocal() as db:
        db.add(ChatMessage(session_id=session_id, role="assistant", content=answer))
        db.commit()

    # D) 尝试刷新摘要（中期记忆）。
    _maybe_update_summary(session_id)

    # E) 旁路增强：结构化记忆抽取（失败不影响主流程）。
    dialogue = [
        {"role": "user", "content": req.message},
        {"role": "assistant", "content": answer},
    ]
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

    # F) 统一返回给前端：答案 + 可选草案 + 可选引用 + trace。
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
