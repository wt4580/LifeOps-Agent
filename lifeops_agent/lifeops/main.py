from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from sqlalchemy import select

from .db import SessionLocal, init_db
from .models import ChatMessage, TodoItem, MemoryCandidate
from .settings import settings
from .llm_qwen import chat_completion
from .memory_extractor import extract_memory_from_dialogue
from .planner import propose_plan, detect_plan_intent, detect_affirmation
from .retrieval import index_documents, search_chunks, rag_answer


logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)

app = FastAPI()
app.mount("/static", StaticFiles(directory="lifeops/static"), name="static")
templates = Jinja2Templates(directory="lifeops/templates")


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class PlanProposeRequest(BaseModel):
    text: str


class PlanConfirmRequest(BaseModel):
    proposal_id: str


class AskRequest(BaseModel):
    question: str


class TodoSummaryResponse(BaseModel):
    items: list[dict]
    summary: str | None = None


class ChatResponse(BaseModel):
    answer: str
    session_id: str
    proposal_id: str | None = None
    proposal: dict | None = None


proposal_cache: dict[str, list[dict]] = {}
pending_intents: dict[str, str] = {}


@app.on_event("startup")
def on_startup() -> None:
    init_db()


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


def _load_recent_dialogue(session_id: str, limit: int = 12) -> list[dict]:
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


def _get_last_user_message(session_id: str) -> str | None:
    with SessionLocal() as db:
        rows = (
            db.execute(
                select(ChatMessage)
                .where(ChatMessage.session_id == session_id)
                .where(ChatMessage.role == "user")
                .order_by(ChatMessage.created_at.desc())
                .limit(2)
            )
            .scalars()
            .all()
        )
    if len(rows) < 2:
        return None
    return rows[1].content


@app.post("/api/chat")
def chat(req: ChatRequest):
    session_id = req.session_id or str(uuid4())
    with SessionLocal() as db:
        db.add(ChatMessage(session_id=session_id, role="user", content=req.message))
        db.commit()

    answer = None
    proposal_id = None
    proposal_payload = None

    if detect_plan_intent(req.message):
        pending_intents[session_id] = req.message
        answer = "我可以为你设置提醒。需要我为你生成待办草案吗？回复“需要/好/可以”即可。"
    elif detect_affirmation(req.message):
        pending_text = pending_intents.pop(session_id, None)
        if pending_text:
            proposal_id, proposal = propose_plan(pending_text)
            proposal_payload = proposal.model_dump()
            proposal_cache[proposal_id] = [item.model_dump() for item in proposal.items]
            answer = "已生成待办草案，请确认是否加入待办。"

    if answer is None:
        messages = [
            {"role": "system", "content": "You are a helpful life assistant."},
            *(_load_recent_dialogue(session_id)),
        ]
        answer = chat_completion(messages)

    with SessionLocal() as db:
        db.add(ChatMessage(session_id=session_id, role="assistant", content=answer))
        db.commit()

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

    return {
        "answer": answer,
        "session_id": session_id,
        "proposal_id": proposal_id,
        "proposal": proposal_payload,
    }


@app.post("/api/plan/propose")
def plan_propose(req: PlanProposeRequest):
    proposal_id, proposal = propose_plan(req.text)
    proposal_cache[proposal_id] = [item.model_dump() for item in proposal.items]
    return {"proposal_id": proposal_id, "proposal": proposal.model_dump()}


@app.post("/api/plan/confirm")
def plan_confirm(req: PlanConfirmRequest):
    items = proposal_cache.get(req.proposal_id, [])
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


@app.get("/api/todos/today")
def todos_today():
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
    summary = None
    if items:
        prompt = "Summarize today's todo list in one short sentence."
        summary = chat_completion([
            {"role": "system", "content": prompt},
            {"role": "user", "content": json.dumps(items, ensure_ascii=False)},
        ], temperature=0.2)
    return {"items": items, "summary": summary}


@app.get("/api/todos/upcoming")
def todos_upcoming(days: int = 7):
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
    return {"items": [
        {"id": r.id, "title": r.title, "due_at": r.due_at.isoformat() if r.due_at else None}
        for r in rows
    ]}


@app.post("/api/index")
def index_docs():
    result = index_documents()
    return result


@app.post("/api/ask")
def ask(req: AskRequest):
    citations = search_chunks(req.question, top_k=5)
    answer = rag_answer(req.question, citations) if citations else "No relevant context found."
    return {
        "answer": answer,
        "citations": [
            {"path": c.path, "page": c.page, "snippet": c.snippet, "score": c.score}
            for c in citations
        ],
    }
