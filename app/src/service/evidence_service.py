"""Bounded, identity-scoped user statements for personalized RAG.

A statement is not automatically a fact: it may be a question, denial or scenario.
Knowledge files and assistant answers never enter this channel.
"""
from sqlalchemy import select

from ..common.config.db_config import SessionLocal
from ..domain.entity.chat_entity import ChatMessage


def needs_user_history(question):
    text = (question or "").lower()
    return any(term in text for term in ("我", "刚才", "刚刚", "之前", "今天", "昨天", "早餐", "午餐", "晚餐",
                                         "饮食", "吃得", "吃的", "运动", "睡眠", "my ", "i "))


def load_user_statements(session_id, *, before_message_id=None, limit=20):
    if not session_id:
        return []
    limit = max(1, min(int(limit), 20))
    with SessionLocal() as db:
        query = select(ChatMessage).where(ChatMessage.session_id == session_id, ChatMessage.role == "user")
        if before_message_id is not None:
            query = query.where(ChatMessage.id < before_message_id)
        rows = db.scalars(query.order_by(ChatMessage.id.desc()).limit(limit)).all()
        remaining, statements = 6000, []
        for row in rows:
            if remaining <= 0:
                break
            original = row.content or ""
            content = original[:min(1000, remaining)]
            statements.append({"message_id": row.id, "evidence_role": "user_statement",
                               "recorded_at_utc": row.created_at.isoformat(timespec="seconds") + "Z" if row.created_at else None,
                               "content": content, "content_truncated": len(content) < len(original)})
            remaining -= len(content)
        return list(reversed(statements))
