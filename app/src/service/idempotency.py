"""Atomic business receipts; no write transaction is held across model calls."""
from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

from sqlalchemy import update, event, select
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy.dialects.sqlite import insert

from ..domain.entity.chat_entity import ChatMessage, ChatRequest, ChatExecution, WriteOperation
from ..common.domain.request_errors import ChatRequestRejected

MISSING = object()


class IdempotencyConflict(ValueError):
    pass


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _saved_response(value, session_id):
    if value is None:
        return None
    try:
        result = json.loads(value)
    except (TypeError, ValueError):
        result = None
    if not isinstance(result, dict) or result.get("session_id") != session_id or not isinstance(result.get("answer"), str):
        raise ChatRequestRejected("request_result_corrupt", "请求回执无法校验，已阻止重新执行，请核对实际记录或恢复数据。")
    return result


@dataclass
class WriteScope:
    session_id: str
    key: str
    ordinal: int = 0
    committed: dict = field(default_factory=dict, repr=False)


_scope = ContextVar("lifeops_write_scope", default=None)


@contextmanager
def write_scope(session_id, key):
    scope = WriteScope(session_id, key)
    token = _scope.set(scope)
    try:
        yield scope
    finally:
        _scope.reset(token)


def begin_write(db, session_id, kind, payload, operation_key=None):
    scope = _scope.get()
    if scope is not None and scope.session_id != session_id:
        raise IdempotencyConflict("写入身份与执行上下文不一致")
    if operation_key is None and scope is not None:
        operation_key = f"{scope.key}:{scope.ordinal}"
        scope.ordinal += 1
    if operation_key is None:
        return None, MISSING
    key = _hash([session_id, kind, operation_key])
    fingerprint = _hash(payload)
    db.execute(insert(WriteOperation).values(id=key, session_id=session_id, payload_hash=fingerprint)
               .on_conflict_do_nothing(index_elements=["id"]))
    row = db.get(WriteOperation, key)
    if row.payload_hash != fingerprint:
        raise IdempotencyConflict("同一操作标识不能用于不同的写入内容")
    db.info.setdefault("turn_write_receipts", {})[key] = {"kind": kind, "payload": payload}
    if row.result_json is not None:
        _remember_committed(key, {"kind": kind, "payload": payload, "result": json.loads(row.result_json)})
        return key, json.loads(row.result_json)
    return key, MISSING


def finish_write(db, key, result):
    if key is not None:
        entry = db.info.get("turn_write_receipts", {}).get(key)
        if entry is not None:
            entry["result"] = result
        db.execute(update(WriteOperation).where(WriteOperation.id == key)
                   .values(result_json=json.dumps(result, ensure_ascii=False)))


def _remember_committed(key, entry):
    from ..common.harness.context import get_turn_context
    turn = get_turn_context(None)
    snapshot = json.loads(json.dumps(entry, ensure_ascii=False, default=str))
    scope = _scope.get()
    if scope is not None:
        scope.committed[key] = snapshot
    if turn is not None:
        turn.committed_writes[key] = snapshot


@event.listens_for(Session, "after_commit")
def _remember_turn_writes(db):
    for key, entry in db.info.pop("turn_write_receipts", {}).items():
        if "result" in entry:
            _remember_committed(key, entry)


@event.listens_for(Session, "after_rollback")
def _discard_uncommitted_writes(db):
    db.info.pop("turn_write_receipts", None)


def saved_write(factory, session_id, kind, operation_key):
    with factory() as db:
        row = db.get(WriteOperation, _hash([session_id, kind, operation_key]))
        return json.loads(row.result_json) if row and row.result_json is not None else MISSING


def start_request(factory, session_id, request_id, message, proposal_id):
    """Persist input once, and return a saved response for completed retries."""
    key = _hash([session_id, request_id])
    fingerprint = request_fingerprint(message, proposal_id)
    with factory() as db:
        db.execute(insert(ChatRequest).values(id=key, session_id=session_id, payload_hash=fingerprint)
                   .on_conflict_do_nothing(index_elements=["id"]))
        row = db.get(ChatRequest, key)
        if row.payload_hash != fingerprint:
            raise IdempotencyConflict("同一 request_id 不能用于不同的消息或草案")
        if row.message_id is None:
            msg = ChatMessage(session_id=session_id, role="user", content=message)
            db.add(msg)
            db.flush()
            row.message_id = msg.id
            db.add(ChatExecution(request_key=key, session_id=session_id, message_id=msg.id))
        result = _saved_response(row.result_json, session_id)
        message_id = row.message_id
        db.commit()
    return key, message_id, result


def request_fingerprint(message, proposal_id):
    return _hash([message, proposal_id])


def finish_request(db, key, result):
    db.execute(update(ChatRequest).where(ChatRequest.id == key)
               .values(result_json=json.dumps(result, ensure_ascii=False)))


def claim_request(factory, key, session_id, message_id):
    """At most one dispatcher, including concurrent callers bypassing the FIFO."""
    with factory() as db:
        changed = db.execute(update(ChatExecution).where(
            ChatExecution.request_key == key, ChatExecution.session_id == session_id,
            ChatExecution.message_id == message_id, ChatExecution.claimed_at.is_(None),
        ).values(claimed_at=datetime.utcnow())).rowcount
        db.commit()
        return changed == 1


def request_dispatched(factory, key, session_id, message_id):
    with factory() as db:
        execution = db.get(ChatExecution, key)
        if execution is None:
            return True  # Legacy unfinished requests have no trustworthy dispatch record.
        if execution.session_id != session_id or execution.message_id != message_id:
            raise IdempotencyConflict("请求执行记录身份不匹配")
        return execution.claimed_at is not None


def request_result(factory, key, session_id, message_id):
    with factory() as db:
        value = db.scalar(select(ChatRequest.result_json).where(
            ChatRequest.id == key, ChatRequest.session_id == session_id, ChatRequest.message_id == message_id))
        return _saved_response(value, session_id)


def persist_response(factory, key, session_id, message_id, result):
    """The winning response and assistant message commit together, never twice."""
    with factory() as db:
        changed = db.execute(update(ChatRequest).where(
            ChatRequest.id == key, ChatRequest.session_id == session_id, ChatRequest.message_id == message_id,
            ChatRequest.result_json.is_(None),
        ).values(result_json=json.dumps(result, ensure_ascii=False))).rowcount
        if changed:
            db.execute(insert(ChatExecution).values(request_key=key, session_id=session_id, message_id=message_id,
                                                   claimed_at=datetime.utcnow()).on_conflict_do_nothing(index_elements=["request_key"]))
            db.execute(update(ChatExecution).where(ChatExecution.request_key == key, ChatExecution.session_id == session_id,
                                                  ChatExecution.message_id == message_id, ChatExecution.claimed_at.is_(None))
                       .values(claimed_at=datetime.utcnow()))
            db.add(ChatMessage(session_id=session_id, role="assistant", content=result.get("answer", "")))
            db.commit()
            return result
        saved = db.scalar(select(ChatRequest.result_json).where(
            ChatRequest.id == key, ChatRequest.session_id == session_id, ChatRequest.message_id == message_id))
        if saved is None:
            raise IdempotencyConflict("请求回执不存在或身份不匹配")
        return _saved_response(saved, session_id)
