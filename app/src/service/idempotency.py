"""Atomic business receipts; no write transaction is held across model calls."""
from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from sqlalchemy import update, event
from sqlalchemy.orm import Session
from sqlalchemy.dialects.sqlite import insert

from ..domain.entity.chat_entity import ChatMessage, ChatRequest, WriteOperation

MISSING = object()


class IdempotencyConflict(ValueError):
    pass


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


@dataclass
class WriteScope:
    session_id: str
    key: str
    ordinal: int = 0


_scope = ContextVar("lifeops_write_scope", default=None)


@contextmanager
def write_scope(session_id, key):
    token = _scope.set(WriteScope(session_id, key))
    try:
        yield
    finally:
        _scope.reset(token)


def begin_write(db, session_id, kind, payload, operation_key=None):
    scope = _scope.get()
    if operation_key is None and scope is not None:
        if scope.session_id != session_id:
            raise IdempotencyConflict("写入身份与执行上下文不一致")
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
    if turn is not None:
        turn.committed_writes[key] = json.loads(json.dumps(entry, ensure_ascii=False, default=str))


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
    fingerprint = _hash([message, proposal_id])
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
        result = json.loads(row.result_json) if row.result_json is not None else None
        message_id = row.message_id
        db.commit()
    return key, message_id, result


def finish_request(db, key, result):
    db.execute(update(ChatRequest).where(ChatRequest.id == key)
               .values(result_json=json.dumps(result, ensure_ascii=False)))
