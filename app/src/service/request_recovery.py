"""Immutable completed-response staging in the checkpoint database.

Business writes remain in their original transactions. This journal bridges a
finished graph and a failed response commit; it never authorizes re-execution.
"""
from __future__ import annotations

import json
import sqlite3
from sqlalchemy.exc import SQLAlchemyError
from pathlib import Path

from ..common.config.base_config import settings
from ..common.domain.request_errors import ChatRequestRejected

TABLE = "chat_outcomes_v1"


def _path():
    return Path(settings.checkpointer_db_path).expanduser().resolve()


def _validate(row, session_id, message_id, payload_hash):
    if row[0:3] != (session_id, message_id, payload_hash):
        raise ChatRequestRejected("request_recovery_conflict", "恢复记录与当前请求不匹配，已阻止执行，请核对数据恢复范围。")
    try:
        result = json.loads(row[3])
    except (ValueError, TypeError):
        raise ChatRequestRejected("request_recovery_corrupt", "请求恢复记录无法读取，已阻止执行，请核对实际记录。") from None
    if not isinstance(result, dict) or result.get("session_id") != session_id or not isinstance(result.get("answer"), str):
        raise ChatRequestRejected("request_recovery_corrupt", "请求恢复记录无效，已阻止执行，请核对实际记录。")
    return result


def load_outcome(key, session_id, message_id, payload_hash):
    path = _path()
    if not path.is_file():
        return None
    db = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=2)
    try:
        try:
            row = db.execute(f"SELECT session_id,message_id,payload_hash,result_json FROM {TABLE} WHERE request_key=?", (key,)).fetchone()
        except sqlite3.OperationalError as exc:
            if str(exc) == f"no such table: {TABLE}":
                return None
            raise
        return _validate(row, session_id, message_id, payload_hash) if row else None
    finally:
        db.close()


def stage_outcome(key, session_id, message_id, payload_hash, result):
    path = _path()
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=2)
    try:
        with db:
            db.execute(f"CREATE TABLE IF NOT EXISTS {TABLE} (request_key TEXT PRIMARY KEY, session_id TEXT NOT NULL, "
                       "message_id INTEGER NOT NULL, payload_hash TEXT NOT NULL, result_json TEXT NOT NULL, "
                       "created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)")
            db.execute(f"INSERT INTO {TABLE}(request_key,session_id,message_id,payload_hash,result_json) VALUES (?,?,?,?,?) "
                       "ON CONFLICT(request_key) DO NOTHING",
                       (key, session_id, message_id, payload_hash, json.dumps(result, ensure_ascii=False, allow_nan=False)))
            row = db.execute(f"SELECT session_id,message_id,payload_hash,result_json FROM {TABLE} WHERE request_key=?", (key,)).fetchone()
            return _validate(row, session_id, message_id, payload_hash)
    finally:
        db.close()


def discard_outcome(key, session_id, message_id, payload_hash):
    path = _path()
    if not path.is_file():
        return
    db = sqlite3.connect(path.as_uri() + "?mode=rw", uri=True, timeout=0.1)
    try:
        with db:
            try:
                db.execute(f"DELETE FROM {TABLE} WHERE request_key=? AND session_id=? AND message_id=? AND payload_hash=?",
                           (key, session_id, message_id, payload_hash))
            except sqlite3.OperationalError as exc:
                if str(exc) != f"no such table: {TABLE}":
                    raise
    finally:
        db.close()


async def completed_checkpoint(graph, session_id, message_id, key):
    """A fallback only for a fully finished, exactly bound graph; pending HITL is not completion."""
    snapshot = await graph.aget_state({"configurable": {"thread_id": session_id}})
    values = getattr(snapshot, "values", None) or {}
    if (not isinstance(values, dict) or getattr(snapshot, "tasks", None) or getattr(snapshot, "next", None)
            or values.get("session_id") != session_id or values.get("operation_id") != f"message:{message_id}"
            or values.get("request_key") != key or not isinstance(values.get("answer"), str) or not values["answer"]):
        return None
    trace = values.get("trace") or {}
    if not any(step.get("name") == "finalize" for step in trace.get("steps", [])):
        return None
    # Runtime summary is written after graph completion, so it may be unavailable.
    trace = {**trace, "meta": {**trace.get("meta", {}), "recovered_from_checkpoint": True}}
    return {"answer": values["answer"], "session_id": session_id, "proposal_id": values.get("proposal_id"),
            "proposal": values.get("proposal"), "used_tool": values.get("used_tool"),
            "citations": values.get("citations") or [], "trace": trace, "tool_error": values.get("tool_error"),
            "recovery": {"source": "completed_checkpoint", "execution_metadata_complete": False}}


def record_cancellation(factory, identity):
    """Keep a completed first outcome, otherwise persist a cancellation/tombstone.

    Either store may be unavailable. A successful dispatch claim is enough to
    prohibit blind replay, even if the exact cancellation response cannot save.
    """
    from .idempotency import claim_request, persist_response, request_dispatched, request_result
    key, session_id, message_id, _ = identity
    durable = False
    errors = (sqlite3.Error, SQLAlchemyError, OSError)
    try:
        if request_result(factory, key, session_id, message_id) is not None:
            return True
        durable = claim_request(factory, key, session_id, message_id) or request_dispatched(factory, key, session_id, message_id)
    except errors:
        pass
    result = {"session_id": session_id, "request_status": "cancelled", "answer": "请求已停止；停止前已经提交的操作不会撤销，可先查询记录再补充条件。",
              "proposal_id": None, "proposal": None, "used_tool": None, "citations": [], "trace": None, "tool_error": None}
    try:
        result = stage_outcome(*identity, result)
        durable = True
    except errors:
        pass
    try:
        persist_response(factory, key, session_id, message_id, result)
        durable = True
        discard_outcome(*identity)
    except errors:
        pass
    return durable
