"""Runtime failure envelopes; raw exception messages never enter responses."""
from __future__ import annotations

import sqlite3

from sqlalchemy.exc import SQLAlchemyError
from openai import OpenAIError
from .model_errors import ModelOutputIncompleteError, ModelOutputInvalidError


class ToolExecutionFailure(Exception):
    def __init__(self, action, source_error):
        super().__init__("Tool execution failed")
        self.action = action
        self.source_error = source_error.source_error if isinstance(source_error, ToolExecutionFailure) else source_error


class ToolResultRejected(Exception):
    """A known unusable result, with an application-authored safe explanation."""
    def __init__(self, code, message):
        super().__init__(message)
        self.code, self.message = code, message


def failure_result(error, *, action, effect, committed, completed_steps=0, trace_id=None, call_id=None):
    from ...service.bookkeeping_service import BookkeepingError
    from ...service.idempotency import IdempotencyConflict
    from ...service.weather_service import WeatherServiceError
    from ...service.calendar_service import CalendarServiceError
    source = error.source_error if isinstance(error, ToolExecutionFailure) else error
    kind, code = "execution", "tool_execution_failed"
    message = "当前步骤执行失败，后续步骤已停止。"
    recovery = "check_records" if effect == "write" else "new_read_request"
    outcome = "unknown" if effect == "write" else "not_completed"
    if isinstance(source, ToolResultRejected):
        kind, code, outcome, recovery = "rejection", source.code, "not_applied", "correct_input"
        message = source.message + "后续步骤已停止。"
    elif isinstance(source, IdempotencyConflict):
        kind, code, outcome, recovery = "rejection", "idempotency_conflict", "not_applied", "check_records"
        message = "操作内容与原操作标识不一致，本步骤已拒绝，请先核对原操作记录。"
    elif isinstance(source, BookkeepingError):
        kind, code, outcome, recovery = "rejection", "business_rule_rejected", "not_applied", "correct_input"
        message = "本步骤未通过业务校验，请检查账户是否存在、归属及填写内容，后续步骤已停止。"
    elif isinstance(source, (SQLAlchemyError, sqlite3.Error)):
        code = "storage_unavailable"
        message = "数据存储操作失败，后续步骤已停止。"
    elif isinstance(source, ModelOutputInvalidError):
        code = "model_output_invalid"
        message = "模型返回的规划格式无效，本轮未继续执行工具，请重新发起请求。"
    elif isinstance(source, ModelOutputIncompleteError):
        code = "model_output_incomplete"
        message = "模型输出未完整结束，后续步骤已停止，请缩小问题范围后重新发起请求。"
    elif isinstance(source, OpenAIError):
        code = "model_unavailable"
        message = "模型服务未能完成当前步骤，后续步骤已停止。"
    elif isinstance(source, (WeatherServiceError, CalendarServiceError)):
        code = "dependency_unavailable"
        message = "查询服务不可用，请检查服务配置或稍后重新发起查询，后续步骤已停止。"
    operations = [{"receipt_id": key, "kind": entry.get("kind"),
                   "record_id": entry.get("result", {}).get("id") if isinstance(entry.get("result"), dict) else None}
                  for key, entry in list(committed.items())[:10]]
    if committed:
        outcome, recovery = "some_operations_committed", "check_records"
        message += f"当前步骤已有 {len(committed)} 项操作提交，不能视为全部撤销，请先查询记录。"
    elif outcome == "unknown":
        message += "暂时无法确认是否写入，请先查询记录，不要重复提交相同业务操作。"
    if completed_steps:
        message += f"此前已完成 {completed_steps} 个计划步骤，已提交的操作仍然有效。"
    return {"code": code, "kind": kind, "action": action, "outcome": outcome,
            "retryable": False, "recovery": recovery, "error_type": type(source).__name__,
            "committed_operations_count": len(committed), "committed_operations": operations,
            "completed_plan_steps": completed_steps, "trace_id": trace_id, "call_id": call_id}, message
