"""LLM / 工具调用的拦截器链。

设计原则：
- 拦截器是**同步**的，因为 LangGraph 节点和 ``chat_completion`` 都是同步调用。
- before 正序、after 逆序（洋葱模型），便于 Trace 包裹真实耗时。
- 生产入口通过 invoke 统一调用，finally 收尾保证异常和拒绝也有记录。
- 预算超限抛 ``TurnLimitExceeded``（由图表层收口成兜底回答）；
  安全策略拒绝抛 ``PolicyViolation``（由工具入口收口成「该操作被拦截」并继续走正常汇总）。

当前挂载：
- ``BudgetInterceptor``：单轮 LLM / 工具调用次数与墙钟预算
- ``TraceInterceptor``：每次调用的耗时记录
- ``ToolPolicyInterceptor``：未知动作白名单 + query_database 只读校验
- ``SessionScopeInterceptor``：query_database 必须限定当前 session_id
"""

from __future__ import annotations

import logging
import asyncio
import time
import re
from contextvars import ContextVar
from uuid import uuid4
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

from .context import TurnContext, TurnLimitExceeded
from .sql_scope import check_session_scope

logger = logging.getLogger(__name__)


class Decision(str, Enum):
    CONTINUE = "continue"
    ABORT = "abort"


class PolicyViolation(BaseException):
    """安全策略拒绝执行某次调用。

    与 ``TurnLimitExceeded`` 区分开：策略拒绝只影响当前这一次调用，
    不应该把整轮请求判定为「预算耗尽」。
    """

    def __init__(self, reason: str, *, interceptor: str = "") -> None:
        super().__init__(reason)
        self.reason = reason
        self.interceptor = interceptor


@dataclass
class CallContext:
    kind: str  # "llm" | "tool"
    name: str
    session_id: str
    turn: TurnContext | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    started_at: float = field(default_factory=time.monotonic)
    block_reason: str = ""
    call_id: str = field(default_factory=lambda: uuid4().hex)
    parent_call_id: str | None = None
    status: str = "success"
    error_type: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    recorded: bool = False
    managed: bool = False


class Interceptor(Protocol):
    def before(self, ctx: CallContext) -> Decision: ...

    def after(self, ctx: CallContext, result: Any) -> Any: ...


class BudgetInterceptor:
    """在每次 LLM / 工具调用前扣减本轮预算。"""

    def before(self, ctx: CallContext) -> Decision:
        if ctx.turn is not None:
            ctx.turn.check_active()
            ctx.turn.budget.before_call(ctx.kind)
        return Decision.CONTINUE

    def after(self, ctx: CallContext, result: Any) -> Any:
        return result


class TraceInterceptor:
    """记录每次调用的耗时，供本轮 trace 汇总。"""

    def before(self, ctx: CallContext) -> Decision:
        ctx.started_at = time.monotonic()
        return Decision.CONTINUE

    def after(self, ctx: CallContext, result: Any) -> Any:
        if not ctx.managed:
            self.on_finish(ctx)
        return result

    def on_finish(self, ctx: CallContext) -> None:
        if ctx.recorded:
            return
        ctx.recorded = True
        elapsed_ms = round((time.monotonic() - ctx.started_at) * 1000, 1)
        if ctx.turn is not None:
            ctx.turn.add_record(
                {"call_id": ctx.call_id, "parent_call_id": ctx.parent_call_id,
                 "kind": ctx.kind, "name": ctx.name if re.fullmatch(r"[A-Za-z0-9_.:-]{1,96}", ctx.name) else "invalid_name", "elapsed_ms": elapsed_ms,
                 "started_offset_ms": round(max(0, ctx.started_at - ctx.turn.budget.started_at) * 1000, 1),
                 "status": ctx.status, "error_type": ctx.error_type, **ctx.usage}
            )
        logger.debug(
            "harness call kind=%s name=%s session=%s elapsed_ms=%s",
            ctx.kind,
            ctx.name,
            ctx.session_id,
            elapsed_ms,
        )


# 允许由 LLM 决策层触发的工具动作。
# 新增工具时务必同步这里，否则会被策略层当成未知动作拒绝。
ALLOWED_TOOL_ACTIONS: frozenset[str] = frozenset(
    {
        "normal_chat",
        "query_todos",
        "query_knowledge",
        "query_weather",
        "query_calendar",
        "propose_todo",
        "ask_user_confirm_proposal",
        "revise_proposal",
        "decompose_goal",
        "clarify",
        "revive_topic",
        "record_expense",
        "query_expenses",
        "query_accounts",
        "create_account",
        "ask_user_confirm_balance",
        "query_database",
        # 内部 / HITL 状态，可能通过 forced_action 进入 run_tool
        "plan_proposal",
        "plan_gate_confirm",
        "plan_gate_cancel",
        "plan_gate_wait",
        "confirm_proposal",
    }
)


def _check_read_only(sql: str) -> str | None:
    """复用 db_query_service 的只读校验；失败时返回原因。"""
    try:
        from ...service.db_query_service import validate_read_only
    except Exception:  # pragma: no cover - 极端导入失败时退回执行层校验
        return None
    try:
        validate_read_only(sql)
    except ValueError as exc:
        return str(exc)
    return None


class ToolPolicyInterceptor:
    """工具白名单 + query_database 只读校验。"""

    def before(self, ctx: CallContext) -> Decision:
        if ctx.kind != "tool":
            return Decision.CONTINUE

        action = str(ctx.payload.get("action") or "")
        if action not in ALLOWED_TOOL_ACTIONS:
            ctx.block_reason = f"未知工具动作 {action or '(empty)'}，已拒绝执行。"
            return Decision.ABORT

        if action == "query_database":
            args = ctx.payload.get("args")
            sql = str(args.get("sql") or "") if isinstance(args, dict) else ""
            if sql.strip():
                reason = _check_read_only(sql)
                if reason:
                    ctx.block_reason = reason
                    return Decision.ABORT

        return Decision.CONTINUE

    def after(self, ctx: CallContext, result: Any) -> Any:
        return result


class SessionScopeInterceptor:
    """query_database 必须限定当前会话，防止跨 session 读取。"""

    def before(self, ctx: CallContext) -> Decision:
        if ctx.kind != "tool" or ctx.payload.get("action") != "query_database":
            return Decision.CONTINUE

        args = ctx.payload.get("args")
        sql = str(args.get("sql") or "") if isinstance(args, dict) else ""
        if not sql.strip():
            return Decision.CONTINUE

        reason = check_session_scope(sql, ctx.session_id)
        if reason:
            ctx.block_reason = reason
            return Decision.ABORT
        return Decision.CONTINUE

    def after(self, ctx: CallContext, result: Any) -> Any:
        return result


_CURRENT_CALL: ContextVar[str | None] = ContextVar("harness_current_call", default=None)


class InterceptorChain:
    """洋葱模型：before 正序、after 逆序。"""

    def __init__(self, interceptors: list[Interceptor] | None = None) -> None:
        self._interceptors: list[Interceptor] = list(interceptors or [])

    def register(self, interceptor: Interceptor) -> None:
        self._interceptors.append(interceptor)

    def before(self, ctx: CallContext) -> CallContext:
        for it in self._interceptors:
            if it.before(ctx) == Decision.ABORT:
                raise PolicyViolation(
                    ctx.block_reason or f"blocked by {type(it).__name__}",
                    interceptor=type(it).__name__,
                )
        return ctx

    def after(self, ctx: CallContext, result: Any) -> Any:
        for it in reversed(self._interceptors):
            result = it.after(ctx, result)
        return result

    def invoke(self, ctx: CallContext, operation):
        """Always finalize metadata, including failures in before/operation/after."""
        ctx.parent_call_id = _CURRENT_CALL.get()
        ctx.managed = True
        token = _CURRENT_CALL.set(ctx.call_id)
        try:
            self.before(ctx)
            result = operation()
            if ctx.turn is not None:
                ctx.turn.check_active()
            return self.after(ctx, result)
        except BaseException as exc:
            ctx.error_type = type(exc).__name__
            if isinstance(exc, PolicyViolation):
                ctx.status = "blocked"
            elif isinstance(exc, TurnLimitExceeded):
                ctx.status = "budget_exceeded"
            elif isinstance(exc, asyncio.CancelledError):
                ctx.status = "cancelled"
            else:
                ctx.status = "error"
            raise
        finally:
            try:
                for interceptor in reversed(self._interceptors):
                    finish = getattr(interceptor, "on_finish", None)
                    if finish is not None:
                        try:
                            finish(ctx)
                        except Exception:
                            logger.warning("harness metadata finalization failed call=%s", ctx.call_id)
            finally:
                _CURRENT_CALL.reset(token)


_LLM_CHAIN = InterceptorChain([BudgetInterceptor(), TraceInterceptor()])
_TOOL_CHAIN = InterceptorChain(
    [
        BudgetInterceptor(),
        TraceInterceptor(),
        ToolPolicyInterceptor(),
        SessionScopeInterceptor(),
    ]
)


def llm_chain() -> InterceptorChain:
    return _LLM_CHAIN


def tool_chain() -> InterceptorChain:
    return _TOOL_CHAIN
