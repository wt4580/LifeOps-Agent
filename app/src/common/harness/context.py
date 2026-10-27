"""每次请求（HTTP turn）级别的 harness 上下文：墙钟截止时间 + 单调调用预算。

设计要点：
- 预算按"每次 HTTP 请求"创建，在 ``arun_chat_graph`` / ``run_chat_graph`` 入口建立、
  出口销毁。HITL 的 ``Command(resume=...)`` 是新请求，会重新计时。
- 计数器保存在 harness 侧（本模块注册表 + ``TurnContext``），**不写入 ChatState**。
  历史上 ``retry_count`` 被 agent 逻辑重置，导致终止条件永远不触发；预算必须由
  外部持有，agent 改不到。
- ``TurnLimitExceeded`` 刻意继承 ``BaseException``：业务代码里大量
  ``except Exception`` 会吞掉普通异常，用 BaseException 保证它能一路冒泡到图入口，
  由 harness 统一收口成兜底回答。
"""

from __future__ import annotations

import threading
import asyncio
import time
from uuid import uuid4
from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Iterator

from ..config.base_config import settings


class TurnLimitExceeded(BaseException):
    """本轮请求超出预算（步数 / 耗时）。

    继承 BaseException 是刻意为之，避免被业务层的 ``except Exception`` 静默吞掉。
    """


@dataclass
class TurnBudget:
    """一轮请求内共享的单调预算。"""

    max_llm_calls: int
    max_tool_calls: int
    deadline_seconds: float
    llm_calls: int = 0
    tool_calls: int = 0
    started_at: float = field(default_factory=time.monotonic)
    max_total_tokens: int = 120000
    charged_tokens: int = 0
    reserved_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    usage_missing_calls: int = 0
    finished_at: float | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def deadline_at(self) -> float:
        return self.started_at + self.deadline_seconds

    def remaining_seconds(self) -> float:
        return max(0.0, self.deadline_at - time.monotonic())

    def _check_locked(self, kind: str) -> None:
        """调用方需持有 ``_lock``。超限时抛出 ``TurnLimitExceeded``。"""
        if time.monotonic() > self.deadline_at:
            raise TurnLimitExceeded(
                f"turn deadline exceeded: {self.deadline_seconds:.0f}s"
            )
        if kind == "llm" and self.llm_calls >= self.max_llm_calls:
            raise TurnLimitExceeded(
                f"turn llm budget exceeded: {self.llm_calls}/{self.max_llm_calls}"
            )
        if kind == "tool" and self.tool_calls >= self.max_tool_calls:
            raise TurnLimitExceeded(
                f"turn tool budget exceeded: {self.tool_calls}/{self.max_tool_calls}"
            )

    def before_call(self, kind: str) -> None:
        """在真实调用前扣减一次预算。kind 为 ``llm`` 或 ``tool``。"""
        with self._lock:
            self._check_locked(kind)
            if kind == "llm":
                self.llm_calls += 1
            elif kind == "tool":
                self.tool_calls += 1

    def check_deadline(self) -> None:
        if time.monotonic() >= self.deadline_at:
            raise TurnLimitExceeded("turn deadline exceeded")

    def check_tokens(self) -> None:
        with self._lock:
            if self.charged_tokens > self.max_total_tokens:
                raise TurnLimitExceeded("reported token usage exceeded turn budget")

    def reserve_tokens(self, prompt_estimate: int, output_limit: int) -> tuple[int, int]:
        """Reserve a conservative upper bound before issuing a model request."""
        with self._lock:
            self.check_deadline()
            available = self.max_total_tokens - self.charged_tokens - self.reserved_tokens
            output_limit = min(output_limit, available - prompt_estimate)
            if output_limit <= 0:
                raise TurnLimitExceeded("turn token budget exhausted")
            reservation = prompt_estimate + output_limit
            self.reserved_tokens += reservation
            return output_limit, reservation

    def settle_tokens(self, reservation: int, usage: tuple[int, int] | None, *, attempted: bool) -> None:
        with self._lock:
            self.reserved_tokens = max(0, self.reserved_tokens - reservation)
            if not attempted:
                return
            if usage is None:
                self.usage_missing_calls += 1
                self.charged_tokens += reservation
            else:
                prompt, completion = usage
                self.prompt_tokens += prompt
                self.completion_tokens += completion
                self.charged_tokens += prompt + completion

    def finish(self) -> None:
        with self._lock:
            if self.finished_at is None:
                self.finished_at = time.monotonic()

    def snapshot(self) -> dict:
        with self._lock:
            now = self.finished_at if self.finished_at is not None else time.monotonic()
            return {
                "llm_calls": self.llm_calls,
                "tool_calls": self.tool_calls,
                "elapsed_seconds": round(now - self.started_at, 3),
                "remaining_seconds": round(max(0, self.deadline_at - now), 3),
                "token_limit": self.max_total_tokens,
                "charged_tokens": self.charged_tokens,
                "reserved_tokens": self.reserved_tokens,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "usage_missing_calls": self.usage_missing_calls,
            }


@dataclass
class TurnContext:
    session_id: str
    budget: TurnBudget
    started_at: float = field(default_factory=time.monotonic)
    # 本轮内所有经过拦截链的调用记录（TraceInterceptor 写入）。
    records: list[dict] = field(default_factory=list)
    cancelled: threading.Event = field(default_factory=threading.Event)
    turn_id: str = field(default_factory=lambda: uuid4().hex)
    phase: str = "foreground"
    operation: str = "chat"
    parent_turn_id: str | None = None
    status: str = "running"
    last_state: dict | None = field(default=None, repr=False)
    _records_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add_record(self, record: dict) -> None:
        with self._records_lock:
            self.records.append(record)
            del self.records[:-200]

    def check_active(self) -> None:
        if self.cancelled.is_set():
            raise asyncio.CancelledError("本轮已停止")
        self.budget.check_deadline()
        self.budget.check_tokens()

    def finish(self, status: str) -> None:
        if self.status == "running":
            self.status = status
            self.budget.finish()

    def summary(self) -> dict:
        with self._records_lock:
            records = list(self.records)
        return {"turn_id": self.turn_id, "parent_turn_id": self.parent_turn_id,
                "phase": self.phase, "status": self.status,
                "operation": self.operation,
                "call_status_counts": dict(Counter(record.get("status", "unknown") for record in records)),
                **self.budget.snapshot(), "invocations": records}


_registry: dict[str, TurnContext] = {}
_registry_lock = threading.Lock()
_CURRENT_TURN: ContextVar[TurnContext | None] = ContextVar("lifeops_current_turn", default=None)


def ensure_turn_active() -> None:
    ctx = _CURRENT_TURN.get()
    if ctx is not None and not budget_bypassed():
        ctx.check_active()

# 后台旁路（post_chat_learn / memory_extraction 等）不应消耗本轮预算。
_BUDGET_BYPASS: ContextVar[bool] = ContextVar("lifeops_budget_bypass", default=False)


def _budget_from_settings() -> TurnBudget:
    return TurnBudget(
        max_llm_calls=settings.chat_turn_max_llm_calls,
        max_tool_calls=settings.chat_turn_max_tool_calls,
        deadline_seconds=settings.chat_turn_deadline_seconds,
        max_total_tokens=settings.chat_turn_max_tokens,
    )


def create_turn_context(session_id: str) -> TurnContext:
    ctx = TurnContext(session_id=session_id, budget=_budget_from_settings())
    with _registry_lock:
        _registry[session_id] = ctx
    return ctx


def get_turn_context(session_id: str | None) -> TurnContext | None:
    active = _CURRENT_TURN.get()
    if active is not None and (session_id is None or session_id == active.session_id):
        return active
    if not session_id:
        return None
    return _registry.get(session_id)


def clear_turn_context(session_id: str, ctx: TurnContext | None = None) -> None:
    """清理本轮上下文。

    传入 ``ctx`` 时只清理同一个实例，避免误删同一 session 后续请求新建的上下文。
    """
    with _registry_lock:
        current = _registry.get(session_id)
        if current is None:
            return
        if ctx is None or current is ctx:
            _registry.pop(session_id, None)


@contextmanager
def turn_context(session_id: str, *, register: bool = True, operation: str = "chat") -> Iterator[TurnContext]:
    ctx = create_turn_context(session_id) if register else TurnContext(session_id, _budget_from_settings())
    ctx.operation = operation
    token = _CURRENT_TURN.set(ctx)
    try:
        yield ctx
    except BaseException as exc:
        ctx.finish(_exception_status(exc))
        raise
    finally:
        ctx.finish("success")
        try:
            _write_audit(ctx)
        finally:
            _CURRENT_TURN.reset(token)
            if register:
                clear_turn_context(session_id, ctx)


def _exception_status(exc: BaseException) -> str:
    if isinstance(exc, TurnLimitExceeded):
        return "budget_exceeded"
    if isinstance(exc, asyncio.CancelledError):
        return "cancelled"
    return "error"


def _write_audit(ctx: TurnContext) -> None:
    # Import lazily to keep the context module independent of storage details.
    from .audit import write_turn
    write_turn(ctx.session_id, ctx.summary())


@contextmanager
def background_context(session_id: str, *, parent_turn_id: str | None = None, operation: str = "post_graph") -> Iterator[TurnContext]:
    """A separate task-local budget; never replaces the foreground registry entry."""
    ctx = TurnContext(session_id=session_id, phase="background", operation=operation, parent_turn_id=parent_turn_id, budget=TurnBudget(
        max_llm_calls=settings.background_max_llm_calls, max_tool_calls=0,
        deadline_seconds=settings.background_deadline_seconds,
        max_total_tokens=settings.background_max_tokens,
    ))
    token = _CURRENT_TURN.set(ctx)
    bypass_token = _BUDGET_BYPASS.set(False)
    try:
        yield ctx
    except BaseException as exc:
        ctx.finish(_exception_status(exc))
        raise
    finally:
        ctx.finish("success")
        try:
            _write_audit(ctx)
        finally:
            _BUDGET_BYPASS.reset(bypass_token)
            _CURRENT_TURN.reset(token)


@contextmanager
def budget_bypass() -> Iterator[None]:
    """受信任维护代码的显式豁免；生产后台任务使用 background_context。"""
    token = _BUDGET_BYPASS.set(True)
    try:
        yield
    finally:
        _BUDGET_BYPASS.reset(token)


def budget_bypassed() -> bool:
    return _BUDGET_BYPASS.get()
