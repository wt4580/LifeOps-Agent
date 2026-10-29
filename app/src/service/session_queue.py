from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable

JOB_END = object()
JOB_STOPPED = object()

_GATES: dict[str, "_Gate"] = {}
_accepting = True


class ChatCapacityExceeded(RuntimeError):
    """Admission failed before the job or its input was persisted."""


def accepting_chat_jobs() -> bool:
    return _accepting


def start_chat_jobs() -> None:
    global _accepting
    _accepting = True


class _Gate:
    """单个会话的串行执行门：严格 FIFO，同一时间至多一个任务在跑。

    用途：同一 session_id 的并发消息会被排成队列按顺序执行，
    解决"同一 thread 双跑时 LangGraph checkpoint 互相覆盖"的问题。
    """

    def __init__(self) -> None:
        self._queue: asyncio.Queue = asyncio.Queue()
        self._worker: asyncio.Task | None = None
        self.current: asyncio.Task | None = None
        self._pending = 0
        self.current_request_id: str | None = None
        self.closing = False

    def submit(self, events: asyncio.Queue, factory: Callable[[], Awaitable[Any]], request_id: str | None = None) -> None:
        self._pending += 1
        self._queue.put_nowait((events, factory, request_id))
        if self._worker is None or self._worker.done():
            self._worker = asyncio.create_task(self._run())

    @property
    def is_busy(self) -> bool:
        if self.current is not None and not self.current.done():
            return True
        return self._pending > 0 or not self._queue.empty()

    async def _run(self) -> None:
        try:
            while not self._queue.empty():
                events, factory, self.current_request_id = self._queue.get_nowait()
                self.current = asyncio.create_task(_execute(events, factory))
                self._pending -= 1
                try:
                    await asyncio.shield(self.current)
                except asyncio.CancelledError:
                    self.current.cancel()
                    # Sync tools must finish cancellation cleanup before resources close.
                    await asyncio.gather(self.current, return_exceptions=True)
                    return
                finally:
                    self.current = None
                    self.current_request_id = None
        finally:
            for sid, gate in list(_GATES.items()):
                if gate is self and not self.is_busy:
                    _GATES.pop(sid, None)

    def cancel_current(self) -> asyncio.Task | None:
        t = self.current
        if t is None or t.done():
            return None
        if not t.cancelling():
            t.cancel()
        return t

    async def close(self) -> None:
        self.closing = True
        while not self._queue.empty():
            events, _, _ = self._queue.get_nowait()
            self._pending -= 1
            events.put_nowait(JOB_STOPPED)
            events.put_nowait(JOB_END)
        if self._worker is not None:
            self._worker.cancel()
            await asyncio.gather(self._worker, return_exceptions=True)
            self._worker = None


async def _execute(events: asyncio.Queue, factory: Callable[[], Awaitable[Any]]) -> None:
    try:
        result = await factory()
        events.put_nowait({"type": "final", "result": result})
    except asyncio.CancelledError:
        events.put_nowait(JOB_STOPPED)
    except Exception as exc:
        events.put_nowait({"type": "error", "message": str(exc)})
    finally:
        events.put_nowait(JOB_END)


def _gate_for(session_id: str, *, create: bool = True) -> _Gate | None:
    gate = _GATES.get(session_id)
    if gate is None and create:
        gate = _Gate()
        _GATES[session_id] = gate
    return gate


def is_session_busy(session_id: str) -> bool:
    gate = _gate_for(session_id, create=False)
    return bool(gate) and gate.is_busy


def submit_chat_job(session_id: str, events: asyncio.Queue, factory: Callable[[], Awaitable[Any]], *, request_id: str | None = None) -> bool:
    """提交一条执行任务。

    返回该任务是否属于"排队中"（前面还有任务在处理/等待），用于前端展示排队提示。
    """
    from ..common.config.base_config import settings
    if not _accepting:
        raise ChatCapacityExceeded("服务正在关闭，请稍后重试")
    # No await between admission and submit: atomic within this process/event loop.
    count = sum(g._pending + int(g.current is not None and not g.current.done()) for g in _GATES.values())
    if count >= settings.chat_max_inflight_jobs:
        raise ChatCapacityExceeded("服务繁忙，请稍后重试")
    gate = _gate_for(session_id)
    if gate.closing or gate._pending + int(gate.current is not None and not gate.current.done()) >= settings.chat_max_session_jobs:
        raise ChatCapacityExceeded("当前会话请求过多，请等待已有任务完成")
    queued = gate.is_busy
    gate.submit(events, factory, request_id)
    return queued


async def cancel_and_wait(session_id: str, request_id: str | None = None) -> bool:
    """取消当前正在执行的任务并等待其完全退出。返回是否确有任务被取消。"""
    gate = _gate_for(session_id, create=False)
    if gate is None:
        return False
    if request_id is not None and request_id != gate.current_request_id:
        return False
    task = gate.cancel_current()
    if task is None:
        return False
    await asyncio.gather(task, return_exceptions=True)
    return True


async def close_session(session_id: str) -> None:
    gate = _GATES.get(session_id)
    if gate is not None:
        await gate.close()
        if _GATES.get(session_id) is gate:
            _GATES.pop(session_id, None)


async def shutdown_chat_jobs(grace_seconds: float = 15.0) -> None:
    """Stop admission, drain, then cancel and join before closing graph/DB resources."""
    global _accepting
    _accepting = False
    gates = list(_GATES.items())
    workers = [g._worker for _, g in gates if g._worker is not None and not g._worker.done()]
    if workers:
        await asyncio.wait(workers, timeout=max(0, grace_seconds))
    await asyncio.gather(*(close_session(sid) for sid, _ in gates))
