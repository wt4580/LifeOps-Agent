from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable

JOB_END = object()
JOB_STOPPED = object()

_GATES: dict[str, "_Gate"] = {}


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

    def submit(self, events: asyncio.Queue, factory: Callable[[], Awaitable[Any]]) -> None:
        self._pending += 1
        self._queue.put_nowait((events, factory))
        if self._worker is None or self._worker.done():
            self._worker = asyncio.create_task(self._run())

    @property
    def is_busy(self) -> bool:
        if self.current is not None and not self.current.done():
            return True
        return self._pending > 0 or not self._queue.empty()

    async def _run(self) -> None:
        while True:
            try:
                events, factory = await self._queue.get()
            except asyncio.CancelledError:
                return
            self.current = asyncio.create_task(_execute(events, factory))
            self._pending -= 1
            try:
                await asyncio.wait([self.current])
            except asyncio.CancelledError:
                self.current.cancel()
                return
            finally:
                self.current = None

    def cancel_current(self) -> asyncio.Task | None:
        t = self.current
        if t is None or t.done():
            return None
        t.cancel()
        return t

    async def close(self) -> None:
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


def submit_chat_job(session_id: str, events: asyncio.Queue, factory: Callable[[], Awaitable[Any]]) -> bool:
    """提交一条执行任务。

    返回该任务是否属于"排队中"（前面还有任务在处理/等待），用于前端展示排队提示。
    """
    gate = _gate_for(session_id)
    queued = gate.is_busy
    gate.submit(events, factory)
    return queued


async def cancel_and_wait(session_id: str) -> bool:
    """取消当前正在执行的任务并等待其完全退出。返回是否确有任务被取消。"""
    gate = _gate_for(session_id, create=False)
    if gate is None:
        return False
    task = gate.cancel_current()
    if task is None:
        return False
    await asyncio.gather(task, return_exceptions=True)
    return True


async def close_session(session_id: str) -> None:
    gate = _GATES.pop(session_id, None)
    if gate is not None:
        await gate.close()