from __future__ import annotations

import asyncio

from app.src.service.session_queue import (
    JOB_END,
    JOB_STOPPED,
    cancel_and_wait,
    close_session,
    is_session_busy,
    submit_chat_job,
)


async def _drain(events: asyncio.Queue, timeout: float = 1.5):
    """在同一个事件循环内收集事件队列，直到遇到 JOB_END。

    返回 (items, was_stopped)：was_stopped 表示队列曾收到 JOB_STOPPED。
    """
    items = []
    stopped = False
    while True:
        item = await asyncio.wait_for(events.get(), timeout)
        if item is JOB_END:
            return items, stopped
        if item is JOB_STOPPED:
            stopped = True
            end = await asyncio.wait_for(events.get(), timeout)
            assert end is JOB_END, "JOB_STOPPED 后必须紧跟 JOB_END"
            return items, stopped
        items.append(item)


class TestFifoOrder:
    def test_second_job_is_queued_and_order_preserved(self):
        async def scenario():
            sid = "t-fifo"
            order = []
            e1: asyncio.Queue = asyncio.Queue()
            e2: asyncio.Queue = asyncio.Queue()

            async def factory1(events):
                await asyncio.sleep(0.02)
                order.append("a")
                return "result-a"

            async def factory2(events):
                await asyncio.sleep(0.01)
                order.append("b")
                return "result-b"

            q1 = submit_chat_job(sid, e1, lambda: factory1(e1))
            assert q1 is False, "第一个任务不应排队"
            assert is_session_busy(sid) is True

            q2 = submit_chat_job(sid, e2, lambda: factory2(e2))
            assert q2 is True, "第二个任务应排队"

            items1, _ = await _drain(e1)
            items2, _ = await _drain(e2)
            assert order == ["a", "b"], f"执行顺序应严格 FIFO: {order}"
            assert items1 == [{"type": "final", "result": "result-a"}]
            assert items2 == [{"type": "final", "result": "result-b"}]

            await close_session(sid)

        asyncio.run(scenario())

    def test_not_busy_when_nothing_submitted(self):
        assert asyncio.run(cancel_and_wait("t-none")) is False
        assert is_session_busy("t-none") is False


class TestCancel:
    def test_cancel_current_and_next_still_runs(self):
        async def scenario():
            sid = "t-cancel"
            ran = []
            e1: asyncio.Queue = asyncio.Queue()
            e2: asyncio.Queue = asyncio.Queue()

            gate_go = asyncio.Event()

            async def factory1(events):
                await gate_go.wait()  # 停在等待点，便于取消
                ran.append("a")
                return "a"

            async def factory2(events):
                ran.append("b")
                return "b"

            submit_chat_job(sid, e1, lambda: factory1(e1))
            assert is_session_busy(sid) is True

            # 等任务1真正开始跑（卡在 gate 上）
            await asyncio.sleep(0.02)
            submit_chat_job(sid, e2, lambda: factory2(e2))

            assert await cancel_and_wait(sid) is True

            items1, was_stopped = await _drain(e1)
            assert was_stopped is True, "被取消的任务应发出 JOB_STOPPED"

            # 任务1被取消，不会执行到 ran.append("a")；任务2正常跑完
            items2, _ = await _drain(e2)
            assert ran == ["b"], f"取消后任务2应继续执行: {ran}"
            assert items2 == [{"type": "final", "result": "b"}]

            await close_session(sid)

        asyncio.run(scenario())

    def test_cancel_when_idle_returns_false(self):
        sid = "t-idle-cancel"

        async def scenario():
            assert await cancel_and_wait(sid) is False
            await close_session(sid)

        asyncio.run(scenario())

    def test_stop_does_not_affect_other_sessions(self):
        async def scenario():
            sid_a = "t-session-a"
            sid_b = "t-session-b"
            ran = []
            e_a: asyncio.Queue = asyncio.Queue()
            e_b: asyncio.Queue = asyncio.Queue()

            gate_a = asyncio.Event()

            async def factory_a(events):
                await gate_a.wait()
                ran.append("a")
                return "a"

            async def factory_b(events):
                ran.append("b")
                return "b"

            submit_chat_job(sid_a, e_a, lambda: factory_a(e_a))
            await asyncio.sleep(0.02)
            submit_chat_job(sid_b, e_b, lambda: factory_b(e_b))

            assert await cancel_and_wait(sid_a) is True  # 只打断 A

            items_b, _ = await _drain(e_b)
            assert ran == ["b"], f"B 会话不应被 A 的停止影响: {ran}"
            _, was_stopped_a = await _drain(e_a)
            assert was_stopped_a is True

            await close_session(sid_a)
            await close_session(sid_b)

        asyncio.run(scenario())