"""APScheduler 后台定时任务。

提供三种周期性推送：
1. 晨间简报（08:00 默认）：今日待办 + 天气
2. 晚间复盘（21:00 默认）：今日已完成事项 + 记录
3. 到点待办推送（每分钟）：带具体钟点/周期的待办到期时推送
"""

from __future__ import annotations

import logging
import asyncio
from datetime import date, datetime, timedelta

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from sqlalchemy import select

from ..common.config.base_config import settings

logger = logging.getLogger("LifeOps-Agent")

scheduler: AsyncIOScheduler | None = None

# 已推送过的待办去重：key = "{todo_id}:{occurrence_date}"，防止每分钟扫描重复推送。
_pushed_todo_keys: set[str] = set()


# ---------------------------------------------------------------------------
# 任务定义
# ---------------------------------------------------------------------------

async def _morning_briefing_job():
    today = date.today()
    weekday_cn = ["一", "二", "三", "四", "五", "六", "日"][today.weekday()]
    lines = [f"早上好！今天是 {today.month}月{today.day}日 星期{weekday_cn}"]

    # 天气
    try:
        from .weather_service import weather_service
        weather = await asyncio.to_thread(weather_service.query, settings.default_weather_city)
        if weather:
            live = weather.get("live") or {}
            forecast = (weather.get("forecast") or [{}])[0]
            description = live.get("weather") or forecast.get("dayweather", "未知")
            temperature = live.get("temperature") or forecast.get("daytemp", "--")
            lines.append(f"🌤 天气：{description}，{temperature}℃")
    except Exception as exc:
        logger.debug("morning briefing weather fetch failed: %s", exc)

    # 今日待办（含周期待办与带钟点事项）
    try:
        from ..domain.entity.chat_entity import TodoItem
        from ..common.config.db_config import SessionLocal
        from .todo_service import TodoService

        day_start = datetime.combine(today, datetime.min.time())
        today_dt = datetime.now()
        with SessionLocal() as db:
            rows = db.execute(select(TodoItem).order_by(TodoItem.due_at.asc())).scalars().all()
        todos = [
            t
            for t in rows
            if not TodoService._done_today(t, day_start)
            and (
                (t.due_at is not None and day_start <= t.due_at < day_start + timedelta(days=1))
                or TodoService._is_recurring_active(t, today_dt)
            )
        ]
        if todos:
            lines.append(f"📋 今日待办 ({len(todos)}项)：")
            for t in todos[:6]:
                time_str = t.due_at.strftime("%H:%M") if t.due_at and (t.due_at.hour or t.due_at.minute) else ""
                lines.append(f"  • {t.title}" + (f" {time_str}" if time_str else ""))
        else:
            lines.append("📋 今日暂无待办")
    except Exception as exc:
        logger.debug("morning briefing todos fetch failed: %s", exc)

    body = "\n".join(lines)

    from .push_service import push_notification
    await asyncio.to_thread(push_notification, "LifeOps 晨间简报", body)


async def _daily_review_job():
    today = date.today()
    weekday_cn = ["一", "二", "三", "四", "五", "六", "日"][today.weekday()]
    lines = [f"🌙 晚间复盘 — {today.month}月{today.day}日 星期{weekday_cn}"]

    # 今日已完成待办
    try:
        from ..domain.entity.chat_entity import TodoItem
        from ..common.config.db_config import SessionLocal

        local_now = datetime.now()
        local_start = local_now.replace(hour=0, minute=0, second=0, microsecond=0)
        with SessionLocal() as db:
            done = db.execute(
                select(TodoItem)
                .where(TodoItem.is_completed == True)
                .where(TodoItem.completed_at >= local_start)
                .where(TodoItem.completed_at < local_start + timedelta(days=1))
                .order_by(TodoItem.completed_at.asc())
            ).scalars().all()
            if done:
                lines.append(f"✅ 今日完成 ({len(done)}项)：")
                for t in done[:10]:
                    lines.append(f"  • {t.title}")
            else:
                lines.append("✅ 今日暂无完成事项")
    except Exception as exc:
        logger.debug("daily review todos fetch failed: %s", exc)

    # 今日新增的个人事件（created_at 按 UTC 存储，需用 UTC 日界）
    try:
        from ..domain.entity.chat_entity import LifeEvent
        from ..common.config.db_config import SessionLocal

        from datetime import timezone
        utc_start = datetime.now().astimezone().replace(hour=0, minute=0, second=0, microsecond=0).astimezone(timezone.utc).replace(tzinfo=None)
        with SessionLocal() as db:
            events = db.execute(
                select(LifeEvent)
                .where(LifeEvent.created_at >= utc_start)
                .where(LifeEvent.created_at < utc_start + timedelta(days=1))
            ).scalars().all()
            if events:
                lines.append(f"📝 记录事件 ({len(events)}项)")
    except Exception as exc:
        logger.debug("daily review events fetch failed: %s", exc)

    body = "\n".join(lines)

    from .push_service import push_notification
    await asyncio.to_thread(push_notification, "LifeOps 晚间复盘", body)


# ---------------------------------------------------------------------------
# 到点待办推送：带具体钟点(或周期)的待办在到期时推送一次
# ---------------------------------------------------------------------------

def _dedupe_pushed_keys() -> None:
    """清理过期去重键，防止集合无限增长。"""
    today = date.today().isoformat()
    stale = [k for k in _pushed_todo_keys if k.split(":", 1)[-1] < today]
    for k in stale:
        _pushed_todo_keys.discard(k)


async def _push_due_todos_job():
    now = datetime.now()
    _dedupe_pushed_keys()
    try:
        from ..domain.entity.chat_entity import TodoItem, NotificationReceipt
        from ..common.config.db_config import SessionLocal
        from .todo_service import TodoService
        from .push_service import push_notification

        window_start = now - timedelta(minutes=5)
        with SessionLocal() as db:
            rows = db.execute(select(TodoItem)).scalars().all()
            sent_keys = set(db.execute(select(NotificationReceipt.key)).scalars().all())

        candidates: list[tuple[TodoItem, datetime]] = []  # (待办, 本次触发时间)
        for t in rows:
            if t.due_at is None:
                continue
            if not (t.due_at.hour or t.due_at.minute):
                continue  # 纯日期待办不推送
            if t.repeat_rule in {"daily", "weekly"}:
                for day in {window_start.date(), now.date()}:
                    occurrence = datetime.combine(day, t.due_at.time())
                    day_start = occurrence.replace(hour=0, minute=0, second=0, microsecond=0)
                    if (TodoService._is_recurring_active(t, occurrence)
                            and not TodoService._done_today(t, day_start)
                            and window_start <= occurrence <= now):
                        candidates.append((t, occurrence))
            else:
                if not t.is_completed and window_start <= t.due_at <= now:
                    candidates.append((t, t.due_at))

        pushed = 0
        for t, occurrence in sorted(candidates, key=lambda item: item[1]):
            key = f"{t.id}:{occurrence.isoformat()}"
            if key in sent_keys:
                continue
            if pushed >= 3:
                break
            time_str = occurrence.strftime("%H:%M")
            if not await asyncio.to_thread(push_notification, f"⏰ {t.title}", f"到期时间 {time_str}，该做这件事了"):
                continue
            with SessionLocal() as db:
                db.add(NotificationReceipt(key=key))
                db.commit()
            _pushed_todo_keys.add(key)
            pushed += 1
        if pushed:
            logger.info("scheduler pushed %d due todos", pushed)
    except Exception as exc:
        logger.warning("due todos push failed: %s", exc)


# ---------------------------------------------------------------------------
# 启动 / 停止
# ---------------------------------------------------------------------------

def start_scheduler():
    global scheduler
    if scheduler is not None:
        logger.warning("scheduler already started")
        return

    scheduler = AsyncIOScheduler()

    # 晨间简报
    morning_time = (settings.schedule_morning_briefing or "").strip()
    if morning_time:
        hour, minute = morning_time.split(":")
        scheduler.add_job(
            _morning_briefing_job,
            trigger="cron",
            hour=int(hour),
            minute=int(minute),
            id="morning_briefing",
            replace_existing=True,
        )
        logger.info("scheduler: morning briefing at %s", morning_time)

    # 晚间复盘
    review_time = (settings.schedule_daily_review or "").strip()
    if review_time:
        hour, minute = review_time.split(":")
        scheduler.add_job(
            _daily_review_job,
            trigger="cron",
            hour=int(hour),
            minute=int(minute),
            id="daily_review",
            replace_existing=True,
        )
        logger.info("scheduler: daily review at %s", review_time)

    # 到点待办推送：每分钟扫描一次
    scheduler.add_job(
        _push_due_todos_job,
        trigger="interval",
        minutes=1,
        id="push_due_todos",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
    )
    logger.info("scheduler: due todos push every 1 minute")

    scheduler.start()
    logger.info("APScheduler started")


def stop_scheduler():
    global scheduler
    if scheduler is None:
        return
    scheduler.shutdown(wait=False)
    scheduler = None
    logger.info("APScheduler stopped")
