"""APScheduler 后台定时任务。

提供两种周期性推送：
1. 晨间简报（08:00 默认）：今日待办 + 天气 + 提醒
2. 晚间复盘（21:00 默认）：今日已完成事项 + 总结
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from ..common.config.base_config import settings

logger = logging.getLogger("LifeOps-Agent")

scheduler: AsyncIOScheduler | None = None


# ---------------------------------------------------------------------------
# 任务定义
# ---------------------------------------------------------------------------

async def _morning_briefing_job():
    today = date.today()
    weekday_cn = ["一", "二", "三", "四", "五", "六", "日"][today.weekday()]
    lines = [f"早上好！今天是 {today.month}月{today.day}日 星期{weekday_cn}"]

    # 天气
    try:
        from .weather_service import WeatherService
        ws = WeatherService()
        weather = await ws.get_weather(city=settings.default_weather_city)
        if weather:
            lines.append(f"🌤 天气：{weather.get('weather', '未知')}，{weather.get('temperature', '--')}℃")
    except Exception as exc:
        logger.debug("morning briefing weather fetch failed: %s", exc)

    # 今日待办
    try:
        from .todo_service import TodoService
        from ..common.config.db_config import SessionLocal
        db = SessionLocal()
        try:
            svc = TodoService(db)
            todos = svc.get_todos_for_date(today.isoformat())
            if todos:
                items = [f"  • {t.title}" for t in todos[:5]]
                lines.append(f"📋 今日待办 ({len(todos)}项)：")
                lines.extend(items)
            else:
                lines.append("📋 今日暂无待办")
        finally:
            db.close()
    except Exception as exc:
        logger.debug("morning briefing todos fetch failed: %s", exc)

    # 到期提醒
    try:
        from .reminder_service import ReminderService
        from ..common.config.db_config import SessionLocal
        db2 = SessionLocal()
        try:
            svc2 = ReminderService(db2)
            reminders = svc2.get_due_reminders()
            if reminders:
                lines.append(f"🔔 待处理提醒 ({len(reminders)}项)：")
                for r in reminders[:3]:
                    content = r.get("content", "") if isinstance(r, dict) else str(r)
                    lines.append(f"  • {content}")
        finally:
            db2.close()
    except Exception as exc:
        logger.debug("morning briefing reminders fetch failed: %s", exc)

    body = "\n".join(lines)

    from .push_service import push_notification
    push_notification("LifeOps 晨间简报", body)


async def _daily_review_job():
    today = date.today()
    weekday_cn = ["一", "二", "三", "四", "五", "六", "日"][today.weekday()]
    lines = [f"🌙 晚间复盘 — {today.month}月{today.day}日 星期{weekday_cn}"]

    # 已完成待办
    try:
        from .todo_service import TodoService
        from ..common.config.db_config import SessionLocal
        db = SessionLocal()
        try:
            svc = TodoService(db)
            done = svc.get_completed_todos_for_date(today.isoformat())
            if done:
                lines.append(f"✅ 今日完成 ({len(done)}项)：")
                for t in done[:10]:
                    lines.append(f"  • {t.title}")
            else:
                lines.append("✅ 今日暂无完成事项")
        finally:
            db.close()
    except Exception as exc:
        logger.debug("daily review todos fetch failed: %s", exc)

    # 今日新增的个人事件（粗略统计）
    try:
        from .memory_service import MemoryService
        from ..common.config.db_config import SessionLocal
        db2 = SessionLocal()
        try:
            svc2 = MemoryService(db2)
            events = svc2.get_events_for_date(today.isoformat())
            if events:
                lines.append(f"📝 记录事件 ({len(events)}项)")
        finally:
            db2.close()
    except Exception as exc:
        logger.debug("daily review events fetch failed: %s", exc)

    body = "\n".join(lines)

    from .push_service import push_notification
    push_notification("LifeOps 晚间复盘", body)


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

    scheduler.start()
    logger.info("APScheduler started")


def stop_scheduler():
    global scheduler
    if scheduler is None:
        return
    scheduler.shutdown(wait=False)
    scheduler = None
    logger.info("APScheduler stopped")
