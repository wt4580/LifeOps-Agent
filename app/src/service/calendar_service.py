from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from sqlalchemy import select

from ..common.config.db_config import SessionLocal
from ..common.config.log_config import logger
from ..domain.entity.chat_entity import CalendarEvent
from .chinese_holidays import get_holidays_in_range


class CalendarServiceError(Exception):
    pass


class CalendarService:
    def list_events(
        self,
        start_iso: str,
        end_iso: str,
        *,
        calendar_id: Optional[str] = None,
        max_results: int = 50,
        interactive: bool = True,
    ) -> list[dict[str, Any]]:
        start_dt = datetime.fromisoformat(start_iso.replace("Z", "+00:00")).replace(tzinfo=None)
        end_dt = datetime.fromisoformat(end_iso.replace("Z", "+00:00")).replace(tzinfo=None)

        # 1. 查询用户自己添加的日历事件
        user_events: list[dict[str, Any]] = []
        with SessionLocal() as db:
            rows = (
                db.execute(
                    select(CalendarEvent)
                    .where(CalendarEvent.start_at >= start_dt)
                    .where(CalendarEvent.start_at < end_dt)
                    .order_by(CalendarEvent.start_at.asc())
                    .limit(max_results)
                )
                .scalars()
                .all()
            )
            for ev in rows:
                user_events.append({
                    "summary": ev.title,
                    "start": ev.start_at.isoformat(),
                    "end": ev.end_at.isoformat(),
                    "location": ev.location,
                    "status": "confirmed",
                    "htmlLink": None,
                    "all_day": ev.is_all_day,
                })

        # 2. 动态计算中国传统节日
        holiday_events = get_holidays_in_range(start_dt, end_dt)

        # 3. 合并去重（节日优先，用户事件如果和节日同名只保留一个）
        seen_summaries: set[str] = set()
        merged: list[dict[str, Any]] = []
        for ev in holiday_events + user_events:
            key = f"{ev['summary']}|{ev['start'][:10]}"
            if key not in seen_summaries:
                seen_summaries.add(key)
                merged.append(ev)

        merged.sort(key=lambda x: x["start"])
        return merged

    def add_event(
        self,
        title: str,
        start_at: datetime,
        end_at: datetime,
        *,
        session_id: str | None = None,
        is_all_day: bool = False,
        location: str | None = None,
        notes: str | None = None,
    ) -> CalendarEvent:
        ev = CalendarEvent(
            session_id=session_id,
            title=title,
            start_at=start_at,
            end_at=end_at,
            is_all_day=is_all_day,
            location=location,
            notes=notes,
        )
        with SessionLocal() as db:
            db.add(ev)
            db.commit()
            db.refresh(ev)
        logger.info("Calendar event created: id=%s title=%s", ev.id, ev.title)
        return ev

    def delete_event(self, event_id: int) -> bool:
        with SessionLocal() as db:
            ev = db.execute(
                select(CalendarEvent).where(CalendarEvent.id == event_id)
            ).scalar_one_or_none()
            if not ev:
                return False
            db.delete(ev)
            db.commit()
        logger.info("Calendar event deleted: id=%s", event_id)
        return True
