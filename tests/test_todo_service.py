from __future__ import annotations

from datetime import datetime, timedelta
from uuid import uuid4

import pytest

from app.src.agent.graph_chat import _build_proposal_item, _coerce_repeat, _handle_interrupt_resume, _parse_due_value
from app.src.common.config.db_config import SessionLocal
from app.src.domain.entity.chat_entity import TodoItem
from app.src.service.todo_service import TodoService


@pytest.fixture(scope="module", autouse=True)
def _ensure_db_schema():
    from app.src.common.config.db_config import init_db

    init_db()


@pytest.fixture
def clean_session():
    session_id = f"test-todo-{uuid4().hex[:8]}"
    yield session_id
    with SessionLocal() as db:
        db.execute(TodoItem.__table__.delete().where(TodoItem.session_id == session_id))
        db.commit()


def _insert(session_id: str, **kwargs):
    with SessionLocal() as db:
        item = TodoItem(session_id=session_id, title=kwargs.pop("title", "测试待办"), **kwargs)
        db.add(item)
        db.commit()
        db.refresh(item)
        return item.id


def _count(session_id: str) -> int:
    with SessionLocal() as db:
        from sqlalchemy import func
        return db.execute(func.count().select().where(TodoItem.session_id == session_id)).scalar_one()


class TestNullDueVisible:
    def test_null_due_pure_checklist_shows_everywhere(self, clean_session):
        _insert(clean_session, due_at=None)
        assert any(t["due_at"] is None for t in TodoService().get_all_todos(clean_session)["pending"])
        assert any(t["due_at"] is None for t in TodoService().get_upcoming_todos(clean_session))

    def test_null_due_not_in_today_without_recurrence(self, clean_session):
        _insert(clean_session, due_at=None)
        assert TodoService().get_todos_today(clean_session)["pending"] == []

    def test_completed_null_due_hidden(self, clean_session):
        _insert(clean_session, due_at=None, is_completed=True, completed_at=datetime.now())
        assert TodoService().get_all_todos(clean_session)["pending"] == []


class TestRecurringToday:
    def test_daily_active_shows_today(self, clean_session):
        _insert(clean_session, due_at=None, repeat_rule="daily", repeat_until=datetime.now() + timedelta(days=5))
        svc = TodoService()
        assert len(svc.get_todos_today(clean_session)["pending"]) == 1
        assert len(svc.get_upcoming_todos(clean_session)) == 1

    def test_daily_after_until_hidden(self, clean_session):
        _insert(clean_session, due_at=None, repeat_rule="daily", repeat_until=datetime.now() - timedelta(days=1))
        assert TodoService().get_todos_today(clean_session)["pending"] == []

    def test_weekly_only_on_anchor_weekday(self, clean_session):
        now = datetime.now()
        svc = TodoService()
        _insert(clean_session, due_at=now, repeat_rule="weekly")
        _insert(clean_session, due_at=now + timedelta(days=1), repeat_rule="weekly")
        assert len(svc.get_todos_today(clean_session)["pending"]) == 1  # 只有锚点在今天的生效

    def test_daily_completed_yesterday_resets_today(self, clean_session):
        _insert(
            clean_session,
            due_at=None,
            repeat_rule="daily",
            repeat_until=datetime.now() + timedelta(days=2),
            is_completed=True,
            completed_at=datetime.now() - timedelta(days=1),
        )
        pending = TodoService().get_todos_today(clean_session)["pending"]
        assert len(pending) == 1

    def test_daily_completed_today_goes_to_completed(self, clean_session):
        _insert(
            clean_session,
            due_at=None,
            repeat_rule="daily",
            repeat_until=datetime.now() + timedelta(days=2),
            is_completed=True,
            completed_at=datetime.now(),
        )
        today = TodoService().get_todos_today(clean_session)
        assert today["pending"] == []
        assert len(today["completed"]) == 1


class TestProposalBuild:
    def test_maps_due_from_start_date_and_repeat(self):
        item = _build_proposal_item(
            {
                "title": "练LeetCode",
                "start_date": "2026-08-31",
                "end_date": "2026-09-06",
                "repeat": "daily",
                "duration_min": 120,
            }
        )
        assert item["due_at"] == "2026-08-31"
        assert item["repeat"] == "daily"
        assert item["repeat_until"] == "2026-09-06"
        assert item["duration_min"] == 120

    def test_maps_due_time_with_date(self):
        item = _build_proposal_item({"title": "开会", "due_date": "2026-08-31", "due_time": "21:00"})
        assert item["due_at"] == "2026-08-31 21:00"

    def test_no_time_no_date_due_absent(self):
        item = _build_proposal_item({"title": "写周报"})
        assert "due_at" not in item
        assert "repeat" not in item

    def test_time_only_falls_to_nearest_hour(self):
        item = _build_proposal_item({"title": "吃药", "due_time": "12:00"})
        assert item["due_at"] is not None

    def test_parse_due_value_variants(self):
        assert _parse_due_value("2026-08-31") == datetime(2026, 8, 31)
        assert _parse_due_value("2026-08-31 21:00:00") == datetime(2026, 8, 31, 21, 0)
        assert _parse_due_value("2026/08/31") == datetime(2026, 8, 31)
        assert _parse_due_value(None) is None
        assert _parse_due_value("abc") is None

    def test_coerce_repeat_mapping(self):
        assert _coerce_repeat("每天") == "daily"
        assert _coerce_repeat("每周") == "weekly"
        assert _coerce_repeat("daily") == "daily"
        assert _coerce_repeat("yearly") is None
        assert _coerce_repeat(None) is None


class TestResumeWritesMergeFields:
    def test_confirm_resume_saves_due_repeat_source(self, clean_session):
        _insert_count = _count(clean_session)
        state = {
            "session_id": clean_session,
            "user_message": "确认",
            "decision": {"action": "propose_todo", "args": {}},
            "proposal_id": "p-test",
            "proposal": {
                "items": [
                    {
                        "title": "练LeetCode",
                        "due_at": "2026-08-31",
                        "repeat": "daily",
                        "repeat_until": "2026-09-06",
                    }
                ]
            },
        }
        out = _handle_interrupt_resume(state, "确认")
        assert out["proposal_confirmed"] is True
        assert _count(clean_session) == _insert_count + 1
        with SessionLocal() as db:
            row = (
                db.execute(TodoItem.__table__.select().where(TodoItem.session_id == clean_session))
                .first()
            )
            assert row.repeat_rule == "daily"
            assert row.source == "plan_proposal"
            assert row.due_at == datetime(2026, 8, 31)