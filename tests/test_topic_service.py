from __future__ import annotations

from datetime import datetime, timedelta

from app.src.service.topic_service import (
    _classify_turn,
    _make_topic_label,
    _should_open_new,
)


class TestClassifyTurn:
    def test_propose_turn_is_task(self):
        state = {"used_tool": "plan_proposal", "decision": {"action": "propose_todo"}}
        assert _classify_turn(state) == "task"

    def test_decompose_goal_is_task(self):
        state = {"used_tool": "decompose_goal", "decision": {"action": "decompose_goal"}}
        assert _classify_turn(state) == "task"

    def test_confirm_turn_is_close(self):
        state = {"used_tool": "plan_gate_confirm", "proposal_confirmed": True}
        assert _classify_turn(state) == "close"

    def test_cancel_turn_is_close(self):
        state = {"used_tool": "plan_gate_cancel"}
        assert _classify_turn(state) == "close"

    def test_chatter_is_chatter(self):
        state = {"used_tool": None, "decision": {"action": "normal_chat"}}
        assert _classify_turn(state) == "chatter"

    def test_quick_chat_is_chatter(self):
        state = {"used_tool": None, "decision": {"action": "normal_chat"}, "mode": "quick"}
        assert _classify_turn(state) == "chatter"

    def test_weather_query_is_chatter(self):
        state = {"used_tool": "query_weather", "decision": {"action": "query_weather"}}
        assert _classify_turn(state) == "chatter"

    def test_stale_proposal_does_not_make_task(self):
        # checkpointer 会带过来上一轮的 proposal，但本轮只是普通聊天，不应误判成任务段
        state = {"used_tool": None, "decision": {"action": "normal_chat"}, "proposal": {"items": []}}
        assert _classify_turn(state) == "chatter"

    def test_record_expense_is_chatter(self):
        state = {"used_tool": "record_expense", "decision": {"action": "record_expense"}}
        assert _classify_turn(state) == "chatter"


class TestMakeTopicLabel:
    def test_truncate_long(self):
        assert _make_topic_label("请帮我把下周的会议预约安排到周五下午三点并追加到日历") == "请帮我把下周的会议预约安排到周五下午三点并追加到日历"[:24]

    def test_collapse_whitespace(self):
        assert _make_topic_label("  安排\n会议  ") == "安排 会议"

    def test_empty_fallback(self):
        assert _make_topic_label("") == "未命名任务"


class TestShouldOpenNew:
    def test_no_active(self):
        assert _should_open_new(None, "你好") is True

    def test_switch_marker(self):
        active = {"last_active_at": datetime.utcnow()}
        assert _should_open_new(active, "先不说这个了，帮我订个体检") is True

    def test_within_gap_keeps_episode(self):
        active = {"last_active_at": datetime.utcnow() - timedelta(minutes=2)}
        assert _should_open_new(active, "会议改到后天") is False

    def test_beyond_gap_opens_new(self):
        active = {"last_active_at": datetime.utcnow() - timedelta(minutes=30)}
        assert _should_open_new(active, "会议改到后天") is True

    def test_bare_continuation(self):
        active = {"last_active_at": datetime.utcnow()}
        assert _should_open_new(active, "好的") is False