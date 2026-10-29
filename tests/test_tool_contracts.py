import json
from unittest.mock import Mock

import pytest

from app.src.domain.tool_registry import TOOLS, tool_catalog, validate_tool_args
from app.src.common.harness.context import turn_context
from app.src.common.harness.interceptors import ALLOWED_TOOL_ACTIONS, CallContext, PolicyViolation, tool_chain


@pytest.mark.parametrize("action,args", [
    ("normal_chat", {}), ("query_todos", {"range_days": "7"}),
    ("query_knowledge", {"question": "报销", "top_k": 5}),
    ("query_weather", {"location": "北京"}),
    ("query_calendar", {"start_date": "2026-10-01", "end_date": "2026-10-31"}),
    ("propose_todo", {"title": "开会", "start_date": "2026-11-01", "due_time": "15:00", "repeat": "once"}),
    ("propose_todo", {"title": "开会", "due_at": "2026-11-01T15:00:00"}),
    ("ask_user_confirm_proposal", {"text": "明天开会"}),
    ("revise_proposal", {"correction": "改到后天"}), ("decompose_goal", {"text": "学习 Python"}),
    ("clarify", {}), ("revive_topic", {"episode_id": 1}),
    ("record_expense", {"amount": "10", "transaction_type": "expense", "transaction_at": "2026-10-29T10:00:00+08:00"}),
    ("query_expenses", {"start_date": "2026-10-01", "group_by": "account"}),
    ("query_accounts", {}), ("create_account", {"name": "现金", "balance": -10}),
    ("ask_user_confirm_balance", {}), ("query_database", {"sql": "SELECT 1", "max_rows": "20"}),
])
def test_supported_contracts(action, args):
    assert isinstance(validate_tool_args(action, args), dict)


@pytest.mark.parametrize("action,args", [
    ("record_expense", {}), ("record_expense", {"amount": 0}),
    ("record_expense", {"amount": True}), ("record_expense", {"amount": "NaN"}),
    ("record_expense", {"amount": float("inf")}),
    ("record_expense", {"amount": 10, "transaction_type": "incmoe"}),
    ("record_expense", {"amount": 10, "transaction_at": "tomorrow"}),
    ("create_account", {"name": " "}), ("create_account", {"name": "x", "balance": float("nan")}),
    ("query_todos", {"range_days": -7}), ("query_todos", {"range_days": True}),
    ("query_todos", {"range_days": 100000}), ("query_knowledge", {"top_k": 20000}),
    ("query_calendar", {"start_date": "2026-02-30", "end_date": "2026-03-01"}),
    ("query_calendar", {"start_date": "2026-03-02", "end_date": "2026-03-01"}),
    ("query_calendar", {"start_date": "2026-03-01"}),
    ("query_calendar", {"start_date": "2026-01-01", "end_date": "2026-12-31"}),
    ("propose_todo", {"title": "meeting", "due_time": "25:00"}),
    ("propose_todo", {"repeat": "forever"}), ("propose_todo", {"duration_min": -1}),
    ("query_database", {"sql": "SELECT 1", "max_rows": 100000}),
    ("query_accounts", {"session_id": "bob"}), ("query_accounts", []),
    ("query_expenses", {"group_by": "arbitrary"}),
])
def test_invalid_arguments_never_reach_tool_execution(action, args):
    operation = Mock()
    ctx = CallContext("tool", f"tool:{action}", "alice", payload={"action": action, "args": args})
    with turn_context("alice") as turn, pytest.raises(PolicyViolation):
        ctx.turn = turn
        tool_chain().invoke(ctx, operation)
    operation.assert_not_called()
    assert turn.records[-1]["status"] == "blocked"


def test_catalog_is_the_runtime_contract_and_hides_internal_actions():
    assert ALLOWED_TOOL_ACTIONS == frozenset(TOOLS)
    catalog = tool_catalog()
    assert {entry["name"] for entry in catalog} == {spec.name for spec in TOOLS.values() if spec.model_visible}
    todo = next(item for item in catalog if item["name"] == "propose_todo")
    assert "title" in todo["parameters"]["properties"]
    assert todo["parameters"]["additionalProperties"] is False
    assert todo["confirmation"] == "required"
    assert TOOLS["decompose_goal"].effect == "write"
    assert not any("session_id" in entry["parameters"]["properties"] for entry in catalog)
    with pytest.raises(TypeError):
        TOOLS["untrusted"] = None


def test_internal_confirmation_cannot_be_selected_by_model():
    operation = Mock()
    ctx = CallContext("tool", "tool:ask_user_confirm_balance", "alice",
                      payload={"action": "ask_user_confirm_balance", "args": {}})
    with pytest.raises(PolicyViolation):
        tool_chain().invoke(ctx, operation)
    operation.assert_not_called()


def test_graph_normalizes_validated_arguments_and_stops_after_rejection(monkeypatch):
    from app.src.agent import graph_chat as graph
    operation = Mock(side_effect=lambda state: {**state, "answer": "done"})
    monkeypatch.setattr(graph, "_run_tool_impl", operation)
    state = {"session_id": "alice", "decision": {"action": "query_knowledge", "args": {"top_k": "3"}}}
    graph._node_run_tool(state)
    assert operation.call_args.args[0]["decision"]["args"]["top_k"] == 3

    operation.reset_mock()
    plan = [{"action": "record_expense", "args": {"amount": 10}, "result": "已记录"},
            {"action": "create_account", "args": {"name": "x", "balance": "NaN"}},
            {"action": "record_expense", "args": {"amount": 20}}]
    state = {"session_id": "alice", "plan": plan, "plan_index": 1, "decision": plan[1]}
    with turn_context("alice") as turn:
        result = graph._node_run_tool(state)
        assert result["tool_error"]["code"] == "invalid_tool_arguments"
        assert "前序已提交" in result["answer"]
        assert "NaN" not in json.dumps(result["tool_error"])
        graph._node_advance_plan(result)
        assert graph._route_after_advance(result) == "finalize"
        assert result["plan_index"] == 1
        graph._node_verify(result)
        graph._node_quality_gate(result)
        graph._node_proactive_decision(result)
        graph._finish_turn_result(result, turn)
        assert turn.status == "tool_rejected"
    operation.assert_not_called()


def test_planner_receives_generated_catalog(monkeypatch):
    from app.src.agent import graph_chat as graph
    completion = Mock(return_value='{"action":"normal_chat","args":{},"mode":"quick"}')
    monkeypatch.setattr(graph, "chat_completion", completion)
    graph._node_decompose({"session_id": "catalog-user", "user_message": "你好"})
    messages = completion.call_args.args[0]
    payload = json.loads(messages[1]["content"])
    assert payload["tools"] == tool_catalog()
    assert "工具契约" in messages[0]["content"]


@pytest.mark.parametrize("group,expected,excluded", [("account", "现金", "餐饮"), ("category", "餐饮", "现金")])
def test_expense_query_honors_the_advertised_grouping(monkeypatch, group, expected, excluded):
    from app.src.agent import graph_chat as graph
    from app.src.service.bookkeeping_service import bookkeeping_service
    monkeypatch.setattr(bookkeeping_service, "analyze", Mock(return_value={
        "transaction_count": 1, "total_expense": 10, "total_income": 0, "net": -10,
        "by_category": [{"category": "餐饮", "total": -10, "count": 1}],
        "by_account": [{"account_name": "现金", "total": -10, "count": 1}],
    }))
    result = graph._node_run_tool({"session_id": "alice", "user_message": "查询账目",
                                   "decision": {"action": "query_expenses", "args": {"group_by": group}}})
    assert expected in result["answer"] and excluded not in result["answer"]
