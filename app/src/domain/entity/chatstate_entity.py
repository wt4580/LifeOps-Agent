from typing import Any, TypedDict


# ChatState is the data packet flowing through the graph.
class ChatState(TypedDict, total=False):
    session_id: str
    user_message: str
    profile_context: str | None
    pre_advice: str | None
    effective_user_message: str

    history: list[dict[str, str]]
    recent_dialogue: list[dict[str, str]]
    summary: str | None

    decision: dict[str, Any]
    forced_action: str | None
    forced_args: dict[str, Any] | None
    answer: str
    used_tool: str | None

    clarification_item: str | None

    proposal_id: str | None
    proposal: dict[str, Any] | None
    proposal_confirmed: bool

    interrupt_message: str | None
    interrupt_answer: str | None

    recent_events: list[dict[str, Any]]
    upcoming_todos: list[dict[str, Any]]
    event_window_start: str | None
    event_window_end: str | None

    goals: list[dict[str, Any]]

    retry_info: dict[str, Any] | None
    retry_count: int

    plan: list[dict[str, Any]]
    plan_index: int

    citations: list[dict[str, Any]]
    trace: dict[str, Any]
