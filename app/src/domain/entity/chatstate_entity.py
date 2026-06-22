from typing import Any, TypedDict


# ChatState 是"图上流动的数据包"：
# 每个 node 读取/修改其中的字段，再交给下一个 node。
# 理解这个结构，就能理解整个状态机的数据依赖。
class ChatState(TypedDict, total=False):
    # 输入
    session_id: str
    user_message: str
    # 前置画像信息：由 main 在进 graph 前准备。
    profile_context: str | None
    pre_advice: str | None
    # 供路由使用的增强输入（用户原文 + 画像/建议）。
    effective_user_message: str

    # 运行时上下文
    history: list[dict[str, str]]
    recent_dialogue: list[dict[str, str]]
    summary: str | None

    # 路由/执行
    decision: dict[str, Any]
    forced_action: str | None
    forced_args: dict[str, Any] | None
    answer: str
    used_tool: str | None

    # 反问/澄清上下文（记录原始意图，下一轮继续执行）
    pending_clarification: dict | None

    # 提案/HITL（由 checkpoint 持久化）
    pending_text: str | None
    hitl_cancelled: bool
    proposal_id: str | None
    proposal: dict[str, Any] | None
    pending_add_confirmation: bool
    proposal_confirmed: bool

    # 主动决策上下文（用于 postprocess 阶段）
    recent_events: list[dict[str, Any]]
    upcoming_todos: list[dict[str, Any]]
    pending_reminders: list[dict[str, Any]]
    event_window_start: str | None
    event_window_end: str | None

    # 目标列表（供 normal_chat context 使用）
    goals: list[dict[str, Any]]

    # 执行验证 + 重试
    retry_info: dict[str, Any] | None
    retry_count: int

    # 多步分解（plan-and-execute）
    plan: list[dict[str, Any]]
    plan_index: int

    # 建议频率控制（由 ReminderItem 持久化替代，此字段已废弃）

    # Trace
    citations: list[dict[str, Any]]
    trace: dict[str, Any]
