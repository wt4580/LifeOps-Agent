from __future__ import annotations

"""lifeops.agent_router

这是 LifeOps-Agent 的“工具路由器（Router）”。

目标：让大模型真正参与“要不要调用工具、调用哪个工具、工具参数是什么”。

为什么要单独做一个 Router：
- 之前的实现大量依赖关键词 if/else，很像一个脚本，并不“像智能体”。
- ReAct/工具调用智能体的关键是：先由 LLM 做决策（Reason），再执行工具（Act）。

注意：
- 我们不会要求模型输出完整的内部思考链（Chain-of-Thought）。
  这是不可靠且可能违反平台/安全规范的。
- 我们会强制模型输出“可审计的决策轨迹 trace”——包含意图、依据、工具选择与参数。
  这对调试与写简历更有价值。

Router 输出严格 JSON，会用 Pydantic 校验；解析失败时降级到 normal_chat。
"""

import json
import logging
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from .llm_qwen import chat_completion

logger = logging.getLogger(__name__)


Action = Literal[
    "normal_chat",
    "query_todos",
    "propose_todo",
    "ask_user_confirm_proposal",
]


class RouterTrace(BaseModel):
    """给前端/日志看的“决策轨迹”。

    这不是 CoT（不会暴露模型的内部逐步推理），而是可审计的理由摘要。
    """

    intent: str = Field(..., description="例如 schedule_query / todo_create / chitchat")
    signals: list[str] = Field(default_factory=list, description="命中的关键词/句式信号")
    why: str = Field(..., description="为什么选择该 action（简短可读）")


class RouterDecision(BaseModel):
    action: Action
    args: dict[str, Any] = Field(default_factory=dict)
    assistant_message: str = Field(..., description="给用户的第一句回应（可为空字符串）")
    trace: RouterTrace


@dataclass
class RouteContext:
    session_id: str
    user_message: str
    # 最近对话（短期记忆）
    recent_dialogue: list[dict]
    # 摘要（中期记忆）
    summary: str | None
    # 简单信号（关键词/句式）
    signals: list[str]


def _detect_signals(text: str) -> list[str]:
    """弱规则：只做“候选信号”，不直接决定分支。"""

    signals: list[str] = []
    markers = [
        # 查询类
        "有什么",
        "有哪些",
        "查看",
        "查一下",
        "查询",
        "列出",
        "有没有",
        # 新增类
        "记住",
        "提醒",
        "帮我",
        "我要",
        "添加",
        "加入",
        # 时间
        "今天",
        "明天",
        "后天",
        "这周",
        "下周",
        "周一",
        "周二",
        "周三",
        "周四",
        "周五",
        "周六",
        "周日",
        "周天",
    ]
    for m in markers:
        if m in text:
            signals.append(m)
    if "?" in text or "？" in text:
        signals.append("?")
    return signals


def build_route_context(
    *,
    session_id: str,
    user_message: str,
    recent_dialogue: list[dict],
    summary: str | None,
) -> RouteContext:
    return RouteContext(
        session_id=session_id,
        user_message=user_message,
        recent_dialogue=recent_dialogue,
        summary=summary,
        signals=_detect_signals(user_message),
    )


def route_decision(ctx: RouteContext) -> RouterDecision:
    """用 LLM 做一次路由决策。

    - 输出严格 JSON
    - 失败则降级 normal_chat
    """

    tool_spec = {
        "tools": [
            {
                "name": "query_todos",
                "description": "查询用户待办/日程（只读），当用户在问\"我今天/明天/这周/下周有什么安排\"等时使用。",
                "args": {
                    "range_days": "int, default 7. 查询未来多少天",
                },
            },
            {
                "name": "propose_todo",
                "description": "从用户一句话中生成待办草案（不写入数据库），当用户在描述将要做的事/会议/DDL，并希望记录提醒时使用。",
                "args": {
                    "text": "string. 原始用户输入原封不动传入",
                },
            },
            {
                "name": "ask_user_confirm_proposal",
                "description": "当你认为应该生成待办草案，但需要用户确认（Human-in-the-loop）时选择它。",
                "args": {
                    "text": "string. 原始用户输入原封不动传入",
                },
            },
            {
                "name": "normal_chat",
                "description": "普通聊天，不调用任何工具。",
                "args": {},
            },
        ]
    }

    system = (
        "你是一个严谨的生活助理智能体 Router。你的任务是：判断用户输入属于哪种动作，并输出严格 JSON。\n"
        "重要规则：\n"
        "1) 你必须只输出 JSON，不能输出多余文字。\n"
        "2) 你不得虚构待办/日程。要查询已存在的安排，必须选择 query_todos。\n"
        "3) 新增待办必须 Human-in-the-loop：如果用户像是在描述一个未来事件（例如\"明天要去理发\"），优先选择 ask_user_confirm_proposal，先询问用户是否生成草案。\n"
        "4) 只有当用户明确在问\"我有什么安排\"这类问题，才选择 query_todos。\n"
        "5) 你的输出 JSON schema：\n"
        "{\n"
        "  \"action\": \"normal_chat|query_todos|propose_todo|ask_user_confirm_proposal\",\n"
        "  \"args\": { ... },\n"
        "  \"assistant_message\": \"给用户的第一句回应（中文）\",\n"
        "  \"trace\": {\"intent\": \"...\", \"signals\": [...], \"why\": \"...\"}\n"
        "}\n"
        "trace.why 必须是可读的简短理由，不要写长篇推理。"
    )

    payload = {
        "session_id": ctx.session_id,
        "user_message": ctx.user_message,
        "signals": ctx.signals,
        "summary": ctx.summary,
        "recent_dialogue": ctx.recent_dialogue,
        "tool_spec": tool_spec,
    }

    raw = chat_completion(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        temperature=0.0,
    )

    try:
        data = json.loads(raw)
        decision = RouterDecision.model_validate(data)
        return decision
    except (json.JSONDecodeError, ValidationError) as exc:
        logger.warning("RouterDecision invalid, fallback to normal_chat: %s; raw=%r", exc, raw)
        return RouterDecision(
            action="normal_chat",
            args={},
            assistant_message="",
            trace=RouterTrace(intent="fallback", signals=ctx.signals, why="Router 输出无法解析，降级为普通对话"),
        )

