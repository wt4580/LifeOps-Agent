from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import ValidationError

from ..common.config.log_config import logger
from ..common.config.llm_config import chat_completion
from ..domain.dto.router_dto import RouterTrace, RouterDecision
from ..util.retrieval.retrieval import get_available_knowledge_topics, get_document_outline
from .time.time_parser import build_date_context


Action = Literal[
    "normal_chat",
    "query_todos",
    "query_knowledge",
    "propose_todo",
    "ask_user_confirm_proposal",
    "query_calendar",
    "query_weather",
]


@dataclass
class RouteContext:
    session_id: str
    user_message: str
    recent_dialogue: list[dict]
    summary: str | None
    available_topics: list[dict] | None = None
    available_outline: list[dict] | None = None


def build_route_context(
    *,
    session_id: str,
    user_message: str,
    recent_dialogue: list[dict],
    summary: str | None,
) -> RouteContext:
    try:
        topics = get_available_knowledge_topics()
    except Exception:
        topics = None
    try:
        outline = get_document_outline()
    except Exception:
        outline = None
    return RouteContext(
        session_id=session_id,
        user_message=user_message,
        recent_dialogue=recent_dialogue,
        summary=summary,
        available_topics=topics,
        available_outline=outline,
    )


def route_decision(ctx: RouteContext) -> RouterDecision:
    today_ctx = build_date_context(remaining_only=True)

    tool_spec = {
        "tools": [
            {
                "name": "query_todos",
                "description": "查询用户已有的待办/日程安排（只读）。当用户问'我今天有什么安排'、'查一下下周三的事'、'上一条'等时使用。用户可能打错字或表达模糊，请结合对话历史推断。",
                "args": {
                    "target_date": "string|null, YYYY-MM-DD。如果用户提到某个日期/节日/相对时间，你计算后填入这个字段。",
                    "range_days": "int, default 7。如果没有明确日期，查未来多少天的安排。与target_date二选一。",
                },
            },
            {
                "name": "query_knowledge",
                "description": "查询本地知识库文档（RAG）。用户在问文档内容、文件内容、论文观点、某个PDF/PNG/TXT里有什么时使用。也可以用于事实型问题。参考上方可用主题列表判断是否应走此路径。",
                "args": {
                    "question": "string. 原始用户问题",
                    "top_k": "int, default 5",
                },
            },
            {
                "name": "propose_todo",
                "description": "用户描述了一个未来要做的事/会议/DDL，且明确希望记录时，用此工具生成待办草案。",
                "args": {
                    "text": "string. 原始用户输入原封不动传入",
                },
            },
            {
                "name": "ask_user_confirm_proposal",
                "description": "用户描述了一个需要跟进/处理的事，或表达对某事的担忧/等待/未完成（如'导员没回我报销'、'快递还没到'、'该交作业了'）。先询问用户是否生成待办草案。也用于模糊的未来计划（如'明天下午开会'）。",
                "args": {
                    "text": "string. 原始用户输入原封不动传入",
                },
            },
            {
                "name": "normal_chat",
                "description": "普通聊天，不调用任何工具。用于闲聊、问候、个人近况讨论等。",
                "args": {},
            },
            {
                "name": "query_calendar",
                "description": "查询日历中的事件，包括中国传统节日和用户创建的日程。用户问'端午节是哪一天'、'中秋什么时候'、'我本周的会议'时使用。",
                "args": {
                    "start_date": "string, YYYY-MM-DD。区间起点，你根据用户说法自行计算。",
                    "end_date": "string, YYYY-MM-DD。区间终点，你根据用户说法自行计算。",
                },
            },
            {
                "name": "query_weather",
                "description": "查询天气（实况+预报）。用户问天气、气温、是否下雨、穿什么等时使用。",
                "args": {
                    "city": "string|null. 城市名。从对话中识别，否则留空。",
                },
            },
        ]
    }

    system = (
        "你是一个生活助理的决策引擎。你的任务是：理解用户意图，选择最合适的工具，并输出严格 JSON。\n\n"
        "今天日期和今年剩余节日如下（请用此信息计算所有时间参数）：\n"
        f"{json.dumps(today_ctx, ensure_ascii=False, indent=2)}\n\n"
        "知识库中可用的主题列表：\n"
        f"{json.dumps(ctx.available_topics or [], ensure_ascii=False, indent=2)}\n"
        "文档大纲（标题层级）：\n"
        f"{json.dumps(ctx.available_outline or [], ensure_ascii=False, indent=2)}\n"
        "如果用户问题与上述主题或文档大纲中的章节相关，优先使用 query_knowledge。\n\n"
        "决策原则：\n"
        "- 你的 args 里的所有日期字段必须是你自己计算后的具体 YYYY-MM-DD，不要传'明天'、'端午节'等文本。\n"
        "- 用户可能用各种说法提到时间：传统节日（端午节/中秋/春节）、相对日期（后天/下周/月底）、具体日期。你都要算出公历日期。\n"
        "- query_todos 查用户自建的待办，query_calendar 查节日和日历事件。如果提到节日名优先用 query_calendar。\n"
        "- 用户表达困扰/等待/未完成（如'导员没回报销'、'快递没到'、'该交作业了'、'还没审核'），优先 ask_user_confirm_proposal 生成跟进待办。\n"
        "- 新增待办建议先 ask_user_confirm_proposal 让用户确认，除非用户非常明确（如'帮我记下来'）。\n"
        "- 天气/文档/闲聊等正常识别。\n"
        "- 用户在后续回复中可能打错字或用模糊短语（如'上一条'说成'上一台哦'），请参考 recent_dialogue 上下文推断真实意图。\n"
        "- 如果上一条 assistant 消息显示了待办列表，用户回复'上一条'/'上一个'类短语应路由到 query_todos 查待办。\n\n"
        "输出 JSON 格式：\n"
        "{\n"
        '  "action": "normal_chat|query_todos|query_knowledge|propose_todo|ask_user_confirm_proposal|query_calendar|query_weather",\n'
        '  "args": { ... },\n'
        '  "assistant_message": "给用户的第一句回应（中文，可留空）",\n'
        '  "trace": {"intent": "简要意图", "why": "简短选择理由"}\n'
        "}"
    )

    payload = {
        "session_id": ctx.session_id,
        "user_message": ctx.user_message,
        "summary": ctx.summary,
        "recent_dialogue": ctx.recent_dialogue,
        "tool_spec": tool_spec,
        "available_topics": ctx.available_topics or [],
        "available_outline": ctx.available_outline or [],
    }

    raw = chat_completion(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        temperature=0.0,
        runtime_context={"scenario": "router_decision", "session_id": ctx.session_id},
    )

    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = cleaned.strip()

    try:
        data = json.loads(cleaned)
        decision = RouterDecision.model_validate(data)
        return decision
    except (json.JSONDecodeError, ValidationError) as exc:
        logger.warning("RouterDecision invalid, fallback to normal_chat: %s; raw=%r", exc, raw)
        return RouterDecision(
            action="normal_chat",
            args={},
            assistant_message="",
            trace=RouterTrace(intent="fallback", why="Router 输出无法解析，降级为普通对话"),
        )
