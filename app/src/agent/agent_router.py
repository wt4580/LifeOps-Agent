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
                "description": "查询用户已有的待办/日程安排（只读）。返回用户之前创建的待办事项列表。",
                "args": {
                    "target_date": "string|null, YYYY-MM-DD。如果用户提到某个日期/节日/相对时间，你计算后填入。",
                    "range_days": "int, default 7。如果没有明确日期，查未来多少天的安排。与target_date二选一。",
                },
            },
            {
                "name": "query_knowledge",
                "description": "在本地知识库中搜索信息（RAG）。用户问文档内容、事实型问题时使用。仅在 available_topics 中存在相关主题时才使用。",
                "args": {
                    "question": "string. 原始用户问题",
                    "top_k": "int, default 5",
                },
            },
            {
                "name": "propose_todo",
                "description": "将用户所述的未来计划/安排/承诺生成一条待办记录（写操作）。用户在说某个日期要做某件事时使用。",
                "args": {
                    "text": "string. 原始用户输入原封不动传入",
                },
            },
            {
                "name": "ask_user_confirm_proposal",
                "description": "用户表达了模糊需求或需要再确认的需求时，先反问用户确认再生成待办。当你不确定用户是否真的想创建待办时使用。",
                "args": {
                    "text": "string. 原始用户输入原封不动传入",
                },
            },
            {
                "name": "normal_chat",
                "description": "不调用任何工具的普通对话。用于闲聊、问候、个人近况讨论、或没有匹配工具时的兜底。",
                "args": {},
            },
            {
                "name": "query_calendar",
                "description": "查询日历中已存在的事件和节日（只读）。用户问节日日期、查已有日程时使用。",
                "args": {
                    "start_date": "string, YYYY-MM-DD。区间起点，你根据用户说法自行计算。",
                    "end_date": "string, YYYY-MM-DD。区间终点，你根据用户说法自行计算。",
                },
            },
            {
                "name": "query_weather",
                "description": "查询天气信息（实况+预报）。用户问天气、气温、是否下雨、穿什么时使用。",
                "args": {
                    "city": "string|null. 城市名。从对话中识别，否则留空。",
                },
            },
        ]
    }

    system = (
        "你是一个生活助理的决策引擎。你的任务：理解用户意图，从以下工具中选择最合适的。\n\n"
        "## 参考日历\n"
        "今天日期和今年剩余节日如下（请用此信息计算所有时间参数）：\n"
        f"{json.dumps(today_ctx, ensure_ascii=False, indent=2)}\n\n"
        "知识库中可用的主题列表：\n"
        f"{json.dumps(ctx.available_topics or [], ensure_ascii=False, indent=2)}\n"
        "文档大纲（标题层级）：\n"
        f"{json.dumps(ctx.available_outline or [], ensure_ascii=False, indent=2)}\n"
        "如果用户问题与上述主题或文档大纲中的章节相关，优先使用 query_knowledge。\n\n"
        "## 判断要点\n"
        "- 工具名暗示能力边界。query_* 只读查询，propose_* / ask_* 会写数据。\n"
        "- args 里的日期必须是你自己算好的 YYYY-MM-DD，不要传'明天''端午节'等文本。\n"
        "- 用户说的日期可能是传统节日、相对日期（后天/下周/月底）、或具体日期，自行识别。\n"
        "- 如果拿不准用户意图，优先选 ask_user_confirm_proposal（反问确认）。\n"
        "- 如果没有匹配的工具，选 normal_chat。\n"
        "- 知识库仅在用户问题与可用主题明确相关时才用。\n\n"
        "## 输出格式\n"
        "输出 JSON：\n"
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
        logger.error("RouterDecision LLM returned invalid JSON: %s; raw=%r", exc, raw)
        raise
