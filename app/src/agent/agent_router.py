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
    "record_expense",
    "query_expenses",
    "query_accounts",
    "create_account",
    "ask_user_confirm_balance",
    "query_database",
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
            {
                "name": "record_expense",
                "description": "记录一笔支出或收入。用户说花了/用了/付了/收了多少钱时使用。从用户描述中自动提取金额、分类、账户、描述。支出金额为正，系统自动记为负数；收入金额直接记录。",
                "args": {
                    "amount": "float. 金额（正数）。系统会自动将支出记为负数。",
                    "category": "string|null. 分类。支出可选：餐饮/交通/购物/娱乐/医疗/教育/住房/日用品/服饰/通讯/其他。收入可选：工资/红包/投资/兼职/退款/其他。从用户描述推断。",
                    "description": "string. 具体用途描述，如'午饭''买书''打车'",
                    "account_name": "string|null. 账户名。从对话中识别用户提到的账户，如'微信''支付宝''现金'",
                    "transaction_type": "string. 'expense' 或 'income'",
                },
            },
            {
                "name": "query_expenses",
                "description": "查询和分析记账记录。用户问花了多少钱、支出情况、账单分析时使用。",
                "args": {
                    "start_date": "string|null, YYYY-MM-DD。查询起始日期，根据用户说法自行计算。",
                    "end_date": "string|null, YYYY-MM-DD。查询结束日期，根据用户说法自行计算。",
                    "category": "string|null. 筛选特定分类。",
                    "group_by": "string, default 'category'。汇总方式：category / account / month",
                },
            },
            {
                "name": "query_accounts",
                "description": "查询用户的所有账户及余额。用户问我的账户、我的余额、还有多少钱时使用。",
                "args": {},
            },
            {
                "name": "create_account",
                "description": "创建新账户。用户说'我的XX有XX元''我XX账户有XX钱'时使用。账户已存在会检测余额是否一致，不一致时设置 pending_balance_update 状态。",
                "args": {
                    "name": "string. 账户名称，如'微信''支付宝''现金'",
                    "balance": "float. 账户余额（正数）",
                },
            },
            {
                "name": "ask_user_confirm_balance",
                "description": "询问用户确认是否更新已有账户的余额。仅当 create_account 检测到余额不一致并设置了 pending_balance_update 后使用。",
                "args": {},
            },
            {
                "name": "query_database",
                "description": "对业务数据库或知识库执行只读 SQL 查询。用于灵活查看数据，如列出表结构、统计记录数、按条件筛选内容等。支持 SELECT/PRAGMA/EXPLAIN。会自动尝试 business 和 knowledge 两个数据库。",
                "args": {
                    "sql": "string. 合法的 SELECT/PRAGMA/EXPLAIN 语句",
                    "max_rows": "int, default 50. 最大返回行数",
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
        "- 工具名暗示能力边界。query_* 只读查询，propose_* / ask_* / record_* 会写数据。\n"
        "- args 里的日期必须是你自己算好的 YYYY-MM-DD，不要传'明天''端午节'等文本。\n"
        "- 用户说的日期可能是传统节日、相对日期（后天/下周/月底）、或具体日期，自行识别。\n"
        "- 如果拿不准用户意图，优先选 ask_user_confirm_proposal（反问确认）。\n"
        "- 如果没有匹配的工具，选 normal_chat。\n"
        "- 知识库仅在用户问题与可用主题明确相关时才用。\n"
        "- 记账相关：用户说'花了xx' '付了xx' '收入xx' '用了xx钱' 等涉及金额的描述，优先用 record_expense。用户说'查账单''花了多少钱''分析支出' 用 query_expenses。用户说'我的账户''我的余额' 用 query_accounts。用户说'我的XX有XX元''我XX账户有XX钱' 用 create_account。如果 create_account 提示余额不一致，接下来用 ask_user_confirm_balance 让用户确认。\n"
        "- query_database：当用户想查看数据统计、列表、表结构，或现有专用工具无法满足的灵活查询时使用。例如'知识库有哪些文件''这个月记了多少笔账''user_profiles表有哪些字段'。优先使用专用工具，专用工具不足时用此工具兜底。\n\n"
        "## 输出格式\n"
        "输出 JSON：\n"
        "{\n"
        '  "action": "normal_chat|query_todos|query_knowledge|propose_todo|ask_user_confirm_proposal|query_calendar|query_weather|record_expense|query_expenses|query_accounts|create_account|ask_user_confirm_balance|query_database",\n'
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
