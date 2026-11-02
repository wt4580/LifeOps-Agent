"""Tool contracts shared by planning and the runtime boundary.

Models propose arguments; only the runtime validates and supplies user identity.
Business services retain transaction, ownership and confirmation enforcement.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time
from types import MappingProxyType
from typing import Annotated, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, StringConstraints, AfterValidator, model_validator


def _not_bool(value):
    if isinstance(value, bool):
        raise ValueError("boolean is not a number")
    return value


def _date(value):
    if value is not None:
        date.fromisoformat(value)
    return value


def _time(value):
    if value is not None:
        time.fromisoformat(value)
    return value


def _timestamp(value):
    if value is not None:
        datetime.fromisoformat(value)
    return value


def _nonzero(value):
    if value == 0:
        raise ValueError("amount must not be zero")
    return value


Text = Annotated[str, StringConstraints(strip_whitespace=True, max_length=8000)]
ShortText = Annotated[str, StringConstraints(strip_whitespace=True, max_length=255)]
Date = Annotated[str, StringConstraints(pattern=r"^\d{4}-\d{2}-\d{2}$"), AfterValidator(_date)]
Clock = Annotated[str, StringConstraints(pattern=r"^\d{2}:\d{2}(:\d{2})?$"), AfterValidator(_time)]
Timestamp = Annotated[str, StringConstraints(max_length=64), AfterValidator(_timestamp)]
Money = Annotated[float, BeforeValidator(_not_bool), Field(allow_inf_nan=False, ge=-1e12, le=1e12)]
Days = Annotated[int, BeforeValidator(_not_bool), Field(ge=1, le=90)]


class Args(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EmptyArgs(Args):
    pass


class TodoQueryArgs(Args):
    range_days: Days = 7


class KnowledgeArgs(Args):
    question: Text | None = None
    top_k: Annotated[int, BeforeValidator(_not_bool), Field(ge=1, le=20)] = 5


class WeatherArgs(Args):
    city: ShortText | None = None
    location: ShortText | None = None


class DateRangeArgs(Args):
    start_date: Date | None = None
    end_date: Date | None = None

    @model_validator(mode="after")
    def ordered_dates(self):
        if self.start_date and self.end_date and self.end_date < self.start_date:
            raise ValueError("end_date must not precede start_date")
        return self


class CalendarArgs(DateRangeArgs):
    range_days: Days = 7

    @model_validator(mode="after")
    def complete_range(self):
        if bool(self.start_date) != bool(self.end_date):
            raise ValueError("start_date and end_date must be supplied together")
        if self.start_date and (date.fromisoformat(self.end_date) - date.fromisoformat(self.start_date)).days >= 90:
            raise ValueError("calendar range must not exceed 90 days")
        return self


class TodoArgs(Args):
    title: ShortText | None = None
    text: Text | None = None
    description: ShortText | None = None
    due_at: Timestamp | None = None
    due_date: Date | None = None
    date: Date | None = None
    start_date: Date | None = None
    end_date: Date | None = None
    due_time: Clock | None = None
    time: Clock | None = None
    start_time: Clock | None = None
    repeat: Literal["once", "daily", "weekly"] | None = None
    repeat_until: Date | None = None
    duration_min: Annotated[int, BeforeValidator(_not_bool), Field(ge=1, le=1440)] | None = None


class GoalArgs(Args):
    text: Text | None = None


class RevisionArgs(Args):
    correction: Text | None = None


class TopicArgs(Args):
    topic: ShortText | None = None
    episode_id: Annotated[int, BeforeValidator(_not_bool), Field(ge=1)] | None = None


class ExpenseArgs(Args):
    amount: Annotated[Money, AfterValidator(_nonzero), Field(description="非零有限金额，交易方向由 transaction_type 指定")]
    transaction_type: Literal["expense", "income"] = "expense"
    account_name: ShortText | None = None
    category: ShortText | None = None
    description: ShortText | None = None
    transaction_at: Timestamp | None = None


class ExpenseQueryArgs(DateRangeArgs):
    category: ShortText | None = None
    group_by: Literal["category", "account"] = "category"


class AccountArgs(Args):
    name: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=64)]
    balance: Money = 0


class DatabaseArgs(Args):
    sql: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=12000)]
    max_rows: Annotated[int, BeforeValidator(_not_bool), Field(ge=1, le=200)] = 50


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    args_model: type[Args]
    effect: Literal["read", "write", "proposal", "conversation"]
    confirmation: Literal["none", "required", "if_existing"] = "none"
    model_visible: bool = True


_SPECS = (
    ToolSpec("normal_chat", "基于上下文回答或闲聊", EmptyArgs, "conversation"),
    ToolSpec("query_todos", "查询已创建的待办", TodoQueryArgs, "read"),
    ToolSpec("query_knowledge", "检索参考资料并引用来源", KnowledgeArgs, "read"),
    ToolSpec("query_weather", "查询城市天气", WeatherArgs, "read"),
    ToolSpec("query_calendar", "查询日历日程", CalendarArgs, "read"),
    ToolSpec("propose_todo", "生成待办草案，用户确认后才写入", TodoArgs, "proposal", "required"),
    ToolSpec("ask_user_confirm_proposal", "生成待办草案并等待确认", TodoArgs, "proposal", "required"),
    ToolSpec("revise_proposal", "按修改要求重新生成服务端已有待办草案", RevisionArgs, "proposal", "required", False),
    ToolSpec("decompose_goal", "拆解并保存长期目标及子目标，不自动创建待办", GoalArgs, "write"),
    ToolSpec("clarify", "询问缺少的关键信息，问题放在 assistant_message", EmptyArgs, "conversation"),
    ToolSpec("revive_topic", "用户主动提起旧话题时找回其记录", TopicArgs, "read"),
    ToolSpec("record_expense", "记录收支；amount 为金额，transaction_type 区分支出和收入", ExpenseArgs, "write"),
    ToolSpec("query_expenses", "查询账目及分类或账户汇总", ExpenseQueryArgs, "read"),
    ToolSpec("query_accounts", "查询当前用户账户", EmptyArgs, "read"),
    ToolSpec("create_account", "创建账户；账户已存在且余额不同则必须用户确认", AccountArgs, "write", "if_existing"),
    ToolSpec("ask_user_confirm_balance", "恢复服务端已有的余额确认流程", EmptyArgs, "conversation", "required", False),
    ToolSpec("query_database", "只读 SQL；仅查询当前用户的数据", DatabaseArgs, "read"),
)
TOOLS = MappingProxyType({spec.name: spec for spec in _SPECS})


def validate_tool_args(action, arguments):
    return TOOLS[action].args_model.model_validate(arguments).model_dump(exclude_none=True)


def tool_catalog():
    """Compact JSON schemas without repeated titles; generated from executable models."""
    def compact(value):
        if isinstance(value, dict):
            return {k: compact(v) for k, v in value.items()
                    if (k != "title" or isinstance(v, dict)) and not (k == "default" and v is None)}
        if isinstance(value, list):
            return [compact(v) for v in value]
        return value
    return [{"name": spec.name, "description": spec.description, "effect": spec.effect,
             "confirmation": spec.confirmation, "parameters": compact(spec.args_model.model_json_schema())}
            for spec in TOOLS.values() if spec.model_visible]
