from datetime import datetime

from pydantic import BaseModel, Field


class AccountCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=64, description="账户名称")
    balance: float = Field(default=0.0, description="初始余额")
    session_id: str | None = None


class AccountResponse(BaseModel):
    id: int
    name: str
    balance: float
    created_at: datetime | None = None


class TransactionCreateRequest(BaseModel):
    account_id: int | None = Field(default=None, description="账户ID（与account_name二选一）")
    account_name: str | None = Field(default=None, description="账户名称（与account_id二选一）")
    amount: float = Field(..., description="金额，支出为负数，收入为正数")
    category: str | None = Field(default=None, max_length=32, description="分类")
    description: str | None = Field(default=None, max_length=255, description="描述")
    transaction_at: str | None = Field(default=None, description="交易时间 ISO 格式，默认当前时间")
    session_id: str | None = None


class TransactionResponse(BaseModel):
    id: int
    account_name: str | None
    amount: float
    category: str | None
    description: str | None
    transaction_at: datetime | None


class ExpenseAnalysisRequest(BaseModel):
    session_id: str | None = None
    start_date: str | None = Field(default=None, description="起始日期 YYYY-MM-DD")
    end_date: str | None = Field(default=None, description="结束日期 YYYY-MM-DD")
    group_by: str = Field(default="category", description="分组方式: category / account / month")


class CategorySummary(BaseModel):
    category: str
    total: float
    count: int


class AccountSummary(BaseModel):
    account_name: str
    total: float
    count: int


class ExpenseAnalysisResponse(BaseModel):
    total_expense: float
    total_income: float
    net: float
    by_category: list[CategorySummary] = Field(default_factory=list)
    by_account: list[AccountSummary] = Field(default_factory=list)
    transaction_count: int
