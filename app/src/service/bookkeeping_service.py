from __future__ import annotations

import logging
from datetime import datetime, date, timedelta, timezone
import math

from sqlalchemy import select, func, and_, update
from sqlalchemy.orm import joinedload

from ..common.config.db_config import SessionLocal
from ..common.config.log_config import logger
from ..domain.entity.chat_entity import Account, Transaction

# 常用支出分类
EXPENSE_CATEGORIES = {"餐饮", "交通", "购物", "娱乐", "医疗", "教育", "住房", "日用品", "服饰", "通讯", "其他"}
# 常用收入分类
INCOME_CATEGORIES = {"工资", "红包", "投资", "兼职", "退款", "其他"}


class BookkeepingError(Exception):
    pass


def _utc_transaction_time(value: str | None = None) -> datetime:
    """Naive user input is local; persisted transaction timestamps remain UTC."""
    parsed = datetime.fromisoformat(value) if value else datetime.now().astimezone()
    return parsed.astimezone(timezone.utc).replace(tzinfo=None)


def _local_transaction_time(value: datetime) -> str:
    return value.replace(tzinfo=timezone.utc).astimezone().isoformat()


class BookkeepingService:
    def add_account(self, name: str, balance: float = 0.0, session_id: str = "") -> dict:
        if not math.isfinite(balance):
            raise BookkeepingError("余额必须为有限数值")
        with SessionLocal() as db:
            existing = db.execute(
                select(Account).where(Account.session_id == session_id, Account.name == name)
            ).scalars().first()
            if existing:
                raise BookkeepingError(f"账户 '{name}' 已存在")
            acc = Account(session_id=session_id, name=name, balance=balance)
            db.add(acc)
            db.commit()
            db.refresh(acc)
            return {"id": acc.id, "name": acc.name, "balance": acc.balance}

    def list_accounts(self, session_id: str = "") -> list[dict]:
        with SessionLocal() as db:
            rows = db.execute(
                select(Account).where(Account.session_id == session_id).order_by(Account.id)
            ).scalars().all()
            return [
                {"id": r.id, "name": r.name, "balance": r.balance,
                 "created_at": r.created_at.isoformat() if r.created_at else None}
                for r in rows
            ]

    def delete_account(self, account_id: int, session_id: str = "") -> bool:
        with SessionLocal() as db:
            acc = db.execute(
                select(Account).where(Account.id == account_id, Account.session_id == session_id)
            ).scalars().first()
            if not acc:
                return False
            db.delete(acc)
            db.commit()
            return True

    def record_transaction(
        self,
        *,
        amount: float,
        session_id: str = "",
        account_id: int | None = None,
        account_name: str | None = None,
        category: str | None = None,
        description: str | None = None,
        transaction_at: str | None = None,
    ) -> dict:
        if not math.isfinite(amount) or amount == 0:
            raise BookkeepingError("金额必须为有限的非零数值")
        try:
            tx_time = _utc_transaction_time(transaction_at)
        except ValueError as exc:
            raise BookkeepingError("交易日期格式不正确") from exc

        with SessionLocal() as db:
            resolved_account_id = account_id
            resolved_account_name = account_name

            if account_id:
                acc = db.execute(select(Account).where(
                    Account.id == account_id, Account.session_id == session_id
                )).scalars().first()
                if acc is None:
                    raise BookkeepingError("账户不存在或不属于当前会话")
                if account_name and account_name != acc.name:
                    raise BookkeepingError("账户 ID 与名称不一致")
                resolved_account_name = acc.name

            if account_name and not account_id:
                acc = db.execute(
                    select(Account).where(Account.session_id == session_id, Account.name == account_name)
                ).scalars().first()
                if acc:
                    resolved_account_id = acc.id
                    resolved_account_name = acc.name
                else:
                    raise BookkeepingError("账户不存在，请先创建账户")

            tx = Transaction(
                session_id=session_id,
                account_id=resolved_account_id,
                account_name=resolved_account_name or account_name,
                amount=amount,
                category=category,
                description=description,
                transaction_at=tx_time,
            )
            db.add(tx)

            if resolved_account_id:
                db.execute(update(Account).where(
                    Account.id == resolved_account_id, Account.session_id == session_id
                ).values(balance=func.round(func.coalesce(Account.balance, 0) + amount, 2)))

            db.commit()
            db.refresh(tx)
            return {
                "id": tx.id,
                "amount": tx.amount,
                "category": tx.category,
                "description": tx.description,
                "account_name": tx.account_name,
                "transaction_at": _local_transaction_time(tx.transaction_at) if tx.transaction_at else None,
            }

    def list_transactions(
        self,
        session_id: str = "",
        *,
        start_date: str | None = None,
        end_date: str | None = None,
        category: str | None = None,
        account_id: int | None = None,
        limit: int = 50,
    ) -> list[dict]:
        with SessionLocal() as db:
            conditions = [Transaction.session_id == session_id]

            if start_date:
                conditions.append(Transaction.transaction_at >= _utc_transaction_time(start_date))
            if end_date:
                conditions.append(
                    Transaction.transaction_at < _utc_transaction_time((datetime.fromisoformat(end_date) + timedelta(days=1)).isoformat())
                )
            if category:
                conditions.append(Transaction.category == category)
            if account_id:
                conditions.append(Transaction.account_id == account_id)

            rows = db.execute(
                select(Transaction)
                .where(and_(*conditions))
                .order_by(Transaction.transaction_at.desc())
                .limit(limit)
            ).scalars().all()

            return [
                {
                    "id": r.id,
                    "amount": r.amount,
                    "category": r.category,
                    "description": r.description,
                    "account_name": r.account_name,
                    "transaction_at": _local_transaction_time(r.transaction_at) if r.transaction_at else None,
                }
                for r in rows
            ]

    def analyze(
        self,
        session_id: str = "",
        *,
        start_date: str | None = None,
        end_date: str | None = None,
        group_by: str = "category",
    ) -> dict:
        with SessionLocal() as db:
            conditions = [Transaction.session_id == session_id]
            if start_date:
                conditions.append(Transaction.transaction_at >= _utc_transaction_time(start_date))
            if end_date:
                conditions.append(
                    Transaction.transaction_at < _utc_transaction_time((datetime.fromisoformat(end_date) + timedelta(days=1)).isoformat())
                )

            rows = db.execute(
                select(Transaction).where(and_(*conditions)).order_by(Transaction.transaction_at.desc())
            ).scalars().all()

            total_expense = sum(r.amount for r in rows if r.amount < 0)
            total_income = sum(r.amount for r in rows if r.amount > 0)

            from collections import defaultdict
            cat_map: dict[str, float] = defaultdict(float)
            cat_count: dict[str, int] = defaultdict(int)
            acct_map: dict[str, float] = defaultdict(float)
            acct_count: dict[str, int] = defaultdict(int)

            for r in rows:
                cat = r.category or "未分类"
                cat_map[cat] += r.amount
                cat_count[cat] += 1
                acct = r.account_name or "未知"
                acct_map[acct] += r.amount
                acct_count[acct] += 1

            by_category = sorted(
                [{"category": k, "total": round(v, 2), "count": cat_count[k]} for k, v in cat_map.items()],
                key=lambda x: x["total"],
            )
            by_account = sorted(
                [{"account_name": k, "total": round(v, 2), "count": acct_count[k]} for k, v in acct_map.items()],
                key=lambda x: x["total"],
            )

            return {
                "total_expense": round(abs(total_expense), 2),
                "total_income": round(total_income, 2),
                "net": round(total_income + total_expense, 2),
                "by_category": by_category,
                "by_account": by_account,
                "transaction_count": len(rows),
            }

    def get_account_balance(self, session_id: str = "") -> list[dict]:
        with SessionLocal() as db:
            rows = db.execute(
                select(Account).where(Account.session_id == session_id).order_by(Account.id)
            ).scalars().all()
            return [
                {"id": r.id, "name": r.name, "balance": r.balance,
                 "updated_at": r.updated_at.isoformat() if r.updated_at else None}
                for r in rows
            ]

    def find_account(self, name: str, session_id: str = "") -> dict | None:
        with SessionLocal() as db:
            acc = db.execute(
                select(Account).where(Account.session_id == session_id, Account.name == name)
            ).scalars().first()
            if not acc:
                return None
            return {"id": acc.id, "name": acc.name, "balance": acc.balance}

    def set_account_balance(self, account_id: int, balance: float, session_id: str = "") -> dict:
        if not math.isfinite(balance):
            raise BookkeepingError("余额必须为有限数值")
        with SessionLocal() as db:
            acc = db.execute(
                select(Account).where(Account.id == account_id, Account.session_id == session_id)
            ).scalars().first()
            if not acc:
                raise BookkeepingError("账户不存在")
            acc.balance = balance
            db.commit()
            db.refresh(acc)
            return {"id": acc.id, "name": acc.name, "balance": acc.balance}


bookkeeping_service = BookkeepingService()
