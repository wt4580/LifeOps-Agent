from fastapi import APIRouter, HTTPException, Query

from ..common.domain.response import success_response
from ..domain.dto.bookkeeping_dto import AccountCreateRequest, TransactionCreateRequest, ExpenseAnalysisRequest
from ..service.bookkeeping_service import bookkeeping_service, BookkeepingError

router = APIRouter()


@router.post("/api/accounts")
def create_account(req: AccountCreateRequest):
    try:
        result = bookkeeping_service.add_account(
            name=req.name, balance=req.balance, session_id=req.session_id or ""
        )
        return success_response(data=result)
    except BookkeepingError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/api/accounts")
def list_accounts(session_id: str = Query(default="", description="会话ID")):
    result = bookkeeping_service.list_accounts(session_id=session_id)
    return success_response(data=result)


@router.get("/api/accounts/balances")
def account_balances(session_id: str = Query(default="", description="会话ID")):
    result = bookkeeping_service.get_account_balance(session_id=session_id)
    return success_response(data=result)


@router.delete("/api/accounts/{account_id}")
def delete_account(account_id: int, session_id: str = Query(default="")):
    ok = bookkeeping_service.delete_account(account_id=account_id, session_id=session_id)
    if not ok:
        raise HTTPException(status_code=404, detail="账户不存在")
    return success_response(data={"deleted": True})


@router.post("/api/transactions")
def create_transaction(req: TransactionCreateRequest):
    try:
        result = bookkeeping_service.record_transaction(
            amount=req.amount,
            session_id=req.session_id or "",
            account_id=req.account_id,
            account_name=req.account_name,
            category=req.category,
            description=req.description,
            transaction_at=req.transaction_at,
        )
        return success_response(data=result)
    except BookkeepingError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/api/transactions")
def list_transactions(
    session_id: str = Query(default=""),
    start_date: str | None = Query(default=None),
    end_date: str | None = Query(default=None),
    category: str | None = Query(default=None),
    account_id: int | None = Query(default=None),
    limit: int = Query(default=50),
):
    result = bookkeeping_service.list_transactions(
        session_id=session_id,
        start_date=start_date,
        end_date=end_date,
        category=category,
        account_id=account_id,
        limit=limit,
    )
    return success_response(data=result)


@router.get("/api/transactions/analysis")
def analyze(
    session_id: str = Query(default=""),
    start_date: str | None = Query(default=None),
    end_date: str | None = Query(default=None),
    group_by: str = Query(default="category"),
):
    result = bookkeeping_service.analyze(
        session_id=session_id,
        start_date=start_date,
        end_date=end_date,
        group_by=group_by,
    )
    return success_response(data=result)
