from fastapi import APIRouter
from ..common.domain.response import success_response

router = APIRouter()

@router.get("/health", include_in_schema=False)
def healthz():
    return success_response(data="System is running")