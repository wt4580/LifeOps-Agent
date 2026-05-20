from typing import Any, Optional, Generic, TypeVar
from pydantic import BaseModel

T = TypeVar('T')

class BaseResponse(BaseModel, Generic[T]):
    success: bool
    resultCode: int
    resultDesc: Optional[str]
    result: Optional[T] = None


def success_response(data: Any = None, message: str = "Success") -> BaseResponse:
    return BaseResponse(
        success=True,
        resultCode=200,
        resultDesc=None,
        result=data
    )

def error_response(code: int, message: str) -> BaseResponse:
    return BaseResponse(
        success=False,
        resultCode=code,
        resultDesc=message,
        result=None
    )