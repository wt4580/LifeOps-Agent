import logging
from fastapi import Request, FastAPI
from fastapi.responses import JSONResponse

from .response import error_response
from .exceptions import (
    GeneralException,
    ServiceInvocationException,
    DatabaseConnectionException
)

logger = logging.getLogger(__name__)


def register_global_exception_handlers(app: FastAPI) -> None:

    @app.exception_handler(Exception)
    async def unexpected_exception_handler(request: Request, exc: Exception):
        trace_id = getattr(request.state, "trace_id", "")
        logger.error("unhandled_request trace_id=%s error_type=%s", trace_id, type(exc).__name__)
        return JSONResponse(status_code=500, content={"detail": "服务暂时不可用，请稍后重试", "trace_id": trace_id},
                            headers={"X-Trace-ID": trace_id})

    # --- 自定义三类异常 ---
    @app.exception_handler(GeneralException)
    async def general_exception_handler(request: Request, exc: GeneralException):
        logger.error(f"GeneralException (500): {exc.detail} | URL: {request.url}", exc_info=True)
        resp = error_response(exc.status_code, exc.detail)
        return JSONResponse(status_code=exc.status_code, content=resp.model_dump())

    @app.exception_handler(ServiceInvocationException)
    async def service_invocation_exception_handler(request: Request, exc: ServiceInvocationException):
        logger.warning(f"ServiceInvocationException (503): {exc.detail} | URL: {request.url}")
        resp = error_response(exc.status_code, exc.detail)
        return JSONResponse(status_code=exc.status_code, content=resp.model_dump())

    @app.exception_handler(DatabaseConnectionException)
    async def db_connection_exception_handler(request: Request, exc: DatabaseConnectionException):
        logger.critical(f"DatabaseConnectionException (504): {exc.detail} | URL: {request.url}", exc_info=True)
        resp = error_response(exc.status_code, exc.detail)
        return JSONResponse(status_code=exc.status_code, content=resp.model_dump())
