import hmac
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response
from sqlalchemy import text
from ..common.domain.response import success_response
from ..common.config.base_config import settings

router = APIRouter()


@router.get("/metrics", include_in_schema=False)
async def operational_metrics(request: Request):
    if not settings.ops_metrics_token:
        raise HTTPException(status_code=404, detail="Not found")
    expected = "Bearer " + settings.ops_metrics_token
    supplied = request.headers.get("authorization", "")
    if not hmac.compare_digest(supplied.encode(), expected.encode()):
        raise HTTPException(status_code=401, detail="Unauthorized", headers={"WWW-Authenticate": "Bearer"})
    from ..common.telemetry import metrics
    from ..service.session_queue import queue_gauges
    return Response(metrics.render(queue_gauges()), media_type="text/plain; version=0.0.4",
                    headers={"Cache-Control": "no-store"})

@router.get("/health", include_in_schema=False)
def healthz():
    return success_response(data={"status": "ok", "mode": "demo" if settings.demo_mode else "live"})


@router.get("/health/ready", include_in_schema=False)
def ready():
    from ..service.session_queue import accepting_chat_jobs
    if not accepting_chat_jobs():
        raise HTTPException(status_code=503, detail="Application is draining")
    from ..common.config.db_config import SessionLocal
    from ..common.config.knowledge_db_config import KnowledgeSessionLocal
    from ..common.config.chat_graph_holder import get_chat_graph
    try:
        for factory in (SessionLocal, KnowledgeSessionLocal):
            with factory() as db:
                db.execute(text("SELECT 1"))
        if get_chat_graph() is None:
            raise RuntimeError("graph unavailable")
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Application is not ready") from exc
    return success_response(data={"status": "ready"})
