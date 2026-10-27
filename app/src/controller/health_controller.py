from fastapi import APIRouter, HTTPException
from sqlalchemy import text
from ..common.domain.response import success_response
from ..common.config.base_config import settings

router = APIRouter()

@router.get("/health", include_in_schema=False)
def healthz():
    return success_response(data={"status": "ok", "mode": "demo" if settings.demo_mode else "live"})


@router.get("/health/ready", include_in_schema=False)
def ready():
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
