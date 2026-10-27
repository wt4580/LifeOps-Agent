"""RAG控制器 - 处理知识库相关的HTTP请求"""

from fastapi import APIRouter, HTTPException, Request
from ..common.harness.context import turn_context, TurnLimitExceeded

from ..common.domain.response import success_response
from ..service.rag_service import rag_service
from ..domain.dto.rag_dto import IndexRequest, AskRequest

router = APIRouter()


@router.post("/api/index")
def index_documents(req: IndexRequest | None = None):
    """索引指定目录下的文档
    
    Args:
        req: 索引请求，可指定 reset=true 重置索引
    """
    try:
        reset = (req or IndexRequest()).reset
        result = rag_service.index_docs(reset=reset)
        return success_response(data=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/ask")
def ask_question(req: AskRequest, request: Request):
    """基于知识库回答问题
    
    Args:
        req: 包含用户问题的请求
    """
    try:
        with turn_context(request.state.username, register=False, operation="rag_ask") as turn:
            try:
                result = rag_service.ask_question(question=req.question, session_id=request.state.username)
            except TurnLimitExceeded:
                turn.finish("budget_exceeded")
                result = {"answer": "本次检索已达到运行预算，请缩小问题范围后继续。", "citations": []}
            turn.finish("success")
            result["trace"] = {"steps": [{"type": "postprocess", "name": "harness_summary", "output": turn.summary()}]}
        return success_response(data=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
