"""RAG控制器 - 处理知识库相关的HTTP请求"""

from fastapi import APIRouter, HTTPException

from ..common.domain.response import success_response
from ..service.rag_service import rag_service
from ..domain.dto.rag_dto import IndexRequest, AskRequest

router = APIRouter()


@router.post("/api/index")
def index_documents(req: IndexRequest | None = None):
    """索引指定目录下的文档
    
    Args:
        req: 可选的文档目录路径，如果为空则使用配置中的默认值
    """
    try:
        result = rag_service.index_docs()
        return success_response(data=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/ask")
def ask_question(req: AskRequest):
    """基于知识库回答问题
    
    Args:
        req: 包含用户问题的请求
    """
    try:
        result = rag_service.ask_question(question=req.question)
        return success_response(data=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
