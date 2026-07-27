from pydantic import BaseModel


class IndexRequest(BaseModel):
    """索引文档请求"""
    reset: bool = False


class AskRequest(BaseModel):
    """知识库问答请求"""
    question: str
