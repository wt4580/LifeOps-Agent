from pydantic import BaseModel


class IndexRequest(BaseModel):
    """索引文档请求（可扩展）"""
    # 可选：添加路径、overwrite 等字段
    path: str | None = None


class AskRequest(BaseModel):
    """知识库问答请求"""
    question: str
