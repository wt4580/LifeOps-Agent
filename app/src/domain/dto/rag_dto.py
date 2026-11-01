from pydantic import BaseModel, ConfigDict, Field


class IndexRequest(BaseModel):
    """索引文档请求"""
    reset: bool = False


class AskRequest(BaseModel):
    """知识库问答请求"""
    model_config = ConfigDict(str_strip_whitespace=True)
    question: str = Field(min_length=1, max_length=1000)
