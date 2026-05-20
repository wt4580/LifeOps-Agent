from fastapi import HTTPException

class GeneralException(HTTPException):
    """通用服务器错误 → 500"""
    def __init__(self, detail: str):
        super().__init__(status_code=500, detail=detail)


class ServiceInvocationException(HTTPException):
    """模型调用 / 外部 API 访问失败 → 503"""
    def __init__(self, detail: str):
        super().__init__(status_code=503, detail=detail)


class DatabaseConnectionException(HTTPException):
    """数据库连接失败 → 504"""
    def __init__(self, detail: str):
        super().__init__(status_code=504, detail=detail)