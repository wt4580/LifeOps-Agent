from pydantic import BaseModel


class PlanProposeRequest(BaseModel):
    """生成计划草案请求"""
    text: str


class PlanConfirmRequest(BaseModel):
    """确认计划请求"""
    proposal_id: str
    session_id: str
