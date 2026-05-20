"""计划控制器 - 处理待办计划相关的HTTP请求"""


from fastapi import APIRouter, HTTPException

from ..common.domain.response import success_response
from ..service.plan_service import plan_service
from ..domain.dto.plan_dto import PlanProposeRequest, PlanConfirmRequest

router = APIRouter()


@router.post("/api/plan/propose")
def propose_plan(req: PlanProposeRequest, session_id: str = "default"):
    """生成待办草案
    
    当用户表达待办意图后，调用此接口生成具体草案
    """
    try:
        result = plan_service.generate_proposal(text=req.text, session_id=session_id)
        return success_response(data=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/plan/confirm")
def confirm_plan(req: PlanConfirmRequest):
    """确认并保存待办草案到数据库
    
    用户点击确认按钮后调用
    """
    try:
        success = plan_service.confirm_proposal(proposal_id=req.proposal_id, session_id=req.session_id)
        if success:
            return success_response(message="待办已保存")
        else:
            raise HTTPException(status_code=404, detail="提案不存在或已过期")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))