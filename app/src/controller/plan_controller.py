"""计划控制器 - 处理待办计划相关的HTTP请求"""


from fastapi import APIRouter, HTTPException
from ..common.harness.context import turn_context, TurnLimitExceeded

from ..common.domain.response import success_response
from ..service.plan_service import plan_service
from ..domain.dto.plan_dto import PlanProposeRequest

router = APIRouter()


@router.post("/api/plan/propose")
def propose_plan(req: PlanProposeRequest, session_id: str = "default"):
    """生成待办草案
    
    当用户表达待办意图后，调用此接口生成具体草案
    """
    try:
        with turn_context(session_id, register=False, operation="plan_proposal") as turn:
            result = plan_service.generate_proposal(text=req.text, session_id=session_id)
        result["trace"] = {"steps": [{"type": "postprocess", "name": "harness_summary", "output": turn.summary()}]}
        return success_response(data=result)
    except TurnLimitExceeded:
        raise HTTPException(status_code=429, detail="本次规划已达到运行预算，请缩小请求范围后继续。")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
