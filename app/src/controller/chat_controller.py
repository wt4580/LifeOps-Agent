from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from ..common.domain.response import success_response,BaseResponse
from ..service.chat_service import ChatService

router = APIRouter()

from ..domain.vo.chat_vo import ChatVo
from ..domain.dto.chat_dto import ChatDto
from ..common.config.log_config import logger


@router.post("/api/chat", response_model=BaseResponse[ChatDto])
async def chat(req: ChatVo):
    """聊天入口（LangGraph 状态机版）。

    主流程：
    Chat -> LLM decide -> Tool -> Update state -> (maybe HITL) -> Final answer

    关键顺序（为什么这样做）：
    1) 先保存用户消息：保证即使后续调用失败，也能保留用户输入审计记录。
    2) 再执行状态机：统一路由、工具调用与回复生成。
    3) 再保存助手消息：形成完整对话对（user/assistant）。
    4) 再做摘要与记忆抽取：这些属于增强能力，不阻塞主回复链路。
    """


    logger.info("#####################对话开始#########################")
    result = await ChatService().chat(message=req.message, session_id=req.session_id)
    logger.info("#####################对话结束#########################")
    return success_response(data=result)


@router.post("/api/chat/stream")
async def chat_stream(req: ChatVo):
    async def event_stream():
        async for event in ChatService().chat_stream(message=req.message, session_id=req.session_id):
            yield event

    return StreamingResponse(event_stream(), media_type="text/event-stream")
