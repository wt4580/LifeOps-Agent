"""语音识别接口。"""

from fastapi import APIRouter, Body, HTTPException
from ..common.config.base_config import settings

from ..common.domain.response import success_response
from ..common.config.log_config import logger
from ..service.asr_service import transcribe

router = APIRouter()


@router.post("/api/speech")
def speech_to_text(body: bytes = Body(...)):
    if settings.demo_mode:
        raise HTTPException(status_code=503, detail="离线演示镜像未启用语音模型，请使用文本输入")
    """接收手机录音字节流，返回识别文本。

    请求体：原始音频字节（audio/webm 等）
    响应：{"success": true, "result": {"text": "识别出的文字"}}
    """
    logger.info("Speech request bytes=%d", len(body))
    text = transcribe(body)
    return success_response(data={"text": text})
