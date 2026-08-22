"""本地语音识别服务（faster-whisper）。

职责：
- 懒加载 WhisperModel 单例，避免每次请求重复初始化。
- 将手机端上传的录音字节流转写为文本。
"""

from __future__ import annotations

import logging
import tempfile
from functools import lru_cache
from pathlib import Path

from ..common.config.base_config import settings

logger = logging.getLogger(__name__)


def _resolve_model_source() -> str:
    """选择模型来源：本地目录优先，其次 HuggingFace 模型名。"""
    local = settings.whisper_model_local
    if local:
        p = Path(local)
        if p.exists():
            logger.info("Whisper model source=local path=%s", str(p))
            return str(p)
        logger.warning(
            "Whisper local model path not found: %s; fallback to remote model=%s",
            local,
            settings.whisper_model,
        )
    return settings.whisper_model


@lru_cache(maxsize=1)
def _model():
    """懒加载 WhisperModel 单例。"""
    from faster_whisper import WhisperModel

    model_path = _resolve_model_source()
    logger.info(
        "Loading Whisper model=%s device=%s compute_type=%s",
        model_path,
        settings.whisper_device,
        settings.whisper_compute_type,
    )
    return WhisperModel(
        model_path,
        device=settings.whisper_device,
        compute_type=settings.whisper_compute_type,
    )


def transcribe(audio_bytes: bytes, extension: str = "webm") -> str:
    """将音频字节流转写为文本。

    :param audio_bytes: 原始录音数据（MediaRecorder 输出的 webm/opus 等）
    :param extension: 临时文件扩展名，用于辅助格式探测
    :return: 识别出的文本（无语音时为空字符串）
    """
    with tempfile.NamedTemporaryFile(suffix=f".{extension}", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        model = _model()
        segments, info = model.transcribe(
            tmp_path,
            language=settings.whisper_language,
            vad_filter=True,
            beam_size=5,
        )
        text = "".join(seg.text for seg in segments).strip()
        logger.info(
            "ASR lang=%s duration=%.1fs text=%r",
            info.language,
            info.duration,
            text[:100],
        )
        return text
    finally:
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            pass
