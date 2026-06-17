from __future__ import annotations

import time
import logging
from datetime import datetime, timezone
from functools import lru_cache

from openai import OpenAI

from .base_config import settings


logger = logging.getLogger(__name__)


@lru_cache()
def get_llm_client() -> OpenAI:
    """Return a cached OpenAI-compatible client configured from settings.

    Centralizing client creation here keeps provider wiring in `common.config`.
    """
    return OpenAI(api_key=settings.qwen_api_key, base_url=settings.base_url)


def get_llm_model() -> str:
    """Return the configured LLM model name from settings."""
    return settings.qwen_model


def get_client() -> OpenAI:
    """Return the configured OpenAI-compatible client from common config."""
    return get_llm_client()


def _build_runtime_context(runtime_context: dict | None) -> dict:
    now_local = datetime.now().astimezone()
    now_utc = datetime.now(timezone.utc)
    merged = {
        "now_local": now_local.isoformat(timespec="seconds"),
        "timezone": str(now_local.tzinfo),
        "now_utc": now_utc.isoformat(timespec="seconds"),
    }
    if runtime_context:
        merged.update(runtime_context)
    return merged


def chat_completion(messages: list[dict], temperature: float = 0.2, runtime_context: dict | None = None) -> str:
    client = get_client()
    start = time.time()

    ctx = _build_runtime_context(runtime_context)
    context_message = {
        "role": "system",
        "content": (
            "RuntimeContext(JSON): "
            + str(ctx)
            + "\n请将当前时间作为事实来源参与推理，避免忽略时间导致建议偏差。"
        ),
    }
    enriched_messages = [context_message, *messages]

    response = client.chat.completions.create(
        model=get_llm_model(),
        messages=enriched_messages,
        temperature=temperature,
    )

    elapsed = time.time() - start
    logger.info(
        "LLM 调用完成，耗时 %.2f 秒, scenario=%s session_id=%s",
        elapsed,
        ctx.get("scenario", "unknown"),
        ctx.get("session_id", "unknown"),
    )

    return response.choices[0].message.content.strip()
