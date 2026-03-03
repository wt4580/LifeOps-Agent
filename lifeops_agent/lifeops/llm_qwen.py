from __future__ import annotations

import time
import logging
from openai import OpenAI

from .settings import settings


logger = logging.getLogger(__name__)


def get_client() -> OpenAI:
    return OpenAI(api_key=settings.qwen_api_key, base_url=settings.base_url)


def chat_completion(messages: list[dict], temperature: float = 0.2) -> str:
    client = get_client()
    start = time.time()
    response = client.chat.completions.create(
        model=settings.qwen_model,
        messages=messages,
        temperature=temperature,
    )
    elapsed = time.time() - start
    logger.info("LLM call finished in %.2fs", elapsed)
    return response.choices[0].message.content.strip()

