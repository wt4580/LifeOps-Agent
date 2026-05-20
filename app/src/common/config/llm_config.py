from functools import lru_cache
from openai import OpenAI

from .base_config import settings


@lru_cache()
def get_llm_client() -> OpenAI:
    """Return a cached OpenAI-compatible client configured from settings.

    Centralizing client creation here keeps provider wiring in `common.config`.
    """
    return OpenAI(api_key=settings.qwen_api_key, base_url=settings.base_url)
