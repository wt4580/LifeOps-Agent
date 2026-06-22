from __future__ import annotations

import time
import logging
import unicodedata
from datetime import datetime, timezone
from functools import lru_cache

from openai import OpenAI

from .base_config import settings


logger = logging.getLogger(__name__)

# 抑制 httpx 的 HTTP 请求/响应日志（太嘈杂）
logging.getLogger("httpx").setLevel(logging.WARNING)


def normalize_llm_output(text: str | None) -> str:
    """LLM 输出标准化工具。

    作用：统一格式化 LLM 原始返回内容，确保下游可靠解析。
    处理：
    - None → 空字符串
    - 去除首尾空白
    - 去除 markdown 代码块包装（```json ... ``` / ``` ... ```）
    - 移除 Unicode 格式控制字符（零宽空格 U+200B、BOM U+FEFF 等 Cf 类）
    """
    if text is None:
        return ""
    text = text.strip()
    # 去除 markdown 代码块包装 ```json ... ``` 或 ``` ... ```
    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1:]
        text = text.strip()
        if text.endswith("```"):
            text = text[:-3].rstrip()
        text = text.strip()
    # 移除 Unicode Cf 类（格式控制字符）
    text = "".join(c for c in text if unicodedata.category(c) != "Cf")
    return text.strip()


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
    finish_reason = response.choices[0].finish_reason
    raw = response.choices[0].message.content
    cleaned = normalize_llm_output(raw)
    preview = (cleaned[:120] + "...") if cleaned and len(cleaned) > 120 else (cleaned or "(empty)")
    if not cleaned and raw:
        logger.warning(
            "LLM 返回仅含不可见字符 finish=%s scenario=%s len=%d",
            finish_reason, ctx.get("scenario", "unknown"), len(raw) if raw else 0,
        )
    logger.info(
        "LLM %s %.2fs %s preview=%s",
        ctx.get("scenario", "?"), elapsed, finish_reason, preview,
    )

    return cleaned
