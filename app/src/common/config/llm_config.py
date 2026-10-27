from __future__ import annotations

import time
import json
import logging
import unicodedata
from collections.abc import Callable
from datetime import datetime, timezone
from functools import lru_cache

from openai import OpenAI

from .base_config import settings
from ..harness.context import budget_bypassed, get_turn_context, ensure_turn_active
from ..harness.interceptors import CallContext, llm_chain


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


def _usage_counts(usage) -> tuple[int, int] | None:
    if usage is None:
        return None
    def field(name):
        return usage.get(name) if isinstance(usage, dict) else getattr(usage, name, None)
    prompt, completion = field("prompt_tokens"), field("completion_tokens")
    if type(prompt) is not int or type(completion) is not int or min(prompt, completion) < 0:
        return None
    total = field("total_tokens")
    if type(total) is int:
        completion = max(completion, total - prompt)
    return prompt, completion


def chat_completion(
    messages: list[dict],
    temperature: float = 0.2,
    runtime_context: dict | None = None,
    on_token: Callable[[str], None] | None = None,
    max_tokens: int | None = None,
) -> str:
    start = time.monotonic()

    ctx = _build_runtime_context(runtime_context)
    context_message = {
        "role": "system",
        "content": (
            "## RuntimeContext\n"
            + str(ctx)
            + "\n\n请将当前时间作为事实来源参与推理，避免忽略时间导致建议偏差。"
        ),
    }
    enriched_messages = [context_message, *messages]

    session_id = ctx.get("session_id")
    turn = None if budget_bypassed() else get_turn_context(session_id)
    call_ctx = CallContext(
        kind="llm",
        name=str(ctx.get("scenario", "unknown")),
        session_id=session_id or "",
        turn=turn,
        payload={"temperature": temperature},
        usage={"usage_source": "not_sent", "prompt_tokens": None, "completion_tokens": None, "charged_tokens": 0},
    )
    finish_reason = "unknown"

    def request():
        nonlocal finish_reason
        output_limit = min(max_tokens if max_tokens is not None else settings.llm_max_output_tokens,
                           settings.llm_max_output_tokens)
        if output_limit <= 0:
            raise ValueError("max_tokens must be positive")
        reservation = 0
        if turn is not None:
            prompt_estimate = len(json.dumps(enriched_messages, ensure_ascii=False).encode("utf-8")) + 32 * len(enriched_messages) + 64
            output_limit, reservation = turn.budget.reserve_tokens(prompt_estimate, output_limit)
        usage = None
        attempted = False
        try:
            if settings.demo_mode:
                from ...demo.provider import DemoClient
                client = DemoClient(ctx)
            else:
                client = get_client()
            if turn is not None:
                client = client.with_options(max_retries=0)
            kwargs = dict(model=get_llm_model(), messages=enriched_messages, temperature=temperature,
                          max_tokens=output_limit,
                          timeout=max(0.001, min(30.0, turn.budget.remaining_seconds())) if turn else 30.0)
            ensure_turn_active()
            if on_token:
                if settings.llm_stream_include_usage:
                    kwargs["stream_options"] = {"include_usage": True}
                attempted = True
                response = client.chat.completions.create(**kwargs, stream=True)
                full_text = ""
                finish_reason = "stream"
                try:
                    for chunk in response:
                        usage = _usage_counts(getattr(chunk, "usage", None)) or usage
                        ensure_turn_active()
                        if not chunk.choices:
                            continue
                        choice = chunk.choices[0]
                        finish_reason = getattr(choice, "finish_reason", None) or finish_reason
                        delta = choice.delta.content or ""
                        full_text += delta
                        if delta:
                            on_token(delta)
                finally:
                    close = getattr(response, "close", None)
                    if close is not None:
                        try:
                            close()
                        except Exception:
                            logger.debug("model response cleanup failed")
                cleaned = normalize_llm_output(full_text)
            else:
                attempted = True
                response = client.chat.completions.create(**kwargs)
                usage = _usage_counts(getattr(response, "usage", None))
                finish_reason = response.choices[0].finish_reason
                cleaned = normalize_llm_output(response.choices[0].message.content)
            ensure_turn_active()
            return cleaned
        finally:
            if turn is not None:
                turn.budget.settle_tokens(reservation, usage, attempted=attempted)
            call_ctx.usage = {
                "usage_source": "reported" if usage is not None else ("reservation" if attempted else "not_sent"),
                "prompt_tokens": usage[0] if usage is not None else None,
                "completion_tokens": usage[1] if usage is not None else None,
                "charged_tokens": sum(usage) if usage is not None else (reservation if attempted else 0),
            }

    cleaned = llm_chain().invoke(call_ctx, request)
    elapsed = time.monotonic() - start
    preview = (cleaned[:300] + "...") if cleaned and len(cleaned) > 300 else (cleaned or "(empty)")
    if not cleaned:
        logger.warning(
            "LLM 返回仅含不可见字符 finish=%s scenario=%s",
            finish_reason, ctx.get("scenario", "unknown"),
        )
    logger.info(
        "LLM %s %.2fs %s preview=%s",
        ctx.get("scenario", "?"), elapsed, finish_reason, preview,
    )

    return cleaned
