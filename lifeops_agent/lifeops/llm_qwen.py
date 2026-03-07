from __future__ import annotations

"""lifeops.llm_qwen

这个模块封装了“调用通义千问（DashScope OpenAI 兼容接口）”的最小能力。

为什么要单独封装：
- 业务代码（planner/router/rag）不需要关心底层 SDK 细节。
- 以后如果更换模型供应商，只需在这里改一层适配。
"""

import time
import logging
from datetime import datetime, timezone
from openai import OpenAI

from .settings import settings


# 模块级日志器：统一记录模型调用耗时与异常排查信息。
logger = logging.getLogger(__name__)


def get_client() -> OpenAI:
    """创建 OpenAI 兼容客户端实例。

    配置来源：
    - api_key: 来自 .env 的 QWEN_API_KEY
    - base_url: DashScope OpenAI 兼容端点
    """

    return OpenAI(api_key=settings.qwen_api_key, base_url=settings.base_url)


def _build_runtime_context(runtime_context: dict | None) -> dict:
    """统一构建运行时上下文，确保每次调用都带有当前时间信息。"""

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
    """执行一次聊天补全调用并返回纯文本结果。

    参数：
    - messages: OpenAI chat 格式消息列表。
    - temperature: 采样温度；越低越稳定，越高越发散。

    返回：
    - 助手回复文本（去除首尾空白）。

    额外行为：
    - 记录调用耗时，用于观察模型链路延迟。
    """

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
        model=settings.qwen_model,
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
