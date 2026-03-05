from __future__ import annotations

"""lifeops.llm_qwen

这个模块封装了“调用通义千问（DashScope OpenAI 兼容接口）”的最小能力。

为什么要单独封装：
- 业务代码（planner/router/rag）不需要关心底层 SDK 细节。
- 以后如果更换模型供应商，只需在这里改一层适配。
"""

import time
import logging
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


def chat_completion(messages: list[dict], temperature: float = 0.2) -> str:
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

    response = client.chat.completions.create(
        model=settings.qwen_model,
        messages=messages,
        temperature=temperature,
    )

    elapsed = time.time() - start
    logger.info("LLM 调用完成，耗时 %.2f 秒", elapsed)

    # choices[0] 是主回复；MVP 场景默认取第一候选即可。
    return response.choices[0].message.content.strip()
