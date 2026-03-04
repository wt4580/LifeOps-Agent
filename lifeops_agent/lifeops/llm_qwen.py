from __future__ import annotations

import time
import logging
from openai import OpenAI

from .settings import settings

# 初始化一个日志记录器，用于记录与 LLM 调用相关的信息
logger = logging.getLogger(__name__)

# 创建并返回一个 OpenAI 客户端实例的函数
def get_client() -> OpenAI:
    # 使用设置中的 API 密钥和基础 URL 初始化 OpenAI 客户端
    return OpenAI(api_key=settings.qwen_api_key, base_url=settings.base_url)

# 与 Qwen LLM 交互并获取聊天补全响应的函数
def chat_completion(messages: list[dict], temperature: float = 0.2) -> str:
    """
    将一组消息发送到 Qwen LLM，并检索助手的响应。

    参数：
        messages (list[dict]): 消息字典列表，每个字典表示对话中的一条消息（例如，角色和内容）。
        temperature (float): LLM 的采样温度。较低的值使输出更确定性，较高的值引入更多随机性。

    返回：
        str: 助手响应的内容。
    """
    # 创建一个 OpenAI 客户端实例
    client = get_client()

    # 记录开始时间以进行性能日志记录
    start = time.time()

    # 将消息发送到 LLM 并请求补全
    response = client.chat.completions.create(
        model=settings.qwen_model,  # 使用的模型，在设置中指定
        messages=messages,          # 对话历史记录
        temperature=temperature,    # 采样温度
    )

    # 计算 LLM 调用的耗时
    elapsed = time.time() - start

    # 记录 LLM 调用所花费的时间
    logger.info("LLM 调用完成，耗时 %.2f 秒", elapsed)

    # 返回响应中第一个选项的内容
    return response.choices[0].message.content.strip()
