from __future__ import annotations

"""lifeops.memory_extractor

这个模块负责把“最近对话”提炼成结构化记忆候选。

定位说明：
- 它提取的是 memory_candidates（候选记忆），不是直接写入 todo。
- 这样做可以避免误提取直接污染主任务表，提升系统可控性。
"""

import json
import logging
from pydantic import BaseModel, Field, ValidationError

from .llm_qwen import chat_completion


logger = logging.getLogger(__name__)


class MemoryItem(BaseModel):
    """单条记忆候选的数据结构。"""

    # kind 约定可取：meeting|deadline|birthday|todo|event
    kind: str = Field(..., description="meeting|deadline|birthday|todo|event")
    title: str
    occurred_at: str | None = None
    notes: str | None = None


class MemoryExtraction(BaseModel):
    """提取结果容器：统一返回 items 列表。"""

    items: list[MemoryItem]


def extract_memory_from_dialogue(dialogue: list[dict]) -> MemoryExtraction | None:
    """从最近对话中抽取结构化记忆。

    参数：
    - dialogue: 聊天消息列表（通常是最近 user/assistant 若干轮）。

    返回：
    - 成功：`MemoryExtraction`
    - 失败：`None`（解析失败或 schema 不合法）

    失败返回 None 的原因：
    - 记忆抽取属于“增强能力”，不应因为抽取失败影响主聊天流程。
    """

    system_prompt = (
        "You extract structured memory items from recent dialogue. "
        "Return ONLY valid JSON following schema: {items:[{kind,title,occurred_at,notes}]}"
    )
    messages = [{"role": "system", "content": system_prompt}] + dialogue
    try:
        content = chat_completion(messages, temperature=0.1)
        data = json.loads(content)
        return MemoryExtraction.model_validate(data)
    except (json.JSONDecodeError, ValidationError) as exc:
        logger.warning("Memory extraction failed: %s", exc)
        return None
