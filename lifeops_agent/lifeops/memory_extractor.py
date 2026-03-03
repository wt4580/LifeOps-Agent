from __future__ import annotations

import json
import logging
from pydantic import BaseModel, Field, ValidationError

from .llm_qwen import chat_completion


logger = logging.getLogger(__name__)


class MemoryItem(BaseModel):
    kind: str = Field(..., description="meeting|deadline|birthday|todo|event")
    title: str
    occurred_at: str | None = None
    notes: str | None = None


class MemoryExtraction(BaseModel):
    items: list[MemoryItem]


def extract_memory_from_dialogue(dialogue: list[dict]) -> MemoryExtraction | None:
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

