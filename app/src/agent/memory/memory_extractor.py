from __future__ import annotations

import json
import logging
from datetime import datetime
from pydantic import ValidationError

from ...common.config.llm_config import chat_completion
from ...domain.dto.memory_dto import MemoryItem, MemoryExtraction

logger = logging.getLogger(__name__)


def extract_memory_from_dialogue(dialogue: list[dict]) -> MemoryExtraction | None:
    system_prompt = (
        "You extract structured memories from recent dialogue. "
        "Return ONLY valid JSON following schema: "
        '{items:[{kind,title,occurred_at,notes,confidence,insight_type}]}\n\n'
        "## Rules\n"
        "- kind: meeting|deadline|birthday|todo|event|preference|pattern\n"
        "- insight_type=explicit: user clearly stated the fact (confidence=1.0)\n"
        "- insight_type=inferred: you deduced a preference/pattern from implicit hints "
        "(confidence 0.3-0.9 based on how certain you are)\n"
        "- For explicit facts, always extract them.\n"
        "- For inferred insights like '雨天后取消运动→不喜欢雨天户外', "
        "set insight_type=inferred, confidence based on evidence strength.\n"
        "- If nothing to extract, return {items:[]}."
    )
    messages = [{"role": "system", "content": system_prompt}] + dialogue
    try:
        content = chat_completion(messages, temperature=0.1)
        data = json.loads(content)
        return MemoryExtraction.model_validate(data)
    except (json.JSONDecodeError, ValidationError) as exc:
        logger.warning("Memory extraction failed dialogue_len=%s err=%s", len(dialogue), exc)
        return None
