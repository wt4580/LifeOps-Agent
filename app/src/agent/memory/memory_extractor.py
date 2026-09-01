from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pydantic import ValidationError

from ...common.config.llm_config import chat_completion
from ...domain.dto.memory_dto import MemoryItem, MemoryExtraction

logger = logging.getLogger(__name__)


def _extract_json_object(content: str) -> dict | None:
    """从 LLM 输出中稳健解析 JSON 对象。

    优先整体解析；失败时查找第一个 { ... } 平衡片段再解析（容忍模型在 JSON 前后夹带散文/代码块）。
    """
    if not content or not content.strip():
        return None
    try:
        obj = json.loads(content)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    # 去掉首尾 ```json ... ``` 包裹
    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", stripped).strip()

    depth = 0
    start = -1
    for i, ch in enumerate(stripped):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                candidate = stripped[start : i + 1]
                try:
                    obj = json.loads(candidate)
                    if isinstance(obj, dict):
                        return obj
                except json.JSONDecodeError:
                    pass
    return None


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
        content = chat_completion(messages, temperature=0.1, runtime_context={"scenario": "memory_extraction"})
        data = _extract_json_object(content)
        if data is None:
            raise json.JSONDecodeError("no JSON object found", content, 0)
        return MemoryExtraction.model_validate(data)
    except (json.JSONDecodeError, ValidationError) as exc:
        logger.warning("Memory extraction failed dialogue_len=%s err=%s", len(dialogue), exc)
        return None
