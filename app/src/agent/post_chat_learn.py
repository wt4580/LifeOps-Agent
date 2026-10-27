from __future__ import annotations

import json
import logging
import os
import threading
import hashlib
from datetime import datetime
from pathlib import Path

from ..common.config.llm_config import chat_completion
from ..common.config.base_config import settings
from ..common.harness.context import ensure_turn_active

logger = logging.getLogger(__name__)

DATA_DIR = Path(settings.personal_memory_dir)
_WRITE_LOCK = threading.Lock()

_LEARN_SYSTEM_PROMPT = (
    "你是个人助手的自我学习引擎。分析本轮对话，判断是否有值得改进或记住的信息。\n\n"
    "## 输出格式\n"
    "输出严格 JSON：\n"
    '{"has_update": false}\n'
    '或 {"has_update": true, "target": "AGENT_MEMO|SKILLS|USER_SUMMARY", "content": "要追加的内容"}'
)


def _write_markdown(target: str, content: str, session_id: str | None = None) -> bool:
    with _WRITE_LOCK:
        return _append_markdown(target, content, session_id)


def _append_markdown(target: str, content: str, session_id: str | None = None) -> bool:
    directory = DATA_DIR if session_id is None else DATA_DIR / "sessions" / hashlib.sha256(session_id.encode("utf-8")).hexdigest()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / target
    try:
        original = path.read_text(encoding="utf-8") if path.exists() else ""
    except OSError as exc:
        logger.warning("读取学习记录失败，保留原文件: %s", exc)
        return False

    ts = datetime.now().isoformat(timespec="seconds")
    lines = original.splitlines()
    metadata = []
    if lines and lines[0].strip() == "---":
        end = next((i for i in range(1, len(lines)) if lines[i].strip() == "---"), None)
        if end is not None:
            metadata = [line for line in lines[1:end] if not line.startswith("updated_at:")]
            original = "\n".join(lines[end + 1:])
    header = "\n".join(["---", *metadata, f"updated_at: {ts}", "---"])
    body = f"{header}\n\n{original.strip()}\n\n---\n{content}"

    try:
        temporary = path.with_suffix(".md.tmp")
        temporary.write_text(body.strip() + "\n", encoding="utf-8")
        ensure_turn_active()
        os.replace(temporary, path)
        logger.info("post_chat_learn: 已更新 %s", target)
        return True
    except OSError as exc:
        logger.warning("post_chat_learn: 写入 %s 失败: %s", target, exc)
        return False


def run_learn(
    user_message: str,
    assistant_answer: str,
    session_id: str | None = None,
) -> None:
    if not assistant_answer:
        return

    payload = {
        "user_message": user_message[:500],
        "assistant_answer": assistant_answer[:1000],
    }

    raw = chat_completion(
        [
            {"role": "system", "content": _LEARN_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        temperature=0.2,
        runtime_context={"scenario": "post_chat_learn", "session_id": session_id},
    )

    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        logger.info("post_chat_learn: LLM 返回非 JSON, 跳过: raw=%s", raw[:120])
        return

    if not data.get("has_update"):
        logger.info("post_chat_learn: 无更新, 跳过")
        return

    target = data.get("target", "")
    content = data.get("content", "")
    if target not in ("AGENT_MEMO", "SKILLS", "USER_SUMMARY"):
        logger.info("post_chat_learn: 未知 target=%s, 跳过", target)
        return
    if not content.strip():
        logger.info("post_chat_learn: 更新内容为空, 跳过")
        return

    logger.info("post_chat_learn: 产生更新 target=%s content=%s", target, content.strip().replace("\n", " ")[:120])

    _write_markdown(f"{target}.md", content.strip(), session_id=session_id)
