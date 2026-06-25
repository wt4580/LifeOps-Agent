from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from ..common.config.llm_config import chat_completion

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

_LEARN_SYSTEM_PROMPT = (
    "你是个人助手的自我学习引擎。分析本轮对话，判断是否有值得改进或记住的信息。\n\n"
    "## 输出格式\n"
    "输出严格 JSON：\n"
    '{"has_update": false}\n'
    '或 {"has_update": true, "target": "AGENT_MEMO|SKILLS|USER_SUMMARY", "content": "要追加的内容"}'
)


def _write_markdown(target: str, content: str) -> bool:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / target
    try:
        original = path.read_text(encoding="utf-8") if path.exists() else ""
    except OSError:
        original = ""

    ts = datetime.now().isoformat(timespec="seconds")
    if "---" in original:
        lines = original.splitlines()
        updated: list[str] = []
        frontmatter_done = False
        for line in lines:
            if line.strip() == "---" and not frontmatter_done:
                updated.append("---")
                updated.append(f"updated_at: {ts}")
                frontmatter_done = True
            elif frontmatter_done and not line.startswith("---"):
                continue
            elif line.strip().startswith("updated_at:"):
                continue
            elif line.strip() == "---" and frontmatter_done:
                break
            else:
                updated.append(line)
        body = "\n".join(updated) + f"\n\n---\n{content}"
    else:
        body = f"---\nupdated_at: {ts}\n---\n\n{original}\n\n---\n{content}"

    try:
        path.write_text(body.strip() + "\n", encoding="utf-8")
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
        logger.debug("post_chat_learn: LLM 返回非 JSON, 跳过: raw=%s", raw[:80])
        return

    if not data.get("has_update"):
        return

    target = data.get("target", "")
    content = data.get("content", "")
    if target not in ("AGENT_MEMO", "SKILLS", "USER_SUMMARY"):
        logger.debug("post_chat_learn: 未知 target=%s, 跳过", target)
        return
    if not content.strip():
        return

    _write_markdown(f"{target}.md", content.strip())
