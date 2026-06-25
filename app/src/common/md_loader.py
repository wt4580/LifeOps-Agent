from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

MARKDOWN_FILES = [
    "AGENT_MEMO.md",
    "SKILLS.md",
    "USER_SUMMARY.md",
]

_TEMPLATES = {
    "AGENT_MEMO.md": "---\nupdated_at:\n---\n\n## 助手备忘录\n\n在这里记录助手需要记住的对话风格、偏好、注意事项。",
    "SKILLS.md": "---\nupdated_at:\n---\n\n## 技能库\n\n每一条技能 = 触发条件 → 执行动作。\n\n```yaml\n- trigger: 触发条件描述\n  action: 执行动作描述\n```",
    "USER_SUMMARY.md": "---\nupdated_at:\n---\n\n## 用户摘要\n\n助手维护的用户简要画像，方便快速回顾。",
}


def ensure_data_files() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for fname in MARKDOWN_FILES:
        path = DATA_DIR / fname
        if not path.exists():
            template = _TEMPLATES.get(fname, f"# {fname}\n")
            try:
                path.write_text(template.strip() + "\n", encoding="utf-8")
                logger.info("md_loader: 已创建 %s", fname)
            except OSError as exc:
                logger.warning("md_loader: 创建 %s 失败: %s", fname, exc)


ensure_data_files()


def _read_file(filename: str) -> Optional[str]:
    path = DATA_DIR / filename
    if not path.exists():
        return None
    try:
        content = path.read_text(encoding="utf-8").strip()
        return content or None
    except OSError as exc:
        logger.warning("md_loader: 读取 %s 失败: %s", filename, exc)
        return None


def load_markdown_context() -> str:
    parts: list[str] = []
    for fname in MARKDOWN_FILES:
        content = _read_file(fname)
        if content:
            parts.append(f"【{fname.replace('.md', '')}】\n{content}")
    if not parts:
        return ""
    return "\n\n".join(parts)
