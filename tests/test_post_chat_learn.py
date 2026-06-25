import json
from pathlib import Path
from unittest.mock import patch

import pytest

from app.src.agent.post_chat_learn import _write_markdown, run_learn

DATA_DIR = Path(__file__).resolve().parent.parent / "app" / "src" / "data"


class TestWriteMarkdown:
    def test_write_updates_frontmatter(self):
        path = DATA_DIR / "AGENT_MEMO.md"
        original = path.read_text(encoding="utf-8")

        result = _write_markdown("AGENT_MEMO.md", "新追加内容")
        assert result is True

        updated = path.read_text(encoding="utf-8")
        assert "updated_at" in updated
        assert "新追加内容" in updated

        path.write_text(original, encoding="utf-8")

    def test_write_nonexistent_target(self):
        target = "NONEXISTENT_TEST.md"
        result = _write_markdown(target, "内容")
        assert result is True
        # cleanup
        (DATA_DIR / target).unlink(missing_ok=True)


class TestRunLearn:
    @patch("app.src.agent.post_chat_learn.chat_completion")
    def test_skip_on_empty_answer(self, mock_llm):
        run_learn(user_message="hi", assistant_answer="", session_id="s1")
        mock_llm.assert_not_called()

    @patch("app.src.agent.post_chat_learn.chat_completion")
    def test_llm_returns_no_update(self, mock_llm):
        mock_llm.return_value = json.dumps({"has_update": False})
        run_learn(user_message="hi", assistant_answer="hello", session_id="s1")
        mock_llm.assert_called_once()

    @patch("app.src.agent.post_chat_learn.chat_completion")
    def test_llm_returns_update(self, mock_llm):
        path = DATA_DIR / "AGENT_MEMO.md"
        original = path.read_text(encoding="utf-8")

        mock_llm.return_value = json.dumps({
            "has_update": True,
            "target": "AGENT_MEMO",
            "content": "用户喜欢简洁回答",
        })
        run_learn(user_message="hi", assistant_answer="hello", session_id="s1")

        content = path.read_text(encoding="utf-8")
        assert "用户喜欢简洁回答" in content

        path.write_text(original, encoding="utf-8")

    @patch("app.src.agent.post_chat_learn.chat_completion")
    def test_llm_bad_json(self, mock_llm):
        mock_llm.return_value = "不是 JSON"
        run_learn(user_message="hi", assistant_answer="hello", session_id="s1")
        mock_llm.assert_called_once()
