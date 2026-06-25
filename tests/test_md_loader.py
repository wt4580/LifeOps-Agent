from pathlib import Path

import pytest

from app.src.common.md_loader import load_markdown_context, _read_file, DATA_DIR, MARKDOWN_FILES


class TestReadFile:
    def test_read_existing_file(self):
        content = _read_file("AGENT_MEMO.md")
        assert content is not None
        assert "助手备忘录" in content

    def test_read_nonexistent_file(self):
        content = _read_file("NONEXISTENT.md")
        assert content is None

    def test_read_all_expected_files_exist(self):
        for fname in MARKDOWN_FILES:
            assert (DATA_DIR / fname).exists(), f"{fname} not found"


class TestLoadMarkdownContext:
    def test_context_contains_all_files(self):
        ctx = load_markdown_context()
        assert "AGENT_MEMO" in ctx
        assert "SKILLS" in ctx
        assert "USER_SUMMARY" in ctx

    def test_context_is_string(self):
        ctx = load_markdown_context()
        assert isinstance(ctx, str)
