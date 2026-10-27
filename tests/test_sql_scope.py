from __future__ import annotations

from app.src.common.harness.sql_scope import check_session_scope, extract_tables


class TestExtractTables:
    def test_from_and_join(self):
        sql = "SELECT * FROM todo_items t JOIN calendar_events c ON c.id = t.id"
        assert extract_tables(sql) == {"todo_items", "calendar_events"}

    def test_schema_and_quotes(self):
        sql = 'SELECT * FROM "main"."todo_items"'
        assert extract_tables(sql) == {"todo_items"}


class TestSessionScope:
    def test_scoped_table_without_filter_is_blocked(self):
        reason = check_session_scope("SELECT * FROM todo_items", "s-1")
        assert reason is not None
        assert "session_id" in reason

    def test_scoped_table_with_filter_is_allowed(self):
        assert check_session_scope("SELECT * FROM todo_items WHERE session_id = 's-1'", "s-1") is None

    def test_alias_filter_is_allowed(self):
        sql = "SELECT * FROM todo_items t WHERE t.session_id = 's-1'"
        assert check_session_scope(sql, "s-1") is None

    def test_in_filter_is_allowed(self):
        sql = "SELECT * FROM chat_messages WHERE session_id IN ('s-1')"
        assert check_session_scope(sql, "s-1") is None

    def test_reversed_filter_is_allowed(self):
        sql = "SELECT * FROM accounts WHERE 's-1' = session_id"
        assert check_session_scope(sql, "s-1") is None

    def test_join_with_filter_is_allowed(self):
        sql = (
            "SELECT * FROM todo_items t JOIN calendar_events c ON c.session_id = t.session_id "
            "WHERE t.session_id = 's-1'"
        )
        assert check_session_scope(sql, "s-1") is None

    def test_other_session_is_blocked(self):
        sql = "SELECT * FROM todo_items WHERE session_id = 's-2'"
        assert check_session_scope(sql, "s-1") is not None

    def test_filter_only_in_comment_is_blocked(self):
        sql = "SELECT * FROM todo_items -- session_id = 's-1'"
        assert check_session_scope(sql, "s-1") is not None

    def test_global_table_is_allowed(self):
        assert check_session_scope("SELECT * FROM documents", "s-1") is None
        assert check_session_scope("SELECT * FROM memory_candidates", "s-1") is None

    def test_pragma_and_explain_are_allowed(self):
        assert check_session_scope("PRAGMA table_info(todo_items)", "s-1") is None
        assert check_session_scope("EXPLAIN SELECT * FROM todo_items", "s-1") is None

    def test_indirect_table_requires_parent(self):
        sql = "SELECT * FROM topic_episode_messages"
        assert check_session_scope(sql, "s-1") is not None

    def test_indirect_table_with_parent_is_allowed(self):
        sql = (
            "SELECT m.* FROM topic_episode_messages m "
            "JOIN topic_episodes e ON e.id = m.episode_id "
            "WHERE e.session_id = 's-1'"
        )
        assert check_session_scope(sql, "s-1") is None

    def test_empty_sql_is_allowed(self):
        assert check_session_scope("", "s-1") is None
