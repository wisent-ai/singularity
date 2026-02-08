"""Tests for DatabaseSkill — SQL database interaction."""

import json
import os
import pytest
import asyncio
import tempfile
from pathlib import Path
from unittest.mock import patch

from singularity.skills.database import DatabaseSkill, _is_read_only, _format_bytes


@pytest.fixture
def skill(tmp_path):
    """Create a DatabaseSkill with temp data directory."""
    with patch("singularity.skills.database.DATA_DIR", tmp_path), \
         patch("singularity.skills.database.DB_REGISTRY_FILE", tmp_path / "database_registry.json"), \
         patch("singularity.skills.database.DEFAULT_DB", tmp_path / "agent_data.db"):
        s = DatabaseSkill()
        yield s
        # Cleanup connections
        for conn in s._connections.values():
            try:
                conn.close()
            except Exception:
                pass


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class TestReadOnlyCheck:
    def test_select_is_readonly(self):
        assert _is_read_only("SELECT * FROM users") is True

    def test_insert_is_not_readonly(self):
        assert _is_read_only("INSERT INTO users VALUES (1)") is False

    def test_pragma_is_readonly(self):
        assert _is_read_only("PRAGMA table_info('users')") is True

    def test_with_select_is_readonly(self):
        assert _is_read_only("WITH cte AS (SELECT 1) SELECT * FROM cte") is True

    def test_drop_is_not_readonly(self):
        assert _is_read_only("DROP TABLE users") is False

    def test_explain_is_readonly(self):
        assert _is_read_only("EXPLAIN SELECT 1") is True


class TestFormatBytes:
    def test_bytes(self):
        assert "B" in _format_bytes(500)

    def test_kilobytes(self):
        assert "KB" in _format_bytes(5000)

    def test_megabytes(self):
        assert "MB" in _format_bytes(5_000_000)


class TestCreateDb:
    def test_create_database(self, skill):
        r = run(skill.execute("create_db", {"name": "testdb", "write_enabled": True}))
        assert r.success is True
        assert r.data["name"] == "testdb"
        assert r.data["write_enabled"] is True

    def test_duplicate_name_fails(self, skill):
        run(skill.execute("create_db", {"name": "dup"}))
        r = run(skill.execute("create_db", {"name": "dup"}))
        assert r.success is False
        assert "already exists" in r.message

    def test_invalid_name_fails(self, skill):
        r = run(skill.execute("create_db", {"name": "bad name!"}))
        assert r.success is False


class TestListDatabases:
    def test_empty_list(self, skill):
        r = run(skill.execute("list_databases", {}))
        assert r.success is True
        assert r.data["count"] == 0

    def test_list_after_create(self, skill):
        run(skill.execute("create_db", {"name": "mydb"}))
        r = run(skill.execute("list_databases", {}))
        assert r.success is True
        assert r.data["count"] == 1
        assert r.data["databases"][0]["name"] == "mydb"


class TestQuery:
    def test_simple_select(self, skill):
        r = run(skill.execute("query", {"sql": "SELECT 1 as val, 'hello' as msg"}))
        assert r.success is True
        assert r.data["rows"] == [{"val": 1, "msg": "hello"}]
        assert r.data["row_count"] == 1

    def test_missing_sql_fails(self, skill):
        r = run(skill.execute("query", {}))
        assert r.success is False

    def test_write_in_query_fails(self, skill):
        r = run(skill.execute("query", {"sql": "DROP TABLE foo"}))
        assert r.success is False
        assert "write operations" in r.message


class TestExecuteWrite:
    def test_write_without_permission_fails(self, skill):
        r = run(skill.execute("execute", {"sql": "CREATE TABLE t (id INTEGER)"}))
        assert r.success is False
        assert "Write mode not enabled" in r.message

    def test_write_with_permission(self, skill):
        run(skill.execute("create_db", {"name": "wdb", "write_enabled": True}))
        r = run(skill.execute("execute", {
            "sql": "CREATE TABLE t (id INTEGER, name TEXT)",
            "db": "wdb",
        }))
        assert r.success is True


class TestImportAndSchema:
    def test_import_json_and_query(self, skill):
        data = [
            {"name": "Alice", "age": 30, "city": "NYC"},
            {"name": "Bob", "age": 25, "city": "LA"},
            {"name": "Carol", "age": 35, "city": "NYC"},
        ]
        r = run(skill.execute("import_data", {"table": "people", "data": data}))
        assert r.success is True
        assert r.data["rows_imported"] == 3

        # Query the imported data
        q = run(skill.execute("query", {"sql": "SELECT * FROM people ORDER BY age", "db": "default"}))
        assert q.success is True
        assert q.data["row_count"] == 3
        assert q.data["rows"][0]["name"] == "Bob"

    def test_import_csv(self, skill):
        csv_text = "name,score\nAlice,95\nBob,87\n"
        r = run(skill.execute("import_data", {"table": "scores", "csv_text": csv_text}))
        assert r.success is True
        assert r.data["rows_imported"] == 2

    def test_schema_analysis(self, skill):
        data = [{"x": 1, "y": "hello"}]
        run(skill.execute("import_data", {"table": "test_tbl", "data": data}))
        r = run(skill.execute("schema", {}))
        assert r.success is True
        assert "test_tbl" in r.data["tables"]
        cols = r.data["tables"]["test_tbl"]["columns"]
        assert len(cols) == 2


class TestStats:
    def test_column_statistics(self, skill):
        data = [{"val": i, "cat": "a" if i % 2 == 0 else "b"} for i in range(10)]
        run(skill.execute("import_data", {"table": "nums", "data": data}))
        r = run(skill.execute("stats", {"table": "nums", "column": "val"}))
        assert r.success is True
        stats = r.data["column_stats"]["val"]
        assert stats["total_rows"] == 10
        assert stats["min"] == 0
        assert stats["max"] == 9

    def test_stats_missing_table(self, skill):
        r = run(skill.execute("stats", {"table": "nonexistent"}))
        assert r.success is False


class TestExport:
    def test_export_json(self, skill):
        data = [{"a": 1}, {"a": 2}]
        run(skill.execute("import_data", {"table": "exp", "data": data}))
        r = run(skill.execute("export", {"sql": "SELECT * FROM exp", "format": "json"}))
        assert r.success is True
        assert r.data["row_count"] == 2

    def test_export_csv(self, skill):
        data = [{"x": 1, "y": 2}]
        run(skill.execute("import_data", {"table": "csv_exp", "data": data}))
        r = run(skill.execute("export", {"sql": "SELECT * FROM csv_exp", "format": "csv"}))
        assert r.success is True
        assert "x,y" in r.data["csv"]


class TestEnableWrite:
    def test_enable_and_write(self, skill):
        run(skill.execute("create_db", {"name": "rw"}))
        # Write should fail
        r1 = run(skill.execute("execute", {"sql": "CREATE TABLE t (id INT)", "db": "rw"}))
        assert r1.success is False
        # Enable write
        run(skill.execute("enable_write", {"db": "rw", "enabled": True}))
        # Now write should succeed
        r2 = run(skill.execute("execute", {"sql": "CREATE TABLE t (id INT)", "db": "rw"}))
        assert r2.success is True


class TestManifest:
    def test_manifest_valid(self, skill):
        m = skill.manifest
        assert m.skill_id == "database"
        assert len(m.actions) == 9
        assert m.category == "data"

    def test_unknown_action(self, skill):
        r = run(skill.execute("nonexistent", {}))
        assert r.success is False
