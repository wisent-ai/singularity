"""Tests for DatabaseMigrationSkill."""
import json
import sqlite3
import pytest
from pathlib import Path
from unittest.mock import patch
from singularity.skills.database_migration import (
    DatabaseMigrationSkill, _load_state, _save_state, _get_schema,
    _diff_schemas, _split_sql, _compute_checksum, DATA_DIR, MIGRATION_FILE,
)


@pytest.fixture
def skill(tmp_path):
    with patch("singularity.skills.database_migration.DATA_DIR", tmp_path), \
         patch("singularity.skills.database_migration.MIGRATION_FILE", tmp_path / "mig.json"), \
         patch("singularity.skills.database_migration.DB_REGISTRY_FILE", tmp_path / "reg.json"):
        s = DatabaseMigrationSkill()
        s._test_tmp = tmp_path
        yield s


def _setup_db(tmp_path):
    """Create default test database at agent_data.db (matching _get_db_path('default'))."""
    db_path = tmp_path / "agent_data.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
    conn.commit()
    conn.close()
    return db_path


@pytest.mark.asyncio
async def test_create_migration(skill):
    result = await skill.execute("create", {
        "name": "add_users_email",
        "up_sql": "ALTER TABLE users ADD COLUMN email TEXT;",
        "down_sql": "-- SQLite cannot drop columns",
        "db": "default",
    })
    assert result.success
    assert result.data["name"] == "add_users_email"
    assert result.data["version"]
    assert result.data["checksum"]


@pytest.mark.asyncio
async def test_create_requires_name(skill):
    result = await skill.execute("create", {"up_sql": "SELECT 1"})
    assert not result.success
    assert "name" in result.message.lower()


@pytest.mark.asyncio
async def test_create_requires_up_sql(skill):
    result = await skill.execute("create", {"name": "test"})
    assert not result.success
    assert "up_sql" in result.message.lower()


@pytest.mark.asyncio
async def test_apply_migration(skill):
    tmp = skill._test_tmp
    db_path = _setup_db(tmp)
    await skill.execute("create", {
        "name": "add_email",
        "up_sql": "ALTER TABLE users ADD COLUMN email TEXT;",
        "db": "default",
    })
    result = await skill.execute("apply", {"db": "default"})
    assert result.success
    assert result.data["applied"] == 1
    conn = sqlite3.connect(str(db_path))
    cols = [c[1] for c in conn.execute("PRAGMA table_info(users)").fetchall()]
    conn.close()
    assert "email" in cols


@pytest.mark.asyncio
async def test_apply_multiple_migrations(skill):
    tmp = skill._test_tmp
    _setup_db(tmp)
    await skill.execute("create", {
        "name": "add_email", "up_sql": "ALTER TABLE users ADD COLUMN email TEXT;", "db": "default"
    })
    await skill.execute("create", {
        "name": "add_age", "up_sql": "ALTER TABLE users ADD COLUMN age INTEGER;", "db": "default"
    })
    result = await skill.execute("apply", {"db": "default"})
    assert result.success
    assert result.data["applied"] == 2


@pytest.mark.asyncio
async def test_apply_idempotent(skill):
    tmp = skill._test_tmp
    _setup_db(tmp)
    await skill.execute("create", {
        "name": "add_email", "up_sql": "ALTER TABLE users ADD COLUMN email TEXT;", "db": "default"
    })
    await skill.execute("apply", {"db": "default"})
    result = await skill.execute("apply", {"db": "default"})
    assert result.success
    assert result.data["applied"] == 0


@pytest.mark.asyncio
async def test_rollback(skill):
    tmp = skill._test_tmp
    _setup_db(tmp)
    await skill.execute("create", {
        "name": "add_posts",
        "up_sql": "CREATE TABLE posts (id INTEGER PRIMARY KEY, title TEXT);",
        "down_sql": "DROP TABLE IF EXISTS posts;",
        "db": "default",
    })
    await skill.execute("apply", {"db": "default"})
    result = await skill.execute("rollback", {"db": "default", "count": 1})
    assert result.success
    assert result.data["rolled_back"] == 1
    db_path = tmp / "agent_data.db"
    conn = sqlite3.connect(str(db_path))
    tables = [t[0] for t in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    conn.close()
    assert "posts" not in tables


@pytest.mark.asyncio
async def test_status(skill):
    tmp = skill._test_tmp
    _setup_db(tmp)
    await skill.execute("create", {
        "name": "m1", "up_sql": "ALTER TABLE users ADD COLUMN c1 TEXT;", "db": "default"
    })
    await skill.execute("create", {
        "name": "m2", "up_sql": "ALTER TABLE users ADD COLUMN c2 TEXT;", "db": "default"
    })
    await skill.execute("apply", {"db": "default"})
    await skill.execute("create", {
        "name": "m3", "up_sql": "ALTER TABLE users ADD COLUMN c3 TEXT;", "db": "default"
    })
    result = await skill.execute("status", {"db": "default"})
    assert result.success
    assert result.data["applied_count"] == 2
    assert result.data["pending_count"] == 1


@pytest.mark.asyncio
async def test_snapshot(skill):
    tmp = skill._test_tmp
    _setup_db(tmp)
    result = await skill.execute("snapshot", {"db": "default"})
    assert result.success
    assert "users" in result.data["tables"]
    assert result.data["table_count"] >= 1


@pytest.mark.asyncio
async def test_validate_clean(skill):
    result = await skill.execute("validate", {"db": "default"})
    assert result.success
    assert result.data["valid"]


@pytest.mark.asyncio
async def test_validate_detects_issues(skill):
    tmp = skill._test_tmp
    _setup_db(tmp)
    await skill.execute("create", {
        "name": "m1", "up_sql": "ALTER TABLE users ADD COLUMN c1 TEXT;", "db": "default"
    })
    await skill.execute("apply", {"db": "default"})
    result = await skill.execute("validate", {"db": "default"})
    assert result.success
    # Should flag missing down_sql
    assert any("rollback" in i.lower() or "without" in i.lower() for i in result.data["issues"])


@pytest.mark.asyncio
async def test_diff_schemas():
    old = {"users": [{"name": "id", "type": "INTEGER", "notnull": 0, "pk": 1, "default": None}]}
    new = {
        "users": [
            {"name": "id", "type": "INTEGER", "notnull": 0, "pk": 1, "default": None},
            {"name": "email", "type": "TEXT", "notnull": 0, "pk": 0, "default": None},
        ],
        "posts": [{"name": "id", "type": "INTEGER", "notnull": 0, "pk": 1, "default": None}],
    }
    stmts = _diff_schemas(old, new)
    assert any("ALTER TABLE" in s for s in stmts)
    assert any("CREATE TABLE posts" in s for s in stmts)


@pytest.mark.asyncio
async def test_history(skill):
    await skill.execute("create", {"name": "m1", "up_sql": "SELECT 1;", "db": "default"})
    result = await skill.execute("history", {"db": "default"})
    assert result.success
    assert len(result.data["history"]) >= 1


def test_split_sql():
    sql = "CREATE TABLE a (id INT);\nINSERT INTO a VALUES (1);"
    stmts = _split_sql(sql)
    assert len(stmts) == 2


def test_checksum_consistency():
    sql = "ALTER TABLE users ADD COLUMN email TEXT;"
    assert _compute_checksum(sql) == _compute_checksum(sql)
    assert _compute_checksum(sql) != _compute_checksum("SELECT 1;")
