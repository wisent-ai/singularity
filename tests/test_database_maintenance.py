"""Tests for DatabaseMaintenanceSkill."""
import json
import sqlite3
import pytest
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import patch
from singularity.skills.database_maintenance import (
    DatabaseMaintenanceSkill, _get_all_databases, DATA_DIR, MAINTENANCE_FILE,
    DEFAULT_RETENTION, _load_state, _save_state,
)


@pytest.fixture
def skill(tmp_path):
    with patch("singularity.skills.database_maintenance.DATA_DIR", tmp_path), \
         patch("singularity.skills.database_maintenance.MAINTENANCE_FILE", tmp_path / "maint.json"), \
         patch("singularity.skills.database_maintenance.DB_REGISTRY_FILE", tmp_path / "reg.json"):
        s = DatabaseMaintenanceSkill()
        # Create a test database
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE logs (id INTEGER PRIMARY KEY, message TEXT, created_at TEXT)")
        conn.execute("CREATE TABLE metrics (id INTEGER PRIMARY KEY, value REAL, timestamp TEXT)")
        for i in range(50):
            ts = (datetime.utcnow() - timedelta(days=i * 10)).isoformat()
            conn.execute("INSERT INTO logs (message, created_at) VALUES (?, ?)", (f"log_{i}", ts))
            conn.execute("INSERT INTO metrics (value, timestamp) VALUES (?, ?)", (float(i), ts))
        conn.commit()
        conn.close()
        s._test_db_path = db_path
        s._test_tmp = tmp_path
        yield s


def _dbs(skill):
    return [{"name": "test", "path": str(skill._test_db_path)}]


@pytest.mark.asyncio
async def test_vacuum(skill):
    with patch("singularity.skills.database_maintenance._get_all_databases", return_value=_dbs(skill)):
        result = await skill.execute("vacuum", {})
    assert result.success
    assert result.data["operation"] == "vacuum"
    assert result.data["databases_processed"] == 1
    assert result.data["results"][0]["success"]


@pytest.mark.asyncio
async def test_analyze(skill):
    with patch("singularity.skills.database_maintenance._get_all_databases", return_value=_dbs(skill)):
        result = await skill.execute("analyze", {})
    assert result.success
    assert result.data["results"][0]["tables_analyzed"] == 2


@pytest.mark.asyncio
async def test_integrity_check(skill):
    with patch("singularity.skills.database_maintenance._get_all_databases", return_value=_dbs(skill)):
        result = await skill.execute("integrity_check", {})
    assert result.success
    assert result.data["all_healthy"]
    assert result.data["results"][0]["healthy"]


@pytest.mark.asyncio
async def test_cleanup_with_retention(skill):
    with patch("singularity.skills.database_maintenance._get_all_databases", return_value=_dbs(skill)):
        result = await skill.execute("cleanup", {"retention_days": 60})
    assert result.success
    assert result.data["total_rows_deleted"] > 0


@pytest.mark.asyncio
async def test_optimize_indexes_suggest(skill):
    # Add enough rows to trigger suggestions (need >= 100)
    conn = sqlite3.connect(str(skill._test_db_path))
    for i in range(60):
        ts = datetime.utcnow().isoformat()
        conn.execute("INSERT INTO logs (message, created_at) VALUES (?, ?)", (f"extra_{i}", ts))
    conn.commit()
    conn.close()
    with patch("singularity.skills.database_maintenance._get_all_databases", return_value=_dbs(skill)):
        result = await skill.execute("optimize_indexes", {"auto_create": False})
    assert result.success
    assert result.data["total_suggestions"] > 0


@pytest.mark.asyncio
async def test_optimize_indexes_auto_create(skill):
    conn = sqlite3.connect(str(skill._test_db_path))
    for i in range(60):
        conn.execute("INSERT INTO logs (message, created_at) VALUES (?, ?)", (f"x_{i}", datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()
    with patch("singularity.skills.database_maintenance._get_all_databases", return_value=_dbs(skill)):
        result = await skill.execute("optimize_indexes", {"auto_create": True})
    assert result.success
    assert result.data["total_indexes_created"] > 0


@pytest.mark.asyncio
async def test_health_report(skill):
    with patch("singularity.skills.database_maintenance._get_all_databases", return_value=_dbs(skill)):
        result = await skill.execute("health", {})
    assert result.success
    report = result.data["reports"][0]
    assert report["healthy"]
    assert report["table_count"] == 2
    assert report["total_rows"] == 100


@pytest.mark.asyncio
async def test_schedule_single(skill):
    result = await skill.execute("schedule", {"operation": "vacuum", "interval_hours": 6})
    assert result.success
    assert result.data["schedules_created"] == 1
    sched = result.data["schedules"][0]
    assert sched["interval_hours"] == 6.0


@pytest.mark.asyncio
async def test_schedule_full(skill):
    result = await skill.execute("schedule", {"operation": "full"})
    assert result.success
    assert result.data["schedules_created"] == 4


@pytest.mark.asyncio
async def test_history(skill):
    with patch("singularity.skills.database_maintenance._get_all_databases", return_value=_dbs(skill)):
        await skill.execute("vacuum", {})
        await skill.execute("analyze", {})
    result = await skill.execute("history", {"limit": 10})
    assert result.success
    assert result.data["total_entries"] >= 2


@pytest.mark.asyncio
async def test_set_retention(skill):
    result = await skill.execute("set_retention", {"category": "logs", "retention_days": 7})
    assert result.success
    assert result.data["retention_days"] == 7
    assert result.data["all_policies"]["logs"] == 7.0


@pytest.mark.asyncio
async def test_unknown_action(skill):
    result = await skill.execute("bogus", {})
    assert not result.success


@pytest.mark.asyncio
async def test_manifest(skill):
    m = skill.manifest()
    assert m.skill_id == "database_maintenance"
    assert len(m.actions) == 9
