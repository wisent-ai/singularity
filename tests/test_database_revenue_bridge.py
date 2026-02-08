"""Tests for DatabaseRevenueBridgeSkill."""
import pytest
import asyncio
import uuid
from singularity.skills.database_revenue_bridge import (
    DatabaseRevenueBridgeSkill, PRICING,
)


def _fresh_skill():
    s = DatabaseRevenueBridgeSkill()
    s._store = {"jobs": [], "reports": {}, "schemas_created": {},
                "revenue": {"total": 0.0, "by_service": {}, "by_customer": {}},
                "stats": {"total_requests": 0, "successful_requests": 0, "failed_requests": 0}}
    return s


@pytest.fixture
def skill():
    return _fresh_skill()


@pytest.fixture
def skill_with_data():
    """Skill with test data loaded via its own internal DatabaseSkill."""
    s = _fresh_skill()
    db = s._get_db_skill()
    dbname = f"test_{uuid.uuid4().hex[:8]}"
    db._create_db({"name": dbname})
    db._enable_write({"db": dbname, "enabled": True})
    db._execute_write({"sql": "CREATE TABLE users (id INTEGER, name TEXT, age INTEGER)", "db": dbname})
    db._execute_write({"sql": "INSERT INTO users VALUES (1, 'Alice', 30)", "db": dbname})
    db._execute_write({"sql": "INSERT INTO users VALUES (2, 'Bob', 25)", "db": dbname})
    db._execute_write({"sql": "INSERT INTO users VALUES (3, 'Charlie', 35)", "db": dbname})
    db._enable_write({"db": dbname, "enabled": False})
    s._test_db = dbname
    return s


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def test_manifest(skill):
    m = skill.manifest()
    assert m.skill_id == "database_revenue_bridge"
    assert len(m.actions) == 7
    action_names = [a.name for a in m.actions]
    assert "analyze" in action_names
    assert "design_schema" in action_names


def test_analyze_success(skill_with_data):
    db = skill_with_data._test_db
    result = run(skill_with_data.execute("analyze", {
        "customer_id": "cust1", "query": "SELECT * FROM users", "db_name": db
    }))
    assert result.success, f"Failed: {result.message}"
    assert result.revenue == PRICING["data_analysis"]
    assert skill_with_data._store["revenue"]["total"] == PRICING["data_analysis"]


def test_analyze_blocks_writes(skill_with_data):
    db = skill_with_data._test_db
    result = run(skill_with_data.execute("analyze", {
        "customer_id": "cust1", "query": "DROP TABLE users", "db_name": db
    }))
    assert not result.success
    assert "Only SELECT" in result.message


def test_analyze_missing_query(skill):
    result = run(skill.execute("analyze", {"customer_id": "cust1"}))
    assert not result.success


def test_design_schema(skill):
    tables = [
        {"name": "products", "columns": [{"name": "id", "type": "INTEGER"}, {"name": "name", "type": "TEXT"}]},
        {"name": "orders", "columns": [{"name": "id", "type": "INTEGER"}, {"name": "product_id", "type": "INTEGER"}]},
    ]
    dbname = f"schema_{uuid.uuid4().hex[:8]}"
    result = run(skill.execute("design_schema", {
        "customer_id": "cust2", "db_name": dbname, "tables": tables
    }))
    assert result.success, f"Failed: {result.message}"
    assert result.data["charged"] == PRICING["schema_design"] * 2


def test_import_data(skill):
    data = [
        {"name": "Widget A", "price": 9.99, "quantity": 100},
        {"name": "Widget B", "price": 19.99, "quantity": 50},
    ]
    dbname = f"import_{uuid.uuid4().hex[:8]}"
    result = run(skill.execute("import_data", {
        "customer_id": "cust3", "table_name": "widgets", "data": data, "db_name": dbname
    }))
    assert result.success, f"Failed: {result.message}"
    assert result.data["rows_imported"] == 2


def test_generate_report(skill_with_data):
    db = skill_with_data._test_db
    queries = [
        {"label": "All Users", "query": "SELECT * FROM users"},
        {"label": "Age Stats", "query": "SELECT AVG(age) as avg_age FROM users"},
    ]
    result = run(skill_with_data.execute("generate_report", {
        "customer_id": "cust4", "db_name": db, "title": "User Report", "queries": queries
    }))
    assert result.success, f"Failed: {result.message}"
    assert result.revenue == PRICING["report_generation"]


def test_transform_data(skill_with_data):
    db = skill_with_data._test_db
    result = run(skill_with_data.execute("transform_data", {
        "customer_id": "cust5", "db_name": db,
        "source_query": "SELECT name, age FROM users WHERE age >= 30",
        "target_table": "senior_users"
    }))
    assert result.success, f"Failed: {result.message}"
    assert result.data["rows_written"] == 2


def test_list_services(skill):
    result = run(skill.execute("list_services", {}))
    assert result.success
    assert len(result.data["services"]) == 5


def test_service_stats(skill_with_data):
    db = skill_with_data._test_db
    run(skill_with_data.execute("analyze", {
        "customer_id": "cust1", "query": "SELECT * FROM users", "db_name": db
    }))
    result = run(skill_with_data.execute("service_stats", {"customer_id": "cust1"}))
    assert result.success
    assert result.data["revenue"]["total"] > 0


def test_unknown_action(skill):
    result = run(skill.execute("nonexistent", {}))
    assert not result.success
    assert "Unknown action" in result.message
