"""Tests for NLDataQuerySkill - natural language to SQL query bridge."""

import json
import pytest
import asyncio
import sqlite3
from pathlib import Path

from singularity.skills.nl_data_query import (
    NLDataQuerySkill,
    NL_QUERY_FILE,
    DATA_DIR,
    SchemaInfo,
    _tokenize,
)


@pytest.fixture(autouse=True)
def clean_data(tmp_path, monkeypatch):
    monkeypatch.setattr("singularity.skills.nl_data_query.NL_QUERY_FILE", tmp_path / "nl_data_query.json")
    monkeypatch.setattr("singularity.skills.nl_data_query.DATA_DIR", tmp_path)
    yield tmp_path


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def _setup_test_db(tmp_path):
    """Create a test SQLite database with sample data."""
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE sales (id INTEGER PRIMARY KEY, region TEXT, amount REAL, product TEXT, sale_date TEXT)")
    conn.execute("INSERT INTO sales VALUES (1, 'North', 100.0, 'Widget', '2024-01-15')")
    conn.execute("INSERT INTO sales VALUES (2, 'South', 200.0, 'Gadget', '2024-01-16')")
    conn.execute("INSERT INTO sales VALUES (3, 'North', 150.0, 'Widget', '2024-02-01')")
    conn.execute("INSERT INTO sales VALUES (4, 'East', 300.0, 'Gizmo', '2024-02-15')")
    conn.execute("INSERT INTO sales VALUES (5, 'South', 250.0, 'Gadget', '2024-03-01')")
    conn.commit()
    conn.close()
    return db_path


class TestTokenize:
    def test_basic(self):
        tokens = _tokenize("show me total sales by region")
        assert "total" in tokens
        assert "sales" in tokens
        assert "region" in tokens
        assert "me" not in tokens  # stop word
        assert "show" not in tokens  # stop word

    def test_empty(self):
        assert _tokenize("") == []
        assert _tokenize("the a an") == []


class TestSchemaInfo:
    def test_find_table(self):
        schema = SchemaInfo()
        schema.add_table("sales", [{"name": "id", "type": "INTEGER"}, {"name": "region", "type": "TEXT"}])
        schema.add_table("users", [{"name": "id", "type": "INTEGER"}, {"name": "name", "type": "TEXT"}])
        assert schema.find_table(["sales"]) == "sales"
        assert schema.find_table(["users"]) == "users"
        assert schema.find_table(["nonexistent"]) is None

    def test_find_columns(self):
        schema = SchemaInfo()
        schema.add_table("sales", [
            {"name": "region", "type": "TEXT"},
            {"name": "amount", "type": "REAL"},
        ])
        cols = schema.find_columns("sales", ["region"])
        assert "region" in cols

    def test_numeric_columns(self):
        schema = SchemaInfo()
        schema.add_table("sales", [
            {"name": "id", "type": "INTEGER"},
            {"name": "amount", "type": "REAL"},
            {"name": "name", "type": "TEXT"},
        ])
        numeric = schema.get_numeric_columns("sales")
        assert "id" in numeric
        assert "amount" in numeric
        assert "name" not in numeric

    def test_text_columns(self):
        schema = SchemaInfo()
        schema.add_table("sales", [
            {"name": "id", "type": "INTEGER"},
            {"name": "region", "type": "TEXT"},
        ])
        text = schema.get_text_columns("sales")
        assert "region" in text
        assert "id" not in text


class TestSQLGeneration:
    def test_aggregate_query(self):
        skill = NLDataQuerySkill()
        schema = SchemaInfo()
        schema.add_table("sales", [
            {"name": "id", "type": "INTEGER"},
            {"name": "region", "type": "TEXT"},
            {"name": "amount", "type": "REAL"},
        ])
        result = skill._generate_sql("total amount by region in sales", schema)
        assert result is not None
        assert "SUM" in result["sql"]
        assert "GROUP BY" in result["sql"]
        assert "region" in result["sql"]

    def test_count_query(self):
        skill = NLDataQuerySkill()
        schema = SchemaInfo()
        schema.add_table("orders", [
            {"name": "id", "type": "INTEGER"},
            {"name": "status", "type": "TEXT"},
        ])
        result = skill._generate_sql("count orders by status", schema)
        assert result is not None
        assert "COUNT" in result["sql"]

    def test_select_query(self):
        skill = NLDataQuerySkill()
        schema = SchemaInfo()
        schema.add_table("users", [
            {"name": "id", "type": "INTEGER"},
            {"name": "name", "type": "TEXT"},
            {"name": "email", "type": "TEXT"},
        ])
        result = skill._generate_sql("show all users", schema)
        assert result is not None
        assert "SELECT" in result["sql"]
        assert "users" in result["sql"]

    def test_limit_detection(self):
        skill = NLDataQuerySkill()
        schema = SchemaInfo()
        schema.add_table("products", [
            {"name": "id", "type": "INTEGER"},
            {"name": "price", "type": "REAL"},
        ])
        result = skill._generate_sql("top 5 products by highest price", schema)
        assert result is not None
        assert "LIMIT 5" in result["sql"]
        assert "DESC" in result["sql"]

    def test_no_table_match(self):
        skill = NLDataQuerySkill()
        schema = SchemaInfo()
        schema.add_table("sales", [{"name": "id", "type": "INTEGER"}])
        result = skill._generate_sql("show me the weather", schema)
        assert result is None


class TestExecuteActions:
    def test_manifest(self):
        skill = NLDataQuerySkill()
        m = skill.manifest
        assert m.skill_id == "nl_data_query"
        assert m.category == "revenue"
        actions = [a.name for a in m.actions]
        assert "query" in actions
        assert "explain" in actions
        assert "discover" in actions
        assert "suggest" in actions
        assert "teach" in actions
        assert "stats" in actions

    def test_teach(self):
        skill = NLDataQuerySkill()
        result = run(skill.execute("teach", {
            "phrase": "revenue breakdown",
            "sql_template": "SELECT category, SUM(amount) FROM {table} GROUP BY category",
        }))
        assert result.success
        assert "revenue breakdown" in skill._store.get("learned_mappings", {})

    def test_teach_rejects_writes(self):
        skill = NLDataQuerySkill()
        result = run(skill.execute("teach", {
            "phrase": "delete everything",
            "sql_template": "DELETE FROM {table}",
        }))
        assert not result.success

    def test_stats_empty(self):
        skill = NLDataQuerySkill()
        result = run(skill.execute("stats", {}))
        assert result.success
        assert result.data["total_revenue"] == 0.0

    def test_unknown_action(self):
        skill = NLDataQuerySkill()
        result = run(skill.execute("nonexistent", {}))
        assert not result.success

    def test_query_missing_question(self):
        skill = NLDataQuerySkill()
        result = run(skill.execute("query", {"customer_id": "test"}))
        assert not result.success

    def test_customers_method(self):
        skill = NLDataQuerySkill()
        skill._store["revenue"] = {"total": 0.03, "query_count": 2, "by_customer": {"c1": 0.015, "c2": 0.015}}
        customers = skill._customers()
        assert len(customers) == 2
        assert customers[0]["source"] == "nl_data_query"

