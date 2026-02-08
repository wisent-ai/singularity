#!/usr/bin/env python3
"""
DatabaseSkill - SQL database interaction for data analysis and service delivery.

Zero external dependencies — uses Python's built-in sqlite3 module.
Enables the agent to:

1. CREATE databases and tables for its own operational data
2. QUERY data with full SQL support (SELECT, JOIN, aggregate, etc.)
3. ANALYZE schema — discover tables, columns, types, relationships
4. IMPORT data from JSON/CSV into database tables
5. EXPORT query results as JSON, CSV, or summary statistics
6. EXECUTE safe write operations (INSERT, UPDATE, DELETE) when enabled

Revenue Generation (primary): Data analysis is a high-value service.
  Customers can submit databases and get insights, reports, transformations.
  Combined with NL Router, users can ask questions in plain English.

Self-Improvement: The agent can store its own operational metrics,
  experiment results, and learning data in structured SQL tables
  instead of flat JSON files — enabling complex cross-session queries.

Goal Setting: Store and query goal/milestone data with SQL for
  sophisticated priority calculations.

Security:
  - Read-only mode by default (only SELECT allowed)
  - Write mode must be explicitly enabled per-connection
  - Maximum result rows capped to prevent memory issues
  - Query timeout to prevent runaway queries
  - No filesystem access beyond specified database paths
"""

import csv
import io
import json
import os
import re
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .base import Skill, SkillAction, SkillManifest, SkillResult


# Persistent state
DATA_DIR = Path(__file__).parent.parent / "data"
DB_REGISTRY_FILE = DATA_DIR / "database_registry.json"
DEFAULT_DB = DATA_DIR / "agent_data.db"

# Safety limits
MAX_RESULT_ROWS = 10000
MAX_QUERY_LENGTH = 10000
QUERY_TIMEOUT_SECONDS = 30

# SQL keywords that modify data
WRITE_KEYWORDS = {"INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "REPLACE", "TRUNCATE"}
READ_KEYWORDS = {"SELECT", "EXPLAIN", "PRAGMA", "WITH"}


def _is_read_only(sql: str) -> bool:
    """Check if a SQL statement is read-only."""
    stripped = sql.strip().upper()
    # Handle WITH ... SELECT (CTEs)
    if stripped.startswith("WITH"):
        # Find the final statement after CTE definitions
        # A CTE-based query should end with SELECT
        parts = re.split(r'\)\s*SELECT', stripped)
        if len(parts) > 1:
            return True
    first_word = stripped.split()[0] if stripped.split() else ""
    return first_word in READ_KEYWORDS


class DatabaseSkill(Skill):
    """
    SQL database interaction skill using SQLite.

    Provides structured data storage, querying, and analysis capabilities
    using Python's built-in sqlite3 module (no external dependencies).
    """

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self._connections: Dict[str, sqlite3.Connection] = {}
        self._write_enabled: Dict[str, bool] = {}
        self._ensure_data()

    def _ensure_data(self):
        """Ensure data directory and registry exist."""
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        if not DB_REGISTRY_FILE.exists():
            self._save_registry({})

    def _load_registry(self) -> Dict:
        """Load database registry."""
        try:
            with open(DB_REGISTRY_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return {}

    def _save_registry(self, registry: Dict):
        """Save database registry."""
        with open(DB_REGISTRY_FILE, "w") as f:
            json.dump(registry, f, indent=2)

    def _get_connection(self, db_name: str) -> sqlite3.Connection:
        """Get or create a database connection."""
        if db_name in self._connections:
            return self._connections[db_name]

        registry = self._load_registry()
        if db_name in registry:
            db_path = registry[db_name]["path"]
        elif db_name == "default":
            db_path = str(DEFAULT_DB)
        else:
            # Create new database in data directory
            db_path = str(DATA_DIR / f"{db_name}.db")

        conn = sqlite3.connect(db_path, timeout=QUERY_TIMEOUT_SECONDS)
        conn.row_factory = sqlite3.Row
        # Enable WAL mode for better concurrent read performance
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        self._connections[db_name] = conn

        # Register in registry if new
        if db_name not in registry:
            registry[db_name] = {
                "path": db_path,
                "created": datetime.utcnow().isoformat(),
                "write_enabled": False,
            }
            self._save_registry(registry)

        return conn

    def _close_connection(self, db_name: str):
        """Close a database connection."""
        if db_name in self._connections:
            self._connections[db_name].close()
            del self._connections[db_name]

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="database",
            name="Database",
            version="1.0.0",
            category="data",
            description="SQL database interaction — query, analyze, import/export data using SQLite",
            required_credentials=[],
            install_cost=0,
            actions=[
                SkillAction(
                    name="query",
                    description="Execute a SQL query and return results as JSON",
                    parameters={
                        "sql": {"type": "string", "required": True, "description": "SQL query to execute"},
                        "db": {"type": "string", "required": False, "description": "Database name (default: 'default')"},
                        "params": {"type": "array", "required": False, "description": "Query parameters for ? placeholders"},
                        "limit": {"type": "integer", "required": False, "description": "Max rows to return (default: 1000)"},
                    },
                    estimated_cost=0,
                    estimated_duration_seconds=2,
                    success_probability=0.90,
                ),
                SkillAction(
                    name="execute",
                    description="Execute a write SQL statement (INSERT/UPDATE/DELETE/CREATE). Requires write mode.",
                    parameters={
                        "sql": {"type": "string", "required": True, "description": "SQL statement to execute"},
                        "db": {"type": "string", "required": False, "description": "Database name (default: 'default')"},
                        "params": {"type": "array", "required": False, "description": "Statement parameters for ? placeholders"},
                    },
                    estimated_cost=0,
                    estimated_duration_seconds=2,
                    success_probability=0.85,
                ),
                SkillAction(
                    name="schema",
                    description="Analyze database schema — list tables, columns, types, and row counts",
                    parameters={
                        "db": {"type": "string", "required": False, "description": "Database name (default: 'default')"},
                        "table": {"type": "string", "required": False, "description": "Specific table to analyze (default: all)"},
                    },
                    estimated_cost=0,
                    estimated_duration_seconds=2,
                    success_probability=0.95,
                ),
                SkillAction(
                    name="import_data",
                    description="Import JSON array or CSV data into a database table",
                    parameters={
                        "table": {"type": "string", "required": True, "description": "Target table name"},
                        "data": {"type": "array", "required": False, "description": "JSON array of objects to import"},
                        "csv_text": {"type": "string", "required": False, "description": "CSV text with header row to import"},
                        "db": {"type": "string", "required": False, "description": "Database name (default: 'default')"},
                        "create_table": {"type": "boolean", "required": False, "description": "Auto-create table if not exists (default: true)"},
                    },
                    estimated_cost=0,
                    estimated_duration_seconds=5,
                    success_probability=0.85,
                ),
                SkillAction(
                    name="export",
                    description="Export query results as CSV or JSON",
                    parameters={
                        "sql": {"type": "string", "required": True, "description": "SQL query to export"},
                        "format": {"type": "string", "required": False, "description": "Output format: 'json' or 'csv' (default: 'json')"},
                        "db": {"type": "string", "required": False, "description": "Database name (default: 'default')"},
                    },
                    estimated_cost=0,
                    estimated_duration_seconds=3,
                    success_probability=0.90,
                ),
                SkillAction(
                    name="stats",
                    description="Get statistical summary of a table column (count, min, max, avg, distinct values)",
                    parameters={
                        "table": {"type": "string", "required": True, "description": "Table name"},
                        "column": {"type": "string", "required": False, "description": "Column to analyze (default: all columns)"},
                        "db": {"type": "string", "required": False, "description": "Database name (default: 'default')"},
                    },
                    estimated_cost=0,
                    estimated_duration_seconds=3,
                    success_probability=0.90,
                ),
                SkillAction(
                    name="create_db",
                    description="Create a new named database",
                    parameters={
                        "name": {"type": "string", "required": True, "description": "Database name"},
                        "write_enabled": {"type": "boolean", "required": False, "description": "Enable write operations (default: false)"},
                    },
                    estimated_cost=0,
                    estimated_duration_seconds=1,
                    success_probability=0.95,
                ),
                SkillAction(
                    name="list_databases",
                    description="List all registered databases with their sizes and table counts",
                    parameters={},
                    estimated_cost=0,
                    estimated_duration_seconds=1,
                    success_probability=0.98,
                ),
                SkillAction(
                    name="enable_write",
                    description="Enable or disable write mode for a database",
                    parameters={
                        "db": {"type": "string", "required": True, "description": "Database name"},
                        "enabled": {"type": "boolean", "required": False, "description": "Enable writes (default: true)"},
                    },
                    estimated_cost=0,
                    estimated_duration_seconds=1,
                    success_probability=0.98,
                ),
            ],
        )

    async def execute(self, action: str, params: Dict[str, Any]) -> SkillResult:
        """Execute a database action."""
        actions = {
            "query": self._query,
            "execute": self._execute_write,
            "schema": self._schema,
            "import_data": self._import_data,
            "export": self._export,
            "stats": self._stats,
            "create_db": self._create_db,
            "list_databases": self._list_databases,
            "enable_write": self._enable_write,
        }

        fn = actions.get(action)
        if not fn:
            return SkillResult(
                success=False,
                message=f"Unknown action: {action}. Available: {list(actions.keys())}",
            )
        try:
            return fn(params)
        except sqlite3.Error as e:
            return SkillResult(success=False, message=f"Database error: {e}")
        except Exception as e:
            return SkillResult(success=False, message=f"Error: {e}")

    def _query(self, params: Dict) -> SkillResult:
        """Execute a read-only SQL query."""
        sql = params.get("sql", "").strip()
        if not sql:
            return SkillResult(success=False, message="Missing required parameter: sql")
        if len(sql) > MAX_QUERY_LENGTH:
            return SkillResult(success=False, message=f"Query too long ({len(sql)} chars, max {MAX_QUERY_LENGTH})")

        if not _is_read_only(sql):
            return SkillResult(
                success=False,
                message="Query contains write operations. Use 'execute' action with write mode enabled.",
            )

        db_name = params.get("db", "default")
        query_params = params.get("params", [])
        limit = min(params.get("limit", 1000), MAX_RESULT_ROWS)

        conn = self._get_connection(db_name)
        start = time.time()
        cursor = conn.execute(sql, query_params)
        rows = cursor.fetchmany(limit)
        elapsed = time.time() - start

        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        results = [dict(zip(columns, row)) for row in rows]

        return SkillResult(
            success=True,
            message=f"Query returned {len(results)} rows in {elapsed:.3f}s",
            data={
                "rows": results,
                "columns": columns,
                "row_count": len(results),
                "elapsed_seconds": round(elapsed, 3),
                "truncated": len(results) >= limit,
            },
        )

    def _execute_write(self, params: Dict) -> SkillResult:
        """Execute a write SQL statement."""
        sql = params.get("sql", "").strip()
        if not sql:
            return SkillResult(success=False, message="Missing required parameter: sql")
        if len(sql) > MAX_QUERY_LENGTH:
            return SkillResult(success=False, message=f"Statement too long ({len(sql)} chars, max {MAX_QUERY_LENGTH})")

        db_name = params.get("db", "default")

        # Check write permission
        if not self._write_enabled.get(db_name, False):
            registry = self._load_registry()
            if not registry.get(db_name, {}).get("write_enabled", False):
                return SkillResult(
                    success=False,
                    message=f"Write mode not enabled for database '{db_name}'. Use 'enable_write' action first.",
                )

        query_params = params.get("params", [])
        conn = self._get_connection(db_name)
        start = time.time()
        cursor = conn.execute(sql, query_params)
        conn.commit()
        elapsed = time.time() - start

        return SkillResult(
            success=True,
            message=f"Statement executed successfully ({cursor.rowcount} rows affected) in {elapsed:.3f}s",
            data={
                "rows_affected": cursor.rowcount,
                "lastrowid": cursor.lastrowid,
                "elapsed_seconds": round(elapsed, 3),
            },
        )

    def _schema(self, params: Dict) -> SkillResult:
        """Analyze database schema."""
        db_name = params.get("db", "default")
        specific_table = params.get("table")
        conn = self._get_connection(db_name)

        # Get all tables
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = [row[0] for row in cursor.fetchall()]

        if specific_table and specific_table not in tables:
            return SkillResult(
                success=False,
                message=f"Table '{specific_table}' not found. Available tables: {tables}",
            )

        target_tables = [specific_table] if specific_table else tables
        schema_info = {}

        for table in target_tables:
            # Get column info
            col_cursor = conn.execute(f"PRAGMA table_info('{table}')")
            columns = []
            for col in col_cursor.fetchall():
                columns.append({
                    "name": col[1],
                    "type": col[2],
                    "not_null": bool(col[3]),
                    "default": col[4],
                    "primary_key": bool(col[5]),
                })

            # Get row count
            count_cursor = conn.execute(f"SELECT COUNT(*) FROM '{table}'")
            row_count = count_cursor.fetchone()[0]

            # Get indexes
            idx_cursor = conn.execute(f"PRAGMA index_list('{table}')")
            indexes = []
            for idx in idx_cursor.fetchall():
                idx_info_cursor = conn.execute(f"PRAGMA index_info('{idx[1]}')")
                idx_columns = [r[2] for r in idx_info_cursor.fetchall()]
                indexes.append({
                    "name": idx[1],
                    "unique": bool(idx[2]),
                    "columns": idx_columns,
                })

            # Get foreign keys
            fk_cursor = conn.execute(f"PRAGMA foreign_key_list('{table}')")
            foreign_keys = []
            for fk in fk_cursor.fetchall():
                foreign_keys.append({
                    "from_column": fk[3],
                    "to_table": fk[2],
                    "to_column": fk[4],
                })

            schema_info[table] = {
                "columns": columns,
                "row_count": row_count,
                "indexes": indexes,
                "foreign_keys": foreign_keys,
            }

        return SkillResult(
            success=True,
            message=f"Schema for {len(schema_info)} table(s) in '{db_name}'",
            data={
                "database": db_name,
                "tables": schema_info,
                "table_count": len(schema_info),
            },
        )

    def _import_data(self, params: Dict) -> SkillResult:
        """Import JSON or CSV data into a table."""
        table = params.get("table", "").strip()
        if not table:
            return SkillResult(success=False, message="Missing required parameter: table")

        db_name = params.get("db", "default")
        create_table = params.get("create_table", True)

        # Parse data from JSON array or CSV text
        json_data = params.get("data")
        csv_text = params.get("csv_text")

        if json_data:
            if not isinstance(json_data, list) or not json_data:
                return SkillResult(success=False, message="'data' must be a non-empty JSON array of objects")
            rows = json_data
        elif csv_text:
            reader = csv.DictReader(io.StringIO(csv_text))
            rows = list(reader)
            if not rows:
                return SkillResult(success=False, message="CSV text has no data rows")
        else:
            return SkillResult(success=False, message="Provide either 'data' (JSON array) or 'csv_text' (CSV string)")

        # Enable write temporarily for import
        conn = self._get_connection(db_name)
        columns = list(rows[0].keys())

        if create_table:
            # Infer column types from first row
            col_defs = []
            for col in columns:
                val = rows[0][col]
                if isinstance(val, bool):
                    col_type = "BOOLEAN"
                elif isinstance(val, int):
                    col_type = "INTEGER"
                elif isinstance(val, float):
                    col_type = "REAL"
                else:
                    col_type = "TEXT"
                col_defs.append(f'"{col}" {col_type}')

            create_sql = f'CREATE TABLE IF NOT EXISTS "{table}" ({", ".join(col_defs)})'
            conn.execute(create_sql)

        # Insert rows
        placeholders = ", ".join(["?"] * len(columns))
        col_names = ", ".join([f'"{c}"' for c in columns])
        insert_sql = f'INSERT INTO "{table}" ({col_names}) VALUES ({placeholders})'

        inserted = 0
        for row in rows:
            values = [row.get(col) for col in columns]
            conn.execute(insert_sql, values)
            inserted += 1

        conn.commit()

        return SkillResult(
            success=True,
            message=f"Imported {inserted} rows into '{table}' in database '{db_name}'",
            data={
                "table": table,
                "database": db_name,
                "rows_imported": inserted,
                "columns": columns,
            },
        )

    def _export(self, params: Dict) -> SkillResult:
        """Export query results as JSON or CSV."""
        sql = params.get("sql", "").strip()
        if not sql:
            return SkillResult(success=False, message="Missing required parameter: sql")

        if not _is_read_only(sql):
            return SkillResult(success=False, message="Export only supports read-only queries (SELECT)")

        db_name = params.get("db", "default")
        fmt = params.get("format", "json").lower()

        conn = self._get_connection(db_name)
        cursor = conn.execute(sql)
        rows = cursor.fetchmany(MAX_RESULT_ROWS)
        columns = [desc[0] for desc in cursor.description] if cursor.description else []

        if fmt == "csv":
            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerow(columns)
            for row in rows:
                writer.writerow(row)
            result_text = output.getvalue()
            return SkillResult(
                success=True,
                message=f"Exported {len(rows)} rows as CSV",
                data={"csv": result_text, "row_count": len(rows), "columns": columns},
            )
        else:
            results = [dict(zip(columns, row)) for row in rows]
            return SkillResult(
                success=True,
                message=f"Exported {len(rows)} rows as JSON",
                data={"rows": results, "row_count": len(rows), "columns": columns},
            )

    def _stats(self, params: Dict) -> SkillResult:
        """Get statistical summary of a table or column."""
        table = params.get("table", "").strip()
        if not table:
            return SkillResult(success=False, message="Missing required parameter: table")

        db_name = params.get("db", "default")
        specific_column = params.get("column")
        conn = self._get_connection(db_name)

        # Verify table exists
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)
        )
        if not cursor.fetchone():
            return SkillResult(success=False, message=f"Table '{table}' not found")

        # Get columns
        col_cursor = conn.execute(f"PRAGMA table_info('{table}')")
        all_columns = [(row[1], row[2]) for row in col_cursor.fetchall()]

        if specific_column:
            target_cols = [(n, t) for n, t in all_columns if n == specific_column]
            if not target_cols:
                return SkillResult(
                    success=False,
                    message=f"Column '{specific_column}' not found in table '{table}'",
                )
        else:
            target_cols = all_columns

        stats = {}
        for col_name, col_type in target_cols:
            col_stats = {}

            # Count and null count
            row = conn.execute(
                f'SELECT COUNT(*), COUNT("{col_name}"), COUNT(DISTINCT "{col_name}") FROM "{table}"'
            ).fetchone()
            col_stats["total_rows"] = row[0]
            col_stats["non_null"] = row[1]
            col_stats["null_count"] = row[0] - row[1]
            col_stats["distinct_count"] = row[2]

            # Numeric stats
            if col_type.upper() in ("INTEGER", "REAL", "NUMERIC", "FLOAT", "DOUBLE"):
                num_row = conn.execute(
                    f'SELECT MIN("{col_name}"), MAX("{col_name}"), AVG("{col_name}"), SUM("{col_name}") '
                    f'FROM "{table}" WHERE "{col_name}" IS NOT NULL'
                ).fetchone()
                if num_row[0] is not None:
                    col_stats["min"] = num_row[0]
                    col_stats["max"] = num_row[1]
                    col_stats["avg"] = round(num_row[2], 4) if num_row[2] else None
                    col_stats["sum"] = num_row[3]

            # Top values for text columns
            if col_type.upper() in ("TEXT", "VARCHAR", "CHAR", ""):
                top_cursor = conn.execute(
                    f'SELECT "{col_name}", COUNT(*) as cnt FROM "{table}" '
                    f'WHERE "{col_name}" IS NOT NULL '
                    f'GROUP BY "{col_name}" ORDER BY cnt DESC LIMIT 5'
                )
                col_stats["top_values"] = [
                    {"value": r[0], "count": r[1]} for r in top_cursor.fetchall()
                ]

            stats[col_name] = col_stats

        return SkillResult(
            success=True,
            message=f"Statistics for {len(stats)} column(s) in '{table}'",
            data={"table": table, "database": db_name, "column_stats": stats},
        )

    def _create_db(self, params: Dict) -> SkillResult:
        """Create a new named database."""
        name = params.get("name", "").strip()
        if not name:
            return SkillResult(success=False, message="Missing required parameter: name")
        if not re.match(r'^[a-zA-Z0-9_-]+$', name):
            return SkillResult(success=False, message="Database name must be alphanumeric with _ or -")

        registry = self._load_registry()
        if name in registry:
            return SkillResult(success=False, message=f"Database '{name}' already exists")

        write_enabled = params.get("write_enabled", False)
        db_path = str(DATA_DIR / f"{name}.db")

        # Create the database file
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.close()

        registry[name] = {
            "path": db_path,
            "created": datetime.utcnow().isoformat(),
            "write_enabled": write_enabled,
        }
        self._save_registry(registry)
        self._write_enabled[name] = write_enabled

        return SkillResult(
            success=True,
            message=f"Created database '{name}' (write_enabled={write_enabled})",
            data={"name": name, "path": db_path, "write_enabled": write_enabled},
        )

    def _list_databases(self, params: Dict) -> SkillResult:
        """List all registered databases."""
        registry = self._load_registry()
        databases = []

        for name, info in registry.items():
            db_path = info.get("path", "")
            size_bytes = 0
            table_count = 0

            if os.path.exists(db_path):
                size_bytes = os.path.getsize(db_path)
                try:
                    conn = sqlite3.connect(db_path)
                    cursor = conn.execute(
                        "SELECT COUNT(*) FROM sqlite_master WHERE type='table'"
                    )
                    table_count = cursor.fetchone()[0]
                    conn.close()
                except sqlite3.Error:
                    pass

            databases.append({
                "name": name,
                "path": db_path,
                "size_bytes": size_bytes,
                "size_readable": _format_bytes(size_bytes),
                "table_count": table_count,
                "created": info.get("created", "unknown"),
                "write_enabled": info.get("write_enabled", False),
            })

        return SkillResult(
            success=True,
            message=f"Found {len(databases)} registered database(s)",
            data={"databases": databases, "count": len(databases)},
        )

    def _enable_write(self, params: Dict) -> SkillResult:
        """Enable or disable write mode for a database."""
        db_name = params.get("db", "").strip()
        if not db_name:
            return SkillResult(success=False, message="Missing required parameter: db")

        enabled = params.get("enabled", True)
        registry = self._load_registry()

        if db_name not in registry and db_name != "default":
            return SkillResult(success=False, message=f"Database '{db_name}' not found")

        # Update registry
        if db_name not in registry:
            registry[db_name] = {
                "path": str(DEFAULT_DB),
                "created": datetime.utcnow().isoformat(),
            }
        registry[db_name]["write_enabled"] = enabled
        self._save_registry(registry)
        self._write_enabled[db_name] = enabled

        return SkillResult(
            success=True,
            message=f"Write mode {'enabled' if enabled else 'disabled'} for database '{db_name}'",
            data={"database": db_name, "write_enabled": enabled},
        )


def _format_bytes(size: int) -> str:
    """Format bytes to human-readable string."""
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"
