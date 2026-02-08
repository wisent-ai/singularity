#!/usr/bin/env python3
"""
DatabaseMigrationSkill - Schema versioning and migration management.

Bridges DatabaseSkill + DatabaseMaintenanceSkill to provide:

  1. Create Migration: Generate versioned migration files (up/down SQL)
  2. Apply Migrations: Run pending migrations in order with transaction safety
  3. Rollback: Revert last N migrations using down scripts
  4. Status: Show which migrations are applied vs pending
  5. History: Full migration audit trail with timing and results
  6. Diff: Compare current schema against target and generate migration
  7. Snapshot: Capture current schema state for comparison
  8. Validate: Check migration chain integrity (no gaps, no conflicts)

Revenue: Customers pay for managed database evolution - schema changes
  are high-risk operations that benefit from automation and safety.

Self-Improvement: The agent can evolve its own database schemas as
  capabilities grow - add new tables, columns, indexes without manual
  intervention. Combined with SchedulerSkill, migrations can run
  automatically when schema changes are detected.

Pillar: Self-Improvement (primary), Revenue (supporting)
"""

import json
import hashlib
import os
import re
import sqlite3
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .base import Skill, SkillAction, SkillManifest, SkillResult

DATA_DIR = Path(__file__).parent.parent / "data"
MIGRATION_FILE = DATA_DIR / "database_migrations.json"
DB_REGISTRY_FILE = DATA_DIR / "database_registry.json"
MAX_HISTORY = 500


def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _load_state() -> Dict:
    try:
        if MIGRATION_FILE.exists():
            with open(MIGRATION_FILE, "r") as f:
                return json.load(f)
    except (json.JSONDecodeError, IOError):
        pass
    return {
        "migrations": {},     # db_name -> [migration records]
        "snapshots": {},      # db_name -> [schema snapshots]
        "history": [],        # global operation log
        "stats": {
            "total_migrations_created": 0,
            "total_migrations_applied": 0,
            "total_rollbacks": 0,
            "total_failures": 0,
        },
    }


def _save_state(state: Dict):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if len(state.get("history", [])) > MAX_HISTORY:
        state["history"] = state["history"][-MAX_HISTORY:]
    try:
        with open(MIGRATION_FILE, "w") as f:
            json.dump(state, f, indent=2, default=str)
    except IOError:
        pass


def _get_db_path(db_name: str) -> Path:
    """Resolve database name to file path."""
    if db_name == "default" or not db_name:
        return DATA_DIR / "agent_data.db"
    # Check registry
    try:
        if DB_REGISTRY_FILE.exists():
            with open(DB_REGISTRY_FILE, "r") as f:
                registry = json.load(f)
            for entry in registry.get("databases", []):
                if entry.get("name") == db_name:
                    return Path(entry["path"])
    except (json.JSONDecodeError, IOError):
        pass
    # Assume it's in data dir
    return DATA_DIR / f"{db_name}.db"


def _get_schema(db_path: Path) -> Dict[str, List[Dict]]:
    """Get current schema of a database as {table: [columns]}."""
    if not db_path.exists():
        return {}
    schema = {}
    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
        for t in tables:
            tname = t["name"]
            cols = conn.execute(f"PRAGMA table_info('{tname}')").fetchall()
            schema[tname] = [
                {"name": c["name"], "type": c["type"], "notnull": c["notnull"],
                 "pk": c["pk"], "default": c["dflt_value"]}
                for c in cols
            ]
            # Include indexes
            idxs = conn.execute(f"PRAGMA index_list('{tname}')").fetchall()
            for idx in idxs:
                idx_info = conn.execute(f"PRAGMA index_info('{idx['name']}')").fetchall()
                schema[f"__index__{tname}__{idx['name']}"] = [
                    {"column": ii["name"], "seqno": ii["seqno"]}
                    for ii in idx_info
                ]
        conn.close()
    except sqlite3.Error:
        pass
    return schema


def _ensure_migration_table(db_path: Path):
    """Create the _migrations tracking table if it doesn't exist."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS _migrations (
            version TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            applied_at TEXT NOT NULL,
            checksum TEXT,
            execution_ms REAL,
            status TEXT DEFAULT 'applied'
        )
    """)
    conn.commit()
    conn.close()


def _get_applied_versions(db_path: Path) -> List[str]:
    """Get list of applied migration versions from the database."""
    _ensure_migration_table(db_path)
    conn = sqlite3.connect(str(db_path))
    rows = conn.execute(
        "SELECT version FROM _migrations WHERE status='applied' ORDER BY version"
    ).fetchall()
    conn.close()
    return [r[0] for r in rows]


def _compute_checksum(sql: str) -> str:
    """Compute checksum of SQL for integrity verification."""
    return hashlib.sha256(sql.strip().encode()).hexdigest()[:16]


def _generate_version() -> str:
    """Generate a version string based on timestamp."""
    return datetime.utcnow().strftime("%Y%m%d%H%M%S") + uuid.uuid4().hex[:4]


def _diff_schemas(old_schema: Dict, new_schema: Dict) -> List[str]:
    """Generate SQL statements to migrate from old_schema to new_schema."""
    statements = []
    old_tables = {k for k in old_schema if not k.startswith("__index__")}
    new_tables = {k for k in new_schema if not k.startswith("__index__")}

    # New tables
    for table in sorted(new_tables - old_tables):
        cols = new_schema[table]
        col_defs = []
        for c in cols:
            d = f"{c['name']} {c['type']}"
            if c.get("pk"):
                d += " PRIMARY KEY"
            if c.get("notnull"):
                d += " NOT NULL"
            if c.get("default") is not None:
                d += f" DEFAULT {c['default']}"
            col_defs.append(d)
        statements.append(f"CREATE TABLE {table} ({', '.join(col_defs)});")

    # Dropped tables
    for table in sorted(old_tables - new_tables):
        statements.append(f"DROP TABLE IF EXISTS {table};")

    # Modified tables - add new columns (SQLite can't drop columns easily)
    for table in sorted(old_tables & new_tables):
        old_cols = {c["name"] for c in old_schema[table]}
        new_cols = {c["name"] for c in new_schema[table]}
        for col_name in sorted(new_cols - old_cols):
            col = next(c for c in new_schema[table] if c["name"] == col_name)
            d = f"ALTER TABLE {table} ADD COLUMN {col['name']} {col['type']}"
            if col.get("default") is not None:
                d += f" DEFAULT {col['default']}"
            statements.append(d + ";")

    return statements


class DatabaseMigrationSkill(Skill):
    """Schema versioning and migration management for databases."""

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="database_migration",
            name="Database Migration",
            version="1.0.0",
            category="infrastructure",
            description="Schema versioning and migration management for databases",
            actions=self._get_actions(),
            required_credentials=[],
        )

    def _get_actions(self) -> List[SkillAction]:
        return [
            SkillAction(
                name="create",
                description="Create a new migration with up/down SQL",
                parameters={
                    "name": {"type": "string", "required": True, "description": "Migration name (e.g. add_users_table)"},
                    "up_sql": {"type": "string", "required": True, "description": "SQL to apply migration"},
                    "down_sql": {"type": "string", "required": False, "description": "SQL to revert migration"},
                    "db": {"type": "string", "required": False, "description": "Database name (default: default)"},
                },
            ),
            SkillAction(
                name="apply",
                description="Apply pending migrations to a database",
                parameters={
                    "db": {"type": "string", "required": False, "description": "Database name"},
                    "target_version": {"type": "string", "required": False, "description": "Apply up to this version"},
                },
            ),
            SkillAction(
                name="rollback",
                description="Rollback last N migrations",
                parameters={
                    "db": {"type": "string", "required": False, "description": "Database name"},
                    "count": {"type": "integer", "required": False, "description": "Number to rollback (default: 1)"},
                },
            ),
            SkillAction(
                name="status",
                description="Show migration status - applied vs pending",
                parameters={
                    "db": {"type": "string", "required": False, "description": "Database name"},
                },
            ),
            SkillAction(
                name="history",
                description="View migration operation audit trail",
                parameters={
                    "db": {"type": "string", "required": False, "description": "Filter by database"},
                    "limit": {"type": "integer", "required": False, "description": "Max entries (default: 20)"},
                },
            ),
            SkillAction(
                name="diff",
                description="Compare current schema against target and generate migration",
                parameters={
                    "db": {"type": "string", "required": False, "description": "Database name"},
                    "target_schema": {"type": "object", "required": True, "description": "Target schema {table: [columns]}"},
                },
            ),
            SkillAction(
                name="snapshot",
                description="Capture current schema state for comparison",
                parameters={
                    "db": {"type": "string", "required": False, "description": "Database name"},
                },
            ),
            SkillAction(
                name="validate",
                description="Check migration chain integrity",
                parameters={
                    "db": {"type": "string", "required": False, "description": "Database name"},
                },
            ),
        ]

    async def execute(self, action: str, params: Dict[str, Any]) -> SkillResult:
        handlers = {
            "create": self._create,
            "apply": self._apply,
            "rollback": self._rollback,
            "status": self._status,
            "history": self._history,
            "diff": self._diff,
            "snapshot": self._snapshot,
            "validate": self._validate,
        }
        handler = handlers.get(action)
        if not handler:
            return SkillResult(success=False, message=f"Unknown action: {action}")
        try:
            return await handler(params)
        except Exception as e:
            return SkillResult(success=False, message=f"Migration error: {e}")

    async def _create(self, params: Dict) -> SkillResult:
        """Create a new migration record."""
        name = params.get("name", "").strip()
        up_sql = params.get("up_sql", "").strip()
        down_sql = params.get("down_sql", "").strip()
        db_name = params.get("db", "default")

        if not name:
            return SkillResult(success=False, message="Migration name is required")
        if not up_sql:
            return SkillResult(success=False, message="up_sql is required")
        # Sanitize name
        name = re.sub(r'[^a-zA-Z0-9_]', '_', name).lower()

        version = _generate_version()
        checksum = _compute_checksum(up_sql)

        state = _load_state()
        if db_name not in state["migrations"]:
            state["migrations"][db_name] = []

        migration = {
            "version": version,
            "name": name,
            "up_sql": up_sql,
            "down_sql": down_sql,
            "checksum": checksum,
            "created_at": _now_iso(),
            "status": "pending",
        }
        state["migrations"][db_name].append(migration)
        state["stats"]["total_migrations_created"] += 1
        state["history"].append({
            "op": "create",
            "db": db_name,
            "version": version,
            "name": name,
            "timestamp": _now_iso(),
        })
        _save_state(state)

        return SkillResult(
            success=True,
            message=f"Migration {version}_{name} created for {db_name}",
            data={"version": version, "name": name, "db": db_name, "checksum": checksum},
        )

    async def _apply(self, params: Dict) -> SkillResult:
        """Apply pending migrations to a database."""
        db_name = params.get("db", "default")
        target_version = params.get("target_version")
        db_path = _get_db_path(db_name)

        state = _load_state()
        migrations = state.get("migrations", {}).get(db_name, [])
        if not migrations:
            return SkillResult(success=True, message=f"No migrations defined for {db_name}", data={"applied": 0})

        # Get already applied
        _ensure_migration_table(db_path)
        applied = set(_get_applied_versions(db_path))

        # Find pending in order
        pending = [m for m in migrations if m["version"] not in applied and m["status"] != "rolled_back"]
        pending.sort(key=lambda m: m["version"])

        if target_version:
            pending = [m for m in pending if m["version"] <= target_version]

        if not pending:
            return SkillResult(success=True, message="All migrations already applied", data={"applied": 0})

        results = []
        applied_count = 0
        for mig in pending:
            start = time.time()
            try:
                conn = sqlite3.connect(str(db_path))
                conn.execute("BEGIN")
                # Execute each statement in the up_sql
                for stmt in _split_sql(mig["up_sql"]):
                    if stmt.strip():
                        conn.execute(stmt)
                elapsed_ms = (time.time() - start) * 1000
                # Record in _migrations table
                conn.execute(
                    "INSERT INTO _migrations (version, name, applied_at, checksum, execution_ms, status) VALUES (?, ?, ?, ?, ?, ?)",
                    (mig["version"], mig["name"], _now_iso(), mig["checksum"], elapsed_ms, "applied"),
                )
                conn.commit()
                conn.close()
                mig["status"] = "applied"
                mig["applied_at"] = _now_iso()
                applied_count += 1
                results.append({"version": mig["version"], "name": mig["name"], "success": True, "ms": round(elapsed_ms, 2)})
            except sqlite3.Error as e:
                elapsed_ms = (time.time() - start) * 1000
                try:
                    conn.rollback()
                    conn.close()
                except Exception:
                    pass
                state["stats"]["total_failures"] += 1
                results.append({"version": mig["version"], "name": mig["name"], "success": False, "error": str(e)})
                # Stop on first failure
                break

        state["stats"]["total_migrations_applied"] += applied_count
        state["history"].append({
            "op": "apply",
            "db": db_name,
            "applied": applied_count,
            "total_pending": len(pending),
            "timestamp": _now_iso(),
        })
        _save_state(state)

        all_ok = all(r["success"] for r in results)
        return SkillResult(
            success=all_ok,
            message=f"Applied {applied_count}/{len(pending)} migrations to {db_name}",
            data={"applied": applied_count, "results": results, "db": db_name},
        )

    async def _rollback(self, params: Dict) -> SkillResult:
        """Rollback last N migrations."""
        db_name = params.get("db", "default")
        count = int(params.get("count", 1))
        db_path = _get_db_path(db_name)

        state = _load_state()
        migrations = state.get("migrations", {}).get(db_name, [])
        if not migrations:
            return SkillResult(success=False, message=f"No migrations for {db_name}")

        # Find applied migrations in reverse order
        applied = [m for m in migrations if m.get("status") == "applied"]
        applied.sort(key=lambda m: m["version"], reverse=True)
        to_rollback = applied[:count]

        if not to_rollback:
            return SkillResult(success=True, message="No migrations to rollback", data={"rolled_back": 0})

        results = []
        rolled_back_count = 0
        for mig in to_rollback:
            if not mig.get("down_sql"):
                results.append({"version": mig["version"], "name": mig["name"], "success": False, "error": "No down_sql defined"})
                continue
            try:
                conn = sqlite3.connect(str(db_path))
                conn.execute("BEGIN")
                for stmt in _split_sql(mig["down_sql"]):
                    if stmt.strip():
                        conn.execute(stmt)
                conn.execute("DELETE FROM _migrations WHERE version = ?", (mig["version"],))
                conn.commit()
                conn.close()
                mig["status"] = "rolled_back"
                mig["rolled_back_at"] = _now_iso()
                rolled_back_count += 1
                results.append({"version": mig["version"], "name": mig["name"], "success": True})
            except sqlite3.Error as e:
                try:
                    conn.rollback()
                    conn.close()
                except Exception:
                    pass
                results.append({"version": mig["version"], "name": mig["name"], "success": False, "error": str(e)})
                break

        state["stats"]["total_rollbacks"] += rolled_back_count
        state["history"].append({
            "op": "rollback",
            "db": db_name,
            "rolled_back": rolled_back_count,
            "timestamp": _now_iso(),
        })
        _save_state(state)

        all_ok = all(r["success"] for r in results)
        return SkillResult(
            success=all_ok,
            message=f"Rolled back {rolled_back_count}/{count} migrations on {db_name}",
            data={"rolled_back": rolled_back_count, "results": results, "db": db_name},
        )

    async def _status(self, params: Dict) -> SkillResult:
        """Show migration status for a database."""
        db_name = params.get("db", "default")
        db_path = _get_db_path(db_name)

        state = _load_state()
        migrations = state.get("migrations", {}).get(db_name, [])

        applied_versions = set()
        if db_path.exists():
            applied_versions = set(_get_applied_versions(db_path))

        status_list = []
        for mig in sorted(migrations, key=lambda m: m["version"]):
            status_list.append({
                "version": mig["version"],
                "name": mig["name"],
                "status": "applied" if mig["version"] in applied_versions else mig.get("status", "pending"),
                "has_down": bool(mig.get("down_sql")),
                "checksum": mig.get("checksum"),
            })

        applied = [s for s in status_list if s["status"] == "applied"]
        pending = [s for s in status_list if s["status"] == "pending"]

        return SkillResult(
            success=True,
            message=f"{db_name}: {len(applied)} applied, {len(pending)} pending",
            data={
                "db": db_name,
                "applied_count": len(applied),
                "pending_count": len(pending),
                "migrations": status_list,
            },
        )

    async def _history(self, params: Dict) -> SkillResult:
        """View migration operation history."""
        db_name = params.get("db")
        limit = int(params.get("limit", 20))

        state = _load_state()
        history = state.get("history", [])
        if db_name:
            history = [h for h in history if h.get("db") == db_name]
        history = history[-limit:]

        return SkillResult(
            success=True,
            message=f"Showing {len(history)} history entries",
            data={"history": history, "stats": state.get("stats", {})},
        )

    async def _diff(self, params: Dict) -> SkillResult:
        """Compare current schema against target and generate migration SQL."""
        db_name = params.get("db", "default")
        target_schema = params.get("target_schema", {})
        db_path = _get_db_path(db_name)

        if not target_schema:
            return SkillResult(success=False, message="target_schema is required")

        current = _get_schema(db_path)
        statements = _diff_schemas(current, target_schema)

        if not statements:
            return SkillResult(
                success=True,
                message="Schemas are identical, no migration needed",
                data={"changes": 0, "sql": []},
            )

        # Generate reverse statements for down_sql
        down_statements = _diff_schemas(target_schema, current)

        return SkillResult(
            success=True,
            message=f"Found {len(statements)} schema changes",
            data={
                "changes": len(statements),
                "up_sql": statements,
                "down_sql": down_statements,
                "current_tables": list(k for k in current if not k.startswith("__")),
                "target_tables": list(k for k in target_schema if not k.startswith("__")),
            },
        )

    async def _snapshot(self, params: Dict) -> SkillResult:
        """Capture current schema state."""
        db_name = params.get("db", "default")
        db_path = _get_db_path(db_name)

        if not db_path.exists():
            return SkillResult(success=False, message=f"Database {db_name} not found at {db_path}")

        schema = _get_schema(db_path)
        snapshot = {
            "id": str(uuid.uuid4())[:8],
            "db": db_name,
            "captured_at": _now_iso(),
            "tables": list(k for k in schema if not k.startswith("__")),
            "table_count": len([k for k in schema if not k.startswith("__")]),
            "schema": schema,
        }

        state = _load_state()
        if db_name not in state["snapshots"]:
            state["snapshots"][db_name] = []
        state["snapshots"][db_name].append(snapshot)
        # Keep last 10 snapshots per db
        if len(state["snapshots"][db_name]) > 10:
            state["snapshots"][db_name] = state["snapshots"][db_name][-10:]
        _save_state(state)

        return SkillResult(
            success=True,
            message=f"Snapshot captured for {db_name}: {snapshot['table_count']} tables",
            data=snapshot,
        )

    async def _validate(self, params: Dict) -> SkillResult:
        """Validate migration chain integrity."""
        db_name = params.get("db", "default")
        db_path = _get_db_path(db_name)

        state = _load_state()
        migrations = state.get("migrations", {}).get(db_name, [])
        if not migrations:
            return SkillResult(success=True, message=f"No migrations for {db_name}", data={"valid": True, "issues": []})

        issues = []
        # Check version ordering
        versions = [m["version"] for m in migrations]
        if versions != sorted(versions):
            issues.append("Migrations are not in chronological order")

        # Check for duplicate versions
        seen = set()
        for v in versions:
            if v in seen:
                issues.append(f"Duplicate version: {v}")
            seen.add(v)

        # Check applied versions match database
        if db_path.exists():
            db_applied = set(_get_applied_versions(db_path))
            state_applied = {m["version"] for m in migrations if m.get("status") == "applied"}
            orphaned = db_applied - {m["version"] for m in migrations}
            if orphaned:
                issues.append(f"Orphaned migrations in database: {orphaned}")
            missing = state_applied - db_applied
            if missing:
                issues.append(f"Migrations marked applied but missing from database: {missing}")

        # Check checksums
        for mig in migrations:
            if mig.get("status") == "applied" and mig.get("up_sql"):
                expected = _compute_checksum(mig["up_sql"])
                if mig.get("checksum") and mig["checksum"] != expected:
                    issues.append(f"Checksum mismatch for {mig['version']}: migration SQL may have been modified")

        # Check for missing down_sql
        no_down = [m["version"] for m in migrations if not m.get("down_sql") and m.get("status") == "applied"]
        if no_down:
            issues.append(f"Applied migrations without rollback SQL: {no_down}")

        valid = len(issues) == 0
        return SkillResult(
            success=True,
            message=f"Validation {'passed' if valid else 'failed'}: {len(issues)} issues found",
            data={"valid": valid, "issues": issues, "total_migrations": len(migrations)},
        )

    async def estimate_cost(self, action: str, params: Dict) -> float:
        return 0.0


def _split_sql(sql: str) -> List[str]:
    """Split SQL into individual statements, handling semicolons."""
    statements = []
    current = []
    for line in sql.split("\n"):
        stripped = line.strip()
        if stripped.startswith("--"):
            continue
        current.append(line)
        if stripped.endswith(";"):
            stmt = "\n".join(current).strip()
            if stmt:
                statements.append(stmt)
            current = []
    # Handle last statement without semicolon
    if current:
        stmt = "\n".join(current).strip()
        if stmt:
            statements.append(stmt)
    return statements
