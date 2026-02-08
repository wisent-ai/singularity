#!/usr/bin/env python3
"""
DatabaseMaintenanceSkill - Autonomous database health maintenance via SchedulerSkill.

Bridges DatabaseSkill + SchedulerSkill to provide automatic database upkeep:

  1. Vacuum: Reclaim space from deleted rows, defragment database files
  2. Analyze: Update SQLite statistics for query optimizer performance
  3. Integrity Check: Detect and report database corruption early
  4. Stale Data Cleanup: Remove old records beyond configurable retention
  5. Index Optimization: Auto-create indexes on frequently queried columns
  6. Health Report: Unified database health assessment with actionable recommendations
  7. Schedule: Set up recurring maintenance via SchedulerSkill
  8. History: View maintenance operation audit trail

Revenue impact: Keeps paid database services (DatabaseRevenueBridgeSkill)
  running smoothly. A corrupted or bloated database = lost revenue.

Self-Improvement: The agent proactively maintains its own infrastructure
  without human intervention — a key autonomous behavior.

Pillar: Self-Improvement (primary), Revenue (supporting)
"""

import json
import os
import sqlite3
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import Skill, SkillAction, SkillManifest, SkillResult

DATA_DIR = Path(__file__).parent.parent / "data"
MAINTENANCE_FILE = DATA_DIR / "database_maintenance.json"
DB_REGISTRY_FILE = DATA_DIR / "database_registry.json"
MAX_HISTORY = 500

# Default retention periods (seconds)
DEFAULT_RETENTION = {
    "logs": 30 * 86400,        # 30 days
    "metrics": 90 * 86400,     # 90 days
    "events": 60 * 86400,      # 60 days
    "temp": 7 * 86400,         # 7 days
    "default": 180 * 86400,    # 180 days
}

# Maintenance schedule defaults (seconds)
DEFAULT_SCHEDULES = {
    "vacuum": 86400,           # Daily
    "analyze": 43200,          # Every 12 hours
    "integrity_check": 604800, # Weekly
    "cleanup": 86400,    # Daily
}


def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _load_state() -> Dict:
    try:
        if MAINTENANCE_FILE.exists():
            with open(MAINTENANCE_FILE, "r") as f:
                return json.load(f)
    except (json.JSONDecodeError, IOError):
        pass
    return {
        "history": [],
        "schedules": {},
        "retention_policies": dict(DEFAULT_RETENTION),
        "index_suggestions": {},
        "stats": {
            "total_operations": 0,
            "successful_operations": 0,
            "failed_operations": 0,
            "total_space_reclaimed_bytes": 0,
            "total_rows_cleaned": 0,
            "total_indexes_created": 0,
        },
    }


def _save_state(state: Dict):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if len(state.get("history", [])) > MAX_HISTORY:
        state["history"] = state["history"][-MAX_HISTORY:]
    try:
        with open(MAINTENANCE_FILE, "w") as f:
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
            if db_name in registry:
                return Path(registry[db_name]["path"])
    except (json.JSONDecodeError, IOError, KeyError):
        pass
    return DATA_DIR / f"{db_name}.db"


def _get_all_databases() -> List[Dict]:
    """List all known databases."""
    dbs = []
    # Default database
    default_db = DATA_DIR / "agent_data.db"
    if default_db.exists():
        dbs.append({"name": "default", "path": str(default_db)})
    # Registry databases
    try:
        if DB_REGISTRY_FILE.exists():
            with open(DB_REGISTRY_FILE, "r") as f:
                registry = json.load(f)
            for name, info in registry.items():
                p = Path(info.get("path", ""))
                if p.exists():
                    dbs.append({"name": name, "path": str(p)})
    except (json.JSONDecodeError, IOError):
        pass
    # .db files in data dir
    for f in DATA_DIR.glob("*.db"):
        fname = f.stem
        if not any(d["path"] == str(f) for d in dbs):
            dbs.append({"name": fname, "path": str(f)})
    return dbs


def _record_operation(state: Dict, op_type: str, db_name: str, success: bool, details: Dict):
    """Record a maintenance operation in history."""
    state["history"].append({
        "id": str(uuid.uuid4())[:8],
        "type": op_type,
        "db_name": db_name,
        "success": success,
        "timestamp": _now_iso(),
        "details": details,
    })
    state["stats"]["total_operations"] += 1
    if success:
        state["stats"]["successful_operations"] += 1
    else:
        state["stats"]["failed_operations"] += 1


class DatabaseMaintenanceSkill(Skill):
    """Autonomous database maintenance bridging DatabaseSkill and SchedulerSkill."""

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self._state = _load_state()

    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="database_maintenance",
            name="Database Maintenance",
            version="1.0.0",
            category="self_improvement",
            description="Autonomous database health: vacuum, analyze, integrity checks, stale cleanup, index optimization",
            required_credentials=[],
            actions=[
                SkillAction(
                    name="vacuum",
                    description="Reclaim space and defragment a database",
                    parameters={
                        "db_name": {"type": "string", "required": False, "description": "Database name (default: all)"},
                    },
                ),
                SkillAction(
                    name="analyze",
                    description="Update query optimizer statistics for a database",
                    parameters={
                        "db_name": {"type": "string", "required": False, "description": "Database name (default: all)"},
                    },
                ),
                SkillAction(
                    name="integrity_check",
                    description="Run integrity check to detect corruption",
                    parameters={
                        "db_name": {"type": "string", "required": False, "description": "Database name (default: all)"},
                    },
                ),
                SkillAction(
                    name="cleanup",
                    description="Remove stale data beyond retention period",
                    parameters={
                        "db_name": {"type": "string", "required": False, "description": "Database name (default: all)"},
                        "table": {"type": "string", "required": False, "description": "Specific table to clean (default: all tables)"},
                        "retention_days": {"type": "number", "required": False, "description": "Override retention in days"},
                    },
                ),
                SkillAction(
                    name="optimize_indexes",
                    description="Analyze query patterns and suggest or create indexes",
                    parameters={
                        "db_name": {"type": "string", "required": False, "description": "Database name (default: all)"},
                        "auto_create": {"type": "boolean", "required": False, "description": "Auto-create suggested indexes (default: false)"},
                    },
                ),
                SkillAction(
                    name="health",
                    description="Generate unified health report for all databases",
                    parameters={
                        "db_name": {"type": "string", "required": False, "description": "Database name (default: all)"},
                    },
                ),
                SkillAction(
                    name="schedule",
                    description="Set up recurring maintenance jobs via SchedulerSkill",
                    parameters={
                        "operation": {"type": "string", "required": True, "description": "Operation: vacuum, analyze, integrity_check, cleanup, full"},
                        "interval_hours": {"type": "number", "required": False, "description": "Interval in hours (uses defaults if omitted)"},
                        "db_name": {"type": "string", "required": False, "description": "Target database (default: all)"},
                    },
                ),
                SkillAction(
                    name="history",
                    description="View maintenance operation audit trail",
                    parameters={
                        "limit": {"type": "number", "required": False, "description": "Number of entries (default: 20)"},
                        "op_type": {"type": "string", "required": False, "description": "Filter by operation type"},
                    },
                ),
                SkillAction(
                    name="set_retention",
                    description="Configure retention policy for a data category",
                    parameters={
                        "category": {"type": "string", "required": True, "description": "Category: logs, metrics, events, temp, or custom name"},
                        "retention_days": {"type": "number", "required": True, "description": "Retention period in days"},
                    },
                ),
            ],
        )

    async def execute(self, action: str, params: Dict[str, Any]) -> SkillResult:
        try:
            if action == "vacuum":
                return await self._vacuum(params)
            elif action == "analyze":
                return await self._analyze(params)
            elif action == "integrity_check":
                return await self._integrity_check(params)
            elif action == "cleanup":
                return await self._cleanup(params)
            elif action == "optimize_indexes":
                return await self._optimize_indexes(params)
            elif action == "health":
                return await self._health(params)
            elif action == "schedule":
                return await self._schedule(params)
            elif action == "history":
                return await self._history(params)
            elif action == "set_retention":
                return await self._set_retention(params)
            else:
                return SkillResult(success=False, data={"error": f"Unknown action: {action}"})
        except Exception as e:
            return SkillResult(success=False, data={"error": str(e)})

    async def _vacuum(self, params: Dict) -> SkillResult:
        """Run VACUUM on databases to reclaim space."""
        db_name = params.get("db_name")
        databases = self._resolve_databases(db_name)
        if not databases:
            return SkillResult(success=False, data={"error": "No databases found"})

        results = []
        total_reclaimed = 0
        for db in databases:
            try:
                path = Path(db["path"])
                size_before = path.stat().st_size if path.exists() else 0
                conn = sqlite3.connect(str(path))
                conn.execute("VACUUM")
                conn.close()
                size_after = path.stat().st_size if path.exists() else 0
                reclaimed = max(0, size_before - size_after)
                total_reclaimed += reclaimed
                _record_operation(self._state, "vacuum", db["name"], True, {
                    "size_before": size_before,
                    "size_after": size_after,
                    "reclaimed_bytes": reclaimed,
                })
                results.append({
                    "db": db["name"],
                    "success": True,
                    "size_before": size_before,
                    "size_after": size_after,
                    "reclaimed_bytes": reclaimed,
                    "reclaimed_mb": round(reclaimed / (1024 * 1024), 3),
                })
            except Exception as e:
                _record_operation(self._state, "vacuum", db["name"], False, {"error": str(e)})
                results.append({"db": db["name"], "success": False, "error": str(e)})

        self._state["stats"]["total_space_reclaimed_bytes"] += total_reclaimed
        _save_state(self._state)
        return SkillResult(success=True, data={
            "operation": "vacuum",
            "databases_processed": len(results),
            "total_reclaimed_bytes": total_reclaimed,
            "total_reclaimed_mb": round(total_reclaimed / (1024 * 1024), 3),
            "results": results,
        })

    async def _analyze(self, params: Dict) -> SkillResult:
        """Run ANALYZE to update query optimizer statistics."""
        db_name = params.get("db_name")
        databases = self._resolve_databases(db_name)
        if not databases:
            return SkillResult(success=False, data={"error": "No databases found"})

        results = []
        for db in databases:
            try:
                conn = sqlite3.connect(str(db["path"]))
                conn.execute("ANALYZE")
                # Get table count for reporting
                cursor = conn.execute(
                    "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
                )
                table_count = cursor.fetchone()[0]
                conn.close()
                _record_operation(self._state, "analyze", db["name"], True, {
                    "tables_analyzed": table_count,
                })
                results.append({
                    "db": db["name"],
                    "success": True,
                    "tables_analyzed": table_count,
                })
            except Exception as e:
                _record_operation(self._state, "analyze", db["name"], False, {"error": str(e)})
                results.append({"db": db["name"], "success": False, "error": str(e)})

        _save_state(self._state)
        return SkillResult(success=True, data={
            "operation": "analyze",
            "databases_processed": len(results),
            "results": results,
        })

    async def _integrity_check(self, params: Dict) -> SkillResult:
        """Run PRAGMA integrity_check to detect corruption."""
        db_name = params.get("db_name")
        databases = self._resolve_databases(db_name)
        if not databases:
            return SkillResult(success=False, data={"error": "No databases found"})

        results = []
        issues_found = 0
        for db in databases:
            try:
                conn = sqlite3.connect(str(db["path"]))
                cursor = conn.execute("PRAGMA integrity_check")
                check_result = cursor.fetchall()
                conn.close()
                is_ok = len(check_result) == 1 and check_result[0][0] == "ok"
                if not is_ok:
                    issues_found += len(check_result)
                issues = [row[0] for row in check_result] if not is_ok else []
                _record_operation(self._state, "integrity_check", db["name"], is_ok, {
                    "result": "ok" if is_ok else "issues_found",
                    "issues": issues,
                })
                results.append({
                    "db": db["name"],
                    "healthy": is_ok,
                    "issues": issues,
                })
            except Exception as e:
                _record_operation(self._state, "integrity_check", db["name"], False, {"error": str(e)})
                results.append({"db": db["name"], "healthy": False, "issues": [str(e)]})
                issues_found += 1

        _save_state(self._state)
        all_healthy = issues_found == 0
        return SkillResult(success=True, data={
            "operation": "integrity_check",
            "databases_checked": len(results),
            "all_healthy": all_healthy,
            "issues_found": issues_found,
            "results": results,
        })

    async def _cleanup(self, params: Dict) -> SkillResult:
        """Remove stale data beyond retention period."""
        db_name = params.get("db_name")
        table_filter = params.get("table")
        retention_days = params.get("retention_days")
        databases = self._resolve_databases(db_name)
        if not databases:
            return SkillResult(success=False, data={"error": "No databases found"})

        results = []
        total_rows_deleted = 0
        for db in databases:
            try:
                conn = sqlite3.connect(str(db["path"]))
                # Get all tables
                cursor = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
                )
                tables = [row[0] for row in cursor.fetchall()]
                if table_filter:
                    tables = [t for t in tables if t == table_filter]

                db_deleted = 0
                table_results = []
                for table in tables:
                    # Find timestamp columns
                    cursor = conn.execute(f"PRAGMA table_info([{table}])")
                    columns = cursor.fetchall()
                    ts_cols = [c[1] for c in columns if any(
                        kw in c[1].lower() for kw in ["timestamp", "created_at", "date", "time", "updated_at"]
                    )]
                    if not ts_cols:
                        continue

                    # Determine retention
                    category = self._categorize_table(table)
                    if retention_days:
                        retention_sec = retention_days * 86400
                    else:
                        retention_sec = self._state.get("retention_policies", DEFAULT_RETENTION).get(
                            category, DEFAULT_RETENTION["default"]
                        )
                    cutoff = datetime.utcnow() - timedelta(seconds=retention_sec)
                    cutoff_str = cutoff.isoformat()

                    # Try each timestamp column
                    for ts_col in ts_cols:
                        try:
                            cursor = conn.execute(
                                f"SELECT COUNT(*) FROM [{table}] WHERE [{ts_col}] < ?",
                                (cutoff_str,)
                            )
                            count = cursor.fetchone()[0]
                            if count > 0:
                                conn.execute(
                                    f"DELETE FROM [{table}] WHERE [{ts_col}] < ?",
                                    (cutoff_str,)
                                )
                                conn.commit()
                                db_deleted += count
                                table_results.append({
                                    "table": table,
                                    "column": ts_col,
                                    "rows_deleted": count,
                                    "retention_days": round(retention_sec / 86400),
                                })
                                break  # Only use first matching timestamp column
                        except sqlite3.OperationalError:
                            continue

                conn.close()
                total_rows_deleted += db_deleted
                _record_operation(self._state, "cleanup", db["name"], True, {
                    "rows_deleted": db_deleted,
                    "tables_cleaned": len(table_results),
                })
                results.append({
                    "db": db["name"],
                    "success": True,
                    "rows_deleted": db_deleted,
                    "tables": table_results,
                })
            except Exception as e:
                _record_operation(self._state, "cleanup", db["name"], False, {"error": str(e)})
                results.append({"db": db["name"], "success": False, "error": str(e)})

        self._state["stats"]["total_rows_cleaned"] += total_rows_deleted
        _save_state(self._state)
        return SkillResult(success=True, data={
            "operation": "cleanup",
            "databases_processed": len(results),
            "total_rows_deleted": total_rows_deleted,
            "results": results,
        })

    async def _optimize_indexes(self, params: Dict) -> SkillResult:
        """Analyze tables and suggest/create indexes for performance."""
        db_name = params.get("db_name")
        auto_create = params.get("auto_create", False)
        databases = self._resolve_databases(db_name)
        if not databases:
            return SkillResult(success=False, data={"error": "No databases found"})

        results = []
        total_created = 0
        for db in databases:
            try:
                conn = sqlite3.connect(str(db["path"]))
                # Get tables
                cursor = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
                )
                tables = [row[0] for row in cursor.fetchall()]

                # Get existing indexes
                cursor = conn.execute(
                    "SELECT name, tbl_name FROM sqlite_master WHERE type='index'"
                )
                existing_indexes = {row[0]: row[1] for row in cursor.fetchall()}

                suggestions = []
                created = []
                for table in tables:
                    cursor = conn.execute(f"PRAGMA table_info([{table}])")
                    columns = cursor.fetchall()

                    # Get row count
                    cursor = conn.execute(f"SELECT COUNT(*) FROM [{table}]")
                    row_count = cursor.fetchone()[0]

                    # Only suggest indexes for tables with significant data
                    if row_count < 100:
                        continue

                    for col in columns:
                        col_name = col[1]
                        col_type = col[2] or ""
                        is_pk = col[5]  # Primary key flag

                        if is_pk:
                            continue  # PKs already indexed

                        # Suggest indexes for common patterns
                        should_index = False
                        reason = ""
                        if any(kw in col_name.lower() for kw in ["_id", "user_id", "customer_id", "session_id"]):
                            should_index = True
                            reason = "Foreign key column"
                        elif any(kw in col_name.lower() for kw in ["timestamp", "created_at", "updated_at", "date"]):
                            should_index = True
                            reason = "Timestamp column (range queries)"
                        elif any(kw in col_name.lower() for kw in ["status", "type", "category", "state"]):
                            should_index = True
                            reason = "Low-cardinality filter column"
                        elif col_name.lower() in ["email", "name", "slug", "key"]:
                            should_index = True
                            reason = "Lookup column"

                        if not should_index:
                            continue

                        idx_name = f"idx_{table}_{col_name}"
                        if idx_name in existing_indexes:
                            continue  # Already indexed

                        suggestion = {
                            "table": table,
                            "column": col_name,
                            "index_name": idx_name,
                            "reason": reason,
                            "row_count": row_count,
                        }
                        suggestions.append(suggestion)

                        if auto_create:
                            try:
                                conn.execute(f"CREATE INDEX IF NOT EXISTS [{idx_name}] ON [{table}] ([{col_name}])")
                                conn.commit()
                                created.append(idx_name)
                                total_created += 1
                            except sqlite3.OperationalError as e:
                                suggestion["create_error"] = str(e)

                conn.close()
                _record_operation(self._state, "optimize_indexes", db["name"], True, {
                    "suggestions": len(suggestions),
                    "created": len(created),
                })
                results.append({
                    "db": db["name"],
                    "success": True,
                    "suggestions": suggestions,
                    "created_indexes": created,
                })
            except Exception as e:
                _record_operation(self._state, "optimize_indexes", db["name"], False, {"error": str(e)})
                results.append({"db": db["name"], "success": False, "error": str(e)})

        self._state["stats"]["total_indexes_created"] += total_created
        # Store suggestions for reference
        for r in results:
            if r.get("suggestions"):
                self._state["index_suggestions"][r["db"]] = r["suggestions"]
        _save_state(self._state)
        return SkillResult(success=True, data={
            "operation": "optimize_indexes",
            "databases_processed": len(results),
            "total_suggestions": sum(len(r.get("suggestions", [])) for r in results),
            "total_indexes_created": total_created,
            "auto_create": auto_create,
            "results": results,
        })

    async def _health(self, params: Dict) -> SkillResult:
        """Generate comprehensive health report for databases."""
        db_name = params.get("db_name")
        databases = self._resolve_databases(db_name)
        if not databases:
            return SkillResult(success=False, data={"error": "No databases found"})

        reports = []
        recommendations = []
        for db in databases:
            try:
                path = Path(db["path"])
                size_bytes = path.stat().st_size if path.exists() else 0

                conn = sqlite3.connect(str(db["path"]))

                # Table info
                cursor = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
                )
                tables = [row[0] for row in cursor.fetchall()]

                # Index count
                cursor = conn.execute(
                    "SELECT COUNT(*) FROM sqlite_master WHERE type='index'"
                )
                index_count = cursor.fetchone()[0]

                # Total rows
                total_rows = 0
                table_sizes = []
                for table in tables:
                    try:
                        cursor = conn.execute(f"SELECT COUNT(*) FROM [{table}]")
                        count = cursor.fetchone()[0]
                        total_rows += count
                        table_sizes.append({"table": table, "rows": count})
                    except sqlite3.OperationalError:
                        pass

                # Integrity
                cursor = conn.execute("PRAGMA integrity_check")
                integrity = cursor.fetchall()
                is_healthy = len(integrity) == 1 and integrity[0][0] == "ok"

                # Page info
                cursor = conn.execute("PRAGMA page_count")
                page_count = cursor.fetchone()[0]
                cursor = conn.execute("PRAGMA freelist_count")
                freelist_count = cursor.fetchone()[0]
                fragmentation = round(freelist_count / max(page_count, 1) * 100, 1)

                conn.close()

                # Generate recommendations
                if fragmentation > 10:
                    recommendations.append({
                        "db": db["name"],
                        "action": "vacuum",
                        "reason": f"Fragmentation at {fragmentation}% (threshold: 10%)",
                        "priority": "high" if fragmentation > 25 else "medium",
                    })
                if size_bytes > 100 * 1024 * 1024:  # 100MB
                    recommendations.append({
                        "db": db["name"],
                        "action": "cleanup",
                        "reason": f"Database size {round(size_bytes / (1024*1024), 1)}MB exceeds 100MB",
                        "priority": "medium",
                    })
                if not is_healthy:
                    recommendations.append({
                        "db": db["name"],
                        "action": "integrity_check",
                        "reason": "Integrity check failed",
                        "priority": "critical",
                    })

                # Sort tables by size desc
                table_sizes.sort(key=lambda x: x["rows"], reverse=True)

                reports.append({
                    "db": db["name"],
                    "healthy": is_healthy,
                    "size_bytes": size_bytes,
                    "size_mb": round(size_bytes / (1024 * 1024), 3),
                    "table_count": len(tables),
                    "index_count": index_count,
                    "total_rows": total_rows,
                    "fragmentation_pct": fragmentation,
                    "page_count": page_count,
                    "freelist_pages": freelist_count,
                    "largest_tables": table_sizes[:5],
                })
            except Exception as e:
                reports.append({"db": db["name"], "healthy": False, "error": str(e)})

        _save_state(self._state)
        return SkillResult(success=True, data={
            "operation": "health",
            "databases_assessed": len(reports),
            "all_healthy": all(r.get("healthy", False) for r in reports),
            "total_size_mb": round(sum(r.get("size_mb", 0) for r in reports), 3),
            "total_rows": sum(r.get("total_rows", 0) for r in reports),
            "recommendations": recommendations,
            "reports": reports,
            "maintenance_stats": self._state["stats"],
        })

    async def _schedule(self, params: Dict) -> SkillResult:
        """Set up recurring maintenance via SchedulerSkill integration."""
        operation = params.get("operation")
        if not operation:
            return SkillResult(success=False, data={"error": "operation is required"})

        db_name = params.get("db_name", "all")

        if operation == "full":
            # Schedule all maintenance operations
            ops = ["vacuum", "analyze", "integrity_check", "cleanup"]
        else:
            if operation not in DEFAULT_SCHEDULES:
                return SkillResult(success=False, data={
                    "error": f"Unknown operation: {operation}. Valid: vacuum, analyze, integrity_check, cleanup, full"
                })
            ops = [operation]

        schedules_created = []
        for op in ops:
            interval_hours = params.get("interval_hours")
            if interval_hours:
                interval_sec = interval_hours * 3600
            else:
                interval_sec = DEFAULT_SCHEDULES[op]

            schedule_id = f"db_maint_{op}_{db_name}"
            schedule_info = {
                "id": schedule_id,
                "operation": op,
                "db_name": db_name,
                "interval_seconds": interval_sec,
                "interval_hours": round(interval_sec / 3600, 1),
                "created_at": _now_iso(),
                "scheduler_task": {
                    "skill_id": "database_maintenance",
                    "action": op,
                    "params": {"db_name": db_name} if db_name != "all" else {},
                    "schedule_type": "recurring",
                    "interval_seconds": interval_sec,
                },
            }
            self._state["schedules"][schedule_id] = schedule_info
            schedules_created.append(schedule_info)

        _save_state(self._state)
        return SkillResult(success=True, data={
            "operation": "schedule",
            "schedules_created": len(schedules_created),
            "schedules": schedules_created,
            "note": "Schedules registered. Use SchedulerSkill to activate recurring execution.",
        })

    async def _history(self, params: Dict) -> SkillResult:
        """View maintenance operation history."""
        limit = min(params.get("limit", 20), 100)
        op_type = params.get("op_type")

        history = self._state.get("history", [])
        if op_type:
            history = [h for h in history if h.get("type") == op_type]

        recent = history[-limit:] if history else []
        recent.reverse()  # Most recent first

        return SkillResult(success=True, data={
            "operation": "history",
            "total_entries": len(history),
            "showing": len(recent),
            "filter": op_type,
            "entries": recent,
            "lifetime_stats": self._state["stats"],
        })

    async def _set_retention(self, params: Dict) -> SkillResult:
        """Configure retention policy for a data category."""
        category = params.get("category")
        retention_days = params.get("retention_days")
        if not category or retention_days is None:
            return SkillResult(success=False, data={"error": "category and retention_days required"})
        if retention_days < 1:
            return SkillResult(success=False, data={"error": "retention_days must be >= 1"})

        retention_sec = int(retention_days * 86400)
        if "retention_policies" not in self._state:
            self._state["retention_policies"] = dict(DEFAULT_RETENTION)
        old_val = self._state["retention_policies"].get(category)
        self._state["retention_policies"][category] = retention_sec
        _save_state(self._state)

        return SkillResult(success=True, data={
            "category": category,
            "retention_days": retention_days,
            "retention_seconds": retention_sec,
            "previous_days": round(old_val / 86400, 1) if old_val else None,
            "all_policies": {
                k: round(v / 86400, 1) for k, v in self._state["retention_policies"].items()
            },
        })

    def _resolve_databases(self, db_name: Optional[str]) -> List[Dict]:
        """Resolve database name to list of database info dicts."""
        if db_name and db_name != "all":
            path = _get_db_path(db_name)
            if path.exists():
                return [{"name": db_name, "path": str(path)}]
            return []
        return _get_all_databases()

    def _categorize_table(self, table_name: str) -> str:
        """Categorize a table for retention policy lookup."""
        name = table_name.lower()
        if any(kw in name for kw in ["log", "audit", "trace"]):
            return "logs"
        if any(kw in name for kw in ["metric", "stat", "perf"]):
            return "metrics"
        if any(kw in name for kw in ["event", "notification", "alert"]):
            return "events"
        if any(kw in name for kw in ["temp", "tmp", "cache", "scratch"]):
            return "temp"
        return "default"
