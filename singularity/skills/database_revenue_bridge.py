#!/usr/bin/env python3
"""
DatabaseRevenueBridgeSkill - Wire DatabaseSkill into revenue-generating paid services.

Connects DatabaseSkill to paid services so the agent can earn money from data:

  1. Data Analysis Service: Customers pay to run analytical SQL queries
  2. Schema Design Service: Customers pay for database schema creation/optimization
  3. Data Import Service: Customers pay to import JSON/CSV into structured tables
  4. Report Generation Service: Customers pay for formatted data reports with stats
  5. Data Transformation Service: Customers pay for ETL-style data transforms

Revenue flow:
  Customer -> ServiceAPI -> DatabaseRevenueBridgeSkill -> DatabaseSkill -> SQLite
                                                       -> BillingPipeline -> Revenue

Pillar: Revenue Generation - turns data analysis into a paid service.
"""

import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from .base import Skill, SkillResult, SkillManifest, SkillAction


DATA_DIR = Path(__file__).parent.parent / "data"
BRIDGE_FILE = DATA_DIR / "database_revenue_bridge.json"
MAX_HISTORY = 500

PRICING = {
    "data_analysis": 0.01,      # Per query
    "schema_design": 0.02,      # Per schema operation
    "data_import": 0.005,       # Per import job
    "report_generation": 0.015, # Per report
    "data_transform": 0.008,    # Per transform
}


def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _load_store() -> Dict:
    try:
        if BRIDGE_FILE.exists():
            with open(BRIDGE_FILE, "r") as f:
                return json.load(f)
    except (json.JSONDecodeError, IOError):
        pass
    return {
        "jobs": [],
        "reports": {},
        "schemas_created": {},
        "revenue": {"total": 0.0, "by_service": {}, "by_customer": {}},
        "stats": {"total_requests": 0, "successful_requests": 0, "failed_requests": 0},
    }


def _save_store(store: Dict):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if len(store.get("jobs", [])) > MAX_HISTORY:
        store["jobs"] = store["jobs"][-MAX_HISTORY:]
    try:
        with open(BRIDGE_FILE, "w") as f:
            json.dump(store, f, indent=2, default=str)
    except IOError:
        pass


class DatabaseRevenueBridgeSkill(Skill):
    """Bridges DatabaseSkill into paid revenue-generating data services."""

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self._store = _load_store()
        self._db_skill = None

    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="database_revenue_bridge",
            name="Database Revenue Bridge",
            version="1.0.0",
            category="revenue",
            description="Paid data services: analysis queries, schema design, data import, reports, transforms",
            required_credentials=[],
            actions=[
                SkillAction(
                    name="analyze",
                    description="Run a paid analytical SQL query for a customer",
                    parameters={
                        "customer_id": {"type": "string", "required": True, "description": "Customer identifier"},
                        "query": {"type": "string", "required": True, "description": "SQL SELECT query to run"},
                        "db_name": {"type": "string", "required": False, "description": "Database name (default: 'default')"},
                        "format": {"type": "string", "required": False, "description": "Output format: json, csv, summary (default: json)"},
                    },
                    estimated_cost=PRICING["data_analysis"],
                ),
                SkillAction(
                    name="design_schema",
                    description="Create database tables from a schema specification",
                    parameters={
                        "customer_id": {"type": "string", "required": True, "description": "Customer identifier"},
                        "db_name": {"type": "string", "required": True, "description": "Database name to create tables in"},
                        "tables": {"type": "array", "required": True,
                                   "description": "List of table defs: [{name, columns: [{name, type, constraints}]}]"},
                    },
                    estimated_cost=PRICING["schema_design"],
                ),
                SkillAction(
                    name="import_data",
                    description="Import JSON or CSV data into a database table",
                    parameters={
                        "customer_id": {"type": "string", "required": True, "description": "Customer identifier"},
                        "db_name": {"type": "string", "required": False, "description": "Database name (default: 'default')"},
                        "table_name": {"type": "string", "required": True, "description": "Target table name"},
                        "data": {"type": "array", "required": True, "description": "List of row dicts to import"},
                        "create_table": {"type": "boolean", "required": False, "description": "Auto-create table from data (default: True)"},
                    },
                    estimated_cost=PRICING["data_import"],
                ),
                SkillAction(
                    name="generate_report",
                    description="Generate a formatted data report with statistics",
                    parameters={
                        "customer_id": {"type": "string", "required": True, "description": "Customer identifier"},
                        "db_name": {"type": "string", "required": False, "description": "Database name (default: 'default')"},
                        "title": {"type": "string", "required": True, "description": "Report title"},
                        "queries": {"type": "array", "required": True,
                                    "description": "List of {label, query} objects for report sections"},
                    },
                    estimated_cost=PRICING["report_generation"],
                ),
                SkillAction(
                    name="transform_data",
                    description="Run a data transformation: filter, aggregate, or pivot data between tables",
                    parameters={
                        "customer_id": {"type": "string", "required": True, "description": "Customer identifier"},
                        "db_name": {"type": "string", "required": False, "description": "Database name (default: 'default')"},
                        "source_query": {"type": "string", "required": True, "description": "SELECT query for source data"},
                        "target_table": {"type": "string", "required": True, "description": "Table to write results to"},
                        "mode": {"type": "string", "required": False, "description": "replace or append (default: replace)"},
                    },
                    estimated_cost=PRICING["data_transform"],
                ),
                SkillAction(
                    name="list_services",
                    description="List available paid data services and pricing",
                    parameters={},
                ),
                SkillAction(
                    name="service_stats",
                    description="Get revenue and usage statistics",
                    parameters={
                        "customer_id": {"type": "string", "required": False, "description": "Filter stats by customer"},
                    },
                ),
            ],
        )

    async def initialize(self) -> bool:
        self._store = _load_store()
        return True

    def _get_db_skill(self):
        if self._db_skill is None:
            from .database import DatabaseSkill
            self._db_skill = DatabaseSkill()
        return self._db_skill

    def _record_revenue(self, service: str, customer_id: str, amount: float):
        rev = self._store.setdefault("revenue", {"total": 0.0, "by_service": {}, "by_customer": {}})
        rev["total"] = rev.get("total", 0.0) + amount
        rev["by_service"][service] = rev.get("by_service", {}).get(service, 0.0) + amount
        rev["by_customer"][customer_id] = rev.get("by_customer", {}).get(customer_id, 0.0) + amount

    def _record_job(self, service: str, customer_id: str, detail: str, success: bool, charged: float):
        self._store.setdefault("jobs", []).append({
            "id": str(uuid.uuid4())[:8],
            "timestamp": _now_iso(),
            "service": service,
            "customer_id": customer_id,
            "detail": detail[:500],
            "success": success,
            "charged": charged,
        })

    def _record_request(self, success: bool):
        stats = self._store.setdefault("stats", {"total_requests": 0, "successful_requests": 0, "failed_requests": 0})
        stats["total_requests"] = stats.get("total_requests", 0) + 1
        if success:
            stats["successful_requests"] = stats.get("successful_requests", 0) + 1
        else:
            stats["failed_requests"] = stats.get("failed_requests", 0) + 1

    async def execute(self, action: str, params: Dict) -> SkillResult:
        actions = {
            "analyze": self._analyze,
            "design_schema": self._design_schema,
            "import_data": self._import_data,
            "generate_report": self._generate_report,
            "transform_data": self._transform_data,
            "list_services": self._list_services,
            "service_stats": self._service_stats,
        }
        handler = actions.get(action)
        if not handler:
            return SkillResult(success=False, message=f"Unknown action: {action}. Available: {list(actions.keys())}")
        try:
            result = await handler(params)
            _save_store(self._store)
            return result
        except Exception as e:
            self._record_request(False)
            _save_store(self._store)
            return SkillResult(success=False, message=f"Error in {action}: {str(e)}")

    async def _analyze(self, params: Dict) -> SkillResult:
        """Run a paid analytical SQL query."""
        customer_id = params.get("customer_id", "anonymous")
        query = params.get("query")
        if not query:
            return SkillResult(success=False, message="query is required")

        db_name = params.get("db_name", "default")
        output_format = params.get("format", "json")

        # Only allow read queries
        query_upper = query.strip().upper()
        first_word = query_upper.split()[0] if query_upper.split() else ""
        if first_word not in ("SELECT", "WITH", "EXPLAIN", "PRAGMA"):
            return SkillResult(
                success=False,
                message="Only SELECT/WITH/EXPLAIN/PRAGMA queries are allowed for analysis. Use design_schema or transform_data for write operations."
            )

        db = self._get_db_skill()
        result = db._query({"sql": query, "db": db_name})

        if result.success:
            price = PRICING["data_analysis"]
            self._record_revenue("data_analysis", customer_id, price)
            self._record_request(True)
            self._record_job("data_analysis", customer_id, f"Query: {query[:200]}", True, price)
            return SkillResult(
                success=True,
                message=f"Analysis complete. {result.data.get('row_count', 0)} rows returned. Charged ${price}.",
                data={
                    "results": result.data,
                    "charged": price,
                    "customer_id": customer_id,
                    "format": output_format,
                },
                revenue=price,
            )
        else:
            self._record_request(False)
            self._record_job("data_analysis", customer_id, f"Failed: {result.message}", False, 0)
            return SkillResult(success=False, message=f"Query failed: {result.message}")

    async def _design_schema(self, params: Dict) -> SkillResult:
        """Create database tables from specification."""
        customer_id = params.get("customer_id", "anonymous")
        db_name = params.get("db_name")
        tables = params.get("tables")

        if not db_name:
            return SkillResult(success=False, message="db_name is required")
        if not tables or not isinstance(tables, list):
            return SkillResult(success=False, message="tables must be a non-empty list of table definitions")

        db = self._get_db_skill()

        # First ensure the database exists
        create_result = db._create_db({"name": db_name})
        # Ignore if already exists

        # Enable writes for schema creation
        db._enable_write({"db": db_name, "enabled": True})

        created_tables = []
        errors = []

        for table_def in tables:
            tname = table_def.get("name")
            columns = table_def.get("columns", [])
            if not tname or not columns:
                errors.append(f"Table missing name or columns: {table_def}")
                continue

            col_defs = []
            for col in columns:
                col_name = col.get("name")
                col_type = col.get("type", "TEXT")
                constraints = col.get("constraints", "")
                if col_name:
                    col_defs.append(f"{col_name} {col_type} {constraints}".strip())

            if not col_defs:
                errors.append(f"Table {tname} has no valid columns")
                continue

            sql = f"CREATE TABLE IF NOT EXISTS {tname} ({', '.join(col_defs)})"
            write_result = db._execute_write({"sql": sql, "db": db_name})
            if write_result.success:
                created_tables.append(tname)
            else:
                errors.append(f"Table {tname}: {write_result.message}")

        # Disable writes after schema creation
        db._enable_write({"db": db_name, "enabled": False})

        if created_tables:
            price = PRICING["schema_design"] * len(created_tables)
            self._record_revenue("schema_design", customer_id, price)
            self._record_request(True)
            self._record_job("schema_design", customer_id, f"Created tables: {', '.join(created_tables)}", True, price)

            # Track schema
            self._store.setdefault("schemas_created", {})[db_name] = {
                "tables": created_tables,
                "customer_id": customer_id,
                "created_at": _now_iso(),
            }

            return SkillResult(
                success=True,
                message=f"Created {len(created_tables)} tables in '{db_name}'. Charged ${price:.3f}.",
                data={
                    "created_tables": created_tables,
                    "errors": errors,
                    "charged": price,
                    "customer_id": customer_id,
                    "db_name": db_name,
                },
                revenue=price,
            )
        else:
            self._record_request(False)
            self._record_job("schema_design", customer_id, f"Failed: {'; '.join(errors)}", False, 0)
            return SkillResult(success=False, message=f"Schema creation failed: {'; '.join(errors)}")

    async def _import_data(self, params: Dict) -> SkillResult:
        """Import data into a database table."""
        customer_id = params.get("customer_id", "anonymous")
        db_name = params.get("db_name", "default")
        table_name = params.get("table_name")
        data = params.get("data")
        create_table = params.get("create_table", True)

        if not table_name:
            return SkillResult(success=False, message="table_name is required")
        if not data or not isinstance(data, list):
            return SkillResult(success=False, message="data must be a non-empty list of row dicts")

        db = self._get_db_skill()

        # Ensure db exists
        db._create_db({"name": db_name})
        db._enable_write({"db": db_name, "enabled": True})

        rows_imported = 0
        errors = []

        # Auto-create table from first row if needed
        if create_table and data:
            first_row = data[0]
            if isinstance(first_row, dict):
                col_defs = []
                for key in first_row.keys():
                    val = first_row[key]
                    if isinstance(val, int):
                        col_type = "INTEGER"
                    elif isinstance(val, float):
                        col_type = "REAL"
                    else:
                        col_type = "TEXT"
                    col_defs.append(f"{key} {col_type}")
                create_sql = f"CREATE TABLE IF NOT EXISTS {table_name} ({', '.join(col_defs)})"
                db._execute_write({"sql": create_sql, "db": db_name})

        # Insert rows
        for row in data:
            if not isinstance(row, dict):
                errors.append(f"Skipped non-dict row: {type(row)}")
                continue
            columns = list(row.keys())
            placeholders = ", ".join(["?" for _ in columns])
            col_names = ", ".join(columns)
            values = [row[c] for c in columns]
            sql = f"INSERT INTO {table_name} ({col_names}) VALUES ({placeholders})"
            result = db._execute_write({"sql": sql, "db": db_name, "params": values})
            if result.success:
                rows_imported += 1
            else:
                errors.append(f"Row insert failed: {result.message}")

        db._enable_write({"db": db_name, "enabled": False})

        if rows_imported > 0:
            price = PRICING["data_import"] * max(1, rows_imported // 100)  # Per 100 rows
            self._record_revenue("data_import", customer_id, price)
            self._record_request(True)
            self._record_job("data_import", customer_id, f"Imported {rows_imported} rows into {table_name}", True, price)
            return SkillResult(
                success=True,
                message=f"Imported {rows_imported} rows into '{table_name}'. Charged ${price:.3f}.",
                data={
                    "rows_imported": rows_imported,
                    "errors": errors[:10],
                    "charged": price,
                    "customer_id": customer_id,
                    "table_name": table_name,
                    "db_name": db_name,
                },
                revenue=price,
            )
        else:
            self._record_request(False)
            self._record_job("data_import", customer_id, f"Failed: {'; '.join(errors[:5])}", False, 0)
            return SkillResult(success=False, message=f"Import failed: {'; '.join(errors[:5])}")

    async def _generate_report(self, params: Dict) -> SkillResult:
        """Generate a formatted data report with multiple query sections."""
        customer_id = params.get("customer_id", "anonymous")
        db_name = params.get("db_name", "default")
        title = params.get("title", "Data Report")
        queries = params.get("queries")

        if not queries or not isinstance(queries, list):
            return SkillResult(success=False, message="queries must be a non-empty list of {label, query} objects")

        db = self._get_db_skill()
        sections = []
        total_rows = 0

        for q in queries:
            label = q.get("label", "Section")
            sql = q.get("query", "")
            if not sql:
                sections.append({"label": label, "error": "No query provided"})
                continue

            # Only allow read queries
            first_word = sql.strip().upper().split()[0] if sql.strip() else ""
            if first_word not in ("SELECT", "WITH", "EXPLAIN", "PRAGMA"):
                sections.append({"label": label, "error": "Only SELECT queries allowed in reports"})
                continue

            result = db._query({"sql": sql, "db": db_name})
            if result.success:
                rows = result.data.get("rows", [])
                row_count = result.data.get("row_count", len(rows))
                total_rows += row_count

                # Compute basic stats for numeric columns
                stats = {}
                if rows:
                    columns = result.data.get("columns", [])
                    for col in columns:
                        values = [r.get(col) for r in rows if isinstance(r.get(col), (int, float))]
                        if values:
                            stats[col] = {
                                "min": min(values),
                                "max": max(values),
                                "avg": round(sum(values) / len(values), 4),
                                "count": len(values),
                                "sum": round(sum(values), 4),
                            }

                sections.append({
                    "label": label,
                    "row_count": row_count,
                    "columns": result.data.get("columns", []),
                    "data": rows[:100],  # Cap preview at 100 rows
                    "stats": stats,
                })
            else:
                sections.append({"label": label, "error": result.message})

        if sections:
            price = PRICING["report_generation"]
            self._record_revenue("report_generation", customer_id, price)
            self._record_request(True)
            self._record_job("report_generation", customer_id, f"Report: {title}", True, price)

            report = {
                "report_id": str(uuid.uuid4())[:8],
                "title": title,
                "generated_at": _now_iso(),
                "customer_id": customer_id,
                "db_name": db_name,
                "sections": sections,
                "total_rows": total_rows,
                "section_count": len(sections),
            }

            # Store report
            self._store.setdefault("reports", {})[report["report_id"]] = {
                "title": title,
                "customer_id": customer_id,
                "generated_at": report["generated_at"],
                "section_count": len(sections),
                "total_rows": total_rows,
            }

            return SkillResult(
                success=True,
                message=f"Report '{title}' generated with {len(sections)} sections, {total_rows} total rows. Charged ${price:.3f}.",
                data={
                    "report": report,
                    "charged": price,
                },
                revenue=price,
            )
        else:
            self._record_request(False)
            self._record_job("report_generation", customer_id, f"Report failed: no valid sections", False, 0)
            return SkillResult(success=False, message="Report generation failed: no valid query sections")

    async def _transform_data(self, params: Dict) -> SkillResult:
        """Run a data transformation: query source and write to target table."""
        customer_id = params.get("customer_id", "anonymous")
        db_name = params.get("db_name", "default")
        source_query = params.get("source_query")
        target_table = params.get("target_table")
        mode = params.get("mode", "replace")

        if not source_query:
            return SkillResult(success=False, message="source_query is required")
        if not target_table:
            return SkillResult(success=False, message="target_table is required")

        # Validate source query is read-only
        first_word = source_query.strip().upper().split()[0] if source_query.strip() else ""
        if first_word not in ("SELECT", "WITH"):
            return SkillResult(success=False, message="source_query must be a SELECT statement")

        db = self._get_db_skill()

        # Ensure db exists and enable writes
        db._create_db({"name": db_name})
        db._enable_write({"db": db_name, "enabled": True})

        # Run the source query to get data
        source_result = db._query({"sql": source_query, "db": db_name})
        if not source_result.success:
            db._enable_write({"db": db_name, "enabled": False})
            self._record_request(False)
            self._record_job("data_transform", customer_id, f"Source query failed: {source_result.message}", False, 0)
            return SkillResult(success=False, message=f"Source query failed: {source_result.message}")

        rows = source_result.data.get("rows", [])
        columns = source_result.data.get("columns", [])

        if not rows:
            db._enable_write({"db": db_name, "enabled": False})
            self._record_request(False)
            self._record_job("data_transform", customer_id, "Source query returned 0 rows", False, 0)
            return SkillResult(success=False, message="Source query returned no rows to transform")

        # Create target table from columns
        col_defs = ", ".join([f"{c} TEXT" for c in columns])
        if mode == "replace":
            db._execute_write({"sql": f"DROP TABLE IF EXISTS {target_table}", "db": db_name})
        db._execute_write({"sql": f"CREATE TABLE IF NOT EXISTS {target_table} ({col_defs})", "db": db_name})

        # Insert rows
        rows_written = 0
        for row in rows:
            if isinstance(row, dict):
                values = [row.get(c) for c in columns]
            else:
                values = list(row)
            placeholders = ", ".join(["?" for _ in columns])
            col_names = ", ".join(columns)
            sql = f"INSERT INTO {target_table} ({col_names}) VALUES ({placeholders})"
            result = db._execute_write({"sql": sql, "db": db_name, "params": values})
            if result.success:
                rows_written += 1

        db._enable_write({"db": db_name, "enabled": False})

        if rows_written > 0:
            price = PRICING["data_transform"]
            self._record_revenue("data_transform", customer_id, price)
            self._record_request(True)
            self._record_job("data_transform", customer_id,
                             f"Transformed {rows_written} rows into {target_table}", True, price)
            return SkillResult(
                success=True,
                message=f"Transformed {rows_written} rows into '{target_table}' ({mode}). Charged ${price:.3f}.",
                data={
                    "rows_written": rows_written,
                    "source_rows": len(rows),
                    "target_table": target_table,
                    "mode": mode,
                    "charged": price,
                    "customer_id": customer_id,
                    "db_name": db_name,
                },
                revenue=price,
            )
        else:
            self._record_request(False)
            self._record_job("data_transform", customer_id, "No rows written", False, 0)
            return SkillResult(success=False, message="Transform failed: no rows could be written")

    async def _list_services(self, params: Dict) -> SkillResult:
        """List available paid data services and pricing."""
        services = [
            {
                "name": "Data Analysis",
                "action": "analyze",
                "price": PRICING["data_analysis"],
                "description": "Run analytical SQL queries on customer databases",
                "unit": "per query",
            },
            {
                "name": "Schema Design",
                "action": "design_schema",
                "price": PRICING["schema_design"],
                "description": "Create optimized database schemas from specifications",
                "unit": "per table",
            },
            {
                "name": "Data Import",
                "action": "import_data",
                "price": PRICING["data_import"],
                "description": "Import JSON/CSV data into structured database tables",
                "unit": "per 100 rows",
            },
            {
                "name": "Report Generation",
                "action": "generate_report",
                "price": PRICING["report_generation"],
                "description": "Generate formatted data reports with statistics",
                "unit": "per report",
            },
            {
                "name": "Data Transformation",
                "action": "transform_data",
                "price": PRICING["data_transform"],
                "description": "ETL-style data transformations between tables",
                "unit": "per transform",
            },
        ]
        total_rev = self._store.get("revenue", {}).get("total", 0.0)
        return SkillResult(
            success=True,
            message=f"{len(services)} paid data services available. Total revenue earned: ${total_rev:.3f}.",
            data={
                "services": services,
                "total_revenue": total_rev,
            },
        )

    async def _service_stats(self, params: Dict) -> SkillResult:
        """Get revenue and usage statistics."""
        customer_id = params.get("customer_id")
        rev = self._store.get("revenue", {"total": 0.0, "by_service": {}, "by_customer": {}})
        stats = self._store.get("stats", {"total_requests": 0, "successful_requests": 0, "failed_requests": 0})
        reports = self._store.get("reports", {})
        schemas = self._store.get("schemas_created", {})

        data = {
            "revenue": rev,
            "stats": stats,
            "reports_generated": len(reports),
            "schemas_created": len(schemas),
            "success_rate": (
                round(stats.get("successful_requests", 0) / stats["total_requests"] * 100, 1)
                if stats.get("total_requests", 0) > 0 else 0
            ),
        }

        if customer_id:
            customer_rev = rev.get("by_customer", {}).get(customer_id, 0.0)
            customer_jobs = [j for j in self._store.get("jobs", []) if j.get("customer_id") == customer_id]
            data["customer_filter"] = customer_id
            data["customer_revenue"] = customer_rev
            data["customer_jobs"] = len(customer_jobs)
            data["customer_recent_jobs"] = customer_jobs[-10:]

        return SkillResult(
            success=True,
            message=f"Revenue: ${rev.get('total', 0.0):.3f} | Requests: {stats.get('total_requests', 0)} | Success rate: {data['success_rate']}%",
            data=data,
        )
