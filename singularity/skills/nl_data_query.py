#!/usr/bin/env python3
"""
NLDataQuerySkill - Natural language to SQL query bridge for paid data services.

This is the critical revenue feature that lets external users ask plain-English
questions about their data and get SQL results without knowing SQL. It bridges
the NaturalLanguageRouter (for understanding intent) with the DatabaseRevenueBridge
(for paid query execution).

Architecture:
  User Question (plain English)
    -> NLDataQuerySkill.query()
      -> Intent classification (what type of query?)
      -> Schema discovery (what tables/columns exist?)
      -> SQL generation (translate NL to SQL)
      -> DatabaseRevenueBridge.analyze() (execute + charge)
    -> Formatted results returned to user

Revenue flow:
  Customer -> "Show total sales by region" -> NLDataQuerySkill
    -> generates SELECT SUM(amount), region FROM sales GROUP BY region
    -> DatabaseRevenueBridge.analyze() charges $0.015/query (premium over raw SQL)
    -> Returns formatted answer: "Here are your sales by region: ..."

This skill charges a PREMIUM over raw SQL queries because it provides the
natural language understanding layer. Raw SQL via DatabaseRevenueBridge costs
$0.01/query. NL queries via this skill cost $0.015/query (50% markup for the
convenience of not knowing SQL).

Pillar: Revenue Generation (primary), Self-Improvement (supporting)
- Revenue: Premium NL query service ($0.015 vs $0.01 for raw SQL)
- Self-Improvement: Learns from query patterns to improve SQL generation

No external dependencies - uses keyword/pattern matching for SQL generation.
"""

import json
import re
import uuid

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

from .base import Skill, SkillManifest, SkillAction, SkillResult


DATA_DIR = Path(__file__).parent.parent / "data"
NL_QUERY_FILE = DATA_DIR / "nl_data_query.json"
MAX_HISTORY = 500

# Premium pricing: 50% markup over raw SQL ($0.01)
NL_QUERY_PRICE = 0.015
SCHEMA_DISCOVERY_PRICE = 0.005
EXPLAIN_PRICE = 0.005

# SQL generation patterns - maps NL intents to SQL templates
AGGREGATE_WORDS = {
    "total": "SUM", "sum": "SUM", "add up": "SUM",
    "average": "AVG", "avg": "AVG", "mean": "AVG",
    "count": "COUNT", "how many": "COUNT", "number of": "COUNT",
    "maximum": "MAX", "max": "MAX", "highest": "MAX", "largest": "MAX", "biggest": "MAX",
    "minimum": "MIN", "min": "MIN", "lowest": "MIN", "smallest": "MIN",
}

ORDER_WORDS = {
    "top": "DESC", "highest": "DESC", "most": "DESC", "best": "DESC", "largest": "DESC",
    "bottom": "ASC", "lowest": "ASC", "least": "ASC", "worst": "ASC", "smallest": "ASC",
    "ascending": "ASC", "descending": "DESC",
}

LIMIT_PATTERNS = [
    (r"top\s+(\d+)", int),
    (r"first\s+(\d+)", int),
    (r"last\s+(\d+)", int),
    (r"limit\s+(\d+)", int),
    (r"(\d+)\s+(?:results?|rows?|records?|items?|entries)", int),
]

TIME_WORDS = {
    "today": "date('now')",
    "yesterday": "date('now', '-1 day')",
    "this week": "date('now', '-7 days')",
    "last week": "date('now', '-14 days')",
    "this month": "date('now', 'start of month')",
    "last month": "date('now', 'start of month', '-1 month')",
    "this year": "date('now', 'start of year')",
    "last year": "date('now', 'start of year', '-1 year')",
}

# Stop words to filter from column/table matching
STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through", "during",
    "show", "me", "get", "find", "give", "display", "list", "what",
    "which", "where", "when", "how", "all", "each", "every", "please",
    "want", "need", "help", "like", "tell", "about",
}


def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase words, filtering stop words."""
    words = re.findall(r'[a-z0-9_]+', text.lower())
    return [w for w in words if w not in STOP_WORDS and len(w) > 1]


def _load_store() -> Dict:
    try:
        if NL_QUERY_FILE.exists():
            with open(NL_QUERY_FILE, "r") as f:
                return json.load(f)
    except (json.JSONDecodeError, IOError):
        pass
    return {
        "queries": [],
        "patterns": {},
        "revenue": {"total": 0.0, "query_count": 0, "by_customer": {}},
        "learned_mappings": {},
    }


def _save_store(store: Dict):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if len(store.get("queries", [])) > MAX_HISTORY:
        store["queries"] = store["queries"][-MAX_HISTORY:]
    try:
        with open(NL_QUERY_FILE, "w") as f:
            json.dump(store, f, indent=2, default=str)
    except IOError:
        pass


class SchemaInfo:
    """Cached database schema information for SQL generation."""

    def __init__(self):
        self.tables: Dict[str, List[Dict]] = {}  # table_name -> [{name, type}]
        self.table_row_counts: Dict[str, int] = {}

    def add_table(self, name: str, columns: List[Dict], row_count: int = 0):
        self.tables[name] = columns
        self.table_row_counts[name] = row_count

    def find_table(self, query_tokens: List[str]) -> Optional[str]:
        """Find the best matching table name from query tokens."""
        best_table = None
        best_score = 0

        for table_name in self.tables:
            table_tokens = set(re.findall(r'[a-z0-9]+', table_name.lower()))
            # Check direct match
            for token in query_tokens:
                if token in table_tokens or token == table_name.lower():
                    score = 3
                    if score > best_score:
                        best_score = score
                        best_table = table_name
                # Check plural/singular match
                elif token.rstrip('s') in table_tokens or token + 's' in table_tokens:
                    score = 2
                    if score > best_score:
                        best_score = score
                        best_table = table_name
                # Check substring match
                elif any(token in t or t in token for t in table_tokens):
                    score = 1
                    if score > best_score:
                        best_score = score
                        best_table = table_name

        return best_table

    def find_columns(self, table_name: str, query_tokens: List[str]) -> List[str]:
        """Find matching columns from query tokens."""
        if table_name not in self.tables:
            return []

        matched = []
        columns = self.tables[table_name]

        for col in columns:
            col_name = col["name"].lower()
            col_tokens = set(re.findall(r'[a-z0-9]+', col_name))

            for token in query_tokens:
                if token in col_tokens or token == col_name:
                    matched.append(col["name"])
                    break
                elif token.rstrip('s') in col_tokens:
                    matched.append(col["name"])
                    break

        return matched

    def get_numeric_columns(self, table_name: str) -> List[str]:
        """Get columns likely to be numeric (for aggregation)."""
        if table_name not in self.tables:
            return []
        numeric_types = {"INTEGER", "REAL", "FLOAT", "DOUBLE", "NUMERIC", "DECIMAL", "INT", "BIGINT"}
        return [
            col["name"] for col in self.tables[table_name]
            if col.get("type", "").upper().split("(")[0].strip() in numeric_types
        ]

    def get_text_columns(self, table_name: str) -> List[str]:
        """Get columns likely to be text (for grouping/filtering)."""
        if table_name not in self.tables:
            return []
        text_types = {"TEXT", "VARCHAR", "CHAR", "STRING"}
        return [
            col["name"] for col in self.tables[table_name]
            if col.get("type", "").upper().split("(")[0].strip() in text_types
        ]

    def all_column_names(self, table_name: str) -> List[str]:
        """Get all column names for a table."""
        if table_name not in self.tables:
            return []
        return [col["name"] for col in self.tables[table_name]]


class NLDataQuerySkill(Skill):
    """
    Translates natural language questions into SQL queries and executes
    them as paid data analysis services.

    Bridges NaturalLanguageRouter intent understanding with
    DatabaseRevenueBridge paid query execution.
    """

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self._store = _load_store()
        self._db_skill = None
        self._db_bridge = None
        self._schema_cache: Dict[str, SchemaInfo] = {}

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="nl_data_query",
            name="Natural Language Data Query",
            version="1.0.0",
            category="revenue",
            description="Ask data questions in plain English - translates to SQL and executes as a paid service",
            required_credentials=[],
            actions=[
                SkillAction(
                    name="query",
                    description="Ask a question about your data in plain English",
                    parameters={
                        "customer_id": {"type": "string", "required": True, "description": "Customer identifier"},
                        "question": {"type": "string", "required": True, "description": "Your question in plain English (e.g. 'show total sales by region')"},
                        "db_name": {"type": "string", "required": False, "description": "Database name (default: 'default')"},
                    },
                    estimated_cost=NL_QUERY_PRICE,
                    estimated_duration_seconds=5,
                    success_probability=0.8,
                ),
                SkillAction(
                    name="explain",
                    description="Show the SQL that would be generated for a question (without executing)",
                    parameters={
                        "customer_id": {"type": "string", "required": True, "description": "Customer identifier"},
                        "question": {"type": "string", "required": True, "description": "Your question in plain English"},
                        "db_name": {"type": "string", "required": False, "description": "Database name (default: 'default')"},
                    },
                    estimated_cost=EXPLAIN_PRICE,
                    estimated_duration_seconds=2,
                    success_probability=0.85,
                ),
                SkillAction(
                    name="discover",
                    description="Discover what tables and columns are available in a database",
                    parameters={
                        "customer_id": {"type": "string", "required": True, "description": "Customer identifier"},
                        "db_name": {"type": "string", "required": False, "description": "Database name (default: 'default')"},
                    },
                    estimated_cost=SCHEMA_DISCOVERY_PRICE,
                    estimated_duration_seconds=2,
                    success_probability=0.95,
                ),
                SkillAction(
                    name="suggest",
                    description="Get suggested questions you can ask about a database",
                    parameters={
                        "db_name": {"type": "string", "required": False, "description": "Database name (default: 'default')"},
                    },
                    estimated_cost=0,
                    estimated_duration_seconds=2,
                    success_probability=0.9,
                ),
                SkillAction(
                    name="teach",
                    description="Teach the system a mapping from a phrase to a SQL pattern",
                    parameters={
                        "phrase": {"type": "string", "required": True, "description": "Natural language phrase"},
                        "sql_template": {"type": "string", "required": True, "description": "SQL template (use {table}, {column} placeholders)"},
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="stats",
                    description="Get NL query service statistics and revenue",
                    parameters={
                        "customer_id": {"type": "string", "required": False, "description": "Filter by customer"},
                    },
                    estimated_cost=0,
                ),
            ],
        )

    async def initialize(self) -> bool:
        self._store = _load_store()
        return True

    def _get_db_skill(self):
        """Lazy-load DatabaseSkill."""
        if self._db_skill is None:
            from .database import DatabaseSkill
            self._db_skill = DatabaseSkill()
        return self._db_skill

    def _get_schema(self, db_name: str) -> SchemaInfo:
        """Get or build schema info for a database."""
        if db_name in self._schema_cache:
            return self._schema_cache[db_name]

        schema = SchemaInfo()
        db = self._get_db_skill()

        # Get table list
        try:
            result = db._query({"sql": "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'", "db": db_name})
            if result.success and result.data:
                rows = result.data.get("rows", [])
                for row in rows:
                    table_name = row[0] if isinstance(row, (list, tuple)) else row.get("name", "")
                    if not table_name:
                        continue

                    # Get columns for each table
                    col_result = db._query({"sql": f"PRAGMA table_info({table_name})", "db": db_name})
                    if col_result.success and col_result.data:
                        columns = []
                        for col_row in col_result.data.get("rows", []):
                            if isinstance(col_row, (list, tuple)):
                                columns.append({"name": col_row[1], "type": col_row[2] or "TEXT"})
                            elif isinstance(col_row, dict):
                                columns.append({"name": col_row.get("name", ""), "type": col_row.get("type", "TEXT")})

                        # Get row count
                        count_result = db._query({"sql": f"SELECT COUNT(*) FROM {table_name}", "db": db_name})
                        row_count = 0
                        if count_result.success and count_result.data:
                            count_rows = count_result.data.get("rows", [])
                            if count_rows:
                                r = count_rows[0]
                                row_count = r[0] if isinstance(r, (list, tuple)) else 0

                        schema.add_table(table_name, columns, row_count)
        except Exception:
            pass

        self._schema_cache[db_name] = schema
        return schema

    def _detect_aggregate(self, question_lower: str) -> Optional[Tuple[str, str]]:
        """Detect if the question asks for an aggregate function.

        Returns (agg_func, trigger_word) or None.
        """
        for trigger, func in AGGREGATE_WORDS.items():
            if trigger in question_lower:
                return (func, trigger)
        return None

    def _detect_order(self, question_lower: str) -> Optional[str]:
        """Detect sort order from question."""
        for word, direction in ORDER_WORDS.items():
            if word in question_lower:
                return direction
        return None

    def _detect_limit(self, question: str) -> Optional[int]:
        """Detect row limit from question."""
        for pattern, converter in LIMIT_PATTERNS:
            match = re.search(pattern, question.lower())
            if match:
                return converter(match.group(1))
        return None

    def _detect_time_filter(self, question_lower: str) -> Optional[Tuple[str, str]]:
        """Detect time-based filter from question.

        Returns (time_phrase, sql_date_expr) or None.
        """
        for phrase, sql_expr in TIME_WORDS.items():
            if phrase in question_lower:
                return (phrase, sql_expr)
        return None

    def _detect_where_conditions(self, question_lower: str, schema: SchemaInfo, table: str) -> List[str]:
        """Detect WHERE conditions from the question."""
        conditions = []

        # Time filter
        time_filter = self._detect_time_filter(question_lower)
        if time_filter:
            phrase, sql_expr = time_filter
            # Find a date column
            all_cols = schema.all_column_names(table)
            date_cols = [c for c in all_cols if any(d in c.lower() for d in ["date", "time", "created", "updated", "timestamp"])]
            if date_cols:
                conditions.append(f"{date_cols[0]} >= {sql_expr}")

        # Equality filters: "where X is Y" or "for X = Y"
        eq_patterns = [
            r"(?:where|for|with)\s+(\w+)\s+(?:is|=|equals?)\s+['\"]?(\w+)['\"]?",
            r"(\w+)\s*=\s*['\"]?(\w+)['\"]?",
        ]
        all_cols_lower = {c.lower(): c for c in schema.all_column_names(table)}
        for pat in eq_patterns:
            for match in re.finditer(pat, question_lower):
                col_candidate = match.group(1).lower()
                value = match.group(2)
                if col_candidate in all_cols_lower:
                    real_col = all_cols_lower[col_candidate]
                    conditions.append(f"{real_col} = '{value}'")

        return conditions

    def _generate_sql(self, question: str, schema: SchemaInfo) -> Optional[Dict[str, Any]]:
        """
        Generate SQL from a natural language question.

        Returns dict with 'sql', 'explanation', 'confidence' or None if can't generate.
        """
        question_lower = question.lower().strip()
        tokens = _tokenize(question)

        if not tokens:
            return None

        # Check learned mappings first
        for phrase, template in self._store.get("learned_mappings", {}).items():
            if phrase.lower() in question_lower:
                # Try to fill template
                table = schema.find_table(tokens)
                if table:
                    sql = template.replace("{table}", table)
                    cols = schema.find_columns(table, tokens)
                    if cols:
                        sql = sql.replace("{column}", cols[0])
                    return {
                        "sql": sql,
                        "explanation": f"Used learned pattern for '{phrase}'",
                        "confidence": 0.9,
                    }

        # Find the target table
        table = schema.find_table(tokens)
        if not table:
            return None

        # Detect intent
        aggregate = self._detect_aggregate(question_lower)
        order = self._detect_order(question_lower)
        limit = self._detect_limit(question)
        where_conditions = self._detect_where_conditions(question_lower, schema, table)

        # Find relevant columns
        matched_cols = schema.find_columns(table, tokens)
        numeric_cols = schema.get_numeric_columns(table)
        text_cols = schema.get_text_columns(table)
        all_cols = schema.all_column_names(table)

        # Build SQL based on intent
        if aggregate:
            agg_func, trigger = aggregate
            return self._build_aggregate_query(
                table, agg_func, trigger, matched_cols, numeric_cols,
                text_cols, all_cols, tokens, where_conditions, order, limit, question_lower
            )
        elif any(w in question_lower for w in ["list", "show", "get", "find", "display", "all"]):
            return self._build_select_query(
                table, matched_cols, all_cols, where_conditions, order, limit, question_lower
            )
        else:
            # Default: select all with any detected filters
            return self._build_select_query(
                table, matched_cols, all_cols, where_conditions, order, limit, question_lower
            )

    def _build_aggregate_query(
        self, table: str, agg_func: str, trigger: str,
        matched_cols: List[str], numeric_cols: List[str],
        text_cols: List[str], all_cols: List[str],
        tokens: List[str], conditions: List[str],
        order: Optional[str], limit: Optional[int],
        question_lower: str,
    ) -> Dict[str, Any]:
        """Build an aggregate SQL query."""
        # Determine which column to aggregate
        agg_col = None
        for col in matched_cols:
            if col in numeric_cols:
                agg_col = col
                break
        if not agg_col and numeric_cols:
            agg_col = numeric_cols[0]
        if not agg_col:
            agg_col = "*"

        # Determine GROUP BY column
        group_col = None
        group_indicators = ["by", "per", "each", "grouped", "group"]
        for indicator in group_indicators:
            if indicator in question_lower:
                # Find the word after "by"
                idx = question_lower.find(indicator)
                after = question_lower[idx + len(indicator):].strip()
                after_tokens = _tokenize(after)
                if after_tokens:
                    # Try to match to a column
                    for t in after_tokens:
                        for col in all_cols:
                            if t in col.lower() or col.lower() in t or t.rstrip('s') == col.lower() or t == col.lower().rstrip('s'):
                                group_col = col
                                break
                        if group_col:
                            break
                break

        if not group_col and text_cols:
            # If "by" not found but text cols exist, try matched text cols
            for col in matched_cols:
                if col in text_cols:
                    group_col = col
                    break

        # Build SQL
        if group_col:
            agg_expr = f"{agg_func}({agg_col})" if agg_col != "*" else f"{agg_func}(*)"
            select = f"SELECT {group_col}, {agg_expr} AS {agg_func.lower()}_{agg_col.replace('*', 'all')}"
            sql = f"{select} FROM {table}"
            if conditions:
                sql += f" WHERE {' AND '.join(conditions)}"
            sql += f" GROUP BY {group_col}"
            if order:
                sql += f" ORDER BY {agg_expr} {order}"
            if limit:
                sql += f" LIMIT {limit}"
            explanation = f"{agg_func} of {agg_col} grouped by {group_col}"
        else:
            agg_expr = f"{agg_func}({agg_col})" if agg_col != "*" else f"{agg_func}(*)"
            sql = f"SELECT {agg_expr} AS result FROM {table}"
            if conditions:
                sql += f" WHERE {' AND '.join(conditions)}"
            explanation = f"{agg_func} of {agg_col}"

        return {
            "sql": sql,
            "explanation": explanation,
            "confidence": 0.75 if group_col else 0.7,
        }

    def _build_select_query(
        self, table: str, matched_cols: List[str],
        all_cols: List[str], conditions: List[str],
        order: Optional[str], limit: Optional[int],
        question_lower: str,
    ) -> Dict[str, Any]:
        """Build a SELECT query."""
        # Determine columns
        if matched_cols:
            select_cols = ", ".join(matched_cols)
        else:
            select_cols = "*"

        sql = f"SELECT {select_cols} FROM {table}"

        if conditions:
            sql += f" WHERE {' AND '.join(conditions)}"

        if order and matched_cols:
            sql += f" ORDER BY {matched_cols[-1]} {order}"

        if limit:
            sql += f" LIMIT {limit}"
        elif not limit and "all" not in question_lower:
            sql += " LIMIT 100"  # Default safety limit

        explanation = f"Selecting {select_cols} from {table}"
        if conditions:
            explanation += f" where {', '.join(conditions)}"

        return {
            "sql": sql,
            "explanation": explanation,
            "confidence": 0.7 if matched_cols else 0.5,
        }

    def _record_revenue(self, customer_id: str, amount: float):
        """Record revenue from an NL query."""
        rev = self._store.setdefault("revenue", {"total": 0.0, "query_count": 0, "by_customer": {}})
        rev["total"] = rev.get("total", 0.0) + amount
        rev["query_count"] = rev.get("query_count", 0) + 1
        rev["by_customer"][customer_id] = rev.get("by_customer", {}).get(customer_id, 0.0) + amount

    def _record_query(self, customer_id: str, question: str, sql: str, success: bool, charged: float):
        """Record a query in history."""
        self._store.setdefault("queries", []).append({
            "id": str(uuid.uuid4())[:8],
            "timestamp": _now_iso(),
            "customer_id": customer_id,
            "question": question[:500],
            "sql": sql[:500],
            "success": success,
            "charged": charged,
        })

    def _customers(self) -> List[Dict]:
        """Return customer data for revenue dashboard integration."""
        rev = self._store.get("revenue", {})
        by_customer = rev.get("by_customer", {})
        return [
            {"customer_id": cid, "revenue": amount, "source": "nl_data_query"}
            for cid, amount in by_customer.items()
        ]

    async def execute(self, action: str, params: Dict) -> SkillResult:
        """Execute an NL data query action."""
        actions = {
            "query": self._query,
            "explain": self._explain,
            "discover": self._discover,
            "suggest": self._suggest,
            "teach": self._teach,
            "stats": self._stats,
        }
        handler = actions.get(action)
        if not handler:
            return SkillResult(
                success=False,
                message=f"Unknown action: {action}. Available: {list(actions.keys())}",
            )
        try:
            result = await handler(params)
            _save_store(self._store)
            return result
        except Exception as e:
            _save_store(self._store)
            return SkillResult(success=False, message=f"Error in {action}: {str(e)}")

    async def _query(self, params: Dict) -> SkillResult:
        """Execute a natural language data query."""
        customer_id = params.get("customer_id", "anonymous")
        question = params.get("question", "").strip()
        db_name = params.get("db_name", "default")

        if not question:
            return SkillResult(success=False, message="'question' parameter is required")

        # Get schema
        schema = self._get_schema(db_name)
        if not schema.tables:
            self._record_query(customer_id, question, "", False, 0)
            return SkillResult(
                success=False,
                message=f"No tables found in database '{db_name}'. Use 'discover' to check available databases.",
                data={"question": question, "db_name": db_name},
            )

        # Generate SQL
        result = self._generate_sql(question, schema)
        if not result:
            self._record_query(customer_id, question, "", False, 0)
            return SkillResult(
                success=False,
                message="Could not understand the question. Try rephrasing or use 'suggest' to see example questions.",
                data={
                    "question": question,
                    "available_tables": list(schema.tables.keys()),
                },
            )

        sql = result["sql"]
        explanation = result["explanation"]
        confidence = result["confidence"]

        # Execute via database skill
        db = self._get_db_skill()
        exec_result = db._query({"sql": sql, "db": db_name})

        if exec_result.success:
            price = NL_QUERY_PRICE
            self._record_revenue(customer_id, price)
            self._record_query(customer_id, question, sql, True, price)

            return SkillResult(
                success=True,
                message=f"Query executed. {exec_result.data.get('row_count', 0)} rows returned. Charged ${price}.",
                data={
                    "results": exec_result.data,
                    "sql_generated": sql,
                    "explanation": explanation,
                    "confidence": confidence,
                    "charged": price,
                    "customer_id": customer_id,
                },
                revenue=price,
            )
        else:
            self._record_query(customer_id, question, sql, False, 0)
            return SkillResult(
                success=False,
                message=f"Generated SQL failed: {exec_result.message}. SQL was: {sql}",
                data={
                    "sql_generated": sql,
                    "explanation": explanation,
                    "error": exec_result.message,
                    "question": question,
                },
            )

    async def _explain(self, params: Dict) -> SkillResult:
        """Show the SQL that would be generated without executing."""
        customer_id = params.get("customer_id", "anonymous")
        question = params.get("question", "").strip()
        db_name = params.get("db_name", "default")

        if not question:
            return SkillResult(success=False, message="'question' parameter is required")

        schema = self._get_schema(db_name)
        if not schema.tables:
            return SkillResult(
                success=False,
                message=f"No tables found in database '{db_name}'.",
            )

        result = self._generate_sql(question, schema)
        if not result:
            return SkillResult(
                success=False,
                message="Could not understand the question. Try rephrasing.",
                data={"question": question, "available_tables": list(schema.tables.keys())},
            )

        price = EXPLAIN_PRICE
        self._record_revenue(customer_id, price)

        return SkillResult(
            success=True,
            message=f"SQL generated (not executed). Charged ${price}.",
            data={
                "sql": result["sql"],
                "explanation": result["explanation"],
                "confidence": result["confidence"],
                "charged": price,
                "customer_id": customer_id,
            },
            revenue=price,
        )

    async def _discover(self, params: Dict) -> SkillResult:
        """Discover available tables and columns in a database."""
        customer_id = params.get("customer_id", "anonymous")
        db_name = params.get("db_name", "default")

        # Force refresh schema cache
        self._schema_cache.pop(db_name, None)
        schema = self._get_schema(db_name)

        if not schema.tables:
            return SkillResult(
                success=True,
                message=f"No tables found in database '{db_name}'.",
                data={"db_name": db_name, "tables": {}},
            )

        tables_info = {}
        for table_name, columns in schema.tables.items():
            tables_info[table_name] = {
                "columns": columns,
                "row_count": schema.table_row_counts.get(table_name, 0),
            }

        price = SCHEMA_DISCOVERY_PRICE
        self._record_revenue(customer_id, price)

        return SkillResult(
            success=True,
            message=f"Found {len(tables_info)} tables in '{db_name}'. Charged ${price}.",
            data={
                "db_name": db_name,
                "tables": tables_info,
                "charged": price,
                "customer_id": customer_id,
            },
            revenue=price,
        )

    async def _suggest(self, params: Dict) -> SkillResult:
        """Suggest questions based on available schema."""
        db_name = params.get("db_name", "default")

        schema = self._get_schema(db_name)
        if not schema.tables:
            return SkillResult(
                success=True,
                message=f"No tables in '{db_name}' to suggest queries for.",
                data={"suggestions": []},
            )

        suggestions = []
        for table_name, columns in schema.tables.items():
            numeric_cols = schema.get_numeric_columns(table_name)
            text_cols = schema.get_text_columns(table_name)

            # Basic suggestions
            suggestions.append(f"Show all {table_name}")
            suggestions.append(f"How many {table_name} are there?")

            if numeric_cols:
                nc = numeric_cols[0]
                suggestions.append(f"What is the total {nc} in {table_name}?")
                suggestions.append(f"What is the average {nc} in {table_name}?")
                if text_cols:
                    tc = text_cols[0]
                    suggestions.append(f"Show total {nc} by {tc} in {table_name}")
                    suggestions.append(f"Top 10 {table_name} by highest {nc}")

            if text_cols:
                suggestions.append(f"Count {table_name} by {text_cols[0]}")

        return SkillResult(
            success=True,
            message=f"Generated {len(suggestions)} suggested questions.",
            data={
                "suggestions": suggestions[:20],
                "db_name": db_name,
                "tables_available": list(schema.tables.keys()),
            },
        )

    async def _teach(self, params: Dict) -> SkillResult:
        """Teach a new NL-to-SQL mapping."""
        phrase = params.get("phrase", "").strip()
        sql_template = params.get("sql_template", "").strip()

        if not phrase or not sql_template:
            return SkillResult(success=False, message="Both 'phrase' and 'sql_template' are required")

        # Validate SQL template is read-only
        first_word = sql_template.strip().upper().split()[0] if sql_template.strip().split() else ""
        if first_word not in ("SELECT", "WITH", "EXPLAIN", "PRAGMA"):
            return SkillResult(success=False, message="Only SELECT/WITH/EXPLAIN/PRAGMA templates allowed")

        self._store.setdefault("learned_mappings", {})[phrase.lower()] = sql_template

        return SkillResult(
            success=True,
            message=f"Learned mapping: '{phrase}' -> SQL template",
            data={"phrase": phrase, "sql_template": sql_template},
        )

    async def _stats(self, params: Dict) -> SkillResult:
        """Get NL query service statistics."""
        customer_id = params.get("customer_id")

        rev = self._store.get("revenue", {"total": 0.0, "query_count": 0, "by_customer": {}})
        queries = self._store.get("queries", [])

        stats = {
            "total_revenue": rev.get("total", 0.0),
            "total_queries": rev.get("query_count", 0),
            "success_rate": 0.0,
            "avg_revenue_per_query": 0.0,
            "learned_mappings_count": len(self._store.get("learned_mappings", {})),
            "pricing": {
                "nl_query": NL_QUERY_PRICE,
                "explain": EXPLAIN_PRICE,
                "discover": SCHEMA_DISCOVERY_PRICE,
            },
        }

        if queries:
            successful = sum(1 for q in queries if q.get("success"))
            stats["success_rate"] = successful / len(queries) if queries else 0
            stats["total_queries_executed"] = len(queries)

        if rev.get("query_count", 0) > 0:
            stats["avg_revenue_per_query"] = rev["total"] / rev["query_count"]

        if customer_id:
            customer_rev = rev.get("by_customer", {}).get(customer_id, 0.0)
            customer_queries = [q for q in queries if q.get("customer_id") == customer_id]
            stats["customer"] = {
                "customer_id": customer_id,
                "revenue": customer_rev,
                "queries": len(customer_queries),
            }

        return SkillResult(
            success=True,
            message=f"NL Query stats: {stats['total_queries']} queries, ${stats['total_revenue']:.3f} revenue",
            data=stats,
        )
