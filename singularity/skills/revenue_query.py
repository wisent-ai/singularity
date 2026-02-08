#!/usr/bin/env python3
"""
RevenueQuerySkill - Natural language interface for revenue data queries.

This is the missing bridge between plain-English questions and the agent's
revenue data infrastructure. External users (via ServiceAPI) and the agent
itself can ask questions like:

  - "What was total revenue last week?"
  - "Which revenue source earns the most?"
  - "Am I profitable?"
  - "Show me customer breakdown"
  - "What's the revenue forecast?"
  - "Compare revenue sources"

Instead of requiring callers to know exact skill IDs and action names
(revenue_analytics_dashboard:overview, revenue_analytics_dashboard:by_source, etc.),
this skill parses the intent from natural language, routes to the right
revenue data source, and returns a formatted answer.

Architecture:
1. PARSE INTENT: Classify the query into a revenue query type using keyword matching
2. ROUTE: Map the intent to the appropriate revenue skill + action
3. EXECUTE: Call the target skill via SkillContext
4. FORMAT: Transform raw skill output into a human-readable answer
5. LEARN: Track which query patterns map to which intents (persistent)

Query types supported:
  - overview: Total revenue, request counts, active sources
  - by_source: Revenue breakdown per source
  - profitability: Revenue vs costs, margin analysis
  - customers: Customer breakdown and analytics
  - trends: Revenue trends over time
  - forecast: Future revenue projections
  - recommendations: AI-generated revenue improvement suggestions
  - status: System health of revenue pipeline

Pillar: Revenue (primary) - enables external users to query revenue in plain English
        Goal Setting (supporting) - agent can self-query revenue for strategic decisions
"""

import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

from .base import Skill, SkillResult, SkillManifest, SkillAction

DATA_DIR = Path(__file__).parent.parent / "data"
QUERY_STATE_FILE = DATA_DIR / "revenue_query.json"
MAX_QUERY_HISTORY = 500


def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


# Intent patterns: each intent has a list of keyword sets.
# A query matches an intent if it contains keywords from any set.
INTENT_PATTERNS = {
    "overview": [
        ["total", "revenue"],
        ["how", "much"],
        ["revenue", "summary"],
        ["overall"],
        ["dashboard"],
        ["earnings"],
        ["how", "doing"],
        ["revenue", "status"],
    ],
    "by_source": [
        ["by", "source"],
        ["breakdown"],
        ["which", "source"],
        ["per", "source"],
        ["compare", "source"],
        ["best", "source"],
        ["top", "source"],
        ["most", "revenue"],
        ["highest", "earning"],
        ["source", "comparison"],
    ],
    "profitability": [
        ["profit"],
        ["profitable"],
        ["margin"],
        ["cost", "revenue"],
        ["break", "even"],
        ["breakeven"],
        ["roi"],
        ["return", "investment"],
        ["losing", "money"],
        ["making", "money"],
    ],
    "customers": [
        ["customer"],
        ["client"],
        ["who", "paying"],
        ["user", "revenue"],
        ["buyer"],
        ["subscriber"],
    ],
    "trends": [
        ["trend"],
        ["over", "time"],
        ["last", "week"],
        ["last", "month"],
        ["growth"],
        ["decline"],
        ["increasing"],
        ["decreasing"],
        ["history"],
        ["historical"],
    ],
    "forecast": [
        ["forecast"],
        ["predict"],
        ["projection"],
        ["expect"],
        ["future"],
        ["next", "week"],
        ["next", "month"],
        ["estimate"],
        ["will", "earn"],
    ],
    "recommendations": [
        ["recommend"],
        ["suggestion"],
        ["improve", "revenue"],
        ["increase", "revenue"],
        ["how", "earn", "more"],
        ["optimize", "revenue"],
        ["boost", "revenue"],
        ["grow", "revenue"],
    ],
    "status": [
        ["pipeline", "status"],
        ["system", "health"],
        ["revenue", "system"],
        ["data", "fresh"],
        ["sync", "status"],
        ["metrics", "health"],
    ],
}

# Map intents to revenue skill actions
INTENT_TO_ACTION = {
    "overview": ("revenue_analytics_dashboard", "overview"),
    "by_source": ("revenue_analytics_dashboard", "by_source"),
    "profitability": ("revenue_analytics_dashboard", "profitability"),
    "customers": ("revenue_analytics_dashboard", "customers"),
    "trends": ("revenue_analytics_dashboard", "trends"),
    "forecast": ("revenue_analytics_dashboard", "forecast"),
    "recommendations": ("revenue_analytics_dashboard", "recommendations"),
    "status": ("revenue_observability_bridge", "status"),
}

# Human-readable intent descriptions
INTENT_DESCRIPTIONS = {
    "overview": "Revenue overview and totals",
    "by_source": "Revenue breakdown by source",
    "profitability": "Profitability and margin analysis",
    "customers": "Customer analytics and breakdown",
    "trends": "Revenue trends over time",
    "forecast": "Revenue forecast and projections",
    "recommendations": "Revenue improvement recommendations",
    "status": "Revenue pipeline health status",
}


def _tokenize(text: str) -> List[str]:
    """Lowercase and split text into tokens."""
    return re.findall(r'[a-z0-9]+', text.lower())


def _match_score(query_tokens: List[str], pattern: List[str]) -> float:
    """Score how well query tokens match a keyword pattern (0.0 to 1.0)."""
    if not pattern:
        return 0.0
    matched = sum(1 for kw in pattern if kw in query_tokens)
    return matched / len(pattern)


class RevenueQuerySkill(Skill):
    """
    Natural language interface for querying revenue data.

    Parses plain-English questions about revenue, routes to the appropriate
    revenue skill, and returns human-readable answers. Learns from query
    patterns over time to improve intent classification.
    """

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self._state: Optional[Dict] = None

    def _ensure_state(self):
        """Load or initialize state."""
        if self._state is not None:
            return
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        if QUERY_STATE_FILE.exists():
            try:
                self._state = json.loads(QUERY_STATE_FILE.read_text())
            except (json.JSONDecodeError, OSError):
                self._state = self._default_state()
        else:
            self._state = self._default_state()

    def _default_state(self) -> Dict:
        return {
            "query_history": [],
            "intent_overrides": {},  # user corrections: query_hash -> intent
            "stats": {
                "total_queries": 0,
                "by_intent": {},
                "successful_queries": 0,
                "failed_queries": 0,
                "ambiguous_queries": 0,
            },
            "created_at": _now_iso(),
        }

    def _save_state(self):
        """Persist state to disk."""
        if self._state is None:
            return
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        QUERY_STATE_FILE.write_text(json.dumps(self._state, indent=2))

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="revenue_query",
            name="Revenue Query (Natural Language)",
            version="1.0.0",
            category="revenue",
            description="Ask revenue questions in plain English. Routes to the right revenue data source automatically.",
            actions=[
                SkillAction(
                    name="ask",
                    description="Ask a revenue question in natural language",
                    parameters={
                        "query": "str - The question (e.g., 'What was total revenue?')",
                    },
                ),
                SkillAction(
                    name="classify",
                    description="Classify a query's intent without executing it",
                    parameters={
                        "query": "str - The question to classify",
                    },
                ),
                SkillAction(
                    name="correct",
                    description="Correct a misclassified query to teach the system",
                    parameters={
                        "query": "str - The original query",
                        "intent": "str - The correct intent (overview/by_source/profitability/customers/trends/forecast/recommendations/status)",
                    },
                ),
                SkillAction(
                    name="examples",
                    description="Show example queries for each intent",
                    parameters={},
                ),
                SkillAction(
                    name="stats",
                    description="View query statistics and classification accuracy",
                    parameters={},
                ),
                SkillAction(
                    name="history",
                    description="View recent query history",
                    parameters={
                        "limit": "int - Number of recent queries (default 10)",
                    },
                ),
            ],
            required_credentials=[],
        )

    async def execute(self, action: str, params: Dict[str, Any]) -> SkillResult:
        """Route to the appropriate action handler."""
        self._ensure_state()

        handlers = {
            "ask": self._ask,
            "classify": self._classify,
            "correct": self._correct,
            "examples": self._examples,
            "stats": self._stats,
            "history": self._history,
        }

        handler = handlers.get(action)
        if not handler:
            return SkillResult(
                success=False,
                message=f"Unknown action: {action}. Available: {list(handlers.keys())}",
            )

        result = await handler(params)
        self._save_state()
        return result

    def _classify_intent(self, query: str) -> Tuple[str, float, List[Tuple[str, float]]]:
        """
        Classify a query's intent.

        Returns:
            (best_intent, best_score, all_scored_intents)
        """
        tokens = _tokenize(query)

        # Check for user corrections first
        query_key = " ".join(sorted(set(tokens)))
        if query_key in self._state.get("intent_overrides", {}):
            override = self._state["intent_overrides"][query_key]
            return override, 1.0, [(override, 1.0)]

        # Score each intent
        scored: List[Tuple[str, float]] = []
        for intent, patterns in INTENT_PATTERNS.items():
            best_pattern_score = 0.0
            for pattern in patterns:
                score = _match_score(tokens, pattern)
                if score > best_pattern_score:
                    best_pattern_score = score
            if best_pattern_score > 0:
                scored.append((intent, best_pattern_score))

        scored.sort(key=lambda x: x[1], reverse=True)

        if not scored:
            return "overview", 0.0, scored  # default to overview

        return scored[0][0], scored[0][1], scored

    async def _ask(self, params: Dict) -> SkillResult:
        """Ask a revenue question in natural language."""
        query = params.get("query", "").strip()
        if not query:
            return SkillResult(
                success=False,
                message="Please provide a question. Example: 'What was total revenue?'",
            )

        # Classify intent
        intent, confidence, all_scores = self._classify_intent(query)

        # Record the query
        query_record = {
            "query": query,
            "intent": intent,
            "confidence": round(confidence, 3),
            "timestamp": _now_iso(),
            "success": False,
        }

        # Update stats
        self._state["stats"]["total_queries"] += 1
        self._state["stats"]["by_intent"][intent] = (
            self._state["stats"]["by_intent"].get(intent, 0) + 1
        )

        # Check if confidence is too low
        if confidence < 0.5:
            self._state["stats"]["ambiguous_queries"] += 1
            query_record["success"] = False
            self._state["query_history"].append(query_record)
            self._state["query_history"] = self._state["query_history"][-MAX_QUERY_HISTORY:]

            alternatives = [
                f"  - {INTENT_DESCRIPTIONS.get(s[0], s[0])} (score: {s[1]:.0%})"
                for s in all_scores[:4]
            ]
            alt_text = "\n".join(alternatives) if alternatives else "  (no close matches)"

            return SkillResult(
                success=True,
                message=(
                    f"I'm not sure what you're asking about. Best guess: {INTENT_DESCRIPTIONS.get(intent, intent)}\n\n"
                    f"Possible interpretations:\n{alt_text}\n\n"
                    f"Try being more specific, or use 'examples' to see sample queries."
                ),
                data={
                    "intent": intent,
                    "confidence": confidence,
                    "alternatives": all_scores[:4],
                    "ambiguous": True,
                },
            )

        # Route to the target skill
        skill_id, action = INTENT_TO_ACTION.get(intent, ("revenue_analytics_dashboard", "overview"))

        # Try to execute via skill context
        if self.context:
            try:
                result = await self.context.execute_skill(skill_id, action, params.get("extra_params", {}))
                if result.success:
                    self._state["stats"]["successful_queries"] += 1
                    query_record["success"] = True
                    self._state["query_history"].append(query_record)
                    self._state["query_history"] = self._state["query_history"][-MAX_QUERY_HISTORY:]

                    # Format the response
                    formatted = self._format_response(query, intent, confidence, result)
                    return formatted
                else:
                    self._state["stats"]["failed_queries"] += 1
                    query_record["success"] = False
                    query_record["error"] = result.message
                    self._state["query_history"].append(query_record)
                    self._state["query_history"] = self._state["query_history"][-MAX_QUERY_HISTORY:]

                    return SkillResult(
                        success=False,
                        message=f"Query understood ({INTENT_DESCRIPTIONS.get(intent, intent)}) but data fetch failed: {result.message}",
                        data={"intent": intent, "confidence": confidence, "error": result.message},
                    )
            except Exception as e:
                self._state["stats"]["failed_queries"] += 1
                query_record["success"] = False
                query_record["error"] = str(e)
                self._state["query_history"].append(query_record)
                self._state["query_history"] = self._state["query_history"][-MAX_QUERY_HISTORY:]

                return SkillResult(
                    success=False,
                    message=f"Query understood ({INTENT_DESCRIPTIONS.get(intent, intent)}) but execution failed: {e}",
                    data={"intent": intent, "confidence": confidence, "error": str(e)},
                )
        else:
            # No skill context - return classification only
            query_record["success"] = True
            self._state["stats"]["successful_queries"] += 1
            self._state["query_history"].append(query_record)
            self._state["query_history"] = self._state["query_history"][-MAX_QUERY_HISTORY:]

            return SkillResult(
                success=True,
                message=(
                    f"Intent: {INTENT_DESCRIPTIONS.get(intent, intent)} (confidence: {confidence:.0%})\n"
                    f"Would route to: {skill_id}:{action}\n"
                    f"(No skill context available for execution)"
                ),
                data={
                    "intent": intent,
                    "confidence": confidence,
                    "target_skill": skill_id,
                    "target_action": action,
                    "executed": False,
                },
            )

    def _format_response(self, query: str, intent: str, confidence: float, result: SkillResult) -> SkillResult:
        """Format raw skill result into a human-readable response."""
        data = result.data or {}

        # Build header
        header = f"Revenue Query: {INTENT_DESCRIPTIONS.get(intent, intent)}"

        # Format based on intent type
        body_lines = []

        if intent == "overview":
            total = data.get("total_revenue", data.get("total", "N/A"))
            requests = data.get("total_requests", "N/A")
            sources = data.get("active_sources", "N/A")
            body_lines = [
                f"Total Revenue: ${total}" if isinstance(total, (int, float)) else f"Total Revenue: {total}",
                f"Total Requests: {requests}",
                f"Active Sources: {sources}",
            ]

        elif intent == "by_source":
            sources = data.get("sources", data.get("by_source", {}))
            if isinstance(sources, dict):
                for name, amount in sorted(sources.items(), key=lambda x: x[1] if isinstance(x[1], (int, float)) else 0, reverse=True):
                    if isinstance(amount, (int, float)):
                        body_lines.append(f"  {name}: ${amount:.4f}")
                    else:
                        body_lines.append(f"  {name}: {amount}")
            elif isinstance(sources, list):
                for item in sources:
                    if isinstance(item, dict):
                        name = item.get("source", item.get("name", "unknown"))
                        amt = item.get("revenue", item.get("amount", "N/A"))
                        body_lines.append(f"  {name}: ${amt}" if isinstance(amt, (int, float)) else f"  {name}: {amt}")

        elif intent == "profitability":
            revenue = data.get("total_revenue", data.get("revenue", "N/A"))
            costs = data.get("total_costs", data.get("costs", "N/A"))
            margin = data.get("margin", data.get("profit_margin", "N/A"))
            profitable = data.get("profitable", None)
            body_lines = [
                f"Revenue: ${revenue}" if isinstance(revenue, (int, float)) else f"Revenue: {revenue}",
                f"Costs: ${costs}" if isinstance(costs, (int, float)) else f"Costs: {costs}",
                f"Margin: {margin}{'%' if isinstance(margin, (int, float)) else ''}",
            ]
            if profitable is not None:
                body_lines.append(f"Profitable: {'Yes' if profitable else 'No'}")

        elif intent == "forecast":
            projected = data.get("projected_revenue", data.get("forecast", "N/A"))
            period = data.get("period", data.get("forecast_period", "N/A"))
            body_lines = [
                f"Forecast period: {period}",
                f"Projected revenue: ${projected}" if isinstance(projected, (int, float)) else f"Projected: {projected}",
            ]

        # Default: just show the raw data keys
        if not body_lines and data:
            for key, value in list(data.items())[:10]:
                if isinstance(value, (int, float)):
                    body_lines.append(f"  {key}: {value}")
                elif isinstance(value, str) and len(value) < 100:
                    body_lines.append(f"  {key}: {value}")
                elif isinstance(value, (list, dict)):
                    body_lines.append(f"  {key}: ({type(value).__name__}, {len(value)} items)")

        body = "\n".join(body_lines) if body_lines else result.message

        return SkillResult(
            success=True,
            message=f"{header}\n{'=' * len(header)}\n{body}",
            data={
                "intent": intent,
                "confidence": confidence,
                "raw_data": data,
                "formatted": True,
            },
        )

    async def _classify(self, params: Dict) -> SkillResult:
        """Classify a query's intent without executing it."""
        query = params.get("query", "").strip()
        if not query:
            return SkillResult(success=False, message="Provide a 'query' to classify.")

        intent, confidence, all_scores = self._classify_intent(query)
        skill_id, action = INTENT_TO_ACTION.get(intent, ("revenue_analytics_dashboard", "overview"))

        return SkillResult(
            success=True,
            message=(
                f"Query: \"{query}\"\n"
                f"Intent: {INTENT_DESCRIPTIONS.get(intent, intent)}\n"
                f"Confidence: {confidence:.0%}\n"
                f"Would route to: {skill_id}:{action}\n"
                f"All scores: {[(s[0], f'{s[1]:.0%}') for s in all_scores[:5]]}"
            ),
            data={
                "intent": intent,
                "confidence": confidence,
                "target_skill": skill_id,
                "target_action": action,
                "all_scores": all_scores[:5],
            },
        )

    async def _correct(self, params: Dict) -> SkillResult:
        """Correct a misclassified query to teach the system."""
        query = params.get("query", "").strip()
        intent = params.get("intent", "").strip()

        if not query or not intent:
            return SkillResult(success=False, message="Provide both 'query' and 'intent'.")

        valid_intents = list(INTENT_PATTERNS.keys())
        if intent not in valid_intents:
            return SkillResult(
                success=False,
                message=f"Invalid intent '{intent}'. Valid: {valid_intents}",
            )

        tokens = _tokenize(query)
        query_key = " ".join(sorted(set(tokens)))
        self._state["intent_overrides"][query_key] = intent

        return SkillResult(
            success=True,
            message=f"Learned: \"{query}\" → {INTENT_DESCRIPTIONS.get(intent, intent)}",
            data={"query": query, "intent": intent, "key": query_key},
        )

    async def _examples(self, params: Dict) -> SkillResult:
        """Show example queries for each intent."""
        examples = {
            "overview": [
                "What is total revenue?",
                "How much have I earned?",
                "Revenue summary",
                "How is revenue doing?",
            ],
            "by_source": [
                "Which source earns the most?",
                "Revenue breakdown by source",
                "Compare revenue sources",
                "Top earning source",
            ],
            "profitability": [
                "Am I profitable?",
                "What's the profit margin?",
                "Revenue vs costs",
                "Am I making money?",
            ],
            "customers": [
                "Who are my customers?",
                "Customer breakdown",
                "Who is paying?",
            ],
            "trends": [
                "Revenue trend over time",
                "Is revenue growing?",
                "Revenue last week",
                "Historical revenue",
            ],
            "forecast": [
                "Revenue forecast",
                "What will I earn next week?",
                "Revenue projections",
                "Future revenue estimate",
            ],
            "recommendations": [
                "How can I improve revenue?",
                "Revenue recommendations",
                "How to earn more?",
                "Optimize revenue",
            ],
            "status": [
                "Revenue pipeline status",
                "Is revenue data fresh?",
                "Revenue system health",
            ],
        }

        lines = []
        for intent, queries in examples.items():
            lines.append(f"\n{INTENT_DESCRIPTIONS.get(intent, intent)}:")
            for q in queries:
                lines.append(f"  → \"{q}\"")

        return SkillResult(
            success=True,
            message="Example revenue queries:\n" + "\n".join(lines),
            data={"examples": examples},
        )

    async def _stats(self, params: Dict) -> SkillResult:
        """View query statistics."""
        stats = self._state["stats"]
        total = stats["total_queries"]
        success_rate = (
            (stats["successful_queries"] / total * 100) if total > 0 else 0
        )

        lines = [
            f"Total queries: {total}",
            f"Successful: {stats['successful_queries']}",
            f"Failed: {stats['failed_queries']}",
            f"Ambiguous: {stats['ambiguous_queries']}",
            f"Success rate: {success_rate:.1f}%",
            f"Learned corrections: {len(self._state.get('intent_overrides', {}))}",
            "",
            "Queries by intent:",
        ]
        for intent, count in sorted(
            stats["by_intent"].items(), key=lambda x: x[1], reverse=True
        ):
            lines.append(f"  {INTENT_DESCRIPTIONS.get(intent, intent)}: {count}")

        return SkillResult(
            success=True,
            message="\n".join(lines),
            data={"stats": stats, "overrides_count": len(self._state.get("intent_overrides", {}))},
        )

    async def _history(self, params: Dict) -> SkillResult:
        """View recent query history."""
        limit = min(int(params.get("limit", 10)), 50)
        history = self._state["query_history"][-limit:]
        history.reverse()  # newest first

        if not history:
            return SkillResult(success=True, message="No query history yet.", data={"history": []})

        lines = []
        for entry in history:
            status = "✓" if entry.get("success") else "✗"
            conf = entry.get("confidence", 0)
            lines.append(
                f"  {status} [{entry.get('intent', '?')}] ({conf:.0%}) \"{entry.get('query', '')}\""
            )

        return SkillResult(
            success=True,
            message=f"Recent queries (newest first):\n" + "\n".join(lines),
            data={"history": history},
        )
