#!/usr/bin/env python3
"""
ErrorRecoverySkill - Autonomous error detection, recovery, and learning.

When actions fail, the agent needs to:
1. Understand WHY the error happened (classify it)
2. Decide HOW to recover (retry, fallback, skip, escalate)
3. LEARN from the error so it doesn't repeat

This skill maintains an error knowledge base that maps error patterns to
recovery strategies. Over time, the agent gets better at handling errors
because it remembers what worked before.

Pillar: Self-Improvement (critical 'act → fail → recover → learn' loop)

Actions:
- record: Log an error with full context
- classify: Categorize an error by type and severity
- suggest_recovery: Get recovery strategies for an error
- attempt_recovery: Execute a recovery strategy and track outcome
- knowledge: Query the error knowledge base
- patterns: Analyze error patterns across sessions
- reset: Clear error history
"""

import json
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from collections import defaultdict
from .base import Skill, SkillManifest, SkillAction, SkillResult


ERROR_FILE = Path(__file__).parent.parent / "data" / "error_recovery.json"
MAX_ERRORS = 500
MAX_KNOWLEDGE_ENTRIES = 200


class ErrorCategory:
    """Error classification categories."""
    TRANSIENT = "transient"       # Network timeouts, rate limits, temp failures
    CONFIG = "configuration"     # Missing env vars, bad config, wrong params
    RESOURCE = "resource"        # Out of memory, disk full, quota exceeded
    LOGIC = "logic"              # Bugs, assertion failures, unexpected state
    DEPENDENCY = "dependency"    # Missing packages, version conflicts
    PERMISSION = "permission"    # Auth failures, access denied
    INPUT = "input"              # Bad user input, malformed data
    EXTERNAL = "external"        # Third-party API errors, service outages
    UNKNOWN = "unknown"


# Patterns that map error messages to categories
ERROR_PATTERNS = {
    ErrorCategory.TRANSIENT: [
        r"timeout",
        r"timed?\s*out",
        r"rate\s*limit",
        r"429",
        r"503",
        r"502",
        r"connection\s*(reset|refused|aborted)",
        r"temporary\s*(failure|error|unavailable)",
        r"retry",
        r"EAGAIN",
        r"too many requests",
    ],
    ErrorCategory.CONFIG: [
        r"not\s*set",
        r"not\s*configured",
        r"missing\s*(env|config|setting|key|variable)",
        r"invalid\s*config",
        r"environment\s*variable",
        r"\.env",
    ],
    ErrorCategory.RESOURCE: [
        r"out\s*of\s*memory",
        r"MemoryError",
        r"disk\s*(full|space)",
        r"quota\s*(exceeded|limit)",
        r"no\s*space\s*left",
        r"ResourceExhausted",
    ],
    ErrorCategory.LOGIC: [
        r"AssertionError",
        r"TypeError",
        r"ValueError",
        r"KeyError",
        r"IndexError",
        r"AttributeError",
        r"ZeroDivisionError",
        r"unexpected\s*(state|value|type)",
    ],
    ErrorCategory.DEPENDENCY: [
        r"ModuleNotFoundError",
        r"ImportError",
        r"No\s*module\s*named",
        r"package.*not\s*(found|installed)",
        r"version\s*(conflict|mismatch)",
    ],
    ErrorCategory.PERMISSION: [
        r"(401|403)",
        r"Unauthorized",
        r"Forbidden",
        r"access\s*denied",
        r"permission\s*denied",
        r"authentication\s*(failed|required|error)",
        r"invalid\s*(token|api\s*key|credential)",
    ],
    ErrorCategory.INPUT: [
        r"invalid\s*(input|argument|parameter)",
        r"bad\s*request",
        r"400",
        r"malformed",
        r"validation\s*(error|failed)",
    ],
    ErrorCategory.EXTERNAL: [
        r"500",
        r"internal\s*server\s*error",
        r"service\s*unavailable",
        r"upstream",
        r"gateway",
        r"API\s*(error|failure)",
    ],
}

# Recovery strategies by error category
DEFAULT_STRATEGIES = {
    ErrorCategory.TRANSIENT: [
        {"strategy": "retry_with_backoff", "description": "Retry with exponential backoff", "max_retries": 3, "base_delay": 1.0},
        {"strategy": "wait_and_retry", "description": "Wait longer and retry once", "delay": 30},
    ],
    ErrorCategory.CONFIG: [
        {"strategy": "check_environment", "description": "Verify environment variables and config files"},
        {"strategy": "use_defaults", "description": "Fall back to default configuration values"},
    ],
    ErrorCategory.RESOURCE: [
        {"strategy": "reduce_scope", "description": "Reduce batch size or scope of operation"},
        {"strategy": "cleanup_resources", "description": "Free up resources and retry"},
    ],
    ErrorCategory.LOGIC: [
        {"strategy": "validate_inputs", "description": "Check and sanitize inputs before retrying"},
        {"strategy": "try_alternative", "description": "Use an alternative approach or algorithm"},
        {"strategy": "escalate", "description": "Flag for human review - likely a bug"},
    ],
    ErrorCategory.DEPENDENCY: [
        {"strategy": "install_dependency", "description": "Install the missing package"},
        {"strategy": "use_alternative", "description": "Use an alternative package or approach"},
    ],
    ErrorCategory.PERMISSION: [
        {"strategy": "refresh_credentials", "description": "Refresh or re-authenticate credentials"},
        {"strategy": "check_permissions", "description": "Verify access permissions are correct"},
    ],
    ErrorCategory.INPUT: [
        {"strategy": "sanitize_input", "description": "Clean and validate the input data"},
        {"strategy": "request_clarification", "description": "Ask for corrected input"},
    ],
    ErrorCategory.EXTERNAL: [
        {"strategy": "retry_with_backoff", "description": "Retry with exponential backoff", "max_retries": 3, "base_delay": 2.0},
        {"strategy": "use_fallback_service", "description": "Try an alternative service or endpoint"},
    ],
    ErrorCategory.UNKNOWN: [
        {"strategy": "retry_once", "description": "Simple retry - might be transient"},
        {"strategy": "escalate", "description": "Flag for investigation - unknown error type"},
    ],
}


class ErrorRecoverySkill(Skill):
    """
    Autonomous error recovery with pattern learning.

    Maintains a knowledge base mapping error signatures to successful
    recovery strategies. Each time an error occurs and is resolved,
    the resolution is recorded. Future occurrences of similar errors
    can be handled automatically using previously successful strategies.
    """

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self._ensure_data()

    def _ensure_data(self):
        ERROR_FILE.parent.mkdir(parents=True, exist_ok=True)
        if not ERROR_FILE.exists():
            self._save({
                "errors": [],
                "knowledge_base": [],
                "recovery_attempts": [],
                "metadata": {
                    "created_at": datetime.now().isoformat(),
                    "total_errors": 0,
                    "total_recoveries": 0,
                    "successful_recoveries": 0,
                },
            })

    def _load(self) -> Dict:
        try:
            with open(ERROR_FILE, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {
                "errors": [],
                "knowledge_base": [],
                "recovery_attempts": [],
                "metadata": {
                    "created_at": datetime.now().isoformat(),
                    "total_errors": 0,
                    "total_recoveries": 0,
                    "successful_recoveries": 0,
                },
            }

    def _save(self, data: Dict):
        # Trim to prevent unbounded growth
        if len(data.get("errors", [])) > MAX_ERRORS:
            data["errors"] = data["errors"][-MAX_ERRORS:]
        if len(data.get("knowledge_base", [])) > MAX_KNOWLEDGE_ENTRIES:
            data["knowledge_base"] = data["knowledge_base"][-MAX_KNOWLEDGE_ENTRIES:]
        if len(data.get("recovery_attempts", [])) > MAX_ERRORS:
            data["recovery_attempts"] = data["recovery_attempts"][-MAX_ERRORS:]

        data["metadata"]["last_updated"] = datetime.now().isoformat()
        ERROR_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(ERROR_FILE, "w") as f:
            json.dump(data, f, indent=2, default=str)

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="error_recovery",
            name="Error Recovery",
            version="1.0.0",
            category="meta",
            description="Autonomous error detection, classification, recovery, and learning",
            actions=[
                SkillAction(
                    name="record",
                    description="Record an error with full context for analysis",
                    parameters={
                        "skill_id": {"type": "string", "required": True, "description": "Skill that produced the error"},
                        "action": {"type": "string", "required": True, "description": "Action that failed"},
                        "error_message": {"type": "string", "required": True, "description": "The error message"},
                        "error_type": {"type": "string", "required": False, "description": "Exception class name"},
                        "context": {"type": "object", "required": False, "description": "Additional context about the error"},
                    },
                ),
                SkillAction(
                    name="classify",
                    description="Classify an error by category and severity",
                    parameters={
                        "error_message": {"type": "string", "required": True, "description": "The error message to classify"},
                        "error_type": {"type": "string", "required": False, "description": "Exception class name"},
                    },
                ),
                SkillAction(
                    name="suggest_recovery",
                    description="Get recovery strategies for an error, including learned strategies from past successes",
                    parameters={
                        "error_message": {"type": "string", "required": True, "description": "The error message"},
                        "skill_id": {"type": "string", "required": False, "description": "Skill that produced the error"},
                        "action": {"type": "string", "required": False, "description": "Action that failed"},
                    },
                ),
                SkillAction(
                    name="record_recovery",
                    description="Record the outcome of a recovery attempt to build the knowledge base",
                    parameters={
                        "error_message": {"type": "string", "required": True, "description": "Original error message"},
                        "strategy_used": {"type": "string", "required": True, "description": "Recovery strategy that was attempted"},
                        "success": {"type": "boolean", "required": True, "description": "Whether the recovery succeeded"},
                        "skill_id": {"type": "string", "required": False, "description": "Skill that had the error"},
                        "notes": {"type": "string", "required": False, "description": "Additional notes about the recovery"},
                    },
                ),
                SkillAction(
                    name="knowledge",
                    description="Query the error knowledge base for known error-fix mappings",
                    parameters={
                        "query": {"type": "string", "required": False, "description": "Search term to filter knowledge entries"},
                        "category": {"type": "string", "required": False, "description": "Filter by error category"},
                    },
                ),
                SkillAction(
                    name="patterns",
                    description="Analyze error patterns - most frequent errors, worst skills, recovery success rates",
                    parameters={
                        "timeframe_hours": {"type": "number", "required": False, "description": "Hours to look back (default: 168 = 1 week)"},
                    },
                ),
                SkillAction(
                    name="reset",
                    description="Clear error history (preserves knowledge base)",
                    parameters={},
                ),
            ],
            required_credentials=[],
        )

    async def execute(self, action: str, params: Dict = None) -> SkillResult:
        params = params or {}
        actions = {
            "record": self._record_error,
            "classify": self._classify_error,
            "suggest_recovery": self._suggest_recovery,
            "record_recovery": self._record_recovery,
            "knowledge": self._query_knowledge,
            "patterns": self._analyze_patterns,
            "reset": self._reset,
        }
        if action not in actions:
            return SkillResult(
                success=False,
                message=f"Unknown action: {action}. Available: {list(actions.keys())}",
            )
        return await actions[action](params)

    def _classify_error_message(self, error_message: str, error_type: str = "") -> Dict:
        """Classify an error by matching against known patterns."""
        combined = f"{error_type} {error_message}".lower()
        scores = defaultdict(int)

        for category, patterns in ERROR_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, combined, re.IGNORECASE):
                    scores[category] += 1

        if scores:
            best_category = max(scores, key=scores.get)
        else:
            best_category = ErrorCategory.UNKNOWN

        # Determine severity based on category
        severity_map = {
            ErrorCategory.TRANSIENT: "low",
            ErrorCategory.CONFIG: "medium",
            ErrorCategory.RESOURCE: "high",
            ErrorCategory.LOGIC: "medium",
            ErrorCategory.DEPENDENCY: "medium",
            ErrorCategory.PERMISSION: "high",
            ErrorCategory.INPUT: "low",
            ErrorCategory.EXTERNAL: "medium",
            ErrorCategory.UNKNOWN: "medium",
        }

        # Determine if retryable
        retryable_categories = {
            ErrorCategory.TRANSIENT,
            ErrorCategory.EXTERNAL,
            ErrorCategory.RESOURCE,
        }

        return {
            "category": best_category,
            "severity": severity_map.get(best_category, "medium"),
            "retryable": best_category in retryable_categories,
            "confidence": min(scores.get(best_category, 0) / 3.0, 1.0) if scores else 0.0,
            "all_scores": dict(scores),
        }

    def _generate_error_signature(self, error_message: str, error_type: str = "") -> str:
        """Generate a normalized signature for an error to match similar errors."""
        sig = f"{error_type}:" if error_type else ""
        # Normalize the error: remove specific values, keep structure
        normalized = error_message.lower().strip()
        # Remove specific file paths
        normalized = re.sub(r'/[\w/._-]+', '<PATH>', normalized)
        # Remove specific numbers
        normalized = re.sub(r'\b\d+\b', '<N>', normalized)
        # Remove quoted strings
        normalized = re.sub(r"'[^']*'", "'<STR>'", normalized)
        normalized = re.sub(r'"[^"]*"', '"<STR>"', normalized)
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized)
        return sig + normalized

    async def _record_error(self, params: Dict) -> SkillResult:
        """Record an error with full context."""
        skill_id = params.get("skill_id", "unknown")
        action = params.get("action", "unknown")
        error_message = params.get("error_message", "")
        error_type = params.get("error_type", "")
        context = params.get("context", {})

        if not error_message:
            return SkillResult(success=False, message="error_message is required")

        classification = self._classify_error_message(error_message, error_type)
        signature = self._generate_error_signature(error_message, error_type)

        error_record = {
            "timestamp": datetime.now().isoformat(),
            "skill_id": skill_id,
            "action": action,
            "error_message": error_message,
            "error_type": error_type,
            "signature": signature,
            "classification": classification,
            "context": context,
            "recovered": False,
        }

        data = self._load()
        data["errors"].append(error_record)
        data["metadata"]["total_errors"] = data["metadata"].get("total_errors", 0) + 1
        self._save(data)

        return SkillResult(
            success=True,
            message=f"Error recorded: {classification['category']} ({classification['severity']} severity, retryable={classification['retryable']})",
            data={
                "error_id": len(data["errors"]) - 1,
                "classification": classification,
                "signature": signature,
            },
        )

    async def _classify_error(self, params: Dict) -> SkillResult:
        """Classify an error without recording it."""
        error_message = params.get("error_message", "")
        error_type = params.get("error_type", "")

        if not error_message:
            return SkillResult(success=False, message="error_message is required")

        classification = self._classify_error_message(error_message, error_type)
        signature = self._generate_error_signature(error_message, error_type)

        return SkillResult(
            success=True,
            message=f"Classified as {classification['category']} ({classification['severity']} severity)",
            data={
                "classification": classification,
                "signature": signature,
            },
        )

    async def _suggest_recovery(self, params: Dict) -> SkillResult:
        """Suggest recovery strategies, prioritizing learned strategies from past successes."""
        error_message = params.get("error_message", "")
        skill_id = params.get("skill_id")
        action = params.get("action")

        if not error_message:
            return SkillResult(success=False, message="error_message is required")

        classification = self._classify_error_message(error_message)
        signature = self._generate_error_signature(error_message)

        # Check knowledge base for previously successful recoveries
        data = self._load()
        learned_strategies = []
        for entry in data.get("knowledge_base", []):
            # Match by signature similarity
            if self._signatures_match(signature, entry.get("signature", "")):
                learned_strategies.append({
                    "strategy": entry["strategy"],
                    "description": entry.get("description", "Learned from past recovery"),
                    "success_count": entry.get("success_count", 1),
                    "source": "knowledge_base",
                    "notes": entry.get("notes", ""),
                })

        # Also check if this specific skill+action combo has known fixes
        if skill_id:
            for entry in data.get("knowledge_base", []):
                if entry.get("skill_id") == skill_id and entry.get("action") == action:
                    if not any(s["strategy"] == entry["strategy"] for s in learned_strategies):
                        learned_strategies.append({
                            "strategy": entry["strategy"],
                            "description": entry.get("description", "Known fix for this skill"),
                            "success_count": entry.get("success_count", 1),
                            "source": "knowledge_base",
                        })

        # Sort learned strategies by success count
        learned_strategies.sort(key=lambda s: s.get("success_count", 0), reverse=True)

        # Get default strategies for this category
        default = DEFAULT_STRATEGIES.get(
            classification["category"],
            DEFAULT_STRATEGIES[ErrorCategory.UNKNOWN],
        )
        default_strategies = [
            {**s, "source": "default"} for s in default
        ]

        # Combine: learned first, then defaults
        all_strategies = learned_strategies + default_strategies

        return SkillResult(
            success=True,
            message=f"Found {len(learned_strategies)} learned + {len(default_strategies)} default recovery strategies",
            data={
                "classification": classification,
                "strategies": all_strategies,
                "has_learned_strategies": len(learned_strategies) > 0,
                "recommended": all_strategies[0] if all_strategies else None,
            },
        )

    def _signatures_match(self, sig1: str, sig2: str) -> bool:
        """Check if two error signatures are similar enough to match."""
        if sig1 == sig2:
            return True
        # Check if one contains the other
        if sig1 in sig2 or sig2 in sig1:
            return True
        # Split into words and check overlap
        words1 = set(sig1.split())
        words2 = set(sig2.split())
        if not words1 or not words2:
            return False
        overlap = len(words1 & words2) / max(len(words1), len(words2))
        return overlap > 0.6

    async def _record_recovery(self, params: Dict) -> SkillResult:
        """Record the outcome of a recovery attempt and update knowledge base."""
        error_message = params.get("error_message", "")
        strategy_used = params.get("strategy_used", "")
        success = params.get("success", False)
        skill_id = params.get("skill_id", "")
        notes = params.get("notes", "")

        if not error_message or not strategy_used:
            return SkillResult(
                success=False,
                message="error_message and strategy_used are required",
            )

        signature = self._generate_error_signature(error_message)
        data = self._load()

        # Record the attempt
        attempt = {
            "timestamp": datetime.now().isoformat(),
            "error_message": error_message,
            "signature": signature,
            "strategy_used": strategy_used,
            "success": success,
            "skill_id": skill_id,
            "notes": notes,
        }
        data["recovery_attempts"].append(attempt)
        data["metadata"]["total_recoveries"] = data["metadata"].get("total_recoveries", 0) + 1

        if success:
            data["metadata"]["successful_recoveries"] = data["metadata"].get("successful_recoveries", 0) + 1

            # Update knowledge base
            existing = None
            for entry in data.get("knowledge_base", []):
                if entry.get("signature") == signature and entry.get("strategy") == strategy_used:
                    existing = entry
                    break

            if existing:
                existing["success_count"] = existing.get("success_count", 0) + 1
                existing["last_used"] = datetime.now().isoformat()
                if notes:
                    existing["notes"] = notes
            else:
                data["knowledge_base"].append({
                    "signature": signature,
                    "strategy": strategy_used,
                    "description": f"Successful recovery for: {error_message[:100]}",
                    "success_count": 1,
                    "skill_id": skill_id,
                    "notes": notes,
                    "created_at": datetime.now().isoformat(),
                    "last_used": datetime.now().isoformat(),
                })

            # Mark the most recent matching error as recovered
            for error in reversed(data.get("errors", [])):
                if error.get("signature") == signature and not error.get("recovered"):
                    error["recovered"] = True
                    error["recovery_strategy"] = strategy_used
                    break

        self._save(data)

        return SkillResult(
            success=True,
            message=f"Recovery {'succeeded' if success else 'failed'} with strategy '{strategy_used}'",
            data={
                "success": success,
                "strategy": strategy_used,
                "knowledge_base_updated": success,
            },
        )

    async def _query_knowledge(self, params: Dict) -> SkillResult:
        """Query the error knowledge base."""
        query = params.get("query", "").lower()
        category = params.get("category", "")

        data = self._load()
        entries = data.get("knowledge_base", [])

        if query:
            entries = [
                e for e in entries
                if query in e.get("signature", "").lower()
                or query in e.get("description", "").lower()
                or query in e.get("notes", "").lower()
                or query in e.get("strategy", "").lower()
            ]

        if category:
            # Filter by re-classifying the signatures
            filtered = []
            for entry in entries:
                cls = self._classify_error_message(entry.get("description", ""))
                if cls["category"] == category:
                    filtered.append(entry)
            entries = filtered

        # Sort by success count
        entries.sort(key=lambda e: e.get("success_count", 0), reverse=True)

        return SkillResult(
            success=True,
            message=f"Found {len(entries)} knowledge base entries",
            data={
                "entries": entries[:50],
                "total": len(entries),
            },
        )

    async def _analyze_patterns(self, params: Dict) -> SkillResult:
        """Analyze error patterns to find recurring issues."""
        timeframe_hours = params.get("timeframe_hours", 168)  # default 1 week
        cutoff = datetime.now() - timedelta(hours=timeframe_hours)

        data = self._load()
        errors = data.get("errors", [])

        # Filter by timeframe
        recent_errors = []
        for error in errors:
            try:
                ts = datetime.fromisoformat(error["timestamp"])
                if ts >= cutoff:
                    recent_errors.append(error)
            except (KeyError, ValueError):
                continue

        if not recent_errors:
            return SkillResult(
                success=True,
                message="No errors found in the specified timeframe",
                data={"timeframe_hours": timeframe_hours, "total_errors": 0},
            )

        # Analyze patterns
        by_category = defaultdict(int)
        by_skill = defaultdict(int)
        by_signature = defaultdict(int)
        recovered_count = 0

        for error in recent_errors:
            cls = error.get("classification", {})
            by_category[cls.get("category", "unknown")] += 1
            by_skill[error.get("skill_id", "unknown")] += 1
            by_signature[error.get("signature", "unknown")] += 1
            if error.get("recovered"):
                recovered_count += 1

        # Find most problematic patterns
        top_errors = sorted(by_signature.items(), key=lambda x: x[1], reverse=True)[:10]
        worst_skills = sorted(by_skill.items(), key=lambda x: x[1], reverse=True)[:5]

        # Recovery rate
        recovery_rate = recovered_count / len(recent_errors) if recent_errors else 0

        # Recovery attempt stats
        attempts = data.get("recovery_attempts", [])
        recent_attempts = []
        for a in attempts:
            try:
                ts = datetime.fromisoformat(a["timestamp"])
                if ts >= cutoff:
                    recent_attempts.append(a)
            except (KeyError, ValueError):
                continue

        strategy_stats = defaultdict(lambda: {"attempts": 0, "successes": 0})
        for a in recent_attempts:
            s = strategy_stats[a.get("strategy_used", "unknown")]
            s["attempts"] += 1
            if a.get("success"):
                s["successes"] += 1

        strategy_effectiveness = {
            name: {
                **stats,
                "success_rate": stats["successes"] / stats["attempts"] if stats["attempts"] > 0 else 0,
            }
            for name, stats in strategy_stats.items()
        }

        return SkillResult(
            success=True,
            message=f"Analyzed {len(recent_errors)} errors over {timeframe_hours}h: {recovery_rate:.0%} recovery rate",
            data={
                "timeframe_hours": timeframe_hours,
                "total_errors": len(recent_errors),
                "recovered": recovered_count,
                "recovery_rate": recovery_rate,
                "by_category": dict(by_category),
                "worst_skills": worst_skills,
                "top_recurring_errors": top_errors,
                "strategy_effectiveness": strategy_effectiveness,
                "knowledge_base_size": len(data.get("knowledge_base", [])),
            },
        )

    async def _reset(self, params: Dict) -> SkillResult:
        """Clear error history but preserve the knowledge base."""
        data = self._load()
        kb_size = len(data.get("knowledge_base", []))

        data["errors"] = []
        data["recovery_attempts"] = []
        data["metadata"]["total_errors"] = 0
        data["metadata"]["total_recoveries"] = 0
        data["metadata"]["successful_recoveries"] = 0
        self._save(data)

        return SkillResult(
            success=True,
            message=f"Error history cleared. Knowledge base preserved ({kb_size} entries).",
            data={"knowledge_base_preserved": kb_size},
        )
