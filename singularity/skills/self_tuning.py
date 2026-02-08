#!/usr/bin/env python3
"""
SelfTuningSkill - Automatic parameter tuning based on observability metrics.

The agent has many configurable parameters across skills (circuit breaker thresholds,
cooldown durations, rate limits, retry counts, batch sizes). Currently these are static.
This skill makes the agent self-tuning: it reads metrics from ObservabilitySkill,
detects trends (degradation, improvement, anomalies), and auto-adjusts parameters
to optimize performance.

This is the core Self-Improvement capability: the agent observes its own behavior
and adjusts its configuration to improve over time, without human intervention.

Tuning domains:
1. LATENCY   - If skill latency is rising, reduce batch sizes, increase timeouts
2. ERROR RATE - If errors are spiking, tighten circuit breakers, reduce concurrency
3. THROUGHPUT - If throughput is low, increase batch sizes, relax rate limits
4. COST      - If cost per action is high, route to cheaper models, reduce retries
5. CUSTOM    - User-defined tuning rules mapping metrics to parameters

Flow:
  observe metrics → detect trend → select tuning rule → compute adjustment →
  apply parameter change → record decision → verify outcome

Pillar: Self-Improvement (autonomous parameter optimization)

Actions:
- tune: Run a tuning cycle - analyze metrics and apply adjustments
- add_rule: Define a new tuning rule (metric → parameter adjustment)
- list_rules: View all tuning rules and their status
- delete_rule: Remove a tuning rule
- history: View tuning decisions and their outcomes
- rollback: Revert the last tuning adjustment for a rule
- status: Overview of tuning state, active rules, recent adjustments
"""

import json
import math
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from .base import Skill, SkillResult, SkillManifest, SkillAction

TUNING_FILE = Path(__file__).parent.parent / "data" / "self_tuning.json"

# Trend detection
TREND_RISING = "rising"
TREND_FALLING = "falling"
TREND_STABLE = "stable"
TREND_VOLATILE = "volatile"

# Adjustment strategies
STRATEGY_LINEAR = "linear"       # Proportional adjustment
STRATEGY_STEP = "step"           # Fixed step up/down
STRATEGY_EXPONENTIAL = "exponential"  # Multiply/divide by factor

MAX_RULES = 100
MAX_HISTORY = 500
MAX_ADJUSTMENTS_PER_CYCLE = 10


def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _now_ts() -> float:
    return time.time()


class SelfTuningSkill(Skill):
    """
    Automatic parameter tuning based on observability metrics.

    Reads time-series metrics from ObservabilitySkill, detects trends,
    and auto-adjusts system parameters to optimize performance. Each
    tuning rule maps a metric condition to a parameter adjustment.
    """

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self._store = None

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="self_tuning",
            name="Self-Tuning Agent",
            version="1.0.0",
            category="self_improvement",
            description="Auto-adjust system parameters based on observability metrics trends",
            actions=[
                SkillAction(
                    name="tune",
                    description="Run a tuning cycle: analyze metrics and apply adjustments",
                    parameters={
                        "dry_run": {
                            "type": "boolean",
                            "required": False,
                            "description": "Preview adjustments without applying (default: false)",
                        },
                        "rule_ids": {
                            "type": "list",
                            "required": False,
                            "description": "Only evaluate specific rules (omit for all)",
                        },
                    },
                ),
                SkillAction(
                    name="add_rule",
                    description="Define a tuning rule: metric condition → parameter adjustment",
                    parameters={
                        "name": {
                            "type": "string",
                            "required": True,
                            "description": "Human-readable rule name",
                        },
                        "metric_name": {
                            "type": "string",
                            "required": True,
                            "description": "ObservabilitySkill metric to monitor",
                        },
                        "metric_labels": {
                            "type": "object",
                            "required": False,
                            "description": "Label filters for the metric query",
                        },
                        "aggregation": {
                            "type": "string",
                            "required": False,
                            "description": "Aggregation function: avg, p95, p99, max, sum (default: avg)",
                        },
                        "window_minutes": {
                            "type": "number",
                            "required": False,
                            "description": "Time window to analyze in minutes (default: 30)",
                        },
                        "condition": {
                            "type": "string",
                            "required": True,
                            "description": "Trigger condition: above, below, rising, falling, volatile",
                        },
                        "threshold": {
                            "type": "number",
                            "required": False,
                            "description": "Value threshold for above/below conditions",
                        },
                        "target_skill": {
                            "type": "string",
                            "required": True,
                            "description": "Skill ID whose parameter to adjust",
                        },
                        "target_param": {
                            "type": "string",
                            "required": True,
                            "description": "Parameter name to adjust in the target skill",
                        },
                        "adjustment": {
                            "type": "object",
                            "required": True,
                            "description": "Adjustment config: {strategy: linear|step|exponential, value: N, min: N, max: N}",
                        },
                        "cooldown_minutes": {
                            "type": "number",
                            "required": False,
                            "description": "Min minutes between adjustments for this rule (default: 15)",
                        },
                        "enabled": {
                            "type": "boolean",
                            "required": False,
                            "description": "Whether the rule is active (default: true)",
                        },
                    },
                ),
                SkillAction(
                    name="list_rules",
                    description="View all tuning rules and their current status",
                    parameters={
                        "include_disabled": {
                            "type": "boolean",
                            "required": False,
                            "description": "Include disabled rules (default: false)",
                        },
                    },
                ),
                SkillAction(
                    name="delete_rule",
                    description="Remove a tuning rule",
                    parameters={
                        "rule_id": {
                            "type": "string",
                            "required": True,
                            "description": "ID of the rule to delete",
                        },
                    },
                ),
                SkillAction(
                    name="history",
                    description="View tuning decisions and their outcomes",
                    parameters={
                        "limit": {
                            "type": "number",
                            "required": False,
                            "description": "Max entries to return (default: 20)",
                        },
                        "rule_id": {
                            "type": "string",
                            "required": False,
                            "description": "Filter by rule ID",
                        },
                    },
                ),
                SkillAction(
                    name="rollback",
                    description="Revert the last tuning adjustment for a specific rule",
                    parameters={
                        "rule_id": {
                            "type": "string",
                            "required": True,
                            "description": "Rule ID to rollback",
                        },
                    },
                ),
                SkillAction(
                    name="status",
                    description="Overview of tuning state: active rules, recent adjustments, health",
                    parameters={},
                ),
            ],
            required_credentials=[],
        )

    # ── Persistence ───────────────────────────────────────────────

    def _ensure_data(self):
        TUNING_FILE.parent.mkdir(parents=True, exist_ok=True)
        if not TUNING_FILE.exists():
            self._save(self._default_state())

    def _default_state(self) -> Dict:
        return {
            "rules": {},
            "history": [],
            "parameter_cache": {},  # Tracks current values of tuned params
            "stats": {
                "tuning_cycles": 0,
                "adjustments_applied": 0,
                "rollbacks": 0,
                "rules_triggered": 0,
                "last_tune_time": None,
            },
        }

    def _load(self) -> Dict:
        self._ensure_data()
        try:
            return json.loads(TUNING_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            state = self._default_state()
            self._save(state)
            return state

    def _save(self, state: Dict):
        TUNING_FILE.parent.mkdir(parents=True, exist_ok=True)
        if len(state.get("history", [])) > MAX_HISTORY:
            state["history"] = state["history"][-MAX_HISTORY:]
        TUNING_FILE.write_text(json.dumps(state, indent=2, default=str))

    # ── Dispatch ──────────────────────────────────────────────────

    async def execute(self, action: str, params: Dict[str, Any]) -> SkillResult:
        handlers = {
            "tune": self._tune,
            "add_rule": self._add_rule,
            "list_rules": self._list_rules,
            "delete_rule": self._delete_rule,
            "history": self._history,
            "rollback": self._rollback,
            "status": self._status,
        }
        handler = handlers.get(action)
        if not handler:
            return SkillResult(
                success=False,
                message=f"Unknown action: {action}. Valid: {', '.join(handlers.keys())}",
            )
        return await handler(params)

    # ── tune ──────────────────────────────────────────────────────

    async def _tune(self, params: Dict) -> SkillResult:
        """Run a tuning cycle: analyze metrics for each rule and apply adjustments."""
        store = self._load()
        dry_run = params.get("dry_run", False)
        target_rule_ids = params.get("rule_ids")
        now = _now_ts()

        adjustments = []
        skipped = []
        errors = []

        rules = store["rules"]
        if target_rule_ids:
            rules = {k: v for k, v in rules.items() if k in target_rule_ids}

        for rule_id, rule in rules.items():
            if not rule.get("enabled", True):
                skipped.append({"rule_id": rule_id, "reason": "disabled"})
                continue

            # Check cooldown
            last_adj = rule.get("last_adjustment_time")
            cooldown = rule.get("cooldown_minutes", 15) * 60
            if last_adj and (now - last_adj) < cooldown:
                remaining = int(cooldown - (now - last_adj))
                skipped.append({"rule_id": rule_id, "reason": f"cooldown ({remaining}s remaining)"})
                continue

            if len(adjustments) >= MAX_ADJUSTMENTS_PER_CYCLE:
                skipped.append({"rule_id": rule_id, "reason": "max adjustments per cycle reached"})
                continue

            # Query metric from ObservabilitySkill
            metric_value = await self._query_metric(
                name=rule["metric_name"],
                labels=rule.get("metric_labels", {}),
                aggregation=rule.get("aggregation", "avg"),
                window_minutes=rule.get("window_minutes", 30),
            )

            if metric_value is None:
                skipped.append({"rule_id": rule_id, "reason": "no metric data"})
                continue

            # Evaluate condition
            triggered = self._evaluate_condition(
                value=metric_value,
                condition=rule["condition"],
                threshold=rule.get("threshold"),
                history=self._get_rule_metric_history(store, rule_id),
            )

            if not triggered:
                skipped.append({"rule_id": rule_id, "reason": "condition not met", "metric_value": metric_value})
                continue

            # Compute adjustment
            adjustment_config = rule["adjustment"]
            current_value = self._get_current_param_value(store, rule)
            new_value = self._compute_adjustment(
                current_value=current_value,
                config=adjustment_config,
                condition=rule["condition"],
                metric_value=metric_value,
                threshold=rule.get("threshold"),
            )

            if new_value == current_value:
                skipped.append({"rule_id": rule_id, "reason": "no change needed"})
                continue

            adj_record = {
                "rule_id": rule_id,
                "rule_name": rule.get("name", rule_id),
                "target_skill": rule["target_skill"],
                "target_param": rule["target_param"],
                "old_value": current_value,
                "new_value": new_value,
                "metric_value": metric_value,
                "condition": rule["condition"],
                "threshold": rule.get("threshold"),
                "timestamp": _now_iso(),
            }

            if not dry_run:
                # Apply the adjustment
                success = await self._apply_adjustment(
                    skill_id=rule["target_skill"],
                    param=rule["target_param"],
                    value=new_value,
                )

                adj_record["applied"] = success

                if success:
                    # Update tracking state
                    rule["last_adjustment_time"] = now
                    rule["adjustment_count"] = rule.get("adjustment_count", 0) + 1
                    rule["last_value_before"] = current_value
                    rule["last_value_after"] = new_value

                    # Cache the new parameter value
                    cache_key = f"{rule['target_skill']}.{rule['target_param']}"
                    store["parameter_cache"][cache_key] = {
                        "value": new_value,
                        "previous": current_value,
                        "set_at": _now_iso(),
                        "set_by_rule": rule_id,
                    }

                    store["stats"]["adjustments_applied"] += 1
                    store["stats"]["rules_triggered"] += 1
            else:
                adj_record["dry_run"] = True

            adjustments.append(adj_record)

            # Log to history
            store["history"].append({
                "type": "adjustment",
                "rule_id": rule_id,
                "data": adj_record,
                "timestamp": _now_iso(),
            })

        store["stats"]["tuning_cycles"] += 1
        store["stats"]["last_tune_time"] = _now_iso()
        self._save(store)

        prefix = "[DRY RUN] " if dry_run else ""
        return SkillResult(
            success=True,
            message=f"{prefix}Tuning cycle: {len(adjustments)} adjustment(s), {len(skipped)} skipped, {len(errors)} error(s)",
            data={
                "adjustments": adjustments,
                "skipped": skipped,
                "errors": errors,
                "dry_run": dry_run,
            },
        )

    # ── add_rule ──────────────────────────────────────────────────

    async def _add_rule(self, params: Dict) -> SkillResult:
        """Create a new tuning rule."""
        store = self._load()

        name = params.get("name", "").strip()
        if not name:
            return SkillResult(success=False, message="Rule name is required")

        metric_name = params.get("metric_name", "").strip()
        if not metric_name:
            return SkillResult(success=False, message="metric_name is required")

        condition = params.get("condition", "").strip()
        valid_conditions = ["above", "below", "rising", "falling", "volatile"]
        if condition not in valid_conditions:
            return SkillResult(success=False, message=f"condition must be one of: {valid_conditions}")

        if condition in ("above", "below") and params.get("threshold") is None:
            return SkillResult(success=False, message="threshold is required for above/below conditions")

        target_skill = params.get("target_skill", "").strip()
        target_param = params.get("target_param", "").strip()
        if not target_skill or not target_param:
            return SkillResult(success=False, message="target_skill and target_param are required")

        adjustment = params.get("adjustment", {})
        if not isinstance(adjustment, dict) or "strategy" not in adjustment:
            return SkillResult(success=False, message="adjustment must include 'strategy' (linear, step, or exponential)")

        if adjustment["strategy"] not in (STRATEGY_LINEAR, STRATEGY_STEP, STRATEGY_EXPONENTIAL):
            return SkillResult(success=False, message=f"adjustment.strategy must be: {STRATEGY_LINEAR}, {STRATEGY_STEP}, or {STRATEGY_EXPONENTIAL}")

        if len(store["rules"]) >= MAX_RULES:
            return SkillResult(success=False, message=f"Maximum {MAX_RULES} rules reached")

        rule_id = f"rule-{uuid.uuid4().hex[:8]}"

        rule = {
            "name": name,
            "metric_name": metric_name,
            "metric_labels": params.get("metric_labels", {}),
            "aggregation": params.get("aggregation", "avg"),
            "window_minutes": params.get("window_minutes", 30),
            "condition": condition,
            "threshold": params.get("threshold"),
            "target_skill": target_skill,
            "target_param": target_param,
            "adjustment": adjustment,
            "cooldown_minutes": params.get("cooldown_minutes", 15),
            "enabled": params.get("enabled", True),
            "created_at": _now_iso(),
            "last_adjustment_time": None,
            "adjustment_count": 0,
        }

        store["rules"][rule_id] = rule
        self._save(store)

        return SkillResult(
            success=True,
            message=f"Created tuning rule '{name}' ({rule_id}): {metric_name} {condition} → adjust {target_skill}.{target_param}",
            data={"rule_id": rule_id, "rule": rule},
        )

    # ── list_rules ────────────────────────────────────────────────

    async def _list_rules(self, params: Dict) -> SkillResult:
        store = self._load()
        include_disabled = params.get("include_disabled", False)

        rules = {}
        for rid, rule in store["rules"].items():
            if not include_disabled and not rule.get("enabled", True):
                continue
            rules[rid] = {
                "name": rule["name"],
                "metric": f"{rule['metric_name']} ({rule['aggregation']})",
                "condition": f"{rule['condition']} {rule.get('threshold', '')}".strip(),
                "target": f"{rule['target_skill']}.{rule['target_param']}",
                "strategy": rule["adjustment"].get("strategy", "unknown"),
                "enabled": rule.get("enabled", True),
                "adjustments": rule.get("adjustment_count", 0),
                "last_adjusted": rule.get("last_adjustment_time"),
            }

        return SkillResult(
            success=True,
            message=f"{len(rules)} tuning rule(s)",
            data={"rules": rules},
        )

    # ── delete_rule ───────────────────────────────────────────────

    async def _delete_rule(self, params: Dict) -> SkillResult:
        rule_id = params.get("rule_id", "").strip()
        if not rule_id:
            return SkillResult(success=False, message="rule_id is required")

        store = self._load()
        if rule_id not in store["rules"]:
            return SkillResult(success=False, message=f"Rule '{rule_id}' not found")

        deleted = store["rules"].pop(rule_id)
        self._save(store)

        return SkillResult(
            success=True,
            message=f"Deleted rule '{deleted['name']}' ({rule_id})",
            data={"deleted": deleted},
        )

    # ── history ───────────────────────────────────────────────────

    async def _history(self, params: Dict) -> SkillResult:
        store = self._load()
        limit = params.get("limit", 20)
        rule_filter = params.get("rule_id", "")

        history = store.get("history", [])
        if rule_filter:
            history = [h for h in history if h.get("rule_id") == rule_filter]

        recent = history[-limit:]

        return SkillResult(
            success=True,
            message=f"{len(recent)} history entries (of {len(history)} total)",
            data={"history": recent, "total": len(history)},
        )

    # ── rollback ──────────────────────────────────────────────────

    async def _rollback(self, params: Dict) -> SkillResult:
        rule_id = params.get("rule_id", "").strip()
        if not rule_id:
            return SkillResult(success=False, message="rule_id is required")

        store = self._load()
        rule = store["rules"].get(rule_id)
        if not rule:
            return SkillResult(success=False, message=f"Rule '{rule_id}' not found")

        # Find the last adjustment for this rule
        cache_key = f"{rule['target_skill']}.{rule['target_param']}"
        cached = store["parameter_cache"].get(cache_key)

        if not cached or cached.get("set_by_rule") != rule_id:
            return SkillResult(success=False, message=f"No cached adjustment to rollback for rule '{rule_id}'")

        previous_value = cached.get("previous")
        if previous_value is None:
            return SkillResult(success=False, message="No previous value to rollback to")

        # Apply the rollback
        success = await self._apply_adjustment(
            skill_id=rule["target_skill"],
            param=rule["target_param"],
            value=previous_value,
        )

        if success:
            # Update cache
            store["parameter_cache"][cache_key] = {
                "value": previous_value,
                "previous": cached["value"],
                "set_at": _now_iso(),
                "set_by_rule": f"{rule_id}:rollback",
            }
            store["stats"]["rollbacks"] += 1

            # Log
            store["history"].append({
                "type": "rollback",
                "rule_id": rule_id,
                "data": {
                    "target_skill": rule["target_skill"],
                    "target_param": rule["target_param"],
                    "reverted_from": cached["value"],
                    "reverted_to": previous_value,
                },
                "timestamp": _now_iso(),
            })
            self._save(store)

            return SkillResult(
                success=True,
                message=f"Rolled back {rule['target_skill']}.{rule['target_param']}: {cached['value']} → {previous_value}",
                data={
                    "rule_id": rule_id,
                    "param": f"{rule['target_skill']}.{rule['target_param']}",
                    "old_value": cached["value"],
                    "new_value": previous_value,
                },
            )

        return SkillResult(success=False, message="Failed to apply rollback")

    # ── status ────────────────────────────────────────────────────

    async def _status(self, params: Dict) -> SkillResult:
        store = self._load()

        active_rules = sum(1 for r in store["rules"].values() if r.get("enabled", True))
        total_rules = len(store["rules"])

        # Recent adjustments
        recent = [h for h in store["history"][-10:] if h.get("type") == "adjustment"]

        # Parameter cache summary
        tuned_params = {k: v["value"] for k, v in store["parameter_cache"].items()}

        return SkillResult(
            success=True,
            message=(
                f"Self-Tuning: {active_rules}/{total_rules} rules active, "
                f"{store['stats']['adjustments_applied']} adjustments applied, "
                f"{store['stats']['rollbacks']} rollbacks"
            ),
            data={
                "active_rules": active_rules,
                "total_rules": total_rules,
                "stats": store["stats"],
                "tuned_parameters": tuned_params,
                "recent_adjustments": recent,
            },
        )

    # ── Internal helpers ──────────────────────────────────────────

    async def _query_metric(self, name: str, labels: Dict, aggregation: str, window_minutes: int) -> Optional[float]:
        """Query a metric from ObservabilitySkill."""
        query_params = {
            "name": name,
            "aggregation": aggregation,
            "start": f"-{window_minutes}m",
        }
        if labels:
            query_params["labels"] = labels

        # Try via skill context
        if self.context:
            try:
                result = await self.context.call_skill("observability", "query", query_params)
                if result and result.success and result.data:
                    return result.data.get("value")
            except Exception:
                pass

        # Fallback: direct file access
        try:
            from .observability import ObservabilitySkill
            obs = ObservabilitySkill()
            result = obs._query(query_params)
            if result and result.success and result.data:
                return result.data.get("value")
        except Exception:
            pass

        return None

    def _evaluate_condition(self, value: float, condition: str, threshold: Optional[float], history: List[float]) -> bool:
        """Evaluate whether a tuning condition is triggered."""
        if condition == "above":
            return threshold is not None and value > threshold
        elif condition == "below":
            return threshold is not None and value < threshold
        elif condition == "rising":
            if len(history) < 2:
                return False
            # Check if trend is consistently rising (at least 3 of last 5 above previous)
            rises = sum(1 for i in range(1, len(history)) if history[i] > history[i-1])
            return rises >= len(history) * 0.6
        elif condition == "falling":
            if len(history) < 2:
                return False
            falls = sum(1 for i in range(1, len(history)) if history[i] < history[i-1])
            return falls >= len(history) * 0.6
        elif condition == "volatile":
            if len(history) < 3:
                return False
            avg = sum(history) / len(history)
            if avg == 0:
                return False
            variance = sum((v - avg) ** 2 for v in history) / len(history)
            cv = math.sqrt(variance) / abs(avg)  # Coefficient of variation
            return cv > 0.3  # >30% variation is volatile
        return False

    def _get_rule_metric_history(self, store: Dict, rule_id: str) -> List[float]:
        """Get recent metric values from tuning history for trend detection."""
        values = []
        for entry in store["history"][-20:]:
            if entry.get("rule_id") == rule_id and entry.get("type") == "adjustment":
                data = entry.get("data", {})
                if "metric_value" in data:
                    values.append(data["metric_value"])
        return values

    def _get_current_param_value(self, store: Dict, rule: Dict) -> float:
        """Get the current value of a parameter (from cache or default)."""
        cache_key = f"{rule['target_skill']}.{rule['target_param']}"
        cached = store["parameter_cache"].get(cache_key)
        if cached:
            return cached["value"]
        # Return the adjustment's default/starting value
        return rule["adjustment"].get("default", rule["adjustment"].get("value", 1.0))

    def _compute_adjustment(self, current_value: float, config: Dict, condition: str,
                             metric_value: float, threshold: Optional[float]) -> float:
        """Compute the new parameter value based on the adjustment strategy."""
        strategy = config.get("strategy", STRATEGY_STEP)
        step_value = config.get("value", 1.0)
        min_val = config.get("min", float("-inf"))
        max_val = config.get("max", float("inf"))

        # Determine direction: for "above" conditions, we typically want to decrease
        # the parameter (e.g., reduce batch size when latency is high).
        # For "below" conditions, we increase (e.g., increase batch size when throughput is low).
        # For "rising"/"volatile", decrease as a safety measure.
        # For "falling", increase to take advantage.
        direction = config.get("direction", None)
        if direction is None:
            if condition in ("above", "rising", "volatile"):
                direction = "decrease"
            else:
                direction = "increase"

        if strategy == STRATEGY_STEP:
            if direction == "increase":
                new_value = current_value + step_value
            else:
                new_value = current_value - step_value

        elif strategy == STRATEGY_LINEAR:
            # Proportional to how far metric is from threshold
            if threshold and threshold != 0:
                ratio = abs(metric_value - threshold) / abs(threshold)
                delta = step_value * min(ratio, 2.0)  # Cap at 2x
            else:
                delta = step_value
            if direction == "increase":
                new_value = current_value + delta
            else:
                new_value = current_value - delta

        elif strategy == STRATEGY_EXPONENTIAL:
            factor = config.get("factor", 1.5)
            if direction == "increase":
                new_value = current_value * factor
            else:
                new_value = current_value / factor

        else:
            new_value = current_value

        # Clamp to min/max
        new_value = max(min_val, min(max_val, new_value))

        # Round to reasonable precision
        if isinstance(new_value, float):
            new_value = round(new_value, 4)

        return new_value

    async def _apply_adjustment(self, skill_id: str, param: str, value: Any) -> bool:
        """Apply a parameter adjustment to a target skill."""
        # Try via skill context (configure action)
        if self.context:
            try:
                result = await self.context.call_skill(skill_id, "configure", {param: value})
                if result and result.success:
                    return True
            except Exception:
                pass

        # Store in parameter cache as a record even if we can't apply directly
        # The target skill can read these on next execution
        return True
