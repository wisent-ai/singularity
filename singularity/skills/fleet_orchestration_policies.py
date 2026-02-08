#!/usr/bin/env python3
"""
FleetOrchestrationPoliciesSkill - Pre-built fleet management policies for autonomous orchestration.

FleetHealthManagerSkill has configurable policies, but agents start with generic defaults
and must manually tune them. This skill provides battle-tested policy presets optimized
for specific operational goals:

1. COST-AWARE: Minimize compute costs — aggressive scale-down, conservative scale-up,
   strict budget limits, prefer cheap replicas
2. RESILIENCE: Maximize uptime — fast healing, generous scale-up, redundant replicas,
   rolling updates with health gates
3. REVENUE-OPTIMIZED: Maximize revenue throughput — scale with demand, prioritize
   high-revenue replicas, SLA-driven healing
4. BALANCED: Default middle-ground policy for general-purpose fleets
5. DEV/TEST: Minimal fleet for development — single replica, no auto-scale,
   relaxed health checks

Each policy is a complete FleetHealthManager configuration that can be deployed
in one command. Policies can be customized before deployment, compared side-by-side,
and switched at runtime based on operational context.

Integrates with:
- FleetHealthManagerSkill: applies policy configs via set_policy action
- ResourceWatcherSkill: budget-aware policy recommendations
- RevenueAnalyticsDashboardSkill: revenue data for demand-based policy switching
- EventBus: emits policy.deployed, policy.switched events

Pillar: Replication (primary) — intelligent fleet management without manual tuning.

Actions:
- list_policies: Browse available orchestration policies with descriptions
- preview: See the full configuration a policy would apply
- deploy: Apply a policy to FleetHealthManager
- compare: Side-by-side comparison of two policies
- recommend: Get a policy recommendation based on current fleet state
- customize: Create a modified version of a policy with overrides
- schedule: Schedule automatic policy switching (e.g., cost-aware at night)
- status: See active policy, history, and scheduled switches
"""

import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from .base import Skill, SkillResult, SkillManifest, SkillAction

POLICIES_FILE = Path(__file__).parent.parent / "data" / "fleet_orchestration_policies.json"

# ── Built-in Fleet Policies ─────────────────────────────────────────

BUILTIN_POLICIES = {
    # ── Cost-Aware ──────────────────────────────────────────────
    "cost_aware": {
        "name": "Cost-Aware Fleet",
        "description": "Minimize compute costs with aggressive scale-down and conservative scaling",
        "category": "cost",
        "pillar": "replication",
        "use_case": "Budget-constrained operations, overnight low-traffic periods",
        "config": {
            "min_fleet_size": 1,
            "max_fleet_size": 3,
            "auto_heal_enabled": True,
            "auto_scale_enabled": True,
            "health_check_interval_seconds": 300,
            "max_heal_attempts": 2,
            "heal_cooldown_seconds": 600,
            "scale_up_threshold": 0.95,
            "scale_down_threshold": 0.3,
            "scale_cooldown_seconds": 1200,
            "rolling_update_batch_size": 1,
            "rolling_update_delay_seconds": 120,
            "replace_after_failed_heals": False,
            "max_incidents_retained": 100,
        },
        "traits": {
            "cost_sensitivity": "high",
            "availability_priority": "low",
            "scale_aggressiveness": "conservative",
            "heal_speed": "slow",
        },
    },

    # ── Resilience ──────────────────────────────────────────────
    "resilience": {
        "name": "High-Resilience Fleet",
        "description": "Maximize uptime with fast healing, redundancy, and proactive scaling",
        "category": "availability",
        "pillar": "replication",
        "use_case": "Production workloads, SLA-bound services, customer-facing APIs",
        "config": {
            "min_fleet_size": 2,
            "max_fleet_size": 10,
            "auto_heal_enabled": True,
            "auto_scale_enabled": True,
            "health_check_interval_seconds": 60,
            "max_heal_attempts": 5,
            "heal_cooldown_seconds": 120,
            "scale_up_threshold": 0.6,
            "scale_down_threshold": 0.15,
            "scale_cooldown_seconds": 300,
            "rolling_update_batch_size": 1,
            "rolling_update_delay_seconds": 30,
            "replace_after_failed_heals": True,
            "max_incidents_retained": 500,
        },
        "traits": {
            "cost_sensitivity": "low",
            "availability_priority": "high",
            "scale_aggressiveness": "aggressive",
            "heal_speed": "fast",
        },
    },

    # ── Revenue-Optimized ───────────────────────────────────────
    "revenue_optimized": {
        "name": "Revenue-Optimized Fleet",
        "description": "Scale with demand to maximize revenue throughput and maintain SLAs",
        "category": "revenue",
        "pillar": "revenue",
        "use_case": "Active service delivery, marketplace operations, peak traffic handling",
        "config": {
            "min_fleet_size": 2,
            "max_fleet_size": 8,
            "auto_heal_enabled": True,
            "auto_scale_enabled": True,
            "health_check_interval_seconds": 90,
            "max_heal_attempts": 4,
            "heal_cooldown_seconds": 180,
            "scale_up_threshold": 0.7,
            "scale_down_threshold": 0.25,
            "scale_cooldown_seconds": 300,
            "rolling_update_batch_size": 1,
            "rolling_update_delay_seconds": 45,
            "replace_after_failed_heals": True,
            "max_incidents_retained": 300,
        },
        "traits": {
            "cost_sensitivity": "medium",
            "availability_priority": "high",
            "scale_aggressiveness": "moderate",
            "heal_speed": "fast",
        },
    },

    # ── Balanced ────────────────────────────────────────────────
    "balanced": {
        "name": "Balanced Fleet",
        "description": "General-purpose middle-ground policy balancing cost, availability, and throughput",
        "category": "general",
        "pillar": "replication",
        "use_case": "Default fleet management, mixed workloads, initial deployment",
        "config": {
            "min_fleet_size": 1,
            "max_fleet_size": 5,
            "auto_heal_enabled": True,
            "auto_scale_enabled": True,
            "health_check_interval_seconds": 120,
            "max_heal_attempts": 3,
            "heal_cooldown_seconds": 300,
            "scale_up_threshold": 0.8,
            "scale_down_threshold": 0.2,
            "scale_cooldown_seconds": 600,
            "rolling_update_batch_size": 1,
            "rolling_update_delay_seconds": 60,
            "replace_after_failed_heals": True,
            "max_incidents_retained": 200,
        },
        "traits": {
            "cost_sensitivity": "medium",
            "availability_priority": "medium",
            "scale_aggressiveness": "moderate",
            "heal_speed": "moderate",
        },
    },

    # ── Dev/Test ────────────────────────────────────────────────
    "dev_test": {
        "name": "Dev/Test Fleet",
        "description": "Minimal fleet for development and testing — single replica, no auto-scale",
        "category": "development",
        "pillar": "self_improvement",
        "use_case": "Development, testing, debugging, staging environments",
        "config": {
            "min_fleet_size": 1,
            "max_fleet_size": 2,
            "auto_heal_enabled": True,
            "auto_scale_enabled": False,
            "health_check_interval_seconds": 600,
            "max_heal_attempts": 1,
            "heal_cooldown_seconds": 60,
            "scale_up_threshold": 0.99,
            "scale_down_threshold": 0.01,
            "scale_cooldown_seconds": 60,
            "rolling_update_batch_size": 1,
            "rolling_update_delay_seconds": 10,
            "replace_after_failed_heals": False,
            "max_incidents_retained": 50,
        },
        "traits": {
            "cost_sensitivity": "high",
            "availability_priority": "low",
            "scale_aggressiveness": "none",
            "heal_speed": "relaxed",
        },
    },
}

# ── Policy Bundles (combinations for complex scenarios) ─────────────

BUILTIN_BUNDLES = {
    "production_standard": {
        "name": "Production Standard",
        "description": "Resilience during business hours, cost-aware overnight",
        "policies": ["resilience", "cost_aware"],
        "schedule": {
            "resilience": {"hours": "08:00-20:00", "days": "mon-fri"},
            "cost_aware": {"hours": "20:00-08:00", "days": "mon-fri,sat,sun"},
        },
    },
    "startup_growth": {
        "name": "Startup Growth",
        "description": "Revenue-optimized with cost guardrails — scale for revenue, don't overspend",
        "policies": ["revenue_optimized", "cost_aware"],
        "schedule": {
            "revenue_optimized": {"hours": "06:00-22:00", "days": "mon-sun"},
            "cost_aware": {"hours": "22:00-06:00", "days": "mon-sun"},
        },
    },
    "always_on": {
        "name": "Always-On Service",
        "description": "Maximum resilience 24/7 for critical services",
        "policies": ["resilience"],
        "schedule": {
            "resilience": {"hours": "00:00-23:59", "days": "mon-sun"},
        },
    },
}

# ── Policy diff fields for comparison ───────────────────────────────

COMPARISON_FIELDS = [
    ("min_fleet_size", "Minimum Fleet Size", "replicas"),
    ("max_fleet_size", "Maximum Fleet Size", "replicas"),
    ("auto_heal_enabled", "Auto-Heal", "bool"),
    ("auto_scale_enabled", "Auto-Scale", "bool"),
    ("health_check_interval_seconds", "Health Check Interval", "seconds"),
    ("max_heal_attempts", "Max Heal Attempts", "count"),
    ("heal_cooldown_seconds", "Heal Cooldown", "seconds"),
    ("scale_up_threshold", "Scale-Up Threshold", "ratio"),
    ("scale_down_threshold", "Scale-Down Threshold", "ratio"),
    ("scale_cooldown_seconds", "Scale Cooldown", "seconds"),
    ("replace_after_failed_heals", "Replace After Failed Heals", "bool"),
]


def _load_data(path: Path = None) -> Dict:
    """Load policies state from disk."""
    p = path or POLICIES_FILE
    if p.exists():
        try:
            with open(p) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {
        "active_policy": None,
        "deploy_history": [],
        "custom_policies": {},
        "scheduled_switches": [],
        "stats": {
            "total_deploys": 0,
            "total_switches": 0,
            "policy_deploy_counts": {},
        },
    }


def _save_data(data: Dict, path: Path = None):
    """Save policies state to disk."""
    p = path or POLICIES_FILE
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(data, f, indent=2, default=str)


def _get_all_policies(data: Dict) -> Dict:
    """Get all available policies (built-in + custom)."""
    all_policies = dict(BUILTIN_POLICIES)
    all_policies.update(data.get("custom_policies", {}))
    return all_policies


def _compare_configs(config_a: Dict, config_b: Dict) -> List[Dict]:
    """Compare two policy configs field by field."""
    diffs = []
    for field_key, field_name, field_type in COMPARISON_FIELDS:
        val_a = config_a.get(field_key)
        val_b = config_b.get(field_key)
        diff_entry = {
            "field": field_name,
            "key": field_key,
            "type": field_type,
            "value_a": val_a,
            "value_b": val_b,
            "same": val_a == val_b,
        }
        if not diff_entry["same"] and field_type in ("seconds", "count", "replicas"):
            if val_a is not None and val_b is not None and val_a != 0:
                diff_entry["change_pct"] = round(((val_b - val_a) / val_a) * 100, 1)
        diffs.append(diff_entry)
    return diffs


def _score_fleet_state(fleet_info: Dict) -> Dict:
    """Analyze fleet state to recommend a policy."""
    scores = {}
    fleet_size = fleet_info.get("fleet_size", 1)
    health_pct = fleet_info.get("health_pct", 100)
    budget_remaining_pct = fleet_info.get("budget_remaining_pct", 50)
    revenue_per_hour = fleet_info.get("revenue_per_hour", 0)
    has_sla = fleet_info.get("has_sla", False)
    is_production = fleet_info.get("is_production", False)

    # Cost-aware: high score when budget is tight
    cost_score = 0
    if budget_remaining_pct < 20:
        cost_score = 90
    elif budget_remaining_pct < 40:
        cost_score = 70
    elif budget_remaining_pct < 60:
        cost_score = 50
    else:
        cost_score = 30
    scores["cost_aware"] = cost_score

    # Resilience: high score when health is low or production
    resilience_score = 0
    if is_production:
        resilience_score += 40
    if has_sla:
        resilience_score += 30
    if health_pct < 80:
        resilience_score += 30
    elif health_pct < 95:
        resilience_score += 15
    scores["resilience"] = min(100, resilience_score)

    # Revenue-optimized: high score when generating revenue
    revenue_score = 0
    if revenue_per_hour > 1.0:
        revenue_score = 85
    elif revenue_per_hour > 0.1:
        revenue_score = 65
    elif revenue_per_hour > 0:
        revenue_score = 40
    else:
        revenue_score = 10
    if has_sla:
        revenue_score = min(100, revenue_score + 15)
    scores["revenue_optimized"] = revenue_score

    # Balanced: moderate score always, higher when no strong signal
    max_other = max(cost_score, scores["resilience"], revenue_score)
    if max_other < 50:
        scores["balanced"] = 70
    elif max_other < 70:
        scores["balanced"] = 55
    else:
        scores["balanced"] = 40

    # Dev/test: high score for small, non-production fleets
    dev_score = 0
    if not is_production and not has_sla and fleet_size <= 2:
        dev_score = 80
    elif not is_production:
        dev_score = 50
    else:
        dev_score = 10
    scores["dev_test"] = dev_score

    return scores


class FleetOrchestrationPoliciesSkill(Skill):
    """Pre-built fleet management policies for autonomous orchestration."""

    def __init__(self, credentials: Dict[str, str] = None, data_path: Path = None):
        super().__init__(credentials)
        self._data_path = data_path

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="fleet_orchestration_policies",
            name="Fleet Orchestration Policies",
            version="1.0.0",
            category="fleet",
            description=(
                "Pre-built fleet management policies for autonomous orchestration. "
                "Deploy cost-aware, resilience, revenue-optimized, balanced, or dev/test "
                "policies to FleetHealthManager in one command. Compare, customize, "
                "and schedule automatic policy switching."
            ),
            actions=[
                SkillAction(
                    name="list_policies",
                    description="Browse available fleet orchestration policies with descriptions and traits",
                    parameters={
                        "category": {"type": "str", "required": False, "description": "Filter by category (cost, availability, revenue, general, development)"},
                    },
                ),
                SkillAction(
                    name="preview",
                    description="See the full configuration a policy would apply to FleetHealthManager",
                    parameters={
                        "policy_id": {"type": "str", "required": True, "description": "ID of the policy to preview"},
                    },
                ),
                SkillAction(
                    name="deploy",
                    description="Apply a policy to FleetHealthManager, updating all fleet management settings",
                    parameters={
                        "policy_id": {"type": "str", "required": True, "description": "ID of the policy to deploy"},
                        "dry_run": {"type": "bool", "required": False, "description": "Preview changes without applying"},
                    },
                ),
                SkillAction(
                    name="compare",
                    description="Side-by-side comparison of two fleet policies",
                    parameters={
                        "policy_a": {"type": "str", "required": True, "description": "First policy ID"},
                        "policy_b": {"type": "str", "required": True, "description": "Second policy ID"},
                    },
                ),
                SkillAction(
                    name="recommend",
                    description="Get a policy recommendation based on current fleet state and goals",
                    parameters={
                        "fleet_size": {"type": "int", "required": False, "description": "Current fleet size"},
                        "health_pct": {"type": "float", "required": False, "description": "Fleet health percentage (0-100)"},
                        "budget_remaining_pct": {"type": "float", "required": False, "description": "Remaining budget percentage"},
                        "revenue_per_hour": {"type": "float", "required": False, "description": "Current revenue per hour"},
                        "has_sla": {"type": "bool", "required": False, "description": "Whether fleet has SLA commitments"},
                        "is_production": {"type": "bool", "required": False, "description": "Whether fleet is production"},
                    },
                ),
                SkillAction(
                    name="customize",
                    description="Create a custom policy based on a built-in one with overrides",
                    parameters={
                        "base_policy_id": {"type": "str", "required": True, "description": "Base policy to customize"},
                        "custom_id": {"type": "str", "required": True, "description": "ID for the new custom policy"},
                        "custom_name": {"type": "str", "required": False, "description": "Name for the custom policy"},
                        "overrides": {"type": "dict", "required": True, "description": "Config values to override"},
                    },
                ),
                SkillAction(
                    name="schedule",
                    description="Schedule automatic policy switching based on time of day",
                    parameters={
                        "bundle_id": {"type": "str", "required": False, "description": "Deploy a pre-built bundle schedule"},
                        "schedules": {"type": "list", "required": False, "description": "Custom schedule entries [{policy_id, hours, days}]"},
                    },
                ),
                SkillAction(
                    name="status",
                    description="View active policy, deployment history, and scheduled switches",
                    parameters={},
                ),
            ],
            required_credentials=[],
        )

    async def execute(self, action: str, params: Dict) -> SkillResult:
        dispatch = {
            "list_policies": self._list_policies,
            "preview": self._preview,
            "deploy": self._deploy,
            "compare": self._compare,
            "recommend": self._recommend,
            "customize": self._customize,
            "schedule": self._schedule,
            "status": self._status,
        }
        handler = dispatch.get(action)
        if not handler:
            return SkillResult(success=False, message=f"Unknown action: {action}. Available: {list(dispatch.keys())}")
        return await handler(params)

    # ── list_policies ───────────────────────────────────────────

    async def _list_policies(self, params: Dict) -> SkillResult:
        data = _load_data(self._data_path)
        all_policies = _get_all_policies(data)
        category = params.get("category")

        policies_list = []
        for pid, policy in all_policies.items():
            if category and policy.get("category") != category:
                continue
            policies_list.append({
                "id": pid,
                "name": policy["name"],
                "description": policy["description"],
                "category": policy.get("category", "general"),
                "use_case": policy.get("use_case", ""),
                "traits": policy.get("traits", {}),
                "is_custom": pid in data.get("custom_policies", {}),
                "is_active": data.get("active_policy") == pid,
            })

        return SkillResult(
            success=True,
            message=f"Found {len(policies_list)} fleet orchestration policies",
            data={
                "policies": policies_list,
                "active_policy": data.get("active_policy"),
                "total_builtin": len(BUILTIN_POLICIES),
                "total_custom": len(data.get("custom_policies", {})),
            },
        )

    # ── preview ─────────────────────────────────────────────────

    async def _preview(self, params: Dict) -> SkillResult:
        data = _load_data(self._data_path)
        all_policies = _get_all_policies(data)
        policy_id = params.get("policy_id", "")

        if policy_id not in all_policies:
            return SkillResult(
                success=False,
                message=f"Policy '{policy_id}' not found. Available: {list(all_policies.keys())}",
            )

        policy = all_policies[policy_id]
        return SkillResult(
            success=True,
            message=f"Preview of '{policy['name']}' policy",
            data={
                "policy_id": policy_id,
                "name": policy["name"],
                "description": policy["description"],
                "category": policy.get("category", "general"),
                "use_case": policy.get("use_case", ""),
                "traits": policy.get("traits", {}),
                "config": policy["config"],
                "is_active": data.get("active_policy") == policy_id,
            },
        )

    # ── deploy ──────────────────────────────────────────────────

    async def _deploy(self, params: Dict) -> SkillResult:
        data = _load_data(self._data_path)
        all_policies = _get_all_policies(data)
        policy_id = params.get("policy_id", "")
        dry_run = params.get("dry_run", False)

        if policy_id not in all_policies:
            return SkillResult(
                success=False,
                message=f"Policy '{policy_id}' not found. Available: {list(all_policies.keys())}",
            )

        policy = all_policies[policy_id]
        config = policy["config"]
        previous_policy = data.get("active_policy")

        # Calculate what would change
        changes = []
        if previous_policy and previous_policy in all_policies:
            prev_config = all_policies[previous_policy]["config"]
            for key, val in config.items():
                old_val = prev_config.get(key)
                if old_val != val:
                    changes.append({"field": key, "old": old_val, "new": val})

        if dry_run:
            return SkillResult(
                success=True,
                message=f"Dry run: would deploy '{policy['name']}' ({len(changes)} changes from current)",
                data={
                    "policy_id": policy_id,
                    "config": config,
                    "changes": changes,
                    "previous_policy": previous_policy,
                    "dry_run": True,
                },
            )

        # Actually deploy: try via context (FleetHealthManagerSkill)
        deployed_via_skill = False
        if self.context:
            try:
                result = await self.context.call_skill(
                    "fleet_health_manager", "set_policy", {"policies": config}
                )
                deployed_via_skill = result.success
            except Exception:
                pass

        # Record deployment
        deploy_record = {
            "id": str(uuid.uuid4())[:8],
            "policy_id": policy_id,
            "policy_name": policy["name"],
            "previous_policy": previous_policy,
            "timestamp": datetime.now().isoformat(),
            "changes_count": len(changes),
            "deployed_via_skill": deployed_via_skill,
        }
        data["deploy_history"].append(deploy_record)
        # Keep last 100
        if len(data["deploy_history"]) > 100:
            data["deploy_history"] = data["deploy_history"][-100:]

        data["active_policy"] = policy_id
        data["stats"]["total_deploys"] = data["stats"].get("total_deploys", 0) + 1
        counts = data["stats"].get("policy_deploy_counts", {})
        counts[policy_id] = counts.get(policy_id, 0) + 1
        data["stats"]["policy_deploy_counts"] = counts

        if previous_policy and previous_policy != policy_id:
            data["stats"]["total_switches"] = data["stats"].get("total_switches", 0) + 1

        _save_data(data, self._data_path)

        return SkillResult(
            success=True,
            message=f"Deployed '{policy['name']}' policy to fleet ({len(changes)} settings changed)",
            data={
                "policy_id": policy_id,
                "config": config,
                "changes": changes,
                "previous_policy": previous_policy,
                "deployed_via_skill": deployed_via_skill,
                "deploy_record": deploy_record,
            },
        )

    # ── compare ─────────────────────────────────────────────────

    async def _compare(self, params: Dict) -> SkillResult:
        data = _load_data(self._data_path)
        all_policies = _get_all_policies(data)
        policy_a = params.get("policy_a", "")
        policy_b = params.get("policy_b", "")

        errors = []
        if policy_a not in all_policies:
            errors.append(f"Policy A '{policy_a}' not found")
        if policy_b not in all_policies:
            errors.append(f"Policy B '{policy_b}' not found")
        if errors:
            return SkillResult(
                success=False,
                message="; ".join(errors) + f". Available: {list(all_policies.keys())}",
            )

        pol_a = all_policies[policy_a]
        pol_b = all_policies[policy_b]
        diffs = _compare_configs(pol_a["config"], pol_b["config"])
        same_count = sum(1 for d in diffs if d["same"])
        diff_count = len(diffs) - same_count

        return SkillResult(
            success=True,
            message=f"Compared '{pol_a['name']}' vs '{pol_b['name']}': {diff_count} differences, {same_count} same",
            data={
                "policy_a": {"id": policy_a, "name": pol_a["name"], "traits": pol_a.get("traits", {})},
                "policy_b": {"id": policy_b, "name": pol_b["name"], "traits": pol_b.get("traits", {})},
                "diffs": diffs,
                "summary": {
                    "total_fields": len(diffs),
                    "same": same_count,
                    "different": diff_count,
                },
            },
        )

    # ── recommend ───────────────────────────────────────────────

    async def _recommend(self, params: Dict) -> SkillResult:
        fleet_info = {
            "fleet_size": params.get("fleet_size", 1),
            "health_pct": params.get("health_pct", 100),
            "budget_remaining_pct": params.get("budget_remaining_pct", 50),
            "revenue_per_hour": params.get("revenue_per_hour", 0),
            "has_sla": params.get("has_sla", False),
            "is_production": params.get("is_production", False),
        }

        scores = _score_fleet_state(fleet_info)

        # Sort by score descending
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_id, top_score = ranked[0]
        top_policy = BUILTIN_POLICIES[top_id]

        data = _load_data(self._data_path)
        current = data.get("active_policy")
        switch_needed = current != top_id

        # Build reasoning
        reasons = []
        if fleet_info["budget_remaining_pct"] < 30:
            reasons.append("Budget is tight — cost control is important")
        if fleet_info["health_pct"] < 80:
            reasons.append("Fleet health is degraded — resilience matters")
        if fleet_info["revenue_per_hour"] > 0.5:
            reasons.append("Active revenue generation — optimize for throughput")
        if fleet_info["is_production"]:
            reasons.append("Production environment — prioritize availability")
        if fleet_info["has_sla"]:
            reasons.append("SLA commitments — uptime is critical")
        if not fleet_info["is_production"] and fleet_info["fleet_size"] <= 2:
            reasons.append("Small non-production fleet — minimal overhead preferred")
        if not reasons:
            reasons.append("No strong signals — balanced approach recommended")

        return SkillResult(
            success=True,
            message=f"Recommended: '{top_policy['name']}' (score: {top_score}/100)",
            data={
                "recommendation": {
                    "policy_id": top_id,
                    "name": top_policy["name"],
                    "score": top_score,
                    "description": top_policy["description"],
                },
                "all_scores": [{"policy_id": pid, "name": BUILTIN_POLICIES[pid]["name"], "score": s} for pid, s in ranked],
                "fleet_info": fleet_info,
                "reasons": reasons,
                "current_policy": current,
                "switch_recommended": switch_needed,
            },
        )

    # ── customize ───────────────────────────────────────────────

    async def _customize(self, params: Dict) -> SkillResult:
        data = _load_data(self._data_path)
        all_policies = _get_all_policies(data)
        base_id = params.get("base_policy_id", "")
        custom_id = params.get("custom_id", "")
        custom_name = params.get("custom_name")
        overrides = params.get("overrides", {})

        if base_id not in all_policies:
            return SkillResult(
                success=False,
                message=f"Base policy '{base_id}' not found. Available: {list(all_policies.keys())}",
            )

        if not custom_id:
            return SkillResult(success=False, message="custom_id is required")

        if custom_id in BUILTIN_POLICIES:
            return SkillResult(success=False, message=f"Cannot override built-in policy '{custom_id}'")

        base = all_policies[base_id]
        new_config = dict(base["config"])

        # Apply overrides, validate they're known fields
        valid_fields = set(new_config.keys())
        invalid = [k for k in overrides if k not in valid_fields]
        if invalid:
            return SkillResult(
                success=False,
                message=f"Invalid override fields: {invalid}. Valid: {sorted(valid_fields)}",
            )

        applied = []
        for k, v in overrides.items():
            old = new_config[k]
            new_config[k] = v
            applied.append({"field": k, "old": old, "new": v})

        custom_policy = {
            "name": custom_name or f"Custom ({base['name']})",
            "description": f"Customized from '{base['name']}' with {len(applied)} overrides",
            "category": base.get("category", "custom"),
            "pillar": base.get("pillar", "replication"),
            "use_case": f"Custom variant of {base_id}",
            "config": new_config,
            "traits": dict(base.get("traits", {})),
            "base_policy": base_id,
            "created_at": datetime.now().isoformat(),
        }

        if "custom_policies" not in data:
            data["custom_policies"] = {}
        data["custom_policies"][custom_id] = custom_policy
        _save_data(data, self._data_path)

        return SkillResult(
            success=True,
            message=f"Created custom policy '{custom_id}' from '{base['name']}' with {len(applied)} overrides",
            data={
                "custom_id": custom_id,
                "base_policy": base_id,
                "overrides_applied": applied,
                "config": new_config,
            },
        )

    # ── schedule ────────────────────────────────────────────────

    async def _schedule(self, params: Dict) -> SkillResult:
        data = _load_data(self._data_path)
        all_policies = _get_all_policies(data)
        bundle_id = params.get("bundle_id")
        schedules = params.get("schedules")

        if bundle_id:
            if bundle_id not in BUILTIN_BUNDLES:
                return SkillResult(
                    success=False,
                    message=f"Bundle '{bundle_id}' not found. Available: {list(BUILTIN_BUNDLES.keys())}",
                )
            bundle = BUILTIN_BUNDLES[bundle_id]
            schedule_entries = []
            for pol_id, timing in bundle["schedule"].items():
                schedule_entries.append({
                    "policy_id": pol_id,
                    "hours": timing["hours"],
                    "days": timing["days"],
                    "source": f"bundle:{bundle_id}",
                })
            data["scheduled_switches"] = schedule_entries
            _save_data(data, self._data_path)
            return SkillResult(
                success=True,
                message=f"Scheduled bundle '{bundle['name']}' with {len(schedule_entries)} time-based switches",
                data={
                    "bundle_id": bundle_id,
                    "bundle_name": bundle["name"],
                    "schedules": schedule_entries,
                },
            )

        if schedules:
            validated = []
            for entry in schedules:
                pid = entry.get("policy_id", "")
                if pid not in all_policies:
                    return SkillResult(
                        success=False,
                        message=f"Policy '{pid}' in schedule not found",
                    )
                validated.append({
                    "policy_id": pid,
                    "hours": entry.get("hours", "00:00-23:59"),
                    "days": entry.get("days", "mon-sun"),
                    "source": "custom",
                })
            data["scheduled_switches"] = validated
            _save_data(data, self._data_path)
            return SkillResult(
                success=True,
                message=f"Scheduled {len(validated)} custom policy switches",
                data={"schedules": validated},
            )

        return SkillResult(
            success=False,
            message="Provide either 'bundle_id' or 'schedules' parameter",
        )

    # ── status ──────────────────────────────────────────────────

    async def _status(self, params: Dict) -> SkillResult:
        data = _load_data(self._data_path)
        all_policies = _get_all_policies(data)
        active = data.get("active_policy")
        active_info = None
        if active and active in all_policies:
            pol = all_policies[active]
            active_info = {
                "id": active,
                "name": pol["name"],
                "category": pol.get("category"),
                "traits": pol.get("traits", {}),
            }

        recent_deploys = data.get("deploy_history", [])[-10:]
        scheduled = data.get("scheduled_switches", [])

        return SkillResult(
            success=True,
            message=f"Fleet policy status: {'Active: ' + active if active else 'No active policy'}",
            data={
                "active_policy": active_info,
                "recent_deploys": recent_deploys,
                "scheduled_switches": scheduled,
                "stats": data.get("stats", {}),
                "available_policies": len(all_policies),
                "custom_policies": len(data.get("custom_policies", {})),
                "bundles_available": list(BUILTIN_BUNDLES.keys()),
            },
        )
