#!/usr/bin/env python3
"""
AdaptiveCircuitThresholdsSkill - Auto-tune circuit breaker thresholds per skill.

The circuit breaker uses static global thresholds (50% failure rate, 5 consecutive
failures, $0.10 cost/success, 60s cooldown). But different skills have fundamentally
different reliability profiles:

- An LLM API skill might have 10% natural failure rate (rate limits, timeouts)
  → 50% threshold is fine, but cooldown should be shorter (APIs recover fast)
- A file system skill should have 0% failure rate
  → threshold should be lower (3% = something is wrong), longer cooldown
- A web scraping skill might have 30% natural failure rate (sites go down)
  → 50% threshold is too aggressive, should be 60%+
- A payment skill must be ultra-reliable
  → threshold should be very low (5%), long cooldown, more half-open tests

This skill solves this by analyzing each skill's historical performance and
computing per-skill threshold overrides. It uses statistical analysis of the
skill's baseline behavior to set appropriate thresholds.

Algorithm:
1. Collect performance data from circuit breaker records
2. Compute each skill's baseline metrics (mean failure rate, std dev, cost patterns)
3. Set thresholds at baseline + N standard deviations (configurable sensitivity)
4. Apply overrides to circuit breaker via its configure action
5. Track tuning effectiveness over time

Pillar: Self-Improvement (auto-tuning based on observed performance)
"""

import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import Skill, SkillAction, SkillManifest, SkillResult

DATA_DIR = Path(__file__).parent.parent / "data"
THRESHOLDS_FILE = DATA_DIR / "adaptive_thresholds.json"
MAX_HISTORY = 200
MAX_PROFILES = 200


class AdaptiveCircuitThresholdsSkill(Skill):
    """
    Auto-tunes circuit breaker thresholds based on historical skill performance.

    Instead of one-size-fits-all thresholds, each skill gets personalized
    thresholds based on its observed behavior patterns.

    Actions:
    - analyze: Analyze a skill's historical performance and compute optimal thresholds
    - tune: Apply computed thresholds to the circuit breaker for a specific skill
    - tune_all: Analyze and tune all skills with sufficient data
    - profiles: View all skill performance profiles
    - history: View tuning history
    - configure: Update adaptive tuning parameters
    - status: Get current tuning status
    - reset: Reset a skill's profile to use global defaults
    """

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self._profiles: Dict[str, Dict] = {}  # skill_id -> performance profile
        self._overrides: Dict[str, Dict] = {}  # skill_id -> threshold overrides
        self._tuning_history: List[Dict] = []
        self._config = {
            "sensitivity": 2.0,  # N std devs above baseline before triggering
            "min_samples": 10,  # Minimum records needed to compute profile
            "recalc_interval_seconds": 3600,  # Recalculate hourly
            "max_failure_rate_threshold": 0.9,  # Never set threshold above 90%
            "min_failure_rate_threshold": 0.05,  # Never set threshold below 5%
            "max_cooldown_seconds": 600,  # Max 10 minutes cooldown
            "min_cooldown_seconds": 10,  # Min 10 seconds cooldown
            "cost_sensitivity_multiplier": 1.5,  # How much above avg cost triggers
            "enable_auto_apply": False,  # Auto-apply after analysis
        }
        self._load_state()

    def _load_state(self):
        """Load persisted state."""
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        try:
            with open(THRESHOLDS_FILE, "r") as f:
                data = json.load(f)
            self._profiles = data.get("profiles", {})
            self._overrides = data.get("overrides", {})
            self._tuning_history = data.get("history", [])[-MAX_HISTORY:]
            self._config.update(data.get("config", {}))
        except (FileNotFoundError, json.JSONDecodeError):
            pass

    def _save_state(self):
        """Persist state to disk."""
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        data = {
            "profiles": dict(list(self._profiles.items())[:MAX_PROFILES]),
            "overrides": self._overrides,
            "history": self._tuning_history[-MAX_HISTORY:],
            "config": self._config,
            "last_saved": datetime.now().isoformat(),
        }
        with open(THRESHOLDS_FILE, "w") as f:
            json.dump(data, f, indent=2, default=str)

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="adaptive_circuit_thresholds",
            name="Adaptive Circuit Thresholds",
            version="1.0.0",
            category="infrastructure",
            description="Auto-tune circuit breaker thresholds per skill based on historical performance patterns",
            actions=self.get_actions(),
            required_credentials=[],
        )

    def get_actions(self) -> List[SkillAction]:
        return [
            SkillAction(
                name="analyze",
                description="Analyze a skill's performance and compute optimal thresholds",
                parameters={
                    "skill_id": {"type": "string", "required": True, "description": "Skill to analyze"},
                    "circuit_data": {"type": "object", "required": True, "description": "Circuit data from circuit_breaker dashboard/status"},
                },
            ),
            SkillAction(
                name="tune",
                description="Apply computed thresholds for a specific skill",
                parameters={
                    "skill_id": {"type": "string", "required": True, "description": "Skill to tune"},
                },
            ),
            SkillAction(
                name="tune_all",
                description="Analyze and compute optimal thresholds for all skills with data",
                parameters={
                    "circuits_data": {"type": "object", "required": True, "description": "All circuits data from dashboard"},
                },
            ),
            SkillAction(
                name="profiles",
                description="View all skill performance profiles and computed thresholds",
                parameters={
                    "skill_id": {"type": "string", "required": False, "description": "Filter to specific skill"},
                },
            ),
            SkillAction(
                name="history",
                description="View tuning history",
                parameters={
                    "limit": {"type": "integer", "required": False, "description": "Number of entries (default 20)"},
                },
            ),
            SkillAction(
                name="configure",
                description="Update adaptive tuning parameters",
                parameters={
                    "sensitivity": {"type": "float", "required": False, "description": "Std devs above baseline (default 2.0)"},
                    "min_samples": {"type": "integer", "required": False, "description": "Min records needed (default 10)"},
                    "enable_auto_apply": {"type": "boolean", "required": False, "description": "Auto-apply thresholds after analysis"},
                },
            ),
            SkillAction(
                name="status",
                description="Get current adaptive tuning status",
                parameters={},
            ),
            SkillAction(
                name="reset",
                description="Reset a skill's profile to use global defaults",
                parameters={
                    "skill_id": {"type": "string", "required": True, "description": "Skill to reset"},
                },
            ),
        ]

    def check_credentials(self) -> bool:
        return True

    async def execute(self, action: str, params: Dict) -> SkillResult:
        actions = {
            "analyze": self._analyze,
            "tune": self._tune,
            "tune_all": self._tune_all,
            "profiles": self._profiles_action,
            "history": self._history,
            "configure": self._configure,
            "status": self._status,
            "reset": self._reset,
        }
        handler = actions.get(action)
        if not handler:
            return SkillResult(
                success=False,
                message=f"Unknown action: {action}. Available: {list(actions.keys())}",
            )
        return handler(params)

    def _compute_stats(self, records: List[Dict]) -> Dict:
        """Compute statistical profile from circuit records."""
        if not records:
            return {}

        successes = [r for r in records if r.get("success", False)]
        failures = [r for r in records if not r.get("success", False)]

        total = len(records)
        failure_count = len(failures)
        success_count = len(successes)

        # Failure rate
        failure_rate = failure_count / total if total > 0 else 0.0

        # Compute failure rate variance using windowed approach
        # Break records into windows of 10 and compute per-window rates
        window_size = min(10, max(3, total // 5))
        window_rates = []
        for i in range(0, total - window_size + 1, max(1, window_size // 2)):
            window = records[i:i + window_size]
            w_failures = sum(1 for r in window if not r.get("success", False))
            window_rates.append(w_failures / len(window))

        rate_std = 0.0
        if len(window_rates) >= 2:
            mean_rate = sum(window_rates) / len(window_rates)
            variance = sum((r - mean_rate) ** 2 for r in window_rates) / (len(window_rates) - 1)
            rate_std = math.sqrt(variance)

        # Cost analysis
        costs = [r.get("cost", 0.0) for r in records if r.get("cost", 0.0) > 0]
        avg_cost = sum(costs) / len(costs) if costs else 0.0
        cost_std = 0.0
        if len(costs) >= 2:
            cost_variance = sum((c - avg_cost) ** 2 for c in costs) / (len(costs) - 1)
            cost_std = math.sqrt(cost_variance)

        # Cost per success
        total_cost = sum(costs)
        cost_per_success = total_cost / success_count if success_count > 0 else float("inf")

        # Duration analysis
        durations = [r.get("duration_ms", 0.0) for r in records if r.get("duration_ms", 0.0) > 0]
        avg_duration = sum(durations) / len(durations) if durations else 0.0

        # Failure burst analysis - longest consecutive failure streak
        max_streak = 0
        current_streak = 0
        for r in records:
            if not r.get("success", False):
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0

        # Recovery time analysis - avg time between circuit opens
        # (approximated by time between failure bursts ending)
        recovery_times = []
        in_failure = False
        failure_start = 0.0
        for r in records:
            ts = r.get("timestamp", 0.0)
            if not r.get("success", False):
                if not in_failure:
                    failure_start = ts
                    in_failure = True
            else:
                if in_failure and ts > failure_start:
                    recovery_times.append(ts - failure_start)
                    in_failure = False

        avg_recovery_time = (
            sum(recovery_times) / len(recovery_times) if recovery_times else 60.0
        )

        return {
            "total_records": total,
            "failure_count": failure_count,
            "success_count": success_count,
            "failure_rate": round(failure_rate, 4),
            "failure_rate_std": round(rate_std, 4),
            "window_rates": [round(r, 3) for r in window_rates[-10:]],
            "avg_cost": round(avg_cost, 6),
            "cost_std": round(cost_std, 6),
            "cost_per_success": round(cost_per_success, 6) if cost_per_success != float("inf") else "inf",
            "total_cost": round(total_cost, 6),
            "avg_duration_ms": round(avg_duration, 2),
            "max_failure_streak": max_streak,
            "avg_recovery_seconds": round(avg_recovery_time, 1),
            "computed_at": datetime.now().isoformat(),
        }

    def _compute_thresholds(self, stats: Dict) -> Dict:
        """Compute optimal thresholds from statistical profile."""
        sensitivity = self._config["sensitivity"]

        # Failure rate threshold: baseline + N * std_dev
        # This means the threshold triggers only when failure rate is
        # significantly above the skill's normal behavior
        baseline_rate = stats.get("failure_rate", 0.0)
        rate_std = stats.get("failure_rate_std", 0.1)
        # Ensure minimum std dev so threshold isn't too tight
        rate_std = max(rate_std, 0.05)

        failure_rate_threshold = baseline_rate + sensitivity * rate_std
        failure_rate_threshold = max(
            self._config["min_failure_rate_threshold"],
            min(self._config["max_failure_rate_threshold"], failure_rate_threshold),
        )

        # Consecutive failure threshold: based on max observed streak + buffer
        max_streak = stats.get("max_failure_streak", 0)
        # Allow slightly more than historical max (natural variation)
        consecutive_threshold = max(3, max_streak + max(2, int(max_streak * 0.5)))

        # Cooldown: based on observed recovery time
        avg_recovery = stats.get("avg_recovery_seconds", 60.0)
        # Cooldown should be ~1.5x average recovery time
        cooldown = avg_recovery * 1.5
        cooldown = max(
            self._config["min_cooldown_seconds"],
            min(self._config["max_cooldown_seconds"], cooldown),
        )

        # Cost per success threshold: based on observed cost patterns
        avg_cost = stats.get("avg_cost", 0.0)
        cost_std = stats.get("cost_std", 0.0)
        multiplier = self._config["cost_sensitivity_multiplier"]
        if avg_cost > 0:
            cost_threshold = (avg_cost + cost_std) * multiplier
        else:
            cost_threshold = 0.10  # Default if no cost data

        # Min window size: scale with data availability
        total_records = stats.get("total_records", 0)
        # More data = can require more samples for decisions
        min_window = max(3, min(20, total_records // 10))

        return {
            "failure_rate_threshold": round(failure_rate_threshold, 3),
            "consecutive_failure_threshold": consecutive_threshold,
            "cooldown_seconds": round(cooldown, 1),
            "cost_per_success_threshold": round(cost_threshold, 6),
            "min_window_size": min_window,
        }

    def _analyze(self, params: Dict) -> SkillResult:
        """Analyze a skill's performance and compute optimal thresholds."""
        skill_id = params.get("skill_id")
        if not skill_id:
            return SkillResult(success=False, message="skill_id is required")

        circuit_data = params.get("circuit_data", {})

        # Build records from circuit data
        records = circuit_data.get("records", [])
        # Also accept summary stats if raw records aren't available
        if not records and circuit_data:
            # Synthesize from summary data for analysis
            records = self._synthesize_records(circuit_data)

        if len(records) < self._config["min_samples"]:
            return SkillResult(
                success=False,
                message=f"Insufficient data for {skill_id}: {len(records)} records, need {self._config['min_samples']}",
                data={"records_available": len(records), "min_required": self._config["min_samples"]},
            )

        # Compute statistical profile
        stats = self._compute_stats(records)

        # Compute optimal thresholds
        thresholds = self._compute_thresholds(stats)

        # Store profile
        self._profiles[skill_id] = {
            "skill_id": skill_id,
            "stats": stats,
            "thresholds": thresholds,
            "analyzed_at": datetime.now().isoformat(),
            "sample_count": len(records),
        }

        # Log tuning event
        self._tuning_history.append({
            "action": "analyze",
            "skill_id": skill_id,
            "thresholds": thresholds,
            "baseline_failure_rate": stats["failure_rate"],
            "timestamp": datetime.now().isoformat(),
        })

        # Auto-apply if configured
        auto_applied = False
        if self._config["enable_auto_apply"]:
            self._overrides[skill_id] = thresholds
            auto_applied = True

        self._save_state()

        msg = f"Analyzed {skill_id}: baseline failure rate {stats['failure_rate']:.1%} (±{stats['failure_rate_std']:.1%})"
        msg += f"\nComputed thresholds: fail_rate={thresholds['failure_rate_threshold']:.1%}"
        msg += f", consec_fails={thresholds['consecutive_failure_threshold']}"
        msg += f", cooldown={thresholds['cooldown_seconds']:.0f}s"
        msg += f", cost/success=${thresholds['cost_per_success_threshold']:.4f}"
        if auto_applied:
            msg += "\n[Auto-applied to circuit breaker]"

        return SkillResult(
            success=True,
            message=msg,
            data={
                "skill_id": skill_id,
                "stats": stats,
                "thresholds": thresholds,
                "auto_applied": auto_applied,
            },
        )

    def _synthesize_records(self, circuit_data: Dict) -> List[Dict]:
        """Synthesize records from summary circuit data when raw records aren't available."""
        records = []
        total = circuit_data.get("window_size", 0) or circuit_data.get("total_requests", 0)
        failure_rate = circuit_data.get("failure_rate", 0.0)
        if isinstance(failure_rate, str):
            try:
                failure_rate = float(failure_rate)
            except ValueError:
                failure_rate = 0.0

        total_cost = circuit_data.get("total_cost", 0.0)
        if isinstance(total_cost, str):
            try:
                total_cost = float(total_cost)
            except ValueError:
                total_cost = 0.0

        if total == 0:
            return records

        failures = int(total * failure_rate)
        successes = total - failures
        cost_per_record = total_cost / total if total > 0 else 0.0

        now = time.time()
        # Create synthetic records spread over time
        for i in range(total):
            is_failure = i < failures
            records.append({
                "timestamp": now - (total - i) * 60,  # 1 min apart
                "success": not is_failure,
                "cost": cost_per_record,
                "duration_ms": 100.0,
            })

        # Consecutive failures from circuit data
        consec = circuit_data.get("consecutive_failures", 0)
        if consec > 0 and len(records) >= consec:
            # Put consecutive failures at the end
            for i in range(consec):
                records[-(i + 1)]["success"] = False

        return records

    def _tune(self, params: Dict) -> SkillResult:
        """Apply computed thresholds for a specific skill."""
        skill_id = params.get("skill_id")
        if not skill_id:
            return SkillResult(success=False, message="skill_id is required")

        profile = self._profiles.get(skill_id)
        if not profile:
            return SkillResult(
                success=False,
                message=f"No profile found for {skill_id}. Run 'analyze' first.",
            )

        thresholds = profile["thresholds"]
        self._overrides[skill_id] = thresholds

        self._tuning_history.append({
            "action": "tune",
            "skill_id": skill_id,
            "thresholds": thresholds,
            "timestamp": datetime.now().isoformat(),
        })

        self._save_state()

        return SkillResult(
            success=True,
            message=f"Applied adaptive thresholds for {skill_id}: "
                    f"fail_rate={thresholds['failure_rate_threshold']:.1%}, "
                    f"consec_fails={thresholds['consecutive_failure_threshold']}, "
                    f"cooldown={thresholds['cooldown_seconds']:.0f}s",
            data={
                "skill_id": skill_id,
                "thresholds": thresholds,
                "profile": profile,
            },
        )

    def _tune_all(self, params: Dict) -> SkillResult:
        """Analyze and tune all skills with sufficient data."""
        circuits_data = params.get("circuits_data", {})

        if not circuits_data:
            return SkillResult(
                success=False,
                message="circuits_data is required (pass circuit breaker dashboard data)",
            )

        # circuits_data can be a dict of skill_id -> circuit_info
        # or a list of circuit dicts
        if isinstance(circuits_data, list):
            circuits = {c.get("skill_id", f"unknown_{i}"): c for i, c in enumerate(circuits_data)}
        elif isinstance(circuits_data, dict):
            # Could be {skill_id: data} or a dashboard response with worst_skills etc.
            if "worst_skills" in circuits_data:
                # Dashboard format - extract individual circuits
                circuits = {}
                for ws in circuits_data.get("worst_skills", []):
                    sid = ws.get("skill_id")
                    if sid:
                        circuits[sid] = ws
            else:
                circuits = circuits_data
        else:
            return SkillResult(success=False, message="circuits_data must be a dict or list")

        analyzed = []
        skipped = []
        errors = []

        for skill_id, circuit_info in circuits.items():
            result = self._analyze({
                "skill_id": skill_id,
                "circuit_data": circuit_info,
            })
            if result.success:
                self._overrides[skill_id] = result.data["thresholds"]
                analyzed.append({
                    "skill_id": skill_id,
                    "thresholds": result.data["thresholds"],
                    "baseline_rate": result.data["stats"]["failure_rate"],
                })
            else:
                skipped.append({"skill_id": skill_id, "reason": result.message})

        self._save_state()

        msg = f"Tuned {len(analyzed)} skills"
        if skipped:
            msg += f", skipped {len(skipped)} (insufficient data)"

        return SkillResult(
            success=True,
            message=msg,
            data={
                "analyzed": analyzed,
                "skipped": skipped,
                "errors": errors,
                "total_overrides": len(self._overrides),
            },
        )

    def _profiles_action(self, params: Dict) -> SkillResult:
        """View skill performance profiles."""
        skill_id = params.get("skill_id")

        if skill_id:
            profile = self._profiles.get(skill_id)
            if not profile:
                return SkillResult(
                    success=False,
                    message=f"No profile for {skill_id}",
                )
            override = self._overrides.get(skill_id)
            return SkillResult(
                success=True,
                message=f"Profile for {skill_id}: baseline failure rate {profile['stats']['failure_rate']:.1%}",
                data={
                    "profile": profile,
                    "override_active": override is not None,
                    "override": override,
                },
            )

        # List all profiles
        summaries = []
        for sid, profile in self._profiles.items():
            summaries.append({
                "skill_id": sid,
                "baseline_failure_rate": profile["stats"]["failure_rate"],
                "computed_threshold": profile["thresholds"]["failure_rate_threshold"],
                "override_active": sid in self._overrides,
                "sample_count": profile["sample_count"],
                "analyzed_at": profile["analyzed_at"],
            })

        return SkillResult(
            success=True,
            message=f"{len(summaries)} skill profiles computed",
            data={"profiles": summaries, "total_overrides": len(self._overrides)},
        )

    def _history(self, params: Dict) -> SkillResult:
        """View tuning history."""
        limit = int(params.get("limit", 20))
        entries = self._tuning_history[-limit:]

        return SkillResult(
            success=True,
            message=f"Showing {len(entries)} tuning events",
            data={"history": entries, "total": len(self._tuning_history)},
        )

    def _configure(self, params: Dict) -> SkillResult:
        """Update adaptive tuning parameters."""
        updated = []
        valid_keys = {
            "sensitivity", "min_samples", "recalc_interval_seconds",
            "max_failure_rate_threshold", "min_failure_rate_threshold",
            "max_cooldown_seconds", "min_cooldown_seconds",
            "cost_sensitivity_multiplier", "enable_auto_apply",
        }

        for key in valid_keys:
            if key in params:
                val = params[key]
                if key == "sensitivity":
                    val = max(0.5, min(5.0, float(val)))
                elif key == "min_samples":
                    val = max(3, int(val))
                elif key in ("recalc_interval_seconds", "max_cooldown_seconds", "min_cooldown_seconds"):
                    val = max(1, int(val))
                elif key in ("max_failure_rate_threshold", "min_failure_rate_threshold"):
                    val = max(0.01, min(1.0, float(val)))
                elif key == "cost_sensitivity_multiplier":
                    val = max(1.0, min(10.0, float(val)))
                elif key == "enable_auto_apply":
                    val = bool(val)
                self._config[key] = val
                updated.append(key)

        if not updated:
            return SkillResult(
                success=True,
                message="No changes. Current config shown in data.",
                data={"config": self._config},
            )

        self._save_state()
        return SkillResult(
            success=True,
            message=f"Updated {len(updated)} settings: {', '.join(updated)}",
            data={"updated": updated, "config": self._config},
        )

    def _status(self, params: Dict) -> SkillResult:
        """Get current adaptive tuning status."""
        total_profiles = len(self._profiles)
        total_overrides = len(self._overrides)
        total_tuning_events = len(self._tuning_history)

        # Compute summary of threshold adjustments
        adjustments = []
        for sid, override in self._overrides.items():
            profile = self._profiles.get(sid, {})
            stats = profile.get("stats", {})
            adjustments.append({
                "skill_id": sid,
                "baseline_rate": stats.get("failure_rate", 0.0),
                "adaptive_threshold": override.get("failure_rate_threshold", 0.5),
                "adaptive_cooldown": override.get("cooldown_seconds", 60),
                "sample_count": profile.get("sample_count", 0),
            })

        msg_lines = ["=== Adaptive Circuit Thresholds Status ==="]
        msg_lines.append(f"Profiles: {total_profiles} | Overrides active: {total_overrides}")
        msg_lines.append(f"Tuning events: {total_tuning_events}")
        msg_lines.append(f"Sensitivity: {self._config['sensitivity']} std devs")
        msg_lines.append(f"Auto-apply: {'ON' if self._config['enable_auto_apply'] else 'OFF'}")

        if adjustments:
            msg_lines.append("\nActive overrides:")
            for adj in adjustments[:5]:
                msg_lines.append(
                    f"  {adj['skill_id']}: baseline={adj['baseline_rate']:.1%} "
                    f"→ threshold={adj['adaptive_threshold']:.1%} "
                    f"(cooldown={adj['adaptive_cooldown']:.0f}s)"
                )

        return SkillResult(
            success=True,
            message="\n".join(msg_lines),
            data={
                "total_profiles": total_profiles,
                "total_overrides": total_overrides,
                "total_tuning_events": total_tuning_events,
                "config": self._config,
                "overrides": adjustments,
            },
        )

    def _reset(self, params: Dict) -> SkillResult:
        """Reset a skill's profile to use global defaults."""
        skill_id = params.get("skill_id")
        if not skill_id:
            return SkillResult(success=False, message="skill_id is required")

        removed_profile = self._profiles.pop(skill_id, None)
        removed_override = self._overrides.pop(skill_id, None)

        if not removed_profile and not removed_override:
            return SkillResult(
                success=False,
                message=f"No profile or override found for {skill_id}",
            )

        self._tuning_history.append({
            "action": "reset",
            "skill_id": skill_id,
            "timestamp": datetime.now().isoformat(),
        })

        self._save_state()

        return SkillResult(
            success=True,
            message=f"Reset {skill_id} to global defaults"
                    + (". Removed profile." if removed_profile else "")
                    + (" Removed override." if removed_override else ""),
            data={"skill_id": skill_id},
        )

    def get_override_for_skill(self, skill_id: str) -> Optional[Dict]:
        """
        Get the adaptive threshold override for a skill, if any.

        This method is designed to be called by CircuitBreakerSkill
        to get per-skill thresholds instead of using global defaults.

        Returns None if no override exists (use global thresholds).
        """
        return self._overrides.get(skill_id)

    def get_all_overrides(self) -> Dict[str, Dict]:
        """Get all active threshold overrides."""
        return dict(self._overrides)

    def estimate_cost(self, action: str, params: Dict) -> float:
        return 0.0
