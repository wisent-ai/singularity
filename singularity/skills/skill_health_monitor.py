#!/usr/bin/env python3
"""
SkillHealthMonitorSkill - Autonomous health monitoring for all registered skills.

The agent has 80+ registered skills, but has no way to know if they're actually
working. A skill could be broken (bad imports, missing data files, runtime errors)
and the agent would keep trying to use it, failing silently. This is critical for:

1. Self-Improvement: The agent needs to know when its capabilities are degraded
   so it can prioritize fixing broken skills or avoiding them.
2. Revenue: If a revenue-generating skill is broken, the agent is losing money
   without realizing it. Health checks catch this immediately.
3. Goal Setting: Health status feeds into strategic planning — can't pursue goals
   that depend on unhealthy skills.

Architecture:
1. SCAN: Iterate all registered skills and check basic health (manifest, actions)
2. PROBE: Optionally call a lightweight action on each skill to verify runtime
3. TRACK: Store health history for trend analysis (degradation detection)
4. ALERT: Emit metrics to ObservabilitySkill when skills become unhealthy
5. REPORT: Generate health dashboards with pass/fail/degraded per skill

Actions:
- check_all: Run health checks on all registered skills
- check_one: Run health check on a specific skill
- status: View current health status of all skills
- history: View health check history with trend data
- configure: Set check parameters (timeout, probe actions, etc.)
- degraded: List only degraded/unhealthy skills
- emit_metrics: Push skill health metrics to ObservabilitySkill
- stats: Aggregate health statistics

Pillar: Self-Improvement (primary) + Revenue (detect broken revenue skills)
"""

import json
import time
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
from .base import Skill, SkillResult, SkillManifest, SkillAction

DATA_DIR = Path(__file__).parent.parent / "data"
HEALTH_FILE = DATA_DIR / "skill_health_monitor.json"
MAX_HISTORY = 500
MAX_CHECKS_PER_SKILL = 100

# Health states
HEALTHY = "healthy"
DEGRADED = "degraded"
UNHEALTHY = "unhealthy"
UNKNOWN = "unknown"

# Check timeout in seconds
DEFAULT_TIMEOUT = 10.0


def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _now_ts() -> float:
    return time.time()


class SkillHealthMonitorSkill(Skill):
    """
    Monitors the health of all registered skills by checking manifests,
    actions, and optionally probing with test calls. Tracks health over
    time and emits metrics for alerting.
    """

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self._store = None

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="skill_health_monitor",
            name="Skill Health Monitor",
            version="1.0.1",
            category="meta",
            description="Autonomous health monitoring for all registered skills — detects broken, degraded, and unhealthy capabilities",
            actions=self.get_actions(),
            required_credentials=[],
            install_cost=0,
            author="singularity",
        )

    def get_actions(self) -> List[SkillAction]:
        return [
            SkillAction(
                name="check_all",
                description="Run health checks on all registered skills and update status",
                parameters={
                    "probe": {"type": "boolean", "required": False,
                              "description": "Also probe skills with a test action call (default: False)"},
                    "timeout": {"type": "number", "required": False,
                                "description": f"Timeout per skill check in seconds (default: {DEFAULT_TIMEOUT})"},
                },
            ),
            SkillAction(
                name="check_one",
                description="Run health check on a specific skill by ID",
                parameters={
                    "skill_id": {"type": "string", "required": True,
                                 "description": "The skill ID to check"},
                    "probe": {"type": "boolean", "required": False,
                              "description": "Also probe with a test action call (default: False)"},
                },
            ),
            SkillAction(
                name="status",
                description="View current health status of all skills",
                parameters={
                    "filter": {"type": "string", "required": False,
                               "description": "Filter by health state: healthy, degraded, unhealthy, unknown"},
                },
            ),
            SkillAction(
                name="history",
                description="View health check history with trend data",
                parameters={
                    "skill_id": {"type": "string", "required": False,
                                 "description": "Filter history for a specific skill"},
                    "limit": {"type": "number", "required": False,
                              "description": "Max entries to return (default: 20)"},
                },
            ),
            SkillAction(
                name="configure",
                description="Set health check parameters",
                parameters={
                    "default_timeout": {"type": "number", "required": False,
                                        "description": "Default timeout per check in seconds"},
                    "auto_emit_metrics": {"type": "boolean", "required": False,
                                          "description": "Auto-emit metrics to ObservabilitySkill after checks"},
                    "probe_actions": {"type": "object", "required": False,
                                      "description": "Map of skill_id → action name to use for probing"},
                },
            ),
            SkillAction(
                name="degraded",
                description="List only degraded or unhealthy skills",
                parameters={},
            ),
            SkillAction(
                name="emit_metrics",
                description="Push current skill health metrics to ObservabilitySkill",
                parameters={},
            ),
            SkillAction(
                name="stats",
                description="Aggregate health statistics across all skills",
                parameters={},
            ),
        ]

    # ── Persistence ──────────────────────────────────────────────────

    def _default_state(self) -> Dict:
        return {
            "skills": {},  # skill_id → {state, last_check, checks: [...], error, ...}
            "config": {
                "default_timeout": DEFAULT_TIMEOUT,
                "auto_emit_metrics": True,
                "probe_actions": {},  # skill_id → action name for probing
            },
            "global_history": [],  # [{timestamp, total, healthy, degraded, unhealthy}]
            "stats": {
                "total_checks": 0,
                "total_probes": 0,
                "last_full_check": None,
            },
            "metadata": {
                "created_at": _now_iso(),
                "version": "1.0.0",
            },
        }

    def _load(self) -> Dict:
        if self._store is not None:
            return self._store
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        if HEALTH_FILE.exists():
            try:
                with open(HEALTH_FILE, "r") as f:
                    self._store = json.load(f)
                    return self._store
            except (json.JSONDecodeError, OSError):
                pass
        self._store = self._default_state()
        return self._store

    def _save(self, data: Dict):
        self._store = data
        # Trim histories
        if len(data.get("global_history", [])) > MAX_HISTORY:
            data["global_history"] = data["global_history"][-MAX_HISTORY:]
        for skill_id, skill_data in data.get("skills", {}).items():
            checks = skill_data.get("checks", [])
            if len(checks) > MAX_CHECKS_PER_SKILL:
                skill_data["checks"] = checks[-MAX_CHECKS_PER_SKILL:]
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(HEALTH_FILE, "w") as f:
            json.dump(data, f, indent=2, default=str)

    # ── Execute Dispatch ─────────────────────────────────────────────

    async def execute(self, action: str, params: Dict[str, Any]) -> SkillResult:
        handlers = {
            "check_all": self._check_all,
            "check_one": self._check_one,
            "status": self._status,
            "history": self._history,
            "configure": self._configure,
            "degraded": self._degraded,
            "emit_metrics": self._emit_metrics,
            "stats": self._stats,
        }
        handler = handlers.get(action)
        if not handler:
            return SkillResult(success=False, message=f"Unknown action: {action}")
        return await handler(params)

    # ── Core Health Check Logic ──────────────────────────────────────

    def _check_skill_health(self, skill) -> Dict:
        """
        Check a single skill's health without probing.
        Returns a health report dict.
        """
        report = {
            "skill_id": None,
            "state": UNKNOWN,
            "checks_passed": [],
            "checks_failed": [],
            "error": None,
            "timestamp": _now_iso(),
        }

        # Check 1: manifest access
        try:
            m = skill.manifest
            report["skill_id"] = m.skill_id
            report["checks_passed"].append("manifest_access")
            report["name"] = m.name
            report["version"] = m.version
            report["category"] = m.category
        except Exception as e:
            report["checks_failed"].append("manifest_access")
            report["error"] = f"manifest access failed: {str(e)[:200]}"
            report["state"] = UNHEALTHY
            return report

        # Check 2: actions list
        try:
            actions = skill.get_actions()
            if actions and len(actions) > 0:
                report["checks_passed"].append("actions_list")
                report["action_count"] = len(actions)
                report["action_names"] = [a.name for a in actions]
            else:
                report["checks_failed"].append("actions_list")
                report["error"] = "skill has no actions"
        except Exception as e:
            report["checks_failed"].append("actions_list")
            report["error"] = f"actions list failed: {str(e)[:200]}"

        # Check 3: execute method exists and is callable
        if hasattr(skill, "execute") and callable(getattr(skill, "execute")):
            report["checks_passed"].append("execute_callable")
        else:
            report["checks_failed"].append("execute_callable")

        # Determine state
        if len(report["checks_failed"]) == 0:
            report["state"] = HEALTHY
        elif len(report["checks_passed"]) > 0:
            report["state"] = DEGRADED
        else:
            report["state"] = UNHEALTHY

        return report

    async def _probe_skill(self, skill, skill_id: str, probe_action: str = None, timeout: float = DEFAULT_TIMEOUT) -> Dict:
        """
        Probe a skill by actually calling an action.
        Returns probe result dict.
        """
        probe_result = {
            "probed": True,
            "probe_action": None,
            "probe_success": False,
            "probe_duration_ms": 0,
            "probe_error": None,
        }

        # Determine which action to probe with
        if probe_action:
            action_name = probe_action
        else:
            # Use status/stats/health-like actions if available
            actions = []
            try:
                actions = [a.name for a in skill.get_actions()]
            except Exception:
                probe_result["probe_error"] = "couldn't get actions for probing"
                return probe_result

            # Prefer safe read-only actions
            safe_actions = ["status", "stats", "health", "list", "info"]
            action_name = None
            for safe in safe_actions:
                if safe in actions:
                    action_name = safe
                    break

            if not action_name and actions:
                # Skip probe if no safe action found
                probe_result["probe_error"] = "no safe probe action found"
                probe_result["probed"] = False
                return probe_result

        probe_result["probe_action"] = action_name

        start = time.time()
        try:
            result = await asyncio.wait_for(
                skill.execute(action_name, {}),
                timeout=timeout,
            )
            elapsed_ms = (time.time() - start) * 1000
            probe_result["probe_duration_ms"] = round(elapsed_ms, 1)

            if result and result.success:
                probe_result["probe_success"] = True
            else:
                probe_result["probe_error"] = (
                    result.message[:200] if result and result.message else "probe returned failure"
                )
        except asyncio.TimeoutError:
            probe_result["probe_error"] = f"probe timed out after {timeout}s"
            probe_result["probe_duration_ms"] = round(timeout * 1000, 1)
        except Exception as e:
            elapsed_ms = (time.time() - start) * 1000
            probe_result["probe_duration_ms"] = round(elapsed_ms, 1)
            probe_result["probe_error"] = f"probe exception: {str(e)[:200]}"

        return probe_result

    def _get_registered_skills(self) -> Dict[str, Any]:
        """Get all skills from the registry via context."""
        skills = {}
        if self.context and hasattr(self.context, '_registry'):
            registry = self.context._registry
            if hasattr(registry, '_skills'):
                for sid, skill in registry._skills.items():
                    if sid != "skill_health_monitor":  # Don't check ourselves
                        skills[sid] = skill
        return skills

    # ── Action: check_all ────────────────────────────────────────────

    async def _check_all(self, params: Dict) -> SkillResult:
        """Run health checks on all registered skills."""
        do_probe = params.get("probe", False)
        timeout = params.get("timeout", DEFAULT_TIMEOUT)

        store = self._load()
        config = store["config"]
        skills = self._get_registered_skills()

        if not skills:
            return SkillResult(
                success=True,
                message="No skills found in registry. Is the skill context connected?",
                data={"total": 0, "note": "Ensure skill_health_monitor has context set"},
            )

        results = {HEALTHY: [], DEGRADED: [], UNHEALTHY: [], UNKNOWN: []}
        total_checked = 0

        for skill_id, skill in skills.items():
            # Basic health check
            report = self._check_skill_health(skill)
            total_checked += 1

            # Optional probe
            if do_probe:
                probe_action = config.get("probe_actions", {}).get(skill_id)
                probe = await self._probe_skill(skill, skill_id, probe_action, timeout)
                report.update(probe)
                store["stats"]["total_probes"] += 1

                # Downgrade health if probe failed
                if probe.get("probed") and not probe.get("probe_success"):
                    if report["state"] == HEALTHY:
                        report["state"] = DEGRADED

            # Store result
            if skill_id not in store["skills"]:
                store["skills"][skill_id] = {"checks": []}
            store["skills"][skill_id].update({
                "state": report["state"],
                "last_check": report["timestamp"],
                "error": report.get("error") or report.get("probe_error"),
                "name": report.get("name", skill_id),
                "version": report.get("version", "?"),
                "action_count": report.get("action_count", 0),
            })
            store["skills"][skill_id]["checks"].append({
                "state": report["state"],
                "timestamp": report["timestamp"],
                "probe": report.get("probed", False),
                "probe_success": report.get("probe_success"),
                "probe_duration_ms": report.get("probe_duration_ms"),
                "error": report.get("error") or report.get("probe_error"),
            })

            results[report["state"]].append({
                "skill_id": skill_id,
                "state": report["state"],
                "error": report.get("error") or report.get("probe_error"),
            })

        store["stats"]["total_checks"] += total_checked
        store["stats"]["last_full_check"] = _now_iso()

        # Record global snapshot
        snapshot = {
            "timestamp": _now_iso(),
            "total": total_checked,
            "healthy": len(results[HEALTHY]),
            "degraded": len(results[DEGRADED]),
            "unhealthy": len(results[UNHEALTHY]),
            "unknown": len(results[UNKNOWN]),
        }
        store["global_history"].append(snapshot)

        self._save(store)

        # Auto-emit metrics if configured
        if config.get("auto_emit_metrics", True):
            await self._do_emit_metrics(store)

        healthy_count = len(results[HEALTHY])
        degraded_count = len(results[DEGRADED])
        unhealthy_count = len(results[UNHEALTHY])

        return SkillResult(
            success=True,
            message=(
                f"Checked {total_checked} skills: "
                f"{healthy_count} healthy, {degraded_count} degraded, "
                f"{unhealthy_count} unhealthy"
            ),
            data={
                "total": total_checked,
                "healthy": results[HEALTHY],
                "degraded": results[DEGRADED],
                "unhealthy": results[UNHEALTHY],
                "unknown": results[UNKNOWN],
                "probed": do_probe,
            },
        )

    # ── Action: check_one ────────────────────────────────────────────

    async def _check_one(self, params: Dict) -> SkillResult:
        """Check health of a specific skill."""
        skill_id = params.get("skill_id", "").strip()
        if not skill_id:
            return SkillResult(success=False, message="skill_id is required")

        do_probe = params.get("probe", False)
        store = self._load()
        config = store["config"]

        skills = self._get_registered_skills()
        skill = skills.get(skill_id)

        if not skill:
            return SkillResult(
                success=False,
                message=f"Skill '{skill_id}' not found in registry",
                data={"available_skills": sorted(skills.keys())[:20]},
            )

        report = self._check_skill_health(skill)

        if do_probe:
            probe_action = config.get("probe_actions", {}).get(skill_id)
            probe = await self._probe_skill(skill, skill_id, probe_action)
            report.update(probe)
            store["stats"]["total_probes"] += 1

            if probe.get("probed") and not probe.get("probe_success"):
                if report["state"] == HEALTHY:
                    report["state"] = DEGRADED

        # Store
        if skill_id not in store["skills"]:
            store["skills"][skill_id] = {"checks": []}
        store["skills"][skill_id].update({
            "state": report["state"],
            "last_check": report["timestamp"],
            "error": report.get("error") or report.get("probe_error"),
            "name": report.get("name", skill_id),
        })
        store["skills"][skill_id]["checks"].append({
            "state": report["state"],
            "timestamp": report["timestamp"],
            "probe": report.get("probed", False),
            "probe_success": report.get("probe_success"),
            "error": report.get("error") or report.get("probe_error"),
        })

        store["stats"]["total_checks"] += 1
        self._save(store)

        return SkillResult(
            success=True,
            message=f"Skill '{skill_id}' is {report['state']}",
            data=report,
        )

    # ── Action: status ───────────────────────────────────────────────

    async def _status(self, params: Dict) -> SkillResult:
        """View current health status of all skills."""
        state_filter = params.get("filter", "")
        store = self._load()

        skill_statuses = []
        for skill_id, data in store.get("skills", {}).items():
            status_entry = {
                "skill_id": skill_id,
                "state": data.get("state", UNKNOWN),
                "name": data.get("name", skill_id),
                "version": data.get("version", "?"),
                "last_check": data.get("last_check"),
                "action_count": data.get("action_count", 0),
                "error": data.get("error"),
                "check_count": len(data.get("checks", [])),
            }
            if state_filter and status_entry["state"] != state_filter:
                continue
            skill_statuses.append(status_entry)

        skill_statuses.sort(key=lambda x: (
            0 if x["state"] == UNHEALTHY else
            1 if x["state"] == DEGRADED else
            2 if x["state"] == UNKNOWN else 3,
            x["skill_id"]
        ))

        counts = {}
        for s in skill_statuses:
            counts[s["state"]] = counts.get(s["state"], 0) + 1

        return SkillResult(
            success=True,
            message=(
                f"{len(skill_statuses)} skills tracked. "
                f"Healthy: {counts.get(HEALTHY, 0)}, "
                f"Degraded: {counts.get(DEGRADED, 0)}, "
                f"Unhealthy: {counts.get(UNHEALTHY, 0)}"
            ),
            data={
                "skills": skill_statuses,
                "counts": counts,
                "last_full_check": store["stats"].get("last_full_check"),
            },
        )

    # ── Action: history ──────────────────────────────────────────────

    async def _history(self, params: Dict) -> SkillResult:
        """View health check history."""
        skill_id = params.get("skill_id", "")
        limit = params.get("limit", 20)
        store = self._load()

        if skill_id:
            # Per-skill history
            skill_data = store.get("skills", {}).get(skill_id, {})
            checks = skill_data.get("checks", [])
            recent = checks[-limit:]

            # Calculate trend
            trend = "stable"
            if len(checks) >= 3:
                recent_states = [c["state"] for c in checks[-3:]]
                if all(s == HEALTHY for s in recent_states):
                    trend = "stable_healthy"
                elif recent_states[-1] != HEALTHY and recent_states[-2] == HEALTHY:
                    trend = "degrading"
                elif recent_states[-1] == HEALTHY and recent_states[-2] != HEALTHY:
                    trend = "recovering"

            return SkillResult(
                success=True,
                message=f"{len(recent)} checks for '{skill_id}' (trend: {trend})",
                data={
                    "skill_id": skill_id,
                    "checks": recent,
                    "total_checks": len(checks),
                    "trend": trend,
                },
            )
        else:
            # Global history
            history = store.get("global_history", [])
            recent = history[-limit:]

            return SkillResult(
                success=True,
                message=f"{len(recent)} global health snapshots",
                data={"snapshots": recent, "total": len(history)},
            )

    # ── Action: configure ────────────────────────────────────────────

    async def _configure(self, params: Dict) -> SkillResult:
        """Set health check parameters."""
        store = self._load()
        config = store["config"]
        changed = []

        if "default_timeout" in params:
            val = float(params["default_timeout"])
            if 0.5 <= val <= 60:
                config["default_timeout"] = val
                changed.append("default_timeout")

        if "auto_emit_metrics" in params:
            config["auto_emit_metrics"] = bool(params["auto_emit_metrics"])
            changed.append("auto_emit_metrics")

        if "probe_actions" in params:
            pa = params["probe_actions"]
            if isinstance(pa, dict):
                config["probe_actions"].update(pa)
                changed.append("probe_actions")

        if not changed:
            return SkillResult(
                success=True,
                message="No configuration changes specified",
                data={"config": config},
            )

        self._save(store)
        return SkillResult(
            success=True,
            message=f"Updated: {', '.join(changed)}",
            data={"config": config, "changed": changed},
        )

    # ── Action: degraded ─────────────────────────────────────────────

    async def _degraded(self, params: Dict) -> SkillResult:
        """List only degraded or unhealthy skills."""
        store = self._load()

        problems = []
        for skill_id, data in store.get("skills", {}).items():
            state = data.get("state", UNKNOWN)
            if state in (DEGRADED, UNHEALTHY):
                problems.append({
                    "skill_id": skill_id,
                    "state": state,
                    "name": data.get("name", skill_id),
                    "error": data.get("error"),
                    "last_check": data.get("last_check"),
                })

        problems.sort(key=lambda x: 0 if x["state"] == UNHEALTHY else 1)

        if not problems:
            return SkillResult(
                success=True,
                message="All checked skills are healthy!",
                data={"problems": [], "count": 0},
            )

        return SkillResult(
            success=True,
            message=f"{len(problems)} skill(s) need attention",
            data={"problems": problems, "count": len(problems)},
        )

    # ── Action: emit_metrics ─────────────────────────────────────────

    async def _emit_metrics(self, params: Dict) -> SkillResult:
        """Push skill health metrics to ObservabilitySkill."""
        store = self._load()
        emitted = await self._do_emit_metrics(store)
        return SkillResult(
            success=True,
            message=f"Emitted {emitted} health metrics to ObservabilitySkill",
            data={"metrics_emitted": emitted},
        )

    async def _do_emit_metrics(self, store: Dict) -> int:
        """Internal: emit metrics to ObservabilitySkill."""
        if not self.context:
            return 0

        emitted = 0
        skills_data = store.get("skills", {})

        # Count by state
        counts = {HEALTHY: 0, DEGRADED: 0, UNHEALTHY: 0, UNKNOWN: 0}
        for data in skills_data.values():
            state = data.get("state", UNKNOWN)
            counts[state] = counts.get(state, 0) + 1

        total = sum(counts.values())

        # Emit gauge metrics
        metrics_to_emit = [
            ("skill_health.total", total, {}),
            ("skill_health.healthy", counts[HEALTHY], {}),
            ("skill_health.degraded", counts[DEGRADED], {}),
            ("skill_health.unhealthy", counts[UNHEALTHY], {}),
            ("skill_health.health_ratio", (counts[HEALTHY] / total * 100) if total > 0 else 0, {}),
        ]

        for metric_name, value, labels in metrics_to_emit:
            try:
                await self.context.call_skill("observability", "emit", {
                    "metric_name": metric_name,
                    "value": value,
                    "metric_type": "gauge",
                    "labels": {"source": "skill_health_monitor", **labels},
                })
                emitted += 1
            except Exception:
                pass

        return emitted

    # ── Action: stats ────────────────────────────────────────────────

    async def _stats(self, params: Dict) -> SkillResult:
        """Aggregate health statistics."""
        store = self._load()
        skills_data = store.get("skills", {})

        counts = {HEALTHY: 0, DEGRADED: 0, UNHEALTHY: 0, UNKNOWN: 0}
        total_action_count = 0
        categories = {}

        for skill_id, data in skills_data.items():
            state = data.get("state", UNKNOWN)
            counts[state] = counts.get(state, 0) + 1
            total_action_count += data.get("action_count", 0)
            cat = data.get("category", "unknown")
            if cat not in categories:
                categories[cat] = {HEALTHY: 0, DEGRADED: 0, UNHEALTHY: 0}
            if state in categories[cat]:
                categories[cat][state] += 1

        total = sum(counts.values())
        health_pct = (counts[HEALTHY] / total * 100) if total > 0 else 0

        # Trend from global history
        history = store.get("global_history", [])
        trend = "no data"
        if len(history) >= 2:
            prev = history[-2]
            curr = history[-1]
            prev_pct = (prev["healthy"] / prev["total"] * 100) if prev["total"] > 0 else 0
            curr_pct = (curr["healthy"] / curr["total"] * 100) if curr["total"] > 0 else 0
            if curr_pct > prev_pct:
                trend = "improving"
            elif curr_pct < prev_pct:
                trend = "degrading"
            else:
                trend = "stable"

        return SkillResult(
            success=True,
            message=(
                f"{total} skills monitored: {health_pct:.0f}% healthy. "
                f"Trend: {trend}. "
                f"Total checks: {store['stats']['total_checks']}"
            ),
            data={
                "total_skills": total,
                "counts": counts,
                "health_percentage": round(health_pct, 1),
                "total_actions": total_action_count,
                "trend": trend,
                "total_checks_run": store["stats"]["total_checks"],
                "total_probes_run": store["stats"]["total_probes"],
                "last_full_check": store["stats"].get("last_full_check"),
                "categories": categories,
            },
        )
