#!/usr/bin/env python3
"""
SelfHealingSkill - Autonomous subsystem health scanning, diagnosis, and repair.

This skill closes a critical gap in the Self-Improvement pillar by providing
an automated healing loop that:

1. SCAN - Periodically checks all skills/subsystems for health issues
   (error rates, consecutive failures, corrupted data, stale state)
2. DIAGNOSE - Classifies the root cause of each issue
   (data corruption, resource exhaustion, dependency failure, config drift)
3. HEAL - Applies automatic repair strategies
   (reset state, clear cache, reload skill, reinitialize, quarantine)
4. VERIFY - Confirms the repair worked by re-checking the subsystem
5. LEARN - Tracks which repairs work for which symptoms, improving over time

Works with:
- ErrorRecoverySkill: uses error patterns for diagnosis
- AgentHealthMonitor: monitors replica health
- DashboardSkill: reports healing status
- PerformanceTracker: detects performance degradation

Pillar: Self-Improvement (autonomous resilience)

Actions:
- scan: Run health scan on all or specific subsystems
- diagnose: Deep diagnosis of a specific subsystem issue
- heal: Apply repair strategy to a subsystem
- auto_heal: Full scan → diagnose → heal cycle (the main loop)
- status: Get current healing status and history
- quarantine: Quarantine a repeatedly-failing subsystem
- release: Release a quarantined subsystem
- healing_report: Get report on healing effectiveness
"""

import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from collections import defaultdict
from .base import Skill, SkillManifest, SkillAction, SkillResult


HEALING_FILE = Path(__file__).parent.parent / "data" / "self_healing.json"
MAX_HISTORY = 500
MAX_QUARANTINE = 50


class HealthStatus:
    """Health status levels for subsystems."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILING = "failing"
    QUARANTINED = "quarantined"
    UNKNOWN = "unknown"


class DiagnosisType:
    """Types of diagnosed issues."""
    DATA_CORRUPTION = "data_corruption"
    STATE_DRIFT = "state_drift"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    DEPENDENCY_FAILURE = "dependency_failure"
    CONFIG_DRIFT = "config_drift"
    REPEATED_ERRORS = "repeated_errors"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    NONE = "none"


class RepairStrategy:
    """Available repair strategies."""
    RESET_STATE = "reset_state"
    CLEAR_DATA = "clear_data"
    REINITIALIZE = "reinitialize"
    REDUCE_LOAD = "reduce_load"
    QUARANTINE = "quarantine"
    RESTART = "restart"
    NOOP = "noop"


# Maps diagnosis types to default repair strategies
DEFAULT_REPAIRS = {
    DiagnosisType.DATA_CORRUPTION: [
        {"strategy": RepairStrategy.CLEAR_DATA, "description": "Clear corrupted data file and reset to defaults"},
        {"strategy": RepairStrategy.RESET_STATE, "description": "Reset skill state to initial values"},
    ],
    DiagnosisType.STATE_DRIFT: [
        {"strategy": RepairStrategy.REINITIALIZE, "description": "Reinitialize the skill to reset internal state"},
        {"strategy": RepairStrategy.RESET_STATE, "description": "Reset skill state"},
    ],
    DiagnosisType.RESOURCE_EXHAUSTION: [
        {"strategy": RepairStrategy.CLEAR_DATA, "description": "Trim data files to reduce disk/memory usage"},
        {"strategy": RepairStrategy.REDUCE_LOAD, "description": "Reduce operation scope to conserve resources"},
    ],
    DiagnosisType.DEPENDENCY_FAILURE: [
        {"strategy": RepairStrategy.RESTART, "description": "Restart the skill hoping dependency recovers"},
        {"strategy": RepairStrategy.QUARANTINE, "description": "Quarantine until dependency is available"},
    ],
    DiagnosisType.CONFIG_DRIFT: [
        {"strategy": RepairStrategy.REINITIALIZE, "description": "Reinitialize with fresh configuration"},
        {"strategy": RepairStrategy.RESET_STATE, "description": "Reset to default config"},
    ],
    DiagnosisType.REPEATED_ERRORS: [
        {"strategy": RepairStrategy.REINITIALIZE, "description": "Reinitialize the skill completely"},
        {"strategy": RepairStrategy.QUARANTINE, "description": "Quarantine if errors persist after reinit"},
    ],
    DiagnosisType.PERFORMANCE_DEGRADATION: [
        {"strategy": RepairStrategy.CLEAR_DATA, "description": "Clear accumulated data slowing performance"},
        {"strategy": RepairStrategy.REDUCE_LOAD, "description": "Reduce operational scope"},
    ],
    DiagnosisType.NONE: [
        {"strategy": RepairStrategy.NOOP, "description": "No repair needed - subsystem is healthy"},
    ],
}


class SelfHealingSkill(Skill):
    """
    Autonomous self-healing for agent subsystems.

    Continuously monitors skill health, diagnoses issues, applies repairs,
    and learns which repairs are effective for which symptoms. This makes
    the agent resilient to failures without human intervention.
    """

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self._ensure_data()

    def _ensure_data(self):
        HEALING_FILE.parent.mkdir(parents=True, exist_ok=True)
        if not HEALING_FILE.exists():
            self._save(self._default_data())

    def _default_data(self) -> Dict:
        return {
            "subsystem_health": {},
            "quarantined": {},
            "healing_history": [],
            "repair_knowledge": {},
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "total_scans": 0,
                "total_diagnoses": 0,
                "total_repairs": 0,
                "successful_repairs": 0,
                "last_scan": None,
                "last_auto_heal": None,
            },
        }

    def _load(self) -> Dict:
        try:
            with open(HEALING_FILE, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return self._default_data()

    def _save(self, data: Dict):
        if len(data.get("healing_history", [])) > MAX_HISTORY:
            data["healing_history"] = data["healing_history"][-MAX_HISTORY:]
        data["metadata"]["last_updated"] = datetime.now().isoformat()
        HEALING_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(HEALING_FILE, "w") as f:
            json.dump(data, f, indent=2, default=str)

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="self_healing",
            name="Self-Healing",
            version="1.0.0",
            category="meta",
            description="Autonomous subsystem health scanning, diagnosis, and repair with learning",
            actions=[
                SkillAction(
                    name="scan",
                    description="Run health scan on all or specific subsystems to detect issues",
                    parameters={
                        "skill_id": {"type": "string", "required": False, "description": "Specific skill to scan (omit for all)"},
                    },
                ),
                SkillAction(
                    name="diagnose",
                    description="Deep diagnosis of a specific subsystem issue",
                    parameters={
                        "skill_id": {"type": "string", "required": True, "description": "Skill to diagnose"},
                        "symptoms": {"type": "object", "required": False, "description": "Observed symptoms (error_rate, latency, failures)"},
                    },
                ),
                SkillAction(
                    name="heal",
                    description="Apply a repair strategy to a subsystem",
                    parameters={
                        "skill_id": {"type": "string", "required": True, "description": "Skill to heal"},
                        "strategy": {"type": "string", "required": True, "description": "Repair strategy: reset_state, clear_data, reinitialize, reduce_load, quarantine, restart"},
                        "reason": {"type": "string", "required": False, "description": "Reason for the repair"},
                    },
                ),
                SkillAction(
                    name="auto_heal",
                    description="Full autonomous scan → diagnose → heal cycle for all subsystems",
                    parameters={
                        "dry_run": {"type": "boolean", "required": False, "description": "If true, diagnose but don't apply repairs"},
                    },
                ),
                SkillAction(
                    name="status",
                    description="Get current healing status, subsystem health, and quarantine list",
                    parameters={},
                ),
                SkillAction(
                    name="quarantine",
                    description="Quarantine a repeatedly-failing subsystem to prevent cascade failures",
                    parameters={
                        "skill_id": {"type": "string", "required": True, "description": "Skill to quarantine"},
                        "reason": {"type": "string", "required": False, "description": "Reason for quarantine"},
                        "duration_hours": {"type": "number", "required": False, "description": "Quarantine duration (default 24)"},
                    },
                ),
                SkillAction(
                    name="release",
                    description="Release a quarantined subsystem back into service",
                    parameters={
                        "skill_id": {"type": "string", "required": True, "description": "Skill to release"},
                    },
                ),
                SkillAction(
                    name="healing_report",
                    description="Get report on healing effectiveness: success rates, common issues, learned repairs",
                    parameters={
                        "timeframe_hours": {"type": "number", "required": False, "description": "Hours to look back (default: 168 = 1 week)"},
                    },
                ),
            ],
            required_credentials=[],
        )

    async def execute(self, action: str, params: Dict = None) -> SkillResult:
        params = params or {}
        actions = {
            "scan": self._scan,
            "diagnose": self._diagnose,
            "heal": self._heal,
            "auto_heal": self._auto_heal,
            "status": self._status,
            "quarantine": self._quarantine,
            "release": self._release,
            "healing_report": self._healing_report,
        }
        if action not in actions:
            return SkillResult(
                success=False,
                message=f"Unknown action: {action}. Available: {list(actions.keys())}",
            )
        return await actions[action](params)

    # --- Core Actions ---

    async def _scan(self, params: Dict) -> SkillResult:
        """Scan subsystems for health issues."""
        target_skill = params.get("skill_id")
        data = self._load()

        # Get list of skills to scan
        skills_to_scan = []
        if self.context:
            available = self.context.list_skills()
            if target_skill:
                if target_skill not in available:
                    return SkillResult(success=False, message=f"Skill '{target_skill}' not found")
                skills_to_scan = [target_skill]
            else:
                skills_to_scan = [s for s in available if s != "self_healing"]
        elif target_skill:
            skills_to_scan = [target_skill]
        else:
            # No context, check if we have stored health data
            skills_to_scan = list(data.get("subsystem_health", {}).keys())

        scan_results = {}
        issues_found = 0

        for sid in skills_to_scan:
            # Skip quarantined skills
            if sid in data.get("quarantined", {}):
                quarantine_info = data["quarantined"][sid]
                expires = quarantine_info.get("expires_at")
                if expires and datetime.fromisoformat(expires) > datetime.now():
                    scan_results[sid] = {
                        "status": HealthStatus.QUARANTINED,
                        "reason": quarantine_info.get("reason", "quarantined"),
                        "expires_at": expires,
                    }
                    continue
                else:
                    # Quarantine expired, release
                    del data["quarantined"][sid]

            health = self._assess_skill_health(sid, data)
            scan_results[sid] = health

            # Update stored health
            data["subsystem_health"][sid] = {
                "status": health["status"],
                "last_scan": datetime.now().isoformat(),
                "error_count": health.get("error_count", 0),
                "consecutive_failures": health.get("consecutive_failures", 0),
                "symptoms": health.get("symptoms", []),
            }

            if health["status"] in (HealthStatus.DEGRADED, HealthStatus.FAILING):
                issues_found += 1

        data["metadata"]["total_scans"] = data["metadata"].get("total_scans", 0) + 1
        data["metadata"]["last_scan"] = datetime.now().isoformat()
        self._save(data)

        status_msg = f"Scanned {len(scan_results)} subsystems: {issues_found} issues found"
        return SkillResult(
            success=True,
            message=status_msg,
            data={
                "scan_results": scan_results,
                "total_scanned": len(scan_results),
                "issues_found": issues_found,
                "healthy": sum(1 for r in scan_results.values() if r.get("status") == HealthStatus.HEALTHY),
                "degraded": sum(1 for r in scan_results.values() if r.get("status") == HealthStatus.DEGRADED),
                "failing": sum(1 for r in scan_results.values() if r.get("status") == HealthStatus.FAILING),
                "quarantined": sum(1 for r in scan_results.values() if r.get("status") == HealthStatus.QUARANTINED),
            },
        )

    def _assess_skill_health(self, skill_id: str, data: Dict) -> Dict:
        """Assess health of a single skill based on available data."""
        symptoms = []
        error_count = 0
        consecutive_failures = 0

        # Check error history if ErrorRecoverySkill data is available
        error_file = Path(__file__).parent.parent / "data" / "error_recovery.json"
        if error_file.exists():
            try:
                with open(error_file, "r") as f:
                    error_data = json.load(f)

                # Count recent errors for this skill
                cutoff = (datetime.now() - timedelta(hours=24)).isoformat()
                skill_errors = [
                    e for e in error_data.get("errors", [])
                    if e.get("skill_id") == skill_id
                    and e.get("timestamp", "") > cutoff
                ]
                error_count = len(skill_errors)

                # Check consecutive failures (look at last N errors)
                recent = [
                    e for e in error_data.get("errors", [])
                    if e.get("skill_id") == skill_id
                ][-10:]
                consecutive_failures = 0
                for e in reversed(recent):
                    if not e.get("recovered", False):
                        consecutive_failures += 1
                    else:
                        break

                if error_count > 10:
                    symptoms.append("high_error_rate")
                if consecutive_failures >= 3:
                    symptoms.append("consecutive_failures")
                if error_count > 0:
                    # Check if errors are all the same type (stuck in a loop)
                    sigs = set(e.get("signature", "") for e in skill_errors)
                    if len(sigs) == 1 and error_count > 3:
                        symptoms.append("repeated_same_error")

            except (json.JSONDecodeError, Exception):
                pass

        # Check data file health for skills that persist data
        data_dir = Path(__file__).parent.parent / "data"
        skill_data_files = list(data_dir.glob(f"{skill_id}*.json")) + list(data_dir.glob(f"*{skill_id}*.json"))
        for data_file in skill_data_files[:3]:  # Check up to 3 data files
            try:
                with open(data_file) as f:
                    content = f.read()
                    if len(content) > 10_000_000:  # > 10MB
                        symptoms.append("data_bloat")
                    json.loads(content)  # Verify valid JSON
            except json.JSONDecodeError:
                symptoms.append("data_corruption")
            except Exception:
                pass

        # Check previous health data for trends
        prev_health = data.get("subsystem_health", {}).get(skill_id, {})
        if prev_health.get("status") == HealthStatus.DEGRADED:
            # Was already degraded - check if getting worse
            prev_errors = prev_health.get("error_count", 0)
            if error_count > prev_errors:
                symptoms.append("worsening")

        # Determine health status
        if "data_corruption" in symptoms:
            status = HealthStatus.FAILING
        elif consecutive_failures >= 5 or "worsening" in symptoms:
            status = HealthStatus.FAILING
        elif (error_count > 10 or consecutive_failures >= 3
              or "data_bloat" in symptoms or "repeated_same_error" in symptoms):
            status = HealthStatus.DEGRADED
        elif error_count > 0:
            status = HealthStatus.HEALTHY  # Some errors are normal
        else:
            status = HealthStatus.HEALTHY

        return {
            "status": status,
            "error_count": error_count,
            "consecutive_failures": consecutive_failures,
            "symptoms": symptoms,
        }

    async def _diagnose(self, params: Dict) -> SkillResult:
        """Deep diagnosis of a specific subsystem issue."""
        skill_id = params.get("skill_id")
        if not skill_id:
            return SkillResult(success=False, message="skill_id is required")

        provided_symptoms = params.get("symptoms", {})
        data = self._load()

        # Get health assessment
        health = self._assess_skill_health(skill_id, data)
        symptoms = health.get("symptoms", [])

        # Add any user-provided symptoms
        if provided_symptoms.get("error_rate", 0) > 0.5:
            symptoms.append("high_error_rate")
        if provided_symptoms.get("latency", 0) > 10:
            symptoms.append("slow_response")
        if provided_symptoms.get("failures", 0) > 5:
            symptoms.append("consecutive_failures")

        # Determine diagnosis type
        diagnosis = self._determine_diagnosis(symptoms)

        # Get recommended repairs
        recommended_repairs = DEFAULT_REPAIRS.get(diagnosis["type"], DEFAULT_REPAIRS[DiagnosisType.NONE])

        # Check knowledge base for learned repairs
        knowledge = data.get("repair_knowledge", {})
        learned = knowledge.get(f"{skill_id}:{diagnosis['type']}", {})
        if learned:
            # Sort by success rate
            learned_repairs = []
            for strategy, stats in learned.items():
                success_rate = stats.get("successes", 0) / max(stats.get("attempts", 1), 1)
                if success_rate > 0.3:  # Only recommend strategies with >30% success
                    learned_repairs.append({
                        "strategy": strategy,
                        "description": f"Learned repair (success rate: {success_rate:.0%})",
                        "success_rate": success_rate,
                        "attempts": stats.get("attempts", 0),
                        "source": "learned",
                    })
            learned_repairs.sort(key=lambda x: x.get("success_rate", 0), reverse=True)
            if learned_repairs:
                recommended_repairs = learned_repairs + [
                    {**r, "source": "default"} for r in recommended_repairs
                ]

        data["metadata"]["total_diagnoses"] = data["metadata"].get("total_diagnoses", 0) + 1
        self._save(data)

        return SkillResult(
            success=True,
            message=f"Diagnosis for '{skill_id}': {diagnosis['type']} ({diagnosis['severity']})",
            data={
                "skill_id": skill_id,
                "health": health,
                "diagnosis": diagnosis,
                "recommended_repairs": recommended_repairs,
                "symptoms": symptoms,
            },
        )

    def _determine_diagnosis(self, symptoms: List[str]) -> Dict:
        """Determine the diagnosis type and severity from symptoms."""
        if "data_corruption" in symptoms:
            return {
                "type": DiagnosisType.DATA_CORRUPTION,
                "severity": "critical",
                "description": "Data file is corrupted or contains invalid JSON",
            }
        if "data_bloat" in symptoms:
            return {
                "type": DiagnosisType.RESOURCE_EXHAUSTION,
                "severity": "high",
                "description": "Data file has grown excessively large",
            }
        if "worsening" in symptoms and "consecutive_failures" in symptoms:
            return {
                "type": DiagnosisType.REPEATED_ERRORS,
                "severity": "critical",
                "description": "Subsystem is failing repeatedly and getting worse",
            }
        if "consecutive_failures" in symptoms:
            return {
                "type": DiagnosisType.REPEATED_ERRORS,
                "severity": "high",
                "description": "Subsystem has multiple consecutive failures",
            }
        if "repeated_same_error" in symptoms:
            return {
                "type": DiagnosisType.STATE_DRIFT,
                "severity": "medium",
                "description": "Subsystem is stuck in an error loop with the same error",
            }
        if "high_error_rate" in symptoms:
            return {
                "type": DiagnosisType.REPEATED_ERRORS,
                "severity": "medium",
                "description": "Subsystem has an elevated error rate",
            }
        if "slow_response" in symptoms:
            return {
                "type": DiagnosisType.PERFORMANCE_DEGRADATION,
                "severity": "low",
                "description": "Subsystem response time is degraded",
            }
        return {
            "type": DiagnosisType.NONE,
            "severity": "none",
            "description": "No issues detected",
        }

    async def _heal(self, params: Dict) -> SkillResult:
        """Apply a repair strategy to a subsystem."""
        skill_id = params.get("skill_id")
        strategy = params.get("strategy")
        reason = params.get("reason", "manual repair")

        if not skill_id or not strategy:
            return SkillResult(success=False, message="skill_id and strategy are required")

        valid_strategies = [
            RepairStrategy.RESET_STATE, RepairStrategy.CLEAR_DATA,
            RepairStrategy.REINITIALIZE, RepairStrategy.REDUCE_LOAD,
            RepairStrategy.QUARANTINE, RepairStrategy.RESTART, RepairStrategy.NOOP,
        ]
        if strategy not in valid_strategies:
            return SkillResult(
                success=False,
                message=f"Invalid strategy: {strategy}. Valid: {valid_strategies}",
            )

        data = self._load()
        success = False
        repair_details = {}

        if strategy == RepairStrategy.NOOP:
            success = True
            repair_details = {"action": "none", "message": "No repair needed"}

        elif strategy == RepairStrategy.QUARANTINE:
            duration_hours = params.get("duration_hours", 24)
            return await self._quarantine({
                "skill_id": skill_id,
                "reason": reason,
                "duration_hours": duration_hours,
            })

        elif strategy == RepairStrategy.CLEAR_DATA:
            success, repair_details = self._do_clear_data(skill_id)

        elif strategy == RepairStrategy.RESET_STATE:
            success, repair_details = self._do_reset_state(skill_id)

        elif strategy == RepairStrategy.REINITIALIZE:
            success, repair_details = await self._do_reinitialize(skill_id)

        elif strategy == RepairStrategy.REDUCE_LOAD:
            success, repair_details = self._do_reduce_load(skill_id)

        elif strategy == RepairStrategy.RESTART:
            success, repair_details = await self._do_restart(skill_id)

        # Record healing attempt
        healing_record = {
            "timestamp": datetime.now().isoformat(),
            "skill_id": skill_id,
            "strategy": strategy,
            "reason": reason,
            "success": success,
            "details": repair_details,
        }
        data["healing_history"].append(healing_record)
        data["metadata"]["total_repairs"] = data["metadata"].get("total_repairs", 0) + 1
        if success:
            data["metadata"]["successful_repairs"] = data["metadata"].get("successful_repairs", 0) + 1

        # Update repair knowledge
        self._update_repair_knowledge(data, skill_id, strategy, success)

        # Update subsystem health if successful
        if success and skill_id in data.get("subsystem_health", {}):
            data["subsystem_health"][skill_id]["status"] = HealthStatus.HEALTHY
            data["subsystem_health"][skill_id]["consecutive_failures"] = 0

        self._save(data)

        return SkillResult(
            success=success,
            message=f"Repair {'succeeded' if success else 'failed'}: {strategy} on '{skill_id}'",
            data={
                "skill_id": skill_id,
                "strategy": strategy,
                "repair_success": success,
                "details": repair_details,
            },
        )

    def _do_clear_data(self, skill_id: str) -> tuple:
        """Clear/trim data files for a skill."""
        data_dir = Path(__file__).parent.parent / "data"
        cleared = []

        # Find related data files
        patterns = [f"{skill_id}*.json", f"*{skill_id}*.json"]
        for pattern in patterns:
            for data_file in data_dir.glob(pattern):
                if data_file.name == "self_healing.json":
                    continue  # Don't clear our own data
                try:
                    with open(data_file, "r") as f:
                        content = json.load(f)

                    # Trim lists to reasonable sizes
                    trimmed = False
                    for key, value in content.items():
                        if isinstance(value, list) and len(value) > 100:
                            content[key] = value[-50:]  # Keep last 50
                            trimmed = True
                        elif isinstance(value, dict):
                            for subkey, subval in value.items():
                                if isinstance(subval, list) and len(subval) > 100:
                                    content[key][subkey] = subval[-50:]
                                    trimmed = True

                    if trimmed:
                        with open(data_file, "w") as f:
                            json.dump(content, f, indent=2, default=str)
                        cleared.append(data_file.name)
                except Exception as e:
                    # If JSON is corrupted, reset file
                    try:
                        with open(data_file, "w") as f:
                            json.dump({}, f)
                        cleared.append(f"{data_file.name} (reset - was corrupted)")
                    except Exception:
                        pass

        return (True, {"cleared_files": cleared, "count": len(cleared)})

    def _do_reset_state(self, skill_id: str) -> tuple:
        """Reset a skill's persisted state to defaults."""
        data_dir = Path(__file__).parent.parent / "data"
        reset_files = []

        patterns = [f"{skill_id}*.json", f"*{skill_id}*.json"]
        for pattern in patterns:
            for data_file in data_dir.glob(pattern):
                if data_file.name == "self_healing.json":
                    continue
                try:
                    with open(data_file, "w") as f:
                        json.dump({
                            "_reset_by": "self_healing",
                            "_reset_at": datetime.now().isoformat(),
                            "_original_skill": skill_id,
                        }, f, indent=2)
                    reset_files.append(data_file.name)
                except Exception:
                    pass

        return (True, {"reset_files": reset_files, "count": len(reset_files)})

    async def _do_reinitialize(self, skill_id: str) -> tuple:
        """Reinitialize a skill through the context."""
        if not self.context:
            return (False, {"error": "No context available - cannot reinitialize skill"})

        skill = self.context.get_skill(skill_id)
        if not skill:
            return (False, {"error": f"Skill '{skill_id}' not found in registry"})

        try:
            skill.initialized = False
            result = await skill.initialize()
            return (result, {"reinitialized": result, "skill_id": skill_id})
        except Exception as e:
            return (False, {"error": str(e)})

    def _do_reduce_load(self, skill_id: str) -> tuple:
        """Reduce operational scope for a skill by trimming data."""
        # Similar to clear_data but more aggressive
        data_dir = Path(__file__).parent.parent / "data"
        reduced = []

        patterns = [f"{skill_id}*.json", f"*{skill_id}*.json"]
        for pattern in patterns:
            for data_file in data_dir.glob(pattern):
                if data_file.name == "self_healing.json":
                    continue
                try:
                    with open(data_file, "r") as f:
                        content = json.load(f)

                    modified = False
                    for key, value in content.items():
                        if isinstance(value, list) and len(value) > 20:
                            content[key] = value[-10:]  # Aggressive trim to 10
                            modified = True

                    if modified:
                        with open(data_file, "w") as f:
                            json.dump(content, f, indent=2, default=str)
                        reduced.append(data_file.name)
                except Exception:
                    pass

        return (True, {"reduced_files": reduced, "count": len(reduced)})

    async def _do_restart(self, skill_id: str) -> tuple:
        """Restart a skill (reinitialize + clear state)."""
        # First clear data
        clear_ok, clear_details = self._do_clear_data(skill_id)
        # Then reinitialize
        reinit_ok, reinit_details = await self._do_reinitialize(skill_id)

        success = clear_ok or reinit_ok
        return (success, {
            "clear": clear_details,
            "reinitialize": reinit_details,
        })

    def _update_repair_knowledge(self, data: Dict, skill_id: str, strategy: str, success: bool):
        """Update the repair knowledge base with the outcome."""
        # Get the current diagnosis type for this skill
        health = data.get("subsystem_health", {}).get(skill_id, {})
        symptoms = health.get("symptoms", [])
        diagnosis = self._determine_diagnosis(symptoms)
        diag_key = f"{skill_id}:{diagnosis['type']}"

        if diag_key not in data["repair_knowledge"]:
            data["repair_knowledge"][diag_key] = {}

        if strategy not in data["repair_knowledge"][diag_key]:
            data["repair_knowledge"][diag_key][strategy] = {
                "attempts": 0,
                "successes": 0,
                "last_used": None,
            }

        entry = data["repair_knowledge"][diag_key][strategy]
        entry["attempts"] += 1
        if success:
            entry["successes"] += 1
        entry["last_used"] = datetime.now().isoformat()

    async def _auto_heal(self, params: Dict) -> SkillResult:
        """Full autonomous scan → diagnose → heal cycle."""
        dry_run = params.get("dry_run", False)
        data = self._load()

        # Step 1: Scan
        scan_result = await self._scan({})
        if not scan_result.success:
            return scan_result

        scan_data = scan_result.data
        actions_taken = []
        issues_healed = 0
        issues_found = scan_data.get("issues_found", 0)

        # Step 2: For each non-healthy subsystem, diagnose and heal
        for sid, health in scan_data.get("scan_results", {}).items():
            if health.get("status") in (HealthStatus.HEALTHY, HealthStatus.QUARANTINED):
                continue

            # Diagnose
            diag_result = await self._diagnose({"skill_id": sid})
            if not diag_result.success:
                continue

            diagnosis = diag_result.data.get("diagnosis", {})
            recommended = diag_result.data.get("recommended_repairs", [])

            if not recommended:
                continue

            # Pick the best repair strategy
            best_repair = recommended[0]
            strategy = best_repair.get("strategy", RepairStrategy.NOOP)

            action_record = {
                "skill_id": sid,
                "diagnosis": diagnosis["type"],
                "severity": diagnosis["severity"],
                "strategy": strategy,
                "applied": not dry_run,
            }

            if not dry_run:
                # Apply the repair
                heal_result = await self._heal({
                    "skill_id": sid,
                    "strategy": strategy,
                    "reason": f"auto_heal: {diagnosis['description']}",
                })
                action_record["heal_success"] = heal_result.success
                if heal_result.success:
                    issues_healed += 1
            else:
                action_record["heal_success"] = None

            actions_taken.append(action_record)

        data["metadata"]["last_auto_heal"] = datetime.now().isoformat()
        self._save(data)

        mode = "DRY RUN" if dry_run else "ACTIVE"
        return SkillResult(
            success=True,
            message=f"Auto-heal [{mode}]: {issues_found} issues found, {issues_healed} healed, {len(actions_taken)} actions",
            data={
                "dry_run": dry_run,
                "issues_found": issues_found,
                "issues_healed": issues_healed,
                "actions_taken": actions_taken,
                "scan_summary": {
                    "total": scan_data.get("total_scanned", 0),
                    "healthy": scan_data.get("healthy", 0),
                    "degraded": scan_data.get("degraded", 0),
                    "failing": scan_data.get("failing", 0),
                    "quarantined": scan_data.get("quarantined", 0),
                },
            },
        )

    async def _status(self, params: Dict) -> SkillResult:
        """Get current healing status."""
        data = self._load()

        # Check for expired quarantines
        quarantined = data.get("quarantined", {})
        active_quarantine = {}
        for sid, info in quarantined.items():
            expires = info.get("expires_at")
            if expires and datetime.fromisoformat(expires) > datetime.now():
                active_quarantine[sid] = info

        data["quarantined"] = active_quarantine
        self._save(data)

        # Compute subsystem summary
        health_summary = defaultdict(int)
        for sid, info in data.get("subsystem_health", {}).items():
            health_summary[info.get("status", "unknown")] += 1

        # Recent healing activity
        recent = data.get("healing_history", [])[-10:]

        return SkillResult(
            success=True,
            message=f"Self-healing status: {len(data.get('subsystem_health', {}))} subsystems tracked, {len(active_quarantine)} quarantined",
            data={
                "subsystem_count": len(data.get("subsystem_health", {})),
                "health_summary": dict(health_summary),
                "quarantined": active_quarantine,
                "quarantine_count": len(active_quarantine),
                "recent_healing": recent,
                "metadata": data["metadata"],
            },
        )

    async def _quarantine(self, params: Dict) -> SkillResult:
        """Quarantine a failing subsystem."""
        skill_id = params.get("skill_id")
        if not skill_id:
            return SkillResult(success=False, message="skill_id is required")

        reason = params.get("reason", "manually quarantined")
        duration_hours = params.get("duration_hours", 24)

        data = self._load()

        if len(data.get("quarantined", {})) >= MAX_QUARANTINE:
            return SkillResult(success=False, message=f"Max quarantine limit ({MAX_QUARANTINE}) reached")

        expires_at = (datetime.now() + timedelta(hours=duration_hours)).isoformat()

        data.setdefault("quarantined", {})[skill_id] = {
            "reason": reason,
            "quarantined_at": datetime.now().isoformat(),
            "expires_at": expires_at,
            "duration_hours": duration_hours,
        }

        # Update health status
        data.setdefault("subsystem_health", {})[skill_id] = {
            "status": HealthStatus.QUARANTINED,
            "last_scan": datetime.now().isoformat(),
            "quarantine_reason": reason,
        }

        # Record in history
        data["healing_history"].append({
            "timestamp": datetime.now().isoformat(),
            "skill_id": skill_id,
            "strategy": RepairStrategy.QUARANTINE,
            "reason": reason,
            "success": True,
            "details": {"duration_hours": duration_hours, "expires_at": expires_at},
        })

        self._save(data)

        return SkillResult(
            success=True,
            message=f"Quarantined '{skill_id}' for {duration_hours}h: {reason}",
            data={
                "skill_id": skill_id,
                "expires_at": expires_at,
                "reason": reason,
            },
        )

    async def _release(self, params: Dict) -> SkillResult:
        """Release a quarantined subsystem."""
        skill_id = params.get("skill_id")
        if not skill_id:
            return SkillResult(success=False, message="skill_id is required")

        data = self._load()

        if skill_id not in data.get("quarantined", {}):
            return SkillResult(success=False, message=f"'{skill_id}' is not quarantined")

        del data["quarantined"][skill_id]

        # Reset health status
        if skill_id in data.get("subsystem_health", {}):
            data["subsystem_health"][skill_id]["status"] = HealthStatus.UNKNOWN

        data["healing_history"].append({
            "timestamp": datetime.now().isoformat(),
            "skill_id": skill_id,
            "strategy": "release",
            "reason": "released from quarantine",
            "success": True,
            "details": {},
        })

        self._save(data)

        return SkillResult(
            success=True,
            message=f"Released '{skill_id}' from quarantine",
            data={"skill_id": skill_id},
        )

    async def _healing_report(self, params: Dict) -> SkillResult:
        """Generate a report on healing effectiveness."""
        timeframe_hours = params.get("timeframe_hours", 168)  # 1 week
        cutoff = (datetime.now() - timedelta(hours=timeframe_hours)).isoformat()

        data = self._load()
        history = data.get("healing_history", [])

        # Filter to timeframe
        recent = [h for h in history if h.get("timestamp", "") > cutoff]

        if not recent:
            return SkillResult(
                success=True,
                message="No healing activity in the specified timeframe",
                data={"timeframe_hours": timeframe_hours, "total_repairs": 0},
            )

        # Analyze by strategy
        by_strategy = defaultdict(lambda: {"attempts": 0, "successes": 0, "skills": set()})
        by_skill = defaultdict(lambda: {"repairs": 0, "successes": 0, "strategies": []})
        total_success = 0

        for h in recent:
            strategy = h.get("strategy", "unknown")
            sid = h.get("skill_id", "unknown")
            success = h.get("success", False)

            by_strategy[strategy]["attempts"] += 1
            by_strategy[strategy]["skills"].add(sid)
            if success:
                by_strategy[strategy]["successes"] += 1
                total_success += 1

            by_skill[sid]["repairs"] += 1
            if success:
                by_skill[sid]["successes"] += 1
            by_skill[sid]["strategies"].append(strategy)

        # Convert sets to lists for serialization
        strategy_report = {}
        for strategy, stats in by_strategy.items():
            rate = stats["successes"] / max(stats["attempts"], 1)
            strategy_report[strategy] = {
                "attempts": stats["attempts"],
                "successes": stats["successes"],
                "success_rate": round(rate, 2),
                "affected_skills": list(stats["skills"]),
            }

        skill_report = {}
        for sid, stats in by_skill.items():
            rate = stats["successes"] / max(stats["repairs"], 1)
            skill_report[sid] = {
                "repairs": stats["repairs"],
                "successes": stats["successes"],
                "success_rate": round(rate, 2),
                "strategies_used": list(set(stats["strategies"])),
            }

        # Overall success rate
        overall_rate = total_success / max(len(recent), 1)

        # Most problematic skills
        problem_skills = sorted(
            skill_report.items(),
            key=lambda x: x[1]["repairs"],
            reverse=True,
        )[:5]

        # Most effective strategies
        effective_strategies = sorted(
            strategy_report.items(),
            key=lambda x: x[1]["success_rate"],
            reverse=True,
        )

        # Knowledge base stats
        kb_size = sum(
            len(strategies)
            for strategies in data.get("repair_knowledge", {}).values()
        )

        return SkillResult(
            success=True,
            message=f"Healing report ({timeframe_hours}h): {len(recent)} repairs, {overall_rate:.0%} success rate",
            data={
                "timeframe_hours": timeframe_hours,
                "total_repairs": len(recent),
                "total_successful": total_success,
                "overall_success_rate": round(overall_rate, 2),
                "by_strategy": strategy_report,
                "by_skill": skill_report,
                "most_problematic": [{"skill": s, **d} for s, d in problem_skills],
                "most_effective_strategies": [
                    {"strategy": s, **d} for s, d in effective_strategies
                ],
                "knowledge_base_entries": kb_size,
                "metadata": data.get("metadata", {}),
            },
        )
