#!/usr/bin/env python3
"""
AlertIncidentBridgeSkill - Auto-creates incidents from observability alerts.

This is the critical missing link between metric anomalies and incident response.
When ObservabilitySkill detects threshold breaches (alert fires), this skill
automatically creates structured incidents in IncidentResponseSkill, ensuring
no alert goes unhandled. When alerts resolve, corresponding incidents are
auto-resolved too.

This completes the reactive self-healing loop:
  metrics → alerts → incidents → response → postmortem → improvement

Pillar: Self-Improvement (closes the observe → react → learn feedback loop)

Flow:
1. poll_alerts: Check ObservabilitySkill for newly fired/resolved alerts
2. For fired alerts: auto-create incidents via IncidentResponseSkill
3. For resolved alerts: auto-resolve matching incidents
4. Severity mapping: alert severity → incident severity (configurable)
5. Dedup: won't create duplicate incidents for the same alert
6. Emit events through EventBus for downstream automation

Actions:
- poll: Check alerts and auto-create/resolve incidents
- configure: Set severity mappings, auto-triage, and behavior options
- link: Manually link an alert to an existing incident
- unlink: Remove an alert-incident link
- status: View active alert-incident links and bridge health
- history: View recent bridge actions (creates, resolves, dedup skips)
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from .base import Skill, SkillResult, SkillManifest, SkillAction

BRIDGE_DATA_FILE = Path(__file__).parent.parent / "data" / "alert_incident_bridge.json"
MAX_HISTORY = 200
MAX_LINKS = 500


def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _now_ts() -> float:
    return time.time()


# Default severity mapping: alert severity → incident severity
DEFAULT_SEVERITY_MAP = {
    "critical": "sev1",
    "warning": "sev2",
    "info": "sev3",
}


class AlertIncidentBridgeSkill(Skill):
    """
    Automatically creates and resolves incidents from observability alerts.

    Bridges ObservabilitySkill (metric alerts) and IncidentResponseSkill
    (structured incident management) so the agent can autonomously detect
    and respond to operational issues.
    """

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self._store = None

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="alert_incident_bridge",
            name="Alert-to-Incident Bridge",
            version="1.0.0",
            category="meta",
            description="Auto-creates/resolves incidents from observability alerts, closing the observe-react-learn loop",
            actions=self.get_actions(),
            required_credentials=[],
            install_cost=0,
            author="singularity",
        )

    def get_actions(self) -> List[SkillAction]:
        return [
            SkillAction(
                name="poll",
                description="Check alerts and auto-create/resolve incidents for any state changes",
                parameters={
                    "auto_triage": {"type": "boolean", "required": False,
                                    "description": "Auto-triage created incidents (default: True)"},
                    "dry_run": {"type": "boolean", "required": False,
                                "description": "Preview what would happen without executing (default: False)"},
                },
            ),
            SkillAction(
                name="configure",
                description="Set severity mappings, auto-triage, and bridge behavior",
                parameters={
                    "severity_map": {"type": "object", "required": False,
                                     "description": "Map alert severity to incident severity, e.g. {'critical': 'sev1'}"},
                    "auto_triage": {"type": "boolean", "required": False,
                                    "description": "Auto-triage incidents after creation (default: True)"},
                    "auto_resolve": {"type": "boolean", "required": False,
                                     "description": "Auto-resolve incidents when alerts resolve (default: True)"},
                    "emit_events": {"type": "boolean", "required": False,
                                    "description": "Emit EventBus events for bridge actions (default: True)"},
                },
            ),
            SkillAction(
                name="link",
                description="Manually link an alert name to an existing incident ID",
                parameters={
                    "alert_name": {"type": "string", "required": True,
                                   "description": "Alert rule name from ObservabilitySkill"},
                    "incident_id": {"type": "string", "required": True,
                                    "description": "Incident ID from IncidentResponseSkill"},
                },
            ),
            SkillAction(
                name="unlink",
                description="Remove an alert-incident link",
                parameters={
                    "alert_name": {"type": "string", "required": True,
                                   "description": "Alert rule name to unlink"},
                },
            ),
            SkillAction(
                name="status",
                description="View active alert-incident links, config, and bridge health",
                parameters={},
            ),
            SkillAction(
                name="history",
                description="View recent bridge actions",
                parameters={
                    "limit": {"type": "number", "required": False,
                              "description": "Max entries to return (default: 20)"},
                    "action_filter": {"type": "string", "required": False,
                                      "description": "Filter by action type: created, resolved, dedup, error"},
                },
            ),
        ]

    # ── Persistence ──────────────────────────────────────────────────

    def _default_state(self) -> Dict:
        return {
            "links": {},  # alert_name → {incident_id, created_at, alert_severity, ...}
            "config": {
                "severity_map": dict(DEFAULT_SEVERITY_MAP),
                "auto_triage": True,
                "auto_resolve": True,
                "emit_events": True,
            },
            "history": [],
            "stats": {
                "incidents_created": 0,
                "incidents_resolved": 0,
                "dedup_skips": 0,
                "errors": 0,
                "polls": 0,
            },
            "metadata": {
                "created_at": _now_iso(),
                "last_poll": None,
            },
        }

    def _load(self) -> Dict:
        if self._store is not None:
            return self._store
        BRIDGE_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
        if BRIDGE_DATA_FILE.exists():
            try:
                with open(BRIDGE_DATA_FILE, "r") as f:
                    self._store = json.load(f)
                    return self._store
            except (json.JSONDecodeError, OSError):
                pass
        self._store = self._default_state()
        return self._store

    def _save(self, data: Dict):
        self._store = data
        if len(data.get("history", [])) > MAX_HISTORY:
            data["history"] = data["history"][-MAX_HISTORY:]
        BRIDGE_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(BRIDGE_DATA_FILE, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def _log_action(self, store: Dict, action_type: str, details: Dict):
        store["history"].append({
            "action": action_type,
            "details": details,
            "timestamp": _now_iso(),
        })

    # ── Execute Dispatch ─────────────────────────────────────────────

    async def execute(self, action: str, params: Dict[str, Any]) -> SkillResult:
        handlers = {
            "poll": self._poll,
            "configure": self._configure,
            "link": self._link,
            "unlink": self._unlink,
            "status": self._status,
            "history": self._history,
        }
        handler = handlers.get(action)
        if not handler:
            return SkillResult(success=False, message=f"Unknown action: {action}")
        return await handler(params)

    # ── Action: poll ─────────────────────────────────────────────────

    async def _poll(self, params: Dict) -> SkillResult:
        """Check all alerts and create/resolve incidents as needed."""
        auto_triage = params.get("auto_triage", True)
        dry_run = params.get("dry_run", False)

        store = self._load()
        config = store["config"]

        # Step 1: Get current alert state from ObservabilitySkill
        alert_data = await self._get_alert_state()
        if alert_data is None:
            store["stats"]["errors"] += 1
            self._save(store)
            return SkillResult(
                success=False,
                message="Failed to get alert state from ObservabilitySkill. Is it available?",
            )

        store["stats"]["polls"] += 1
        store["metadata"]["last_poll"] = _now_iso()

        alert_rules = alert_data.get("rules", [])
        created = []
        resolved = []
        deduped = []
        errors = []

        for alert in alert_rules:
            alert_name = alert.get("name", "")
            alert_state = alert.get("state", "ok")
            alert_severity = alert.get("severity", "warning")

            if alert_state == "firing":
                # Check if we already have a link for this alert
                if alert_name in store["links"]:
                    deduped.append(alert_name)
                    store["stats"]["dedup_skips"] += 1
                    self._log_action(store, "dedup", {
                        "alert_name": alert_name,
                        "existing_incident": store["links"][alert_name]["incident_id"],
                    })
                    continue

                # Map severity
                incident_severity = config["severity_map"].get(alert_severity, "sev3")

                if dry_run:
                    created.append({
                        "alert_name": alert_name,
                        "would_create_severity": incident_severity,
                        "dry_run": True,
                    })
                    continue

                # Create incident
                incident_result = await self._create_incident(
                    alert_name=alert_name,
                    alert=alert,
                    severity=incident_severity,
                )

                if incident_result and incident_result.get("incident_id"):
                    incident_id = incident_result["incident_id"]
                    store["links"][alert_name] = {
                        "incident_id": incident_id,
                        "alert_severity": alert_severity,
                        "incident_severity": incident_severity,
                        "created_at": _now_iso(),
                        "auto_created": True,
                    }
                    store["stats"]["incidents_created"] += 1
                    created.append({
                        "alert_name": alert_name,
                        "incident_id": incident_id,
                        "severity": incident_severity,
                    })
                    self._log_action(store, "created", {
                        "alert_name": alert_name,
                        "incident_id": incident_id,
                        "severity": incident_severity,
                    })

                    # Auto-triage if enabled
                    if auto_triage and config.get("auto_triage", True):
                        await self._auto_triage_incident(incident_id, incident_severity, alert_name)

                    # Emit event
                    if config.get("emit_events", True):
                        await self._emit_event("alert_bridge.incident_created", {
                            "alert_name": alert_name,
                            "incident_id": incident_id,
                            "severity": incident_severity,
                        })
                else:
                    errors.append({
                        "alert_name": alert_name,
                        "error": "Failed to create incident",
                    })
                    store["stats"]["errors"] += 1
                    self._log_action(store, "error", {
                        "alert_name": alert_name,
                        "error": "incident creation failed",
                    })

            elif alert_state in ("ok", "cooldown"):
                # Check if we have a link that should be resolved
                if alert_name in store["links"] and config.get("auto_resolve", True):
                    link = store["links"][alert_name]
                    incident_id = link["incident_id"]

                    if dry_run:
                        resolved.append({
                            "alert_name": alert_name,
                            "would_resolve_incident": incident_id,
                            "dry_run": True,
                        })
                        continue

                    resolve_ok = await self._resolve_incident(incident_id, alert_name)

                    if resolve_ok:
                        store["stats"]["incidents_resolved"] += 1
                        resolved.append({
                            "alert_name": alert_name,
                            "incident_id": incident_id,
                        })
                        self._log_action(store, "resolved", {
                            "alert_name": alert_name,
                            "incident_id": incident_id,
                        })

                        # Emit event
                        if config.get("emit_events", True):
                            await self._emit_event("alert_bridge.incident_resolved", {
                                "alert_name": alert_name,
                                "incident_id": incident_id,
                            })

                    # Remove link regardless (alert is no longer firing)
                    del store["links"][alert_name]

        # Enforce link limit
        if len(store["links"]) > MAX_LINKS:
            oldest = sorted(store["links"].items(), key=lambda x: x[1].get("created_at", ""))
            for name, _ in oldest[:len(store["links"]) - MAX_LINKS]:
                del store["links"][name]

        self._save(store)

        return SkillResult(
            success=True,
            message=(
                f"Poll complete: {len(created)} incident(s) created, "
                f"{len(resolved)} resolved, {len(deduped)} deduped, "
                f"{len(errors)} error(s)"
            ),
            data={
                "created": created,
                "resolved": resolved,
                "deduped": deduped,
                "errors": errors,
                "active_links": len(store["links"]),
                "dry_run": dry_run,
            },
        )

    # ── Action: configure ────────────────────────────────────────────

    async def _configure(self, params: Dict) -> SkillResult:
        store = self._load()
        config = store["config"]
        changed = []

        if "severity_map" in params:
            smap = params["severity_map"]
            if isinstance(smap, dict):
                config["severity_map"].update(smap)
                changed.append("severity_map")

        if "auto_triage" in params:
            config["auto_triage"] = bool(params["auto_triage"])
            changed.append("auto_triage")

        if "auto_resolve" in params:
            config["auto_resolve"] = bool(params["auto_resolve"])
            changed.append("auto_resolve")

        if "emit_events" in params:
            config["emit_events"] = bool(params["emit_events"])
            changed.append("emit_events")

        if not changed:
            return SkillResult(
                success=True,
                message="No configuration changes specified",
                data={"config": config},
            )

        self._save(store)

        return SkillResult(
            success=True,
            message=f"Updated configuration: {', '.join(changed)}",
            data={"config": config, "changed": changed},
        )

    # ── Action: link ─────────────────────────────────────────────────

    async def _link(self, params: Dict) -> SkillResult:
        alert_name = params.get("alert_name", "").strip()
        incident_id = params.get("incident_id", "").strip()

        if not alert_name:
            return SkillResult(success=False, message="alert_name is required")
        if not incident_id:
            return SkillResult(success=False, message="incident_id is required")

        store = self._load()

        if alert_name in store["links"]:
            existing = store["links"][alert_name]["incident_id"]
            return SkillResult(
                success=False,
                message=f"Alert '{alert_name}' already linked to incident '{existing}'. Unlink first.",
            )

        store["links"][alert_name] = {
            "incident_id": incident_id,
            "alert_severity": "unknown",
            "incident_severity": "unknown",
            "created_at": _now_iso(),
            "auto_created": False,
        }

        self._log_action(store, "manual_link", {
            "alert_name": alert_name,
            "incident_id": incident_id,
        })
        self._save(store)

        return SkillResult(
            success=True,
            message=f"Linked alert '{alert_name}' to incident '{incident_id}'",
            data={"alert_name": alert_name, "incident_id": incident_id},
        )

    # ── Action: unlink ───────────────────────────────────────────────

    async def _unlink(self, params: Dict) -> SkillResult:
        alert_name = params.get("alert_name", "").strip()
        if not alert_name:
            return SkillResult(success=False, message="alert_name is required")

        store = self._load()

        if alert_name not in store["links"]:
            return SkillResult(success=False, message=f"No link found for alert '{alert_name}'")

        removed = store["links"].pop(alert_name)
        self._log_action(store, "unlink", {
            "alert_name": alert_name,
            "incident_id": removed["incident_id"],
        })
        self._save(store)

        return SkillResult(
            success=True,
            message=f"Unlinked alert '{alert_name}' from incident '{removed['incident_id']}'",
            data={"alert_name": alert_name, "removed_link": removed},
        )

    # ── Action: status ───────────────────────────────────────────────

    async def _status(self, params: Dict) -> SkillResult:
        store = self._load()

        active_links = []
        for alert_name, link in store["links"].items():
            active_links.append({
                "alert_name": alert_name,
                "incident_id": link["incident_id"],
                "severity": link.get("incident_severity", "unknown"),
                "created_at": link["created_at"],
                "auto_created": link.get("auto_created", True),
            })

        return SkillResult(
            success=True,
            message=(
                f"{len(active_links)} active link(s). "
                f"Created: {store['stats']['incidents_created']}, "
                f"Resolved: {store['stats']['incidents_resolved']}, "
                f"Polls: {store['stats']['polls']}"
            ),
            data={
                "active_links": active_links,
                "config": store["config"],
                "stats": store["stats"],
                "last_poll": store["metadata"].get("last_poll"),
            },
        )

    # ── Action: history ──────────────────────────────────────────────

    async def _history(self, params: Dict) -> SkillResult:
        limit = params.get("limit", 20)
        action_filter = params.get("action_filter", "")

        store = self._load()
        history = store.get("history", [])

        if action_filter:
            history = [h for h in history if h.get("action") == action_filter]

        recent = history[-limit:]

        return SkillResult(
            success=True,
            message=f"{len(recent)} history entries (of {len(history)} total)",
            data={"history": recent, "total": len(history)},
        )

    # ── Internal: Interact with ObservabilitySkill ───────────────────

    async def _get_alert_state(self) -> Optional[Dict]:
        """Get current alert list from ObservabilitySkill."""
        # Try via skill context (agent runtime)
        if self.context:
            try:
                result = await self.context.call_skill("observability", "alert_list", {})
                if result and result.success:
                    return result.data
            except Exception:
                pass

        # Fallback: read alert file directly
        try:
            from .observability import ALERTS_FILE as obs_alerts_file
            if obs_alerts_file.exists():
                with open(obs_alerts_file, "r") as f:
                    alerts_data = json.load(f)
                rules = []
                for name, rule in alerts_data.get("rules", {}).items():
                    rules.append({
                        "name": name,
                        "state": rule.get("state", "ok"),
                        "severity": rule.get("severity", "warning"),
                        "metric": rule.get("metric_name", ""),
                        "condition": f"{rule.get('condition', '')} {rule.get('threshold', '')}",
                        "fire_count": rule.get("fire_count", 0),
                        "last_fired": rule.get("last_fired"),
                    })
                return {"rules": rules, "total": len(rules)}
        except Exception:
            pass

        return None

    async def _create_incident(self, alert_name: str, alert: Dict, severity: str) -> Optional[Dict]:
        """Create an incident via IncidentResponseSkill."""
        title = f"[Alert] {alert_name}: {alert.get('metric', 'unknown')} {alert.get('condition', '')}"
        description = (
            f"Automatically created by AlertIncidentBridge.\n"
            f"Alert: {alert_name}\n"
            f"Metric: {alert.get('metric', 'unknown')}\n"
            f"Condition: {alert.get('condition', '')}\n"
            f"Current value: {alert.get('current_value', 'N/A')}\n"
            f"Fire count: {alert.get('fire_count', 0)}\n"
            f"Source: observability_alert"
        )

        params = {
            "title": title,
            "description": description,
            "severity": severity,
            "source": "alert_incident_bridge",
        }

        # Try via skill context
        if self.context:
            try:
                result = await self.context.call_skill("incident_response", "detect", params)
                if result and result.success and result.data:
                    return result.data
            except Exception:
                pass

        # Fallback: create directly via IncidentResponseSkill
        try:
            from .incident_response import IncidentResponseSkill
            ir_skill = IncidentResponseSkill()
            result = await ir_skill.execute("detect", params)
            if result.success and result.data:
                return result.data
        except Exception:
            pass

        return None

    async def _auto_triage_incident(self, incident_id: str, severity: str, alert_name: str):
        """Auto-triage a newly created incident."""
        triage_params = {
            "incident_id": incident_id,
            "severity": severity,
            "notes": f"Auto-triaged from alert '{alert_name}'",
        }

        if self.context:
            try:
                await self.context.call_skill("incident_response", "triage", triage_params)
            except Exception:
                pass
        else:
            try:
                from .incident_response import IncidentResponseSkill
                ir_skill = IncidentResponseSkill()
                await ir_skill.execute("triage", triage_params)
            except Exception:
                pass

    async def _resolve_incident(self, incident_id: str, alert_name: str) -> bool:
        """Auto-resolve an incident when its alert resolves."""
        resolve_params = {
            "incident_id": incident_id,
            "resolution": f"Alert '{alert_name}' resolved - metric returned to normal range",
            "root_cause": "Metric threshold breach (auto-resolved)",
        }

        if self.context:
            try:
                result = await self.context.call_skill("incident_response", "resolve", resolve_params)
                return result.success if result else False
            except Exception:
                return False

        try:
            from .incident_response import IncidentResponseSkill
            ir_skill = IncidentResponseSkill()
            result = await ir_skill.execute("resolve", resolve_params)
            return result.success
        except Exception:
            return False

    async def _emit_event(self, topic: str, data: Dict):
        """Emit event via EventBus if available."""
        if not self.context:
            return
        try:
            await self.context.call_skill("event", "publish", {
                "topic": topic,
                "data": data,
                "source": "alert_incident_bridge",
            })
        except Exception:
            pass
