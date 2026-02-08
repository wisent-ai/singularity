"""Tests for AlertIncidentBridgeSkill."""

import pytest
import json
from singularity.skills.alert_incident_bridge import (
    AlertIncidentBridgeSkill, BRIDGE_DATA_FILE, DEFAULT_SEVERITY_MAP,
)
import singularity.skills.alert_incident_bridge as mod
import singularity.skills.observability as obs_mod
import singularity.skills.incident_response as ir_mod


@pytest.fixture
def skill(tmp_path):
    """Create skill with temp data paths."""
    s = AlertIncidentBridgeSkill()
    mod.BRIDGE_DATA_FILE = tmp_path / "alert_incident_bridge.json"
    # Also redirect observability and incident_response data files
    obs_mod.METRICS_FILE = tmp_path / "metrics.json"
    obs_mod.ALERTS_FILE = tmp_path / "alerts.json"
    ir_mod.INCIDENT_FILE = tmp_path / "incidents.json"
    return s


@pytest.mark.asyncio
async def test_manifest(skill):
    m = skill.manifest
    assert m.skill_id == "alert_incident_bridge"
    assert len(m.actions) == 6
    names = [a.name for a in m.actions]
    assert "poll" in names
    assert "configure" in names
    assert "link" in names
    assert "status" in names


@pytest.mark.asyncio
async def test_configure_severity_map(skill):
    r = await skill.execute("configure", {"severity_map": {"critical": "sev1", "info": "sev4"}})
    assert r.success
    assert r.data["config"]["severity_map"]["info"] == "sev4"


@pytest.mark.asyncio
async def test_configure_auto_resolve(skill):
    r = await skill.execute("configure", {"auto_resolve": False})
    assert r.success
    assert r.data["config"]["auto_resolve"] is False


@pytest.mark.asyncio
async def test_configure_no_changes(skill):
    r = await skill.execute("configure", {})
    assert r.success
    assert "No configuration changes" in r.message


@pytest.mark.asyncio
async def test_manual_link(skill):
    r = await skill.execute("link", {"alert_name": "high_cpu", "incident_id": "INC-001"})
    assert r.success
    assert r.data["incident_id"] == "INC-001"


@pytest.mark.asyncio
async def test_link_duplicate_blocked(skill):
    await skill.execute("link", {"alert_name": "high_cpu", "incident_id": "INC-001"})
    r = await skill.execute("link", {"alert_name": "high_cpu", "incident_id": "INC-002"})
    assert not r.success
    assert "already linked" in r.message


@pytest.mark.asyncio
async def test_link_missing_params(skill):
    r = await skill.execute("link", {"alert_name": ""})
    assert not r.success
    r2 = await skill.execute("link", {"alert_name": "x", "incident_id": ""})
    assert not r2.success


@pytest.mark.asyncio
async def test_unlink(skill):
    await skill.execute("link", {"alert_name": "high_cpu", "incident_id": "INC-001"})
    r = await skill.execute("unlink", {"alert_name": "high_cpu"})
    assert r.success
    assert r.data["removed_link"]["incident_id"] == "INC-001"


@pytest.mark.asyncio
async def test_unlink_not_found(skill):
    r = await skill.execute("unlink", {"alert_name": "nonexistent"})
    assert not r.success


@pytest.mark.asyncio
async def test_status_empty(skill):
    r = await skill.execute("status", {})
    assert r.success
    assert r.data["active_links"] == []
    assert r.data["stats"]["incidents_created"] == 0


@pytest.mark.asyncio
async def test_status_with_links(skill):
    await skill.execute("link", {"alert_name": "cpu_alert", "incident_id": "INC-01"})
    r = await skill.execute("status", {})
    assert r.success
    assert len(r.data["active_links"]) == 1
    assert r.data["active_links"][0]["alert_name"] == "cpu_alert"


@pytest.mark.asyncio
async def test_history_empty(skill):
    r = await skill.execute("history", {})
    assert r.success
    assert len(r.data["history"]) == 0


@pytest.mark.asyncio
async def test_history_after_actions(skill):
    await skill.execute("link", {"alert_name": "x", "incident_id": "INC-1"})
    await skill.execute("unlink", {"alert_name": "x"})
    r = await skill.execute("history", {})
    assert r.success
    assert len(r.data["history"]) == 2
    assert r.data["history"][0]["action"] == "manual_link"
    assert r.data["history"][1]["action"] == "unlink"


@pytest.mark.asyncio
async def test_history_filter(skill):
    await skill.execute("link", {"alert_name": "a", "incident_id": "I1"})
    await skill.execute("unlink", {"alert_name": "a"})
    r = await skill.execute("history", {"action_filter": "unlink"})
    assert r.success
    assert len(r.data["history"]) == 1


@pytest.mark.asyncio
async def test_poll_with_firing_alert(skill):
    """Poll creates incidents for firing alerts (via direct file fallback)."""
    from singularity.skills.observability import ObservabilitySkill
    obs = ObservabilitySkill()
    # Create a metric and alert
    await obs.execute("emit", {"name": "error_rate", "value": 95.0})
    await obs.execute("alert_create", {
        "name": "high_errors",
        "metric_name": "error_rate",
        "condition": "above",
        "threshold": 50,
        "severity": "critical",
        "window_minutes": 60,
    })
    # Trigger the alert
    await obs.execute("check_alerts", {})

    # Now poll
    r = await skill.execute("poll", {})
    assert r.success
    assert len(r.data["created"]) == 1
    assert r.data["created"][0]["alert_name"] == "high_errors"
    assert r.data["created"][0]["severity"] == "sev1"  # critical → sev1


@pytest.mark.asyncio
async def test_poll_dedup(skill):
    """Polling twice doesn't create duplicate incidents."""
    from singularity.skills.observability import ObservabilitySkill
    obs = ObservabilitySkill()
    await obs.execute("emit", {"name": "latency", "value": 5000.0})
    await obs.execute("alert_create", {
        "name": "slow_response",
        "metric_name": "latency",
        "condition": "above",
        "threshold": 1000,
        "severity": "warning",
        "window_minutes": 60,
    })
    await obs.execute("check_alerts", {})

    r1 = await skill.execute("poll", {})
    assert len(r1.data["created"]) == 1

    r2 = await skill.execute("poll", {})
    assert len(r2.data["created"]) == 0
    assert len(r2.data["deduped"]) == 1


@pytest.mark.asyncio
async def test_poll_dry_run(skill):
    """Dry run previews without creating incidents."""
    from singularity.skills.observability import ObservabilitySkill
    obs = ObservabilitySkill()
    await obs.execute("emit", {"name": "mem", "value": 99.0})
    await obs.execute("alert_create", {
        "name": "high_mem",
        "metric_name": "mem",
        "condition": "above",
        "threshold": 80,
        "severity": "warning",
        "window_minutes": 60,
    })
    await obs.execute("check_alerts", {})

    r = await skill.execute("poll", {"dry_run": True})
    assert r.success
    assert r.data["dry_run"] is True
    assert len(r.data["created"]) == 1
    assert r.data["created"][0]["dry_run"] is True
    # No actual links created
    status = await skill.execute("status", {})
    assert len(status.data["active_links"]) == 0


@pytest.mark.asyncio
async def test_unknown_action(skill):
    r = await skill.execute("nonexistent", {})
    assert not r.success
