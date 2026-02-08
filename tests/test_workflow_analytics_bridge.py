"""Tests for WorkflowAnalyticsBridgeSkill."""
import json
import pytest
import time
from pathlib import Path
from singularity.skills.workflow_analytics_bridge import (
    WorkflowAnalyticsBridgeSkill, DATA_DIR, BRIDGE_FILE, WORKFLOWS_FILE, TEMPLATE_BRIDGE_FILE,
)


@pytest.fixture(autouse=True)
def clean_data(tmp_path, monkeypatch):
    monkeypatch.setattr("singularity.skills.workflow_analytics_bridge.DATA_DIR", tmp_path)
    monkeypatch.setattr("singularity.skills.workflow_analytics_bridge.BRIDGE_FILE", tmp_path / "workflow_analytics_bridge.json")
    monkeypatch.setattr("singularity.skills.workflow_analytics_bridge.WORKFLOWS_FILE", tmp_path / "workflow_analytics.json")
    monkeypatch.setattr("singularity.skills.workflow_analytics_bridge.TEMPLATE_BRIDGE_FILE", tmp_path / "template_event_bridge.json")
    yield tmp_path


def _seed_executions(tmp_path, template_id="pr_review", count=5, success_rate=0.8):
    """Seed execution data directly."""
    skill = WorkflowAnalyticsBridgeSkill()
    bridge = skill._load_bridge()
    now = time.time()
    for i in range(count):
        success = i < int(count * success_rate)
        bridge["executions"].append({
            "template_id": template_id,
            "deployment_id": f"dep_{i}",
            "steps": [
                {"action": "fetch_code", "success": True, "duration_ms": 100},
                {"action": "run_review", "success": success, "duration_ms": 200},
                {"action": "post_comment", "success": True, "duration_ms": 50},
            ],
            "success": success,
            "trigger_event": "github-push",
            "revenue": 0.05 if success else 0,
            "timestamp": "2026-01-01T00:00:00Z",
            "ts": now - (count - i) * 60,
            "total_duration_ms": 350,
            "step_count": 3,
            "failed_steps": 0 if success else 1,
        })
    skill._save_bridge(bridge)
    return skill


def _seed_deployments(tmp_path, template_ids=None):
    """Seed TemplateEventBridge deployment data."""
    template_ids = template_ids or ["pr_review", "deploy_service"]
    deployments = {}
    for tid in template_ids:
        deployments[f"dep_{tid}"] = {
            "template_id": tid,
            "steps": [{"name": "fetch_code"}, {"name": "run_review"}, {"name": "post_comment"}],
            "status": "active",
        }
    data = {"deployments": deployments, "stats": {}}
    (tmp_path / "template_event_bridge.json").write_text(json.dumps(data))


@pytest.mark.asyncio
async def test_record_execution(clean_data):
    skill = WorkflowAnalyticsBridgeSkill()
    result = await skill.execute("record_execution", {
        "template_id": "pr_review",
        "steps": [{"action": "fetch", "success": True, "duration_ms": 100}],
        "success": True,
        "revenue": 0.10,
    })
    assert result.success
    assert "pr_review" in result.message
    bridge = json.loads((clean_data / "workflow_analytics_bridge.json").read_text())
    assert len(bridge["executions"]) == 1


@pytest.mark.asyncio
async def test_record_writes_to_workflow_analytics(clean_data):
    skill = WorkflowAnalyticsBridgeSkill()
    await skill.execute("record_execution", {
        "template_id": "test_wf",
        "steps": [{"action": "step1", "success": True}],
        "success": True,
    })
    wf = json.loads((clean_data / "workflow_analytics.json").read_text())
    assert len(wf["workflows"]) == 1
    assert wf["metadata"]["total_successes"] == 1


@pytest.mark.asyncio
async def test_template_health(clean_data):
    _seed_executions(clean_data, "pr_review", count=5, success_rate=0.8)
    skill = WorkflowAnalyticsBridgeSkill()
    result = await skill.execute("template_health")
    assert result.success
    scores = result.data["scores"]
    assert "pr_review" in scores
    assert scores["pr_review"]["health_score"] > 0


@pytest.mark.asyncio
async def test_template_health_specific(clean_data):
    _seed_executions(clean_data, "pr_review", count=5)
    _seed_executions(clean_data, "deploy", count=3)
    skill = WorkflowAnalyticsBridgeSkill()
    result = await skill.execute("template_health", {"template_id": "pr_review"})
    assert result.success
    assert "pr_review" in result.data["scores"]
    assert "deploy" not in result.data["scores"]


@pytest.mark.asyncio
async def test_pattern_report(clean_data):
    _seed_executions(clean_data, "pr_review", count=5)
    skill = WorkflowAnalyticsBridgeSkill()
    result = await skill.execute("pattern_report")
    assert result.success
    assert result.data["total_executions"] == 5


@pytest.mark.asyncio
async def test_anti_patterns(clean_data):
    _seed_executions(clean_data, "bad_wf", count=6, success_rate=0.2)
    skill = WorkflowAnalyticsBridgeSkill()
    result = await skill.execute("anti_patterns", {"threshold": 0.3})
    assert result.success
    assert isinstance(result.data["anti_patterns"], list)


@pytest.mark.asyncio
async def test_recommend(clean_data):
    _seed_executions(clean_data, "pr_review", count=10, success_rate=0.9)
    _seed_executions(clean_data, "deploy_svc", count=5, success_rate=0.4)
    skill = WorkflowAnalyticsBridgeSkill()
    result = await skill.execute("recommend", {"top_k": 3})
    assert result.success
    recs = result.data["recommendations"]
    assert len(recs) > 0
    assert recs[0]["template_id"] == "pr_review"  # Higher health


@pytest.mark.asyncio
async def test_enrich_deployments(clean_data):
    _seed_deployments(clean_data, ["pr_review"])
    _seed_executions(clean_data, "pr_review", count=5)
    skill = WorkflowAnalyticsBridgeSkill()
    result = await skill.execute("enrich_deployments")
    assert result.success
    assert result.data["enriched"] == 1


@pytest.mark.asyncio
async def test_performance_dashboard(clean_data):
    _seed_executions(clean_data, "pr_review", count=8, success_rate=0.75)
    skill = WorkflowAnalyticsBridgeSkill()
    result = await skill.execute("performance_dashboard", {"window_hours": 24})
    assert result.success
    assert result.data["total_executions"] == 8
    assert result.data["overall_success_rate"] > 0


@pytest.mark.asyncio
async def test_status(clean_data):
    skill = WorkflowAnalyticsBridgeSkill()
    result = await skill.execute("status")
    assert result.success
    assert "execution_count" in result.data


@pytest.mark.asyncio
async def test_unknown_action(clean_data):
    skill = WorkflowAnalyticsBridgeSkill()
    result = await skill.execute("nonexistent")
    assert not result.success


@pytest.mark.asyncio
async def test_record_requires_template_id(clean_data):
    skill = WorkflowAnalyticsBridgeSkill()
    result = await skill.execute("record_execution", {"steps": [], "success": True})
    assert not result.success
