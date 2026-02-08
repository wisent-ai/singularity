"""Tests for WorkflowPipelineBridgeSkill."""
import json
import pytest
from pathlib import Path
from unittest.mock import patch
from singularity.skills.workflow_pipeline_bridge import WorkflowPipelineBridgeSkill, BRIDGE_DATA_FILE


@pytest.fixture
def bridge(tmp_path):
    """Create a bridge skill with temp data."""
    data_file = tmp_path / "bridge.json"
    with patch("singularity.skills.workflow_pipeline_bridge.BRIDGE_DATA_FILE", data_file):
        skill = WorkflowPipelineBridgeSkill()
        yield skill


@pytest.fixture
def sample_workflow():
    return {
        "name": "deploy-service",
        "description": "Deploy a service",
        "steps": [
            {"index": 0, "skill_id": "shell", "action": "run", "params": {"command": "git status"}, "label": "Check git", "on_failure": "stop", "max_retries": 0, "condition": None},
            {"index": 1, "skill_id": "github", "action": "create_pr", "params": {"title": "Deploy"}, "label": "Create PR", "on_failure": "stop", "max_retries": 1, "condition": None},
            {"index": 2, "skill_id": "shell", "action": "run", "params": {"command": "pytest"}, "label": "Run tests", "on_failure": "skip", "max_retries": 2, "condition": {"ref": "$steps.1.status", "equals": "success"}},
        ],
        "status": "ready",
        "run_count": 0,
    }


@pytest.fixture
def sample_pipeline():
    return [
        {"tool": "shell:run", "params": {"command": "git status"}, "label": "Check git", "timeout_seconds": 30, "max_cost": 0.02, "required": True, "retry_count": 0},
        {"tool": "github:create_pr", "params": {"title": "Deploy"}, "label": "Create PR", "timeout_seconds": 60, "max_cost": 0.05, "required": True, "retry_count": 1},
        {"tool": "shell:run", "params": {"command": "pytest"}, "label": "Run tests", "timeout_seconds": 30, "max_cost": 0.03, "required": False, "retry_count": 0, "condition": {"prev_success": True}},
    ]


class TestWorkflowToPipeline:
    def test_converts_steps(self, bridge, sample_workflow, tmp_path):
        wf_file = tmp_path / "workflows.json"
        wf_file.write_text(json.dumps({"workflows": {"deploy-service": sample_workflow}}))
        with patch("singularity.skills.workflow_pipeline_bridge.Path.__truediv__", return_value=wf_file):
            with patch.object(bridge, "_load_workflow", return_value=sample_workflow):
                result = bridge.execute("workflow_to_pipeline", {"workflow_name": "deploy-service"})
        assert result.success
        pipeline = result.data["pipeline"]
        assert len(pipeline) == 3
        assert pipeline[0]["tool"] == "shell:run"
        assert pipeline[1]["tool"] == "github:create_pr"

    def test_preserves_retries(self, bridge, sample_workflow):
        with patch.object(bridge, "_load_workflow", return_value=sample_workflow):
            result = bridge.execute("workflow_to_pipeline", {"workflow_name": "deploy-service"})
        assert result.success
        assert result.data["pipeline"][1]["retry_count"] == 1

    def test_converts_on_failure_to_required(self, bridge, sample_workflow):
        with patch.object(bridge, "_load_workflow", return_value=sample_workflow):
            result = bridge.execute("workflow_to_pipeline", {"workflow_name": "deploy-service"})
        # "skip" on_failure -> required=False
        assert result.data["pipeline"][2]["required"] is False
        # "stop" on_failure -> required=True
        assert result.data["pipeline"][0]["required"] is True

    def test_missing_workflow(self, bridge):
        with patch.object(bridge, "_load_workflow", return_value=None):
            result = bridge.execute("workflow_to_pipeline", {"workflow_name": "nonexistent"})
        assert not result.success

    def test_cost_budget_scaling(self, bridge, sample_workflow):
        with patch.object(bridge, "_load_workflow", return_value=sample_workflow):
            result = bridge.execute("workflow_to_pipeline", {"workflow_name": "deploy-service", "max_cost": 0.05})
        total = sum(s["max_cost"] for s in result.data["pipeline"])
        assert total <= 0.06  # Allow small rounding


class TestPipelineToWorkflow:
    def test_converts_steps(self, bridge, sample_pipeline):
        result = bridge.execute("pipeline_to_workflow", {"pipeline": sample_pipeline, "workflow_name": "my-workflow"})
        assert result.success
        wf = result.data["workflow_definition"]
        assert wf["name"] == "my-workflow"
        assert len(wf["steps"]) == 3
        assert wf["steps"][0]["skill_id"] == "shell"
        assert wf["steps"][0]["action"] == "run"

    def test_preserves_retries(self, bridge, sample_pipeline):
        result = bridge.execute("pipeline_to_workflow", {"pipeline": sample_pipeline, "workflow_name": "wf"})
        assert result.data["workflow_definition"]["steps"][1]["max_retries"] == 1

    def test_converts_required_to_on_failure(self, bridge, sample_pipeline):
        result = bridge.execute("pipeline_to_workflow", {"pipeline": sample_pipeline, "workflow_name": "wf"})
        steps = result.data["workflow_definition"]["steps"]
        assert steps[0]["on_failure"] == "stop"  # required=True
        assert steps[2]["on_failure"] == "skip"  # required=False

    def test_missing_pipeline(self, bridge):
        result = bridge.execute("pipeline_to_workflow", {"pipeline": [], "workflow_name": "wf"})
        assert not result.success


class TestRecommendEngine:
    def test_recommends_pipeline_for_small(self, bridge):
        pipeline = [{"tool": "shell:run", "params": {}, "timeout_seconds": 10, "max_cost": 0.01}] * 3
        result = bridge.execute("recommend_engine", {"pipeline": pipeline})
        assert result.success
        assert result.data["recommended_engine"] == "pipeline"

    def test_recommends_workflow_for_large(self, bridge):
        pipeline = [{"tool": f"shell:run", "params": {}, "timeout_seconds": 30, "max_cost": 0.05}] * 20
        result = bridge.execute("recommend_engine", {"pipeline": pipeline})
        assert result.success
        assert result.data["recommended_engine"] == "workflow"

    def test_requires_input(self, bridge):
        result = bridge.execute("recommend_engine", {})
        assert not result.success


class TestRecordComparison:
    def test_records_pipeline(self, bridge):
        result = bridge.execute("record_comparison", {"workflow_name": "wf1", "engine": "pipeline", "success": True, "duration_ms": 100})
        assert result.success

    def test_records_workflow(self, bridge):
        result = bridge.execute("record_comparison", {"workflow_name": "wf1", "engine": "workflow", "success": True, "duration_ms": 200})
        assert result.success

    def test_invalid_engine(self, bridge):
        result = bridge.execute("record_comparison", {"workflow_name": "wf1", "engine": "invalid", "success": True, "duration_ms": 100})
        assert not result.success


class TestCompareEngines:
    def test_no_data(self, bridge):
        result = bridge.execute("compare_engines", {})
        assert result.success
        assert result.data["comparisons"] == []

    def test_with_data(self, bridge):
        bridge.execute("record_comparison", {"workflow_name": "wf1", "engine": "pipeline", "success": True, "duration_ms": 100})
        bridge.execute("record_comparison", {"workflow_name": "wf1", "engine": "workflow", "success": True, "duration_ms": 200})
        result = bridge.execute("compare_engines", {})
        assert result.success
        assert result.data["total_data_points"] == 2


class TestStatus:
    def test_empty_status(self, bridge):
        result = bridge.execute("status", {})
        assert result.success
        assert result.data["total_conversions"] == 0

    def test_after_operations(self, bridge, sample_pipeline):
        bridge.execute("pipeline_to_workflow", {"pipeline": sample_pipeline, "workflow_name": "wf"})
        result = bridge.execute("status", {})
        assert result.data["total_conversions"] == 1
        assert result.data["pipeline_to_workflow"] == 1
