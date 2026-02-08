"""Tests for EventDrivenWorkflowSkill - event-driven autonomous workflows."""
import pytest
import asyncio
from pathlib import Path
from singularity.skills.event_workflow import EventDrivenWorkflowSkill


@pytest.fixture
def skill(tmp_path):
    return EventDrivenWorkflowSkill(data_dir=str(tmp_path))


def make_step(skill_id="revenue_services", action="code_review", **kwargs):
    return {"skill_id": skill_id, "action": action, **kwargs}


@pytest.mark.asyncio
async def test_create_rule(skill):
    result = await skill.execute("create_rule", {
        "name": "on-github-push",
        "description": "Run code review on GitHub push events",
        "event_type": "webhook",
        "event_source": "github",
        "steps": [make_step()],
    })
    assert result.success
    assert result.data["name"] == "on-github-push"
    assert result.data["steps_count"] == 1


@pytest.mark.asyncio
async def test_create_rule_duplicate(skill):
    await skill.execute("create_rule", {
        "name": "dup-rule", "event_type": "webhook",
        "steps": [make_step()],
    })
    result = await skill.execute("create_rule", {
        "name": "dup-rule", "event_type": "webhook",
        "steps": [make_step()],
    })
    assert not result.success
    assert "already exists" in result.message


@pytest.mark.asyncio
async def test_create_rule_validation(skill):
    result = await skill.execute("create_rule", {
        "name": "", "event_type": "webhook", "steps": [make_step()],
    })
    assert not result.success

    result = await skill.execute("create_rule", {
        "name": "r", "event_type": "webhook", "steps": [],
    })
    assert not result.success


@pytest.mark.asyncio
async def test_trigger_no_match(skill):
    result = await skill.execute("trigger", {
        "event_type": "webhook", "event_source": "github", "payload": {},
    })
    assert result.success
    assert result.data["matched_rules"] == 0


@pytest.mark.asyncio
async def test_trigger_matching_rule(skill):
    await skill.execute("create_rule", {
        "name": "github-pr", "event_type": "webhook", "event_source": "github",
        "steps": [make_step()],
    })
    result = await skill.execute("trigger", {
        "event_type": "webhook", "event_source": "github",
        "payload": {"action": "opened"},
    })
    assert result.success
    assert result.data["matched_rules"] == 1
    assert result.data["successful"] == 1


@pytest.mark.asyncio
async def test_trigger_with_conditions(skill):
    await skill.execute("create_rule", {
        "name": "pr-opened", "event_type": "webhook", "event_source": "github",
        "conditions": {"action": {"op": "eq", "value": "opened"}},
        "steps": [make_step()],
    })
    # Should match
    r = await skill.execute("trigger", {
        "event_type": "webhook", "event_source": "github",
        "payload": {"action": "opened"},
    })
    assert r.data["matched_rules"] == 1

    # Should NOT match
    r = await skill.execute("trigger", {
        "event_type": "webhook", "event_source": "github",
        "payload": {"action": "closed"},
    })
    assert r.data["matched_rules"] == 0


@pytest.mark.asyncio
async def test_multi_step_workflow(skill):
    await skill.execute("create_rule", {
        "name": "multi-step", "event_type": "internal",
        "steps": [
            make_step(step_id="s1"),
            make_step(skill_id="usage_tracking", action="track", step_id="s2"),
        ],
    })
    r = await skill.execute("trigger", {
        "event_type": "internal", "payload": {"test": True},
    })
    assert r.success
    execs = r.data["executions"]
    assert len(execs) == 1
    assert len(execs[0]["step_results"]) == 2


@pytest.mark.asyncio
async def test_conditional_step_skip(skill):
    await skill.execute("create_rule", {
        "name": "conditional", "event_type": "internal",
        "steps": [
            make_step(step_id="s1"),
            make_step(step_id="s2", condition={
                "field": "should_run", "op": "eq", "value": True,
            }),
        ],
    })
    r = await skill.execute("trigger", {
        "event_type": "internal", "payload": {"should_run": False},
    })
    assert r.success
    steps = r.data["executions"][0]["step_results"]
    assert steps[0]["skipped"] is False
    assert steps[1]["skipped"] is True


@pytest.mark.asyncio
async def test_list_and_get_rules(skill):
    await skill.execute("create_rule", {
        "name": "r1", "event_type": "webhook", "steps": [make_step()],
    })
    await skill.execute("create_rule", {
        "name": "r2", "event_type": "cron", "steps": [make_step()],
    })
    r = await skill.execute("list_rules", {})
    assert r.data["total"] == 2

    r = await skill.execute("list_rules", {"event_type": "cron"})
    assert r.data["total"] == 1

    r = await skill.execute("get_rule", {"name": "r1"})
    assert r.success
    assert r.data["event_type"] == "webhook"


@pytest.mark.asyncio
async def test_update_and_delete_rule(skill):
    await skill.execute("create_rule", {
        "name": "updatable", "event_type": "webhook", "steps": [make_step()],
    })
    r = await skill.execute("update_rule", {"name": "updatable", "enabled": False})
    assert r.success
    assert "enabled" in r.data["updated_fields"]

    # Disabled rule should not match
    r = await skill.execute("trigger", {"event_type": "webhook"})
    assert r.data["matched_rules"] == 0

    r = await skill.execute("delete_rule", {"name": "updatable"})
    assert r.success

    r = await skill.execute("get_rule", {"name": "updatable"})
    assert not r.success


@pytest.mark.asyncio
async def test_execution_history(skill):
    await skill.execute("create_rule", {
        "name": "tracked", "event_type": "internal", "steps": [make_step()],
    })
    await skill.execute("trigger", {"event_type": "internal", "payload": {}})
    await skill.execute("trigger", {"event_type": "internal", "payload": {}})

    r = await skill.execute("get_executions", {"rule_name": "tracked"})
    assert r.data["total"] == 2


@pytest.mark.asyncio
async def test_stats(skill):
    await skill.execute("create_rule", {
        "name": "stat-rule", "event_type": "webhook", "steps": [make_step()],
    })
    await skill.execute("trigger", {"event_type": "webhook"})

    r = await skill.execute("get_stats", {})
    assert r.success
    assert r.data["total_rules"] == 1
    assert r.data["total_executions"] == 1


@pytest.mark.asyncio
async def test_persistence(tmp_path):
    s1 = EventDrivenWorkflowSkill(data_dir=str(tmp_path))
    await s1.execute("create_rule", {
        "name": "persist-test", "event_type": "cron", "steps": [make_step()],
    })
    # Create a new instance - should load saved data
    s2 = EventDrivenWorkflowSkill(data_dir=str(tmp_path))
    r = await s2.execute("get_rule", {"name": "persist-test"})
    assert r.success
    assert r.data["event_type"] == "cron"


@pytest.mark.asyncio
async def test_param_resolution(skill):
    """Test parameter mapping from event payload."""
    await skill.execute("create_rule", {
        "name": "param-test", "event_type": "webhook",
        "steps": [make_step(
            param_mapping={"repo": "event.repository.name", "msg": "event.message"},
            static_params={"format": "brief"},
        )],
    })
    r = await skill.execute("trigger", {
        "event_type": "webhook",
        "payload": {"repository": {"name": "my-repo"}, "message": "hello"},
    })
    assert r.success
    exec_data = r.data["executions"][0]
    # Step should have been executed (dry run in test)
    assert exec_data["step_results"][0]["success"] is True
