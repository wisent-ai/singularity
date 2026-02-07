#!/usr/bin/env python3
"""Tests for ReplicationSkill."""

import pytest
import json
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from singularity.skills.replication import (
    ReplicationSkill, AgentSnapshot, Replica, ReplicaStatus,
)


@pytest.fixture
def mock_agent():
    agent = MagicMock()
    agent.name = "TestAgent"
    agent.ticker = "TEST"
    agent.balance = 100.0
    cognition = MagicMock()
    cognition.system_prompt = "You are a test agent."
    cognition.llm_type = "anthropic"
    cognition.llm_model = "claude-sonnet-4-20250514"
    agent.cognition = cognition
    skills_reg = MagicMock()
    skills_reg.skills = {
        "content": MagicMock(manifest=MagicMock(version="1.0", category="content")),
    }
    agent.skills = skills_reg
    return agent


@pytest.fixture
def skill(mock_agent, tmp_path):
    s = ReplicationSkill()
    s.set_agent(mock_agent)
    s.set_data_dir(str(tmp_path / "replication"))
    return s


@pytest.mark.asyncio
async def test_snapshot_captures_config(skill, mock_agent):
    result = await skill.execute("snapshot", {"label": "v1"})
    assert result.success
    assert "snapshot_id" in result.data
    assert result.data["name"] == "TestAgent"
    assert result.data["label"] == "v1"


@pytest.mark.asyncio
async def test_list_snapshots(skill):
    await skill.execute("snapshot", {"label": "first"})
    await skill.execute("snapshot", {"label": "second"})
    result = await skill.execute("list_snapshots", {})
    assert result.success
    assert result.data["count"] == 2


@pytest.mark.asyncio
async def test_export_import_snapshot(skill):
    snap = await skill.execute("snapshot", {})
    sid = snap.data["snapshot_id"]
    exp = await skill.execute("export_snapshot", {"snapshot_id": sid})
    assert exp.success
    snap_json = exp.data["snapshot_json"]
    # Import into a fresh skill
    skill2 = ReplicationSkill()
    imp = await skill2.execute("import_snapshot", {"snapshot_json": snap_json})
    assert imp.success
    assert imp.data["name"] == "TestAgent"


@pytest.mark.asyncio
async def test_import_invalid_json(skill):
    result = await skill.execute("import_snapshot", {"snapshot_json": "not json"})
    assert not result.success


@pytest.mark.asyncio
async def test_export_nonexistent_snapshot(skill):
    result = await skill.execute("export_snapshot", {"snapshot_id": "nope"})
    assert not result.success


@pytest.mark.asyncio
async def test_spawn_requires_name(skill):
    result = await skill.execute("spawn", {"budget": 10})
    assert not result.success


@pytest.mark.asyncio
async def test_spawn_requires_positive_budget(skill):
    result = await skill.execute("spawn", {"name": "clone", "budget": 0})
    assert not result.success


@pytest.mark.asyncio
async def test_spawn_checks_parent_balance(skill, mock_agent):
    mock_agent.balance = 5.0
    result = await skill.execute("spawn", {"name": "clone", "budget": 50})
    assert not result.success
    assert "Insufficient" in result.message


@pytest.mark.asyncio
async def test_spawn_deducts_budget_on_success(skill, mock_agent):
    with patch("singularity.skills.replication.ReplicationSkill._launch_container",
               new_callable=AsyncMock, return_value="abc123"):
        result = await skill.execute("spawn", {"name": "clone1", "budget": 25.0})
        assert result.success
        assert mock_agent.balance == 75.0
        assert result.data["status"] == "running"


@pytest.mark.asyncio
async def test_spawn_refunds_on_failure(skill, mock_agent):
    with patch("singularity.skills.replication.ReplicationSkill._launch_container",
               new_callable=AsyncMock, side_effect=RuntimeError("docker fail")):
        result = await skill.execute("spawn", {"name": "fail", "budget": 20.0})
        assert not result.success
        assert mock_agent.balance == 100.0  # Refunded


@pytest.mark.asyncio
async def test_list_replicas(skill):
    with patch("singularity.skills.replication.ReplicationSkill._launch_container",
               new_callable=AsyncMock, return_value="abc"):
        await skill.execute("spawn", {"name": "r1", "budget": 10})
    result = await skill.execute("list_replicas", {})
    assert result.success
    assert result.data["total"] >= 1


@pytest.mark.asyncio
async def test_stop_replica(skill):
    with patch("singularity.skills.replication.ReplicationSkill._launch_container",
               new_callable=AsyncMock, return_value="cid123"):
        spawn_result = await skill.execute("spawn", {"name": "r2", "budget": 10})
    rid = spawn_result.data["replica_id"]
    with patch("singularity.skills.replication.ReplicationSkill._stop_container",
               new_callable=AsyncMock):
        result = await skill.execute("stop_replica", {"replica_id": rid})
    assert result.success


@pytest.mark.asyncio
async def test_stop_nonexistent_replica(skill):
    result = await skill.execute("stop_replica", {"replica_id": "nope"})
    assert not result.success


@pytest.mark.asyncio
async def test_inspect_replica(skill):
    with patch("singularity.skills.replication.ReplicationSkill._launch_container",
               new_callable=AsyncMock, return_value="cid"):
        spawn = await skill.execute("spawn", {"name": "ins", "budget": 5})
    rid = spawn.data["replica_id"]
    result = await skill.execute("inspect_replica", {"replica_id": rid})
    assert result.success
    assert result.data["name"] == "ins"


@pytest.mark.asyncio
async def test_spawn_with_mutations(skill, mock_agent):
    with patch("singularity.skills.replication.ReplicationSkill._launch_container",
               new_callable=AsyncMock, return_value="mut123"):
        result = await skill.execute("spawn", {
            "name": "mutant",
            "budget": 15,
            "mutations": {"model": "gpt-4o", "prompt_addition": "Be creative."}
        })
    assert result.success


@pytest.mark.asyncio
async def test_max_replicas_enforced(skill):
    skill._max_replicas = 2
    with patch("singularity.skills.replication.ReplicationSkill._launch_container",
               new_callable=AsyncMock, return_value="c"):
        await skill.execute("spawn", {"name": "r1", "budget": 5})
        await skill.execute("spawn", {"name": "r2", "budget": 5})
        result = await skill.execute("spawn", {"name": "r3", "budget": 5})
    assert not result.success
    assert "Max replicas" in result.message


def test_agent_snapshot_serialization():
    snap = AgentSnapshot(
        name="Test", ticker="TST", system_prompt="hello",
        llm_provider="openai", llm_model="gpt-4", balance=50.0,
        skills_config={"shell": {"version": "1.0"}},
    )
    j = snap.to_json()
    restored = AgentSnapshot.from_dict(json.loads(j))
    assert restored.name == "Test"
    assert restored.llm_model == "gpt-4"
    assert restored.balance == 50.0


def test_unknown_action():
    skill = ReplicationSkill()
    result = asyncio.get_event_loop().run_until_complete(
        skill.execute("nonexistent", {})
    )
    assert not result.success
