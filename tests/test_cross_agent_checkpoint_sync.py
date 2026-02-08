"""Tests for CrossAgentCheckpointSyncSkill."""

import pytest

from singularity.skills.cross_agent_checkpoint_sync import (
    BRIDGE_FILE,
    CrossAgentCheckpointSyncSkill,
)


@pytest.fixture(autouse=True)
def clean_bridge_file():
    if BRIDGE_FILE.exists():
        BRIDGE_FILE.unlink()
    yield
    if BRIDGE_FILE.exists():
        BRIDGE_FILE.unlink()


@pytest.fixture
def skill():
    return CrossAgentCheckpointSyncSkill()


def peer_params(agent_id="agent-replica-1", name="Replica 1", endpoint="http://replica1:8000"):
    return {"agent_id": agent_id, "agent_name": name, "endpoint": endpoint}


# -------------------------------------------------------------------------
# Manifest and basics
# -------------------------------------------------------------------------


class TestManifest:
    def test_manifest_name(self, skill):
        assert skill.manifest.skill_id == "cross_agent_checkpoint_sync"
        assert skill.manifest.name == "Cross-Agent Checkpoint Sync"

    def test_manifest_version(self, skill):
        assert skill.manifest.version == "1.0.0"

    def test_manifest_has_actions(self, skill):
        actions = skill.manifest.actions
        assert len(actions) == 10

    def test_manifest_action_names(self, skill):
        names = {a.name for a in skill.manifest.actions}
        expected = {
            "push",
            "pull",
            "fleet_sync",
            "compare",
            "fleet_progress",
            "register_peer",
            "remove_peer",
            "resolve_conflict",
            "configure",
            "status",
        }
        assert names == expected

    def test_estimate_cost_zero(self, skill):
        assert skill.estimate_cost("push", {}) == 0.0


@pytest.mark.asyncio
async def test_unknown_action(skill):
    result = await skill.execute("nonexistent", {})
    assert not result.success
    assert "Unknown action" in result.message


# -------------------------------------------------------------------------
# Peer registration
# -------------------------------------------------------------------------


class TestRegisterPeer:
    @pytest.mark.asyncio
    async def test_register_peer(self, skill):
        result = await skill.execute("register_peer", peer_params())
        assert result.success
        assert "registered" in result.message
        assert result.data["peer"]["name"] == "Replica 1"
        assert result.data["total_peers"] == 1

    @pytest.mark.asyncio
    async def test_register_duplicate(self, skill):
        await skill.execute("register_peer", peer_params())
        result = await skill.execute("register_peer", peer_params())
        assert not result.success
        assert "already registered" in result.message

    @pytest.mark.asyncio
    async def test_register_missing_params(self, skill):
        result = await skill.execute("register_peer", {"agent_id": "x"})
        assert not result.success
        assert "Required" in result.message

    @pytest.mark.asyncio
    async def test_register_max_peers_limit(self, skill):
        await skill.execute("configure", {"max_peers": 2})
        await skill.execute("register_peer", peer_params("a1", "Agent 1"))
        await skill.execute("register_peer", peer_params("a2", "Agent 2"))
        result = await skill.execute("register_peer", peer_params("a3", "Agent 3"))
        assert not result.success
        assert "Max peers" in result.message

    @pytest.mark.asyncio
    async def test_register_with_optional_endpoint(self, skill):
        result = await skill.execute(
            "register_peer",
            {
                "agent_id": "a1",
                "agent_name": "Agent 1",
            },
        )
        assert result.success
        assert result.data["peer"]["endpoint"] == ""


class TestRemovePeer:
    @pytest.mark.asyncio
    async def test_remove_peer(self, skill):
        await skill.execute("register_peer", peer_params())
        result = await skill.execute("remove_peer", {"agent_id": "agent-replica-1"})
        assert result.success
        assert "removed" in result.message
        assert result.data["remaining_peers"] == 0

    @pytest.mark.asyncio
    async def test_remove_nonexistent_peer(self, skill):
        result = await skill.execute("remove_peer", {"agent_id": "nonexistent"})
        assert not result.success
        assert "not found" in result.message

    @pytest.mark.asyncio
    async def test_remove_missing_agent_id(self, skill):
        result = await skill.execute("remove_peer", {})
        assert not result.success
        assert "Required" in result.message


# -------------------------------------------------------------------------
# Push
# -------------------------------------------------------------------------


class TestPush:
    @pytest.mark.asyncio
    async def test_push_to_peer(self, skill):
        await skill.execute("register_peer", peer_params())
        result = await skill.execute("push", {"target_agent_id": "agent-replica-1"})
        assert result.success
        assert "pushed" in result.message.lower()
        assert result.data["checkpoint_id"] == "latest"

    @pytest.mark.asyncio
    async def test_push_specific_checkpoint(self, skill):
        await skill.execute("register_peer", peer_params())
        result = await skill.execute(
            "push",
            {
                "target_agent_id": "agent-replica-1",
                "checkpoint_id": "cp-20240115-abc",
            },
        )
        assert result.success
        assert result.data["checkpoint_id"] == "cp-20240115-abc"

    @pytest.mark.asyncio
    async def test_push_dry_run(self, skill):
        await skill.execute("register_peer", peer_params())
        result = await skill.execute(
            "push",
            {
                "target_agent_id": "agent-replica-1",
                "dry_run": True,
            },
        )
        assert result.success
        assert "DRY RUN" in result.message
        assert result.data["dry_run"] is True

    @pytest.mark.asyncio
    async def test_push_unregistered_peer(self, skill):
        result = await skill.execute("push", {"target_agent_id": "unknown"})
        assert not result.success
        assert "not registered" in result.message

    @pytest.mark.asyncio
    async def test_push_missing_target(self, skill):
        result = await skill.execute("push", {})
        assert not result.success
        assert "Required" in result.message

    @pytest.mark.asyncio
    async def test_push_updates_stats(self, skill):
        await skill.execute("register_peer", peer_params())
        await skill.execute("push", {"target_agent_id": "agent-replica-1"})
        status = await skill.execute("status", {})
        assert status.data["stats"]["pushes_initiated"] == 1
        assert status.data["stats"]["pushes_successful"] == 1

    @pytest.mark.asyncio
    async def test_push_updates_peer_status(self, skill):
        await skill.execute("register_peer", peer_params())
        await skill.execute("push", {"target_agent_id": "agent-replica-1"})
        status = await skill.execute("status", {})
        peer_info = status.data["peers"]["agent-replica-1"]
        assert peer_info["status"] == "synced"
        assert peer_info["last_sync_at"] is not None


# -------------------------------------------------------------------------
# Pull
# -------------------------------------------------------------------------


class TestPull:
    @pytest.mark.asyncio
    async def test_pull_from_peer(self, skill):
        await skill.execute("register_peer", peer_params())
        result = await skill.execute("pull", {"source_agent_id": "agent-replica-1"})
        assert result.success
        assert "pulled" in result.message.lower()

    @pytest.mark.asyncio
    async def test_pull_specific_checkpoint(self, skill):
        await skill.execute("register_peer", peer_params())
        result = await skill.execute(
            "pull",
            {
                "source_agent_id": "agent-replica-1",
                "checkpoint_id": "cp-remote-abc",
            },
        )
        assert result.success
        assert result.data["checkpoint_id"] == "cp-remote-abc"

    @pytest.mark.asyncio
    async def test_pull_dry_run(self, skill):
        await skill.execute("register_peer", peer_params())
        result = await skill.execute(
            "pull",
            {
                "source_agent_id": "agent-replica-1",
                "dry_run": True,
            },
        )
        assert result.success
        assert "DRY RUN" in result.message

    @pytest.mark.asyncio
    async def test_pull_unregistered_peer(self, skill):
        result = await skill.execute("pull", {"source_agent_id": "unknown"})
        assert not result.success

    @pytest.mark.asyncio
    async def test_pull_missing_source(self, skill):
        result = await skill.execute("pull", {})
        assert not result.success

    @pytest.mark.asyncio
    async def test_pull_updates_stats(self, skill):
        await skill.execute("register_peer", peer_params())
        await skill.execute("pull", {"source_agent_id": "agent-replica-1"})
        status = await skill.execute("status", {})
        assert status.data["stats"]["pulls_initiated"] == 1
        assert status.data["stats"]["pulls_successful"] == 1


# -------------------------------------------------------------------------
# Fleet sync
# -------------------------------------------------------------------------


class TestFleetSync:
    @pytest.mark.asyncio
    async def test_fleet_sync_no_peers(self, skill):
        result = await skill.execute("fleet_sync", {})
        assert result.success
        assert "No peers" in result.message
        assert result.data["peers_count"] == 0

    @pytest.mark.asyncio
    async def test_fleet_sync_with_peers(self, skill):
        await skill.execute("register_peer", peer_params("a1", "Agent 1"))
        await skill.execute("register_peer", peer_params("a2", "Agent 2"))
        result = await skill.execute("fleet_sync", {})
        assert result.success
        assert len(result.data["synced"]) == 2

    @pytest.mark.asyncio
    async def test_fleet_sync_dry_run(self, skill):
        await skill.execute("register_peer", peer_params("a1", "Agent 1"))
        result = await skill.execute("fleet_sync", {"dry_run": True})
        assert result.success
        assert "DRY RUN" in result.message
        assert result.data["synced"][0]["dry_run"] is True

    @pytest.mark.asyncio
    async def test_fleet_sync_skips_inactive(self, skill):
        await skill.execute("register_peer", peer_params("a1", "Agent 1"))
        # Manually set peer to inactive
        data = skill._load()
        data["peers"]["a1"]["status"] = "inactive"
        skill._save(data)
        result = await skill.execute("fleet_sync", {})
        assert result.success
        assert len(result.data["skipped"]) == 1
        assert result.data["skipped"][0]["reason"] == "inactive"

    @pytest.mark.asyncio
    async def test_fleet_sync_push_direction(self, skill):
        await skill.execute("configure", {"sync_direction": "push"})
        await skill.execute("register_peer", peer_params("a1", "Agent 1"))
        result = await skill.execute("fleet_sync", {})
        assert result.success
        status = await skill.execute("status", {})
        assert status.data["stats"]["pushes_initiated"] == 1
        assert status.data["stats"]["pulls_initiated"] == 0

    @pytest.mark.asyncio
    async def test_fleet_sync_pull_direction(self, skill):
        await skill.execute("configure", {"sync_direction": "pull"})
        await skill.execute("register_peer", peer_params("a1", "Agent 1"))
        result = await skill.execute("fleet_sync", {})
        assert result.success
        status = await skill.execute("status", {})
        assert status.data["stats"]["pulls_initiated"] == 1
        assert status.data["stats"]["pushes_initiated"] == 0

    @pytest.mark.asyncio
    async def test_fleet_sync_bidirectional(self, skill):
        await skill.execute("configure", {"sync_direction": "bidirectional"})
        await skill.execute("register_peer", peer_params("a1", "Agent 1"))
        result = await skill.execute("fleet_sync", {})
        assert result.success
        status = await skill.execute("status", {})
        assert status.data["stats"]["pushes_initiated"] == 1
        assert status.data["stats"]["pulls_initiated"] == 1

    @pytest.mark.asyncio
    async def test_fleet_sync_updates_fleet_syncs_stat(self, skill):
        await skill.execute("register_peer", peer_params("a1", "Agent 1"))
        await skill.execute("fleet_sync", {})
        await skill.execute("fleet_sync", {})
        status = await skill.execute("status", {})
        assert status.data["stats"]["fleet_syncs"] == 2

    @pytest.mark.asyncio
    async def test_fleet_sync_respects_max_peers(self, skill):
        await skill.execute("configure", {"max_peers": 20})
        for i in range(5):
            await skill.execute("register_peer", peer_params(f"a{i}", f"Agent {i}"))
        await skill.execute("configure", {"max_peers": 3})
        # fleet_sync only syncs up to max_peers in config
        result = await skill.execute("fleet_sync", {})
        assert result.success
        # We can't guarantee order, but synced count should be <= 3
        assert len(result.data["synced"]) <= 3


# -------------------------------------------------------------------------
# Compare
# -------------------------------------------------------------------------


class TestCompare:
    @pytest.mark.asyncio
    async def test_compare_basic(self, skill):
        await skill.execute("register_peer", peer_params())
        result = await skill.execute("compare", {"peer_agent_id": "agent-replica-1"})
        assert result.success
        assert "in sync" in result.message
        assert result.data["divergence_detected"] is False

    @pytest.mark.asyncio
    async def test_compare_with_specific_checkpoints(self, skill):
        await skill.execute("register_peer", peer_params())
        result = await skill.execute(
            "compare",
            {
                "peer_agent_id": "agent-replica-1",
                "local_checkpoint_id": "cp-local-1",
                "remote_checkpoint_id": "cp-remote-1",
            },
        )
        assert result.success
        assert result.data["local_checkpoint"] == "cp-local-1"
        assert result.data["remote_checkpoint"] == "cp-remote-1"

    @pytest.mark.asyncio
    async def test_compare_unregistered_peer(self, skill):
        result = await skill.execute("compare", {"peer_agent_id": "unknown"})
        assert not result.success
        assert "not registered" in result.message

    @pytest.mark.asyncio
    async def test_compare_missing_peer_id(self, skill):
        result = await skill.execute("compare", {})
        assert not result.success
        assert "Required" in result.message

    @pytest.mark.asyncio
    async def test_compare_updates_stats(self, skill):
        await skill.execute("register_peer", peer_params())
        await skill.execute("compare", {"peer_agent_id": "agent-replica-1"})
        status = await skill.execute("status", {})
        assert status.data["stats"]["comparisons_run"] == 1

    @pytest.mark.asyncio
    async def test_compare_has_pillar_comparison(self, skill):
        await skill.execute("register_peer", peer_params())
        result = await skill.execute("compare", {"peer_agent_id": "agent-replica-1"})
        pillars = result.data["pillar_comparison"]
        assert "self_improvement" in pillars
        assert "revenue" in pillars
        assert "replication" in pillars
        assert "goal_setting" in pillars


# -------------------------------------------------------------------------
# Fleet progress
# -------------------------------------------------------------------------


class TestFleetProgress:
    @pytest.mark.asyncio
    async def test_fleet_progress_no_peers(self, skill):
        result = await skill.execute("fleet_progress", {})
        assert result.success
        assert result.data["peers_count"] == 0
        assert result.data["fleet_health"] == "unknown"

    @pytest.mark.asyncio
    async def test_fleet_progress_with_peers(self, skill):
        await skill.execute("register_peer", peer_params("a1", "Agent 1"))
        await skill.execute("register_peer", peer_params("a2", "Agent 2"))
        result = await skill.execute("fleet_progress", {})
        assert result.success
        assert result.data["total_peers"] == 2
        assert result.data["fleet_grade"] in ("A", "B", "C", "D", "F")

    @pytest.mark.asyncio
    async def test_fleet_progress_grade_with_synced_peers(self, skill):
        await skill.execute("register_peer", peer_params("a1", "Agent 1"))
        # Sync to mark as synced
        await skill.execute("push", {"target_agent_id": "a1"})
        result = await skill.execute("fleet_progress", {})
        assert result.data["synced_peers"] == 1
        assert result.data["fleet_grade"] == "A"  # 1/1 = 100%

    @pytest.mark.asyncio
    async def test_fleet_progress_includes_peer_statuses(self, skill):
        await skill.execute("register_peer", peer_params("a1", "Agent 1"))
        result = await skill.execute("fleet_progress", {})
        assert len(result.data["peer_statuses"]) == 1
        assert result.data["peer_statuses"][0]["agent_id"] == "a1"

    @pytest.mark.asyncio
    async def test_fleet_progress_shows_open_conflicts(self, skill):
        await skill.execute("register_peer", peer_params("a1", "Agent 1"))
        # Manually inject a conflict
        data = skill._load()
        data["conflicts"].append(
            {
                "timestamp": "2024-01-01T00:00:00",
                "peer_agent_id": "a1",
                "peer_name": "Agent 1",
                "progress_delta": 50,
                "resolved": False,
            }
        )
        skill._save(data)
        result = await skill.execute("fleet_progress", {})
        assert result.data["open_conflicts"] == 1


# -------------------------------------------------------------------------
# Resolve conflict
# -------------------------------------------------------------------------


class TestResolveConflict:
    async def _setup_conflict(self, skill):
        """Helper: register peer and inject a conflict."""
        await skill.execute("register_peer", peer_params())
        data = skill._load()
        data["conflicts"].append(
            {
                "timestamp": "2024-01-01T00:00:00",
                "peer_agent_id": "agent-replica-1",
                "peer_name": "Replica 1",
                "progress_delta": 50,
                "resolved": False,
                "resolution_strategy": None,
            }
        )
        skill._save(data)

    @pytest.mark.asyncio
    async def test_resolve_accept_local(self, skill):
        await self._setup_conflict(skill)
        result = await skill.execute(
            "resolve_conflict",
            {
                "conflict_index": 0,
                "strategy": "accept_local",
            },
        )
        assert result.success
        assert "accept_local" in result.message
        assert result.data["conflict"]["resolved"] is True

    @pytest.mark.asyncio
    async def test_resolve_accept_remote(self, skill):
        await self._setup_conflict(skill)
        result = await skill.execute(
            "resolve_conflict",
            {
                "conflict_index": 0,
                "strategy": "accept_remote",
            },
        )
        assert result.success
        assert result.data["conflict"]["resolution_strategy"] == "accept_remote"

    @pytest.mark.asyncio
    async def test_resolve_merge(self, skill):
        await self._setup_conflict(skill)
        result = await skill.execute(
            "resolve_conflict",
            {
                "conflict_index": 0,
                "strategy": "merge",
                "notes": "Manually merged key files",
            },
        )
        assert result.success
        assert result.data["conflict"]["resolution_notes"] == "Manually merged key files"

    @pytest.mark.asyncio
    async def test_resolve_already_resolved(self, skill):
        await self._setup_conflict(skill)
        await skill.execute("resolve_conflict", {"conflict_index": 0, "strategy": "merge"})
        result = await skill.execute("resolve_conflict", {"conflict_index": 0, "strategy": "merge"})
        assert not result.success
        assert "already resolved" in result.message

    @pytest.mark.asyncio
    async def test_resolve_invalid_strategy(self, skill):
        await self._setup_conflict(skill)
        result = await skill.execute(
            "resolve_conflict",
            {
                "conflict_index": 0,
                "strategy": "invalid",
            },
        )
        assert not result.success
        assert "Invalid strategy" in result.message

    @pytest.mark.asyncio
    async def test_resolve_out_of_range(self, skill):
        result = await skill.execute(
            "resolve_conflict",
            {
                "conflict_index": 99,
                "strategy": "merge",
            },
        )
        assert not result.success
        assert "out of range" in result.message

    @pytest.mark.asyncio
    async def test_resolve_missing_params(self, skill):
        result = await skill.execute("resolve_conflict", {})
        assert not result.success
        assert "Required" in result.message

    @pytest.mark.asyncio
    async def test_resolve_negative_index(self, skill):
        result = await skill.execute(
            "resolve_conflict",
            {
                "conflict_index": -1,
                "strategy": "merge",
            },
        )
        assert not result.success

    @pytest.mark.asyncio
    async def test_resolve_updates_stats(self, skill):
        await self._setup_conflict(skill)
        await skill.execute("resolve_conflict", {"conflict_index": 0, "strategy": "merge"})
        status = await skill.execute("status", {})
        assert status.data["stats"]["conflicts_resolved"] == 1

    @pytest.mark.asyncio
    async def test_resolve_decrements_open_conflicts(self, skill):
        await self._setup_conflict(skill)
        progress_before = await skill.execute("fleet_progress", {})
        assert progress_before.data["open_conflicts"] == 1
        await skill.execute("resolve_conflict", {"conflict_index": 0, "strategy": "accept_local"})
        progress_after = await skill.execute("fleet_progress", {})
        assert progress_after.data["open_conflicts"] == 0


# -------------------------------------------------------------------------
# Configure
# -------------------------------------------------------------------------


class TestConfigure:
    @pytest.mark.asyncio
    async def test_configure_single(self, skill):
        result = await skill.execute("configure", {"auto_push_on_save": True})
        assert result.success
        assert result.data["config"]["auto_push_on_save"] is True

    @pytest.mark.asyncio
    async def test_configure_multiple(self, skill):
        result = await skill.execute(
            "configure",
            {
                "auto_push_on_save": True,
                "conflict_threshold": 50,
                "sync_direction": "push",
            },
        )
        assert result.success
        assert result.data["config"]["auto_push_on_save"] is True
        assert result.data["config"]["conflict_threshold"] == 50
        assert result.data["config"]["sync_direction"] == "push"

    @pytest.mark.asyncio
    async def test_configure_invalid_direction(self, skill):
        result = await skill.execute("configure", {"sync_direction": "invalid"})
        assert not result.success
        assert "Invalid sync_direction" in result.message

    @pytest.mark.asyncio
    async def test_configure_negative_threshold(self, skill):
        result = await skill.execute("configure", {"conflict_threshold": -5})
        assert not result.success
        assert "non-negative" in result.message

    @pytest.mark.asyncio
    async def test_configure_no_changes(self, skill):
        result = await skill.execute("configure", {})
        assert result.success
        assert "No changes" in result.message
        assert "current_config" in result.data

    @pytest.mark.asyncio
    async def test_configure_returns_updated_diff(self, skill):
        result = await skill.execute("configure", {"max_peers": 20})
        assert "updated" in result.data
        assert result.data["updated"]["max_peers"]["old"] == 10
        assert result.data["updated"]["max_peers"]["new"] == 20


# -------------------------------------------------------------------------
# Status
# -------------------------------------------------------------------------


class TestStatus:
    @pytest.mark.asyncio
    async def test_status_empty(self, skill):
        result = await skill.execute("status", {})
        assert result.success
        assert result.data["peers_count"] == 0
        assert "config" in result.data
        assert "stats" in result.data

    @pytest.mark.asyncio
    async def test_status_with_peers(self, skill):
        await skill.execute("register_peer", peer_params("a1", "Agent 1"))
        await skill.execute("register_peer", peer_params("a2", "Agent 2"))
        result = await skill.execute("status", {})
        assert result.data["peers_count"] == 2
        assert "a1" in result.data["peers"]
        assert "a2" in result.data["peers"]

    @pytest.mark.asyncio
    async def test_status_shows_recent_syncs(self, skill):
        await skill.execute("register_peer", peer_params("a1", "Agent 1"))
        await skill.execute("push", {"target_agent_id": "a1"})
        result = await skill.execute("status", {})
        assert len(result.data["recent_syncs"]) == 1

    @pytest.mark.asyncio
    async def test_status_shows_recent_events(self, skill):
        await skill.execute("register_peer", peer_params("a1", "Agent 1"))
        result = await skill.execute("status", {})
        assert len(result.data["recent_events"]) >= 1


# -------------------------------------------------------------------------
# Persistence and data integrity
# -------------------------------------------------------------------------


class TestPersistence:
    @pytest.mark.asyncio
    async def test_data_persists_across_instances(self, skill):
        await skill.execute("register_peer", peer_params())
        # Create new instance
        skill2 = CrossAgentCheckpointSyncSkill()
        result = await skill2.execute("status", {})
        assert result.data["peers_count"] == 1

    @pytest.mark.asyncio
    async def test_event_log_truncation(self, skill):
        data = skill._load()
        data["event_log"] = [{"event": f"test-{i}"} for i in range(600)]
        skill._save(data)
        loaded = skill._load()
        assert len(loaded["event_log"]) == 500

    @pytest.mark.asyncio
    async def test_sync_history_truncation(self, skill):
        data = skill._load()
        data["sync_history"] = [{"sync": f"test-{i}"} for i in range(300)]
        skill._save(data)
        loaded = skill._load()
        assert len(loaded["sync_history"]) == 200

    @pytest.mark.asyncio
    async def test_corrupted_file_returns_default(self, skill):
        BRIDGE_FILE.write_text("not json")
        data = skill._load()
        assert "config" in data
        assert "peers" in data
        assert "stats" in data

    @pytest.mark.asyncio
    async def test_default_state_structure(self, skill):
        data = skill._load()
        assert "config" in data
        assert "peers" in data
        assert "sync_history" in data
        assert "conflicts" in data
        assert "event_log" in data
        assert "stats" in data

    @pytest.mark.asyncio
    async def test_default_config_values(self, skill):
        data = skill._load()
        config = data["config"]
        assert config["auto_push_on_save"] is False
        assert config["sync_direction"] == "bidirectional"
        assert config["conflict_threshold"] == 30
        assert config["max_peers"] == 10
        assert config["include_file_contents"] is False


# -------------------------------------------------------------------------
# Integration: multi-step workflows
# -------------------------------------------------------------------------


class TestWorkflows:
    @pytest.mark.asyncio
    async def test_register_push_pull_workflow(self, skill):
        """Full lifecycle: register peer, push, pull, check status."""
        await skill.execute("register_peer", peer_params("a1", "Agent 1"))
        push = await skill.execute("push", {"target_agent_id": "a1"})
        assert push.success
        pull = await skill.execute("pull", {"source_agent_id": "a1"})
        assert pull.success
        status = await skill.execute("status", {})
        assert status.data["stats"]["pushes_successful"] == 1
        assert status.data["stats"]["pulls_successful"] == 1

    @pytest.mark.asyncio
    async def test_fleet_sync_then_progress(self, skill):
        """Register multiple peers, fleet sync, then check progress."""
        for i in range(3):
            await skill.execute("register_peer", peer_params(f"a{i}", f"Agent {i}"))
        sync = await skill.execute("fleet_sync", {})
        assert sync.success
        assert len(sync.data["synced"]) == 3
        progress = await skill.execute("fleet_progress", {})
        assert progress.data["synced_peers"] == 3
        assert progress.data["fleet_grade"] == "A"

    @pytest.mark.asyncio
    async def test_register_compare_resolve_workflow(self, skill):
        """Register peer, compare, detect divergence, resolve."""
        await skill.execute("register_peer", peer_params("a1", "Agent 1"))
        compare = await skill.execute("compare", {"peer_agent_id": "a1"})
        assert compare.success
        # With default data, no divergence. Inject one manually.
        data = skill._load()
        data["conflicts"].append(
            {
                "timestamp": "2024-01-01T00:00:00",
                "peer_agent_id": "a1",
                "peer_name": "Agent 1",
                "progress_delta": 50,
                "resolved": False,
                "resolution_strategy": None,
            }
        )
        skill._save(data)
        resolve = await skill.execute(
            "resolve_conflict",
            {
                "conflict_index": 0,
                "strategy": "accept_local",
            },
        )
        assert resolve.success
        progress = await skill.execute("fleet_progress", {})
        assert progress.data["open_conflicts"] == 0

    @pytest.mark.asyncio
    async def test_configure_then_fleet_sync(self, skill):
        """Configure push-only, then fleet sync should only push."""
        await skill.execute("configure", {"sync_direction": "push"})
        await skill.execute("register_peer", peer_params("a1", "Agent 1"))
        await skill.execute("fleet_sync", {})
        status = await skill.execute("status", {})
        assert status.data["stats"]["pushes_initiated"] == 1
        assert status.data["stats"]["pulls_initiated"] == 0
