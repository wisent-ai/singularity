#!/usr/bin/env python3
"""Tests for PlaybookSharingSkill."""

import json
import pytest
import asyncio
from pathlib import Path
from unittest.mock import patch

from singularity.skills.playbook_sharing import PlaybookSharingSkill, SHARING_FILE


@pytest.fixture
def skill(tmp_path):
    test_file = tmp_path / "playbook_sharing.json"
    with patch("singularity.skills.playbook_sharing.SHARING_FILE", test_file):
        with patch("singularity.skills.playbook_sharing.DATA_DIR", tmp_path):
            s = PlaybookSharingSkill()
            yield s


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def sample_playbook_params(name="deploy_strategy", agent="agent_a"):
    return {
        "playbook_name": name,
        "agent_name": agent,
        "category": "deployment",
        "description": "A deployment strategy for production",
        "playbook_data": {
            "name": name,
            "task_pattern": "deploy * to production",
            "steps": ["Build image", "Run tests", "Push to registry", "Deploy"],
            "pitfalls": ["Forgetting to run tests"],
            "tags": ["deployment", "production", "docker"],
            "effectiveness": 0.8,
            "uses": 5,
        },
    }


class TestPublish:
    def test_publish_playbook(self, skill):
        result = run(skill.execute("publish", sample_playbook_params()))
        assert result.success
        assert "published" in result.message.lower()
        assert result.data["shared_id"]

    def test_publish_requires_name_and_agent(self, skill):
        result = run(skill.execute("publish", {"playbook_name": "x"}))
        assert not result.success

    def test_publish_rejects_empty_steps(self, skill):
        params = sample_playbook_params()
        params["playbook_data"]["steps"] = []
        result = run(skill.execute("publish", params))
        assert not result.success

    def test_publish_rejects_low_effectiveness(self, skill):
        params = sample_playbook_params()
        params["playbook_data"]["effectiveness"] = 0.2
        params["playbook_data"]["uses"] = 10
        result = run(skill.execute("publish", params))
        assert not result.success

    def test_publish_detects_duplicates(self, skill):
        run(skill.execute("publish", sample_playbook_params()))
        result = run(skill.execute("publish", sample_playbook_params()))
        assert not result.success
        assert "already exists" in result.message.lower()

    def test_publish_increments_stats(self, skill):
        run(skill.execute("publish", sample_playbook_params()))
        status = run(skill.execute("status", {}))
        assert status.data["stats"]["total_published"] == 1


class TestBrowse:
    def test_browse_empty(self, skill):
        result = run(skill.execute("browse", {}))
        assert result.success
        assert len(result.data["playbooks"]) == 0

    def test_browse_finds_published(self, skill):
        run(skill.execute("publish", sample_playbook_params()))
        result = run(skill.execute("browse", {}))
        assert len(result.data["playbooks"]) == 1
        assert result.data["playbooks"][0]["name"] == "deploy_strategy"

    def test_browse_filter_by_category(self, skill):
        run(skill.execute("publish", sample_playbook_params()))
        result = run(skill.execute("browse", {"category": "development"}))
        assert len(result.data["playbooks"]) == 0
        result = run(skill.execute("browse", {"category": "deployment"}))
        assert len(result.data["playbooks"]) == 1

    def test_browse_filter_by_tag(self, skill):
        run(skill.execute("publish", sample_playbook_params()))
        result = run(skill.execute("browse", {"tags": ["docker"]}))
        assert len(result.data["playbooks"]) == 1
        result = run(skill.execute("browse", {"tags": ["python"]}))
        assert len(result.data["playbooks"]) == 0

    def test_browse_search(self, skill):
        run(skill.execute("publish", sample_playbook_params()))
        result = run(skill.execute("browse", {"search": "deploy"}))
        assert len(result.data["playbooks"]) == 1
        result = run(skill.execute("browse", {"search": "zzz_nonexistent"}))
        assert len(result.data["playbooks"]) == 0


class TestImport:
    def test_import_playbook(self, skill):
        pub = run(skill.execute("publish", sample_playbook_params()))
        sid = pub.data["shared_id"]
        result = run(skill.execute("import_playbook", {"shared_id": sid, "agent_name": "agent_b"}))
        assert result.success
        assert "imported" in result.message.lower()

    def test_import_requires_shared_id(self, skill):
        result = run(skill.execute("import_playbook", {}))
        assert not result.success

    def test_import_rejects_own_playbook(self, skill):
        pub = run(skill.execute("publish", sample_playbook_params()))
        sid = pub.data["shared_id"]
        result = run(skill.execute("import_playbook", {"shared_id": sid, "agent_name": "agent_a"}))
        assert not result.success

    def test_import_rejects_duplicate(self, skill):
        pub = run(skill.execute("publish", sample_playbook_params()))
        sid = pub.data["shared_id"]
        run(skill.execute("import_playbook", {"shared_id": sid, "agent_name": "agent_b"}))
        result = run(skill.execute("import_playbook", {"shared_id": sid, "agent_name": "agent_b"}))
        assert not result.success
        assert "already imported" in result.message.lower()

    def test_import_increments_count(self, skill):
        pub = run(skill.execute("publish", sample_playbook_params()))
        sid = pub.data["shared_id"]
        run(skill.execute("import_playbook", {"shared_id": sid, "agent_name": "agent_b"}))
        status = run(skill.execute("status", {}))
        assert status.data["stats"]["total_imported"] == 1


class TestRate:
    def test_rate_playbook(self, skill):
        pub = run(skill.execute("publish", sample_playbook_params()))
        sid = pub.data["shared_id"]
        result = run(skill.execute("rate", {"shared_id": sid, "rating": 4.5, "agent_name": "agent_b"}))
        assert result.success
        assert result.data["avg_rating"] == 4.5

    def test_rate_invalid_range(self, skill):
        pub = run(skill.execute("publish", sample_playbook_params()))
        sid = pub.data["shared_id"]
        result = run(skill.execute("rate", {"shared_id": sid, "rating": 6}))
        assert not result.success

    def test_rate_updates_average(self, skill):
        pub = run(skill.execute("publish", sample_playbook_params()))
        sid = pub.data["shared_id"]
        run(skill.execute("rate", {"shared_id": sid, "rating": 5, "agent_name": "a1"}))
        result = run(skill.execute("rate", {"shared_id": sid, "rating": 3, "agent_name": "a2"}))
        assert result.data["avg_rating"] == 4.0

    def test_rate_update_existing(self, skill):
        pub = run(skill.execute("publish", sample_playbook_params()))
        sid = pub.data["shared_id"]
        run(skill.execute("rate", {"shared_id": sid, "rating": 2, "agent_name": "a1"}))
        result = run(skill.execute("rate", {"shared_id": sid, "rating": 5, "agent_name": "a1"}))
        assert result.data["avg_rating"] == 5.0  # Updated, not averaged


class TestTop:
    def test_top_empty(self, skill):
        result = run(skill.execute("top", {}))
        assert result.success
        assert len(result.data["top"]) == 0

    def test_top_returns_rated(self, skill):
        pub = run(skill.execute("publish", sample_playbook_params()))
        sid = pub.data["shared_id"]
        run(skill.execute("rate", {"shared_id": sid, "rating": 5, "agent_name": "b"}))
        result = run(skill.execute("top", {}))
        assert len(result.data["top"]) == 1
        assert result.data["top"][0]["avg_rating"] == 5.0


class TestSync:
    def test_sync_export(self, skill):
        run(skill.execute("publish", sample_playbook_params()))
        result = run(skill.execute("sync", {"mode": "export", "agent_name": "agent_a"}))
        assert result.success
        assert len(result.data["playbooks"]) == 1

    def test_sync_import(self, skill):
        pub = run(skill.execute("publish", sample_playbook_params()))
        export = run(skill.execute("sync", {"mode": "export", "agent_name": "agent_a"}))
        # Import as different agent
        result = run(skill.execute("sync", {
            "mode": "import",
            "agent_name": "agent_c",
            "playbooks": export.data["playbooks"],
        }))
        assert result.success
        # Duplicates are skipped (same content hash)
        assert result.data["skipped"] >= 0


class TestRecommend:
    def test_recommend_by_tags(self, skill):
        run(skill.execute("publish", sample_playbook_params()))
        result = run(skill.execute("recommend", {
            "agent_name": "agent_b",
            "task_tags": ["deployment"],
        }))
        assert result.success
        assert len(result.data["recommendations"]) == 1

    def test_recommend_excludes_own(self, skill):
        run(skill.execute("publish", sample_playbook_params()))
        result = run(skill.execute("recommend", {
            "agent_name": "agent_a",
            "task_tags": ["deployment"],
        }))
        assert len(result.data["recommendations"]) == 0

    def test_recommend_by_gap_areas(self, skill):
        run(skill.execute("publish", sample_playbook_params()))
        result = run(skill.execute("recommend", {
            "agent_name": "agent_b",
            "gap_areas": ["deployment"],
        }))
        assert len(result.data["recommendations"]) >= 1

    def test_recommend_requires_input(self, skill):
        result = run(skill.execute("recommend", {"agent_name": "x"}))
        assert not result.success


class TestStatus:
    def test_status(self, skill):
        result = run(skill.execute("status", {}))
        assert result.success
        assert "stats" in result.data
        assert "config" in result.data


class TestUnknownAction:
    def test_unknown_action(self, skill):
        result = run(skill.execute("nonexistent", {}))
        assert not result.success


class TestPersistence:
    def test_data_persists(self, tmp_path):
        test_file = tmp_path / "playbook_sharing.json"
        with patch("singularity.skills.playbook_sharing.SHARING_FILE", test_file):
            with patch("singularity.skills.playbook_sharing.DATA_DIR", tmp_path):
                s1 = PlaybookSharingSkill()
                run(s1.execute("publish", sample_playbook_params()))

                s2 = PlaybookSharingSkill()
                result = run(s2.execute("browse", {}))
                assert len(result.data["playbooks"]) == 1
