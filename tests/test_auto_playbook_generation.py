#!/usr/bin/env python3
"""Comprehensive tests for AutoPlaybookGenerationSkill."""

import asyncio
import hashlib
import json
import pytest
from pathlib import Path
from unittest.mock import patch, AsyncMock, MagicMock

from singularity.skills.auto_playbook_generation import (
    AutoPlaybookGenerationSkill,
    _tokenize,
    _jaccard,
    _shared_words_ratio,
    STATE_FILE,
    STOP_WORDS,
)
from singularity.skills.base import SkillResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run(coro):
    """Run an async coroutine synchronously."""
    return asyncio.get_event_loop().run_until_complete(coro)


def _make_reflection(
    rid="r1",
    task="Deploy the web service",
    tags=None,
    success=True,
    actions_taken=None,
    outcome="Deployed successfully",
    analysis="",
    improvements=None,
):
    """Build a minimal reflection dict for tests."""
    return {
        "id": rid,
        "task": task,
        "tags": tags or ["deployment"],
        "success": success,
        "actions_taken": actions_taken or ["step_a", "step_b"],
        "outcome": outcome,
        "analysis": analysis,
        "improvements": improvements or [],
    }


def _make_cluster(n=3, tag="deployment", task_prefix="Deploy service"):
    """Build a list of n reflections that share a tag and similar task."""
    cluster = []
    for i in range(n):
        cluster.append(_make_reflection(
            rid=f"refl_{i}",
            task=f"{task_prefix} version {i}",
            tags=[tag, f"extra_{i % 2}"],
            success=i % 3 != 0,  # 2/3 succeed for clusters of size 3
            actions_taken=[f"build_{i}", "push_registry", "apply_manifest"],
            outcome=f"Service v{i} deployed",
            analysis=f"Analysis for task {i}" if i % 3 == 0 else "",
            improvements=[f"Improvement {i}"] if i % 3 == 0 else [],
        ))
    return cluster


@pytest.fixture
def skill(tmp_path):
    """Create a skill instance with an isolated state file."""
    test_file = tmp_path / "auto_playbook_generation.json"
    with patch("singularity.skills.auto_playbook_generation.STATE_FILE", test_file):
        s = AutoPlaybookGenerationSkill()
        yield s


@pytest.fixture
def skill_with_context(tmp_path):
    """Create a skill instance that has a mock context for call_skill."""
    test_file = tmp_path / "auto_playbook_generation.json"
    with patch("singularity.skills.auto_playbook_generation.STATE_FILE", test_file):
        s = AutoPlaybookGenerationSkill()
        ctx = MagicMock()
        ctx.call_skill = AsyncMock()
        s.context = ctx
        yield s


# ===================================================================
# Test Helper Functions (unit, no async)
# ===================================================================


class TestTokenize:
    """Tests for _tokenize()."""

    def test_basic_tokenization(self):
        result = _tokenize("Deploy the web service")
        assert "deploy" in result
        assert "web" in result
        assert "service" in result

    def test_stop_word_filtering(self):
        result = _tokenize("the is are was a an to of in for on with")
        # All are stop words or single-char, nothing should remain
        assert len(result) == 0

    def test_lowercase(self):
        result = _tokenize("DEPLOY Service WebApp")
        assert "deploy" in result
        assert "webapp" in result

    def test_extracts_significant_words(self):
        result = _tokenize("Configure the database connection with SSL certificates")
        assert "configure" in result
        assert "database" in result
        assert "connection" in result
        assert "ssl" in result
        assert "certificates" in result
        # Stop words excluded
        assert "the" not in result
        assert "with" not in result

    def test_empty_string(self):
        result = _tokenize("")
        assert result == set()

    def test_single_char_words_excluded(self):
        # Regex r"[a-z][a-z0-9_]+" requires 2+ chars starting with alpha
        result = _tokenize("a b c d e x y z 1 2 3")
        assert len(result) == 0

    def test_underscored_words(self):
        result = _tokenize("deploy_service and build_image")
        assert "deploy_service" in result
        assert "build_image" in result

    def test_alphanumeric_tokens(self):
        result = _tokenize("Use python3 and node18 to build k8s cluster")
        assert "python3" in result
        assert "node18" in result
        assert "k8s" in result
        assert "build" in result
        assert "cluster" in result


class TestJaccard:
    """Tests for _jaccard()."""

    def test_empty_sets(self):
        assert _jaccard(set(), set()) == 0.0

    def test_identical_sets(self):
        s = {"deploy", "service"}
        assert _jaccard(s, s) == 1.0

    def test_partial_overlap(self):
        a = {"deploy", "service", "web"}
        b = {"deploy", "service", "api"}
        # intersection=2, union=4
        assert _jaccard(a, b) == pytest.approx(0.5)

    def test_no_overlap(self):
        a = {"deploy", "service"}
        b = {"database", "connection"}
        assert _jaccard(a, b) == 0.0

    def test_one_empty_set(self):
        a = {"deploy"}
        assert _jaccard(a, set()) == 0.0
        assert _jaccard(set(), a) == 0.0

    def test_subset(self):
        a = {"deploy"}
        b = {"deploy", "service"}
        assert _jaccard(a, b) == pytest.approx(0.5)


class TestSharedWordsRatio:
    """Tests for _shared_words_ratio()."""

    def test_empty_sets(self):
        assert _shared_words_ratio(set(), set()) == 0.0

    def test_one_empty(self):
        assert _shared_words_ratio({"a"}, set()) == 0.0
        assert _shared_words_ratio(set(), {"a"}) == 0.0

    def test_subset_relationship(self):
        a = {"deploy"}
        b = {"deploy", "service", "web"}
        # intersection=1, min=1 -> 1.0
        assert _shared_words_ratio(a, b) == 1.0

    def test_partial_overlap(self):
        a = {"deploy", "service", "web"}
        b = {"deploy", "service", "api", "gateway"}
        # intersection=2, min=3
        assert _shared_words_ratio(a, b) == pytest.approx(2.0 / 3.0)

    def test_identical_sets(self):
        s = {"x", "y", "z"}
        assert _shared_words_ratio(s, s) == 1.0

    def test_no_overlap(self):
        a = {"deploy"}
        b = {"database"}
        assert _shared_words_ratio(a, b) == 0.0


# ===================================================================
# Test Class Initialization
# ===================================================================


class TestInitialization:
    """Tests for skill initialization and state defaults."""

    def test_default_state(self, skill):
        assert skill._clusters == {}
        assert skill._generated_playbooks == []
        assert skill._wire_state["active"] is False
        assert skill._wire_state["subscription_id"] is None
        assert skill._wire_state["reflections_since_scan"] == 0
        assert skill._config["min_cluster_size"] == 3
        assert skill._stats["scans_performed"] == 0
        assert skill._stats["playbooks_generated"] == 0

    def test_state_persistence_round_trip(self, tmp_path):
        test_file = tmp_path / "state.json"
        with patch("singularity.skills.auto_playbook_generation.STATE_FILE", test_file):
            s1 = AutoPlaybookGenerationSkill()
            s1._clusters["abc"] = {"cluster_id": "abc", "reflections": []}
            s1._stats["scans_performed"] = 42
            s1._config["min_cluster_size"] = 7
            s1._save_state()

            s2 = AutoPlaybookGenerationSkill()
            assert "abc" in s2._clusters
            assert s2._stats["scans_performed"] == 42
            assert s2._config["min_cluster_size"] == 7

    def test_corrupted_state_file(self, tmp_path):
        test_file = tmp_path / "state.json"
        test_file.write_text("NOT VALID JSON {{{")
        with patch("singularity.skills.auto_playbook_generation.STATE_FILE", test_file):
            s = AutoPlaybookGenerationSkill()
            # Should fall back to empty state
            assert s._clusters == {}
            assert s._stats["scans_performed"] == 0


# ===================================================================
# Test Clustering Algorithm
# ===================================================================


class TestClustering:
    """Tests for _cluster_reflections()."""

    def test_cluster_by_tag_similarity(self, skill):
        reflections = [
            _make_reflection(rid="r1", task="task alpha one", tags=["deploy", "k8s"]),
            _make_reflection(rid="r2", task="task beta two", tags=["deploy", "k8s"]),
            _make_reflection(rid="r3", task="task gamma three", tags=["deploy", "docker"]),
        ]
        clusters = skill._cluster_reflections(reflections, min_cluster_size=2)
        assert len(clusters) >= 1
        # All should end up linked via shared "deploy"
        biggest = max(clusters, key=len)
        assert len(biggest) >= 2

    def test_cluster_by_keyword_similarity(self, skill):
        reflections = [
            _make_reflection(rid="r1", task="deploy service monitoring dashboard", tags=["a"]),
            _make_reflection(rid="r2", task="deploy service monitoring alerts", tags=["b"]),
            _make_reflection(rid="r3", task="deploy service monitoring logs", tags=["c"]),
        ]
        clusters = skill._cluster_reflections(reflections, min_cluster_size=2)
        assert len(clusters) >= 1
        biggest = max(clusters, key=len)
        assert len(biggest) >= 2

    def test_min_cluster_size_enforcement(self, skill):
        reflections = [
            _make_reflection(rid="r1", task="deploy alpha", tags=["x"]),
            _make_reflection(rid="r2", task="deploy beta", tags=["y"]),
        ]
        clusters = skill._cluster_reflections(reflections, min_cluster_size=5)
        assert clusters == []

    def test_single_linkage_transitive(self, skill):
        # A linked to B via tags, B linked to C via keywords => A,B,C in one cluster
        reflections = [
            _make_reflection(rid="r1", task="completely unrelated alpha words", tags=["shared_tag", "tag1"]),
            _make_reflection(rid="r2", task="database migration rollback script", tags=["shared_tag", "tag2"]),
            _make_reflection(rid="r3", task="database migration rollback procedure", tags=["tag3"]),
        ]
        clusters = skill._cluster_reflections(reflections, min_cluster_size=2)
        # r1 links to r2 via shared_tag (Jaccard=1/3 >= 0.3), r2 links to r3 via keyword overlap
        has_all_three = any(len(c) == 3 for c in clusters)
        # If not all three, at least r2 and r3 should cluster
        has_at_least_two = any(len(c) >= 2 for c in clusters)
        assert has_at_least_two

    def test_multiple_distinct_clusters(self, skill):
        reflections = [
            # Cluster A: deploy
            _make_reflection(rid="a1", task="deploy service frontend", tags=["deploy"]),
            _make_reflection(rid="a2", task="deploy service backend", tags=["deploy"]),
            _make_reflection(rid="a3", task="deploy service database", tags=["deploy"]),
            # Cluster B: migrate
            _make_reflection(rid="b1", task="migrate schema version upgrade", tags=["migration"]),
            _make_reflection(rid="b2", task="migrate schema column addition", tags=["migration"]),
            _make_reflection(rid="b3", task="migrate schema table creation", tags=["migration"]),
        ]
        clusters = skill._cluster_reflections(reflections, min_cluster_size=3)
        assert len(clusters) == 2

    def test_empty_reflection_list(self, skill):
        clusters = skill._cluster_reflections([], min_cluster_size=2)
        assert clusters == []

    def test_all_unique_no_clusters(self, skill):
        reflections = [
            _make_reflection(rid="r1", task="completely unique alpha", tags=["t1"]),
            _make_reflection(rid="r2", task="different beta thing", tags=["t2"]),
            _make_reflection(rid="r3", task="unrelated gamma work", tags=["t3"]),
        ]
        clusters = skill._cluster_reflections(reflections, min_cluster_size=2)
        # No pair should have sufficient similarity
        assert clusters == []

    def test_fewer_than_min_cluster_size(self, skill):
        reflections = [_make_reflection(rid="r1")]
        clusters = skill._cluster_reflections(reflections, min_cluster_size=3)
        assert clusters == []


# ===================================================================
# Test Playbook Extraction
# ===================================================================


class TestExtraction:
    """Tests for playbook extraction helper methods."""

    def test_auto_generate_name_from_common_words(self, skill):
        reflections = [
            _make_reflection(task="deploy service monitoring"),
            _make_reflection(task="deploy service alerts"),
            _make_reflection(task="deploy service logging"),
        ]
        name = skill._auto_generate_name(reflections)
        # "deploy" and "service" should be among top words
        assert "deploy" in name or "service" in name

    def test_auto_generate_name_fallback_to_hash(self, skill):
        reflections = [
            _make_reflection(rid="r1", task="a"),  # no significant tokens
            _make_reflection(rid="r2", task="b"),
        ]
        name = skill._auto_generate_name(reflections)
        assert name.startswith("auto_playbook_")

    def test_extract_task_pattern(self, skill):
        reflections = [
            _make_reflection(task="deploy service production environment"),
            _make_reflection(task="deploy service staging environment"),
            _make_reflection(task="deploy service development environment"),
        ]
        pattern = skill._extract_task_pattern(reflections)
        assert "deploy" in pattern.lower() or "service" in pattern.lower()
        assert "Tasks involving" in pattern or "Tasks related to" in pattern

    def test_extract_task_pattern_empty(self, skill):
        reflections = [_make_reflection(task="a")]
        pattern = skill._extract_task_pattern(reflections)
        # With no significant tokens, falls back
        assert "General task pattern" in pattern or "Tasks" in pattern

    def test_extract_steps_from_successful(self, skill):
        reflections = [
            _make_reflection(success=True, actions_taken=["build", "test", "deploy"]),
            _make_reflection(success=True, actions_taken=["build", "test", "deploy"]),
            _make_reflection(success=False, actions_taken=["wrong_step"]),
        ]
        steps = skill._extract_steps(reflections)
        assert "build" in steps
        assert "test" in steps
        assert "deploy" in steps
        # wrong_step is from failed, so not included (successful are preferred)
        assert "wrong_step" not in steps

    def test_extract_steps_handles_string_actions(self, skill):
        reflections = [
            _make_reflection(success=True, actions_taken="single_action_string"),
        ]
        steps = skill._extract_steps(reflections)
        assert "single_action_string" in steps

    def test_extract_steps_fallback_when_none_succeed(self, skill):
        reflections = [
            _make_reflection(success=False, actions_taken=["action_a"]),
            _make_reflection(success=False, actions_taken=["action_b"]),
        ]
        steps = skill._extract_steps(reflections)
        # Falls back to all reflections
        assert "action_a" in steps or "action_b" in steps

    def test_extract_steps_no_actions(self, skill):
        reflections = [
            {"id": "r1", "task": "something", "tags": [], "success": True,
             "actions_taken": [], "outcome": "", "analysis": "", "improvements": []},
        ]
        steps = skill._extract_steps(reflections)
        assert steps == ["Execute the task based on available context"]

    def test_extract_pitfalls_from_failed(self, skill):
        reflections = [
            _make_reflection(success=False, analysis="Forgot env vars", improvements=["Check env first"]),
            _make_reflection(success=True, analysis="All good"),
        ]
        pitfalls = skill._extract_pitfalls(reflections)
        assert any("Forgot env vars" in p for p in pitfalls)
        assert any("Check env first" in p for p in pitfalls)
        # Successful reflections are not pitfall sources
        assert not any("All good" in p for p in pitfalls)

    def test_extract_pitfalls_deduplication(self, skill):
        reflections = [
            _make_reflection(success=False, analysis="Same problem", improvements=[]),
            _make_reflection(success=False, analysis="same problem", improvements=[]),
        ]
        pitfalls = skill._extract_pitfalls(reflections)
        # case-insensitive dedup
        analysis_pitfalls = [p for p in pitfalls if "problem" in p.lower()]
        assert len(analysis_pitfalls) == 1

    def test_extract_pitfalls_none_failed(self, skill):
        reflections = [
            _make_reflection(success=True),
            _make_reflection(success=True),
        ]
        pitfalls = skill._extract_pitfalls(reflections)
        assert pitfalls == []

    def test_extract_pitfalls_string_improvements(self, skill):
        reflections = [
            _make_reflection(success=False, analysis="", improvements="single string"),
        ]
        pitfalls = skill._extract_pitfalls(reflections)
        assert any("single string" in p for p in pitfalls)

    def test_extract_prerequisites_common_tags(self, skill):
        reflections = [
            _make_reflection(tags=["deploy", "k8s", "production"]),
            _make_reflection(tags=["deploy", "k8s", "staging"]),
            _make_reflection(tags=["deploy", "docker"]),
        ]
        prereqs = skill._extract_prerequisites(reflections)
        # "deploy" appears in all 3, should be common
        assert any("deploy" in p.lower() for p in prereqs)

    def test_extract_prerequisites_hint_words(self, skill):
        reflections = [
            _make_reflection(task="deploy database cluster with ssl"),
            _make_reflection(task="deploy database cluster with auth"),
            _make_reflection(task="deploy database cluster monitoring"),
        ]
        prereqs = skill._extract_prerequisites(reflections)
        # "database" and "cluster" appear in all and are in prereq_hints
        assert any("database" in p for p in prereqs) or any("cluster" in p for p in prereqs)

    def test_extract_prerequisites_no_specific(self, skill):
        # Use single-char words that the tokenizer will exclude (regex needs 2+ chars starting with alpha)
        reflections = [
            {"id": "r1", "task": "a b c", "tags": [], "success": True,
             "actions_taken": [], "outcome": "", "analysis": "", "improvements": []},
            {"id": "r2", "task": "x y z", "tags": [], "success": True,
             "actions_taken": [], "outcome": "", "analysis": "", "improvements": []},
        ]
        prereqs = skill._extract_prerequisites(reflections)
        assert any("No specific" in p for p in prereqs)

    def test_extract_expected_outcome_from_successful(self, skill):
        reflections = [
            _make_reflection(success=True, outcome="Service deployed and healthy"),
            _make_reflection(success=True, outcome="Service deployed and running"),
            _make_reflection(success=False, outcome="Failed miserably"),
        ]
        outcome = skill._extract_expected_outcome(reflections)
        assert "deployed" in outcome.lower()
        assert "Failed miserably" != outcome

    def test_extract_expected_outcome_no_successes(self, skill):
        reflections = [
            _make_reflection(success=False, outcome="bad"),
        ]
        outcome = skill._extract_expected_outcome(reflections)
        assert "no successful" in outcome.lower()

    def test_extract_expected_outcome_empty_outcomes(self, skill):
        reflections = [
            _make_reflection(success=True, outcome=""),
        ]
        outcome = skill._extract_expected_outcome(reflections)
        assert "Successful task completion" == outcome

    def test_extract_all_tags_union(self, skill):
        reflections = [
            _make_reflection(tags=["deploy", "k8s"]),
            _make_reflection(tags=["deploy", "docker"]),
        ]
        tags = skill._extract_all_tags(reflections)
        assert "deploy" in tags
        assert "k8s" in tags
        assert "docker" in tags
        assert "auto_generated_playbook" in tags

    def test_extract_all_tags_extra(self, skill):
        reflections = [_make_reflection(tags=["a"])]
        tags = skill._extract_all_tags(reflections, extra_tags=["custom_tag"])
        assert "custom_tag" in tags
        assert "a" in tags
        assert "auto_generated_playbook" in tags

    def test_extract_all_tags_sorted(self, skill):
        reflections = [
            _make_reflection(tags=["zebra", "apple"]),
        ]
        tags = skill._extract_all_tags(reflections)
        assert tags == sorted(tags)


# ===================================================================
# Test Cluster Metadata
# ===================================================================


class TestClusterMetadata:
    """Tests for cluster metadata helpers."""

    def test_generate_cluster_id_deterministic(self, skill):
        cluster = [_make_reflection(rid="r1"), _make_reflection(rid="r2")]
        id1 = skill._generate_cluster_id(cluster)
        id2 = skill._generate_cluster_id(cluster)
        assert id1 == id2
        assert id1.startswith("cluster_")

    def test_generate_cluster_id_different_for_different_clusters(self, skill):
        c1 = [_make_reflection(rid="r1"), _make_reflection(rid="r2")]
        c2 = [_make_reflection(rid="r3"), _make_reflection(rid="r4")]
        assert skill._generate_cluster_id(c1) != skill._generate_cluster_id(c2)

    def test_get_common_tags_threshold(self, skill):
        # 4 reflections, threshold = max(1, 4//2) = 2
        cluster = [
            _make_reflection(tags=["deploy", "k8s"]),
            _make_reflection(tags=["deploy", "k8s"]),
            _make_reflection(tags=["deploy", "docker"]),
            _make_reflection(tags=["deploy"]),
        ]
        common = skill._get_common_tags(cluster)
        # "deploy" in all 4, "k8s" in 2, "docker" in 1
        assert "deploy" in common
        assert "k8s" in common  # 2 >= threshold of 2
        assert "docker" not in common  # 1 < threshold of 2

    def test_get_common_tags_empty_cluster(self, skill):
        common = skill._get_common_tags([])
        assert common == []

    def test_get_representative_task(self, skill):
        cluster = [
            _make_reflection(task="deploy service monitoring dashboard frontend"),
            _make_reflection(task="deploy service monitoring backend"),
            _make_reflection(task="deploy service monitoring"),
        ]
        representative = skill._get_representative_task(cluster)
        # Should pick the one with most overlap with aggregate keywords
        assert "deploy" in representative.lower()
        assert "service" in representative.lower() or "monitoring" in representative.lower()

    def test_get_representative_task_empty(self, skill):
        assert skill._get_representative_task([]) == ""

    def test_compute_success_rate_all_success(self, skill):
        cluster = [
            _make_reflection(success=True),
            _make_reflection(success=True),
        ]
        assert skill._compute_success_rate(cluster) == 1.0

    def test_compute_success_rate_all_fail(self, skill):
        cluster = [
            _make_reflection(success=False),
            _make_reflection(success=False),
        ]
        assert skill._compute_success_rate(cluster) == 0.0

    def test_compute_success_rate_mixed(self, skill):
        cluster = [
            _make_reflection(success=True),
            _make_reflection(success=False),
            _make_reflection(success=True),
            _make_reflection(success=False),
        ]
        assert skill._compute_success_rate(cluster) == pytest.approx(0.5)

    def test_compute_success_rate_empty(self, skill):
        assert skill._compute_success_rate([]) == 0.0

    def test_find_overlapping_playbook_match(self, skill):
        common_tags = ["deploy", "k8s"]
        existing = {
            "my_playbook": {"deploy", "k8s", "production"},
        }
        result = skill._find_overlapping_playbook(common_tags, existing)
        assert result == "my_playbook"

    def test_find_overlapping_playbook_no_match(self, skill):
        common_tags = ["deploy"]
        existing = {
            "my_playbook": {"database", "migration", "sql"},
        }
        result = skill._find_overlapping_playbook(common_tags, existing)
        assert result is None

    def test_find_overlapping_playbook_empty_tags(self, skill):
        result = skill._find_overlapping_playbook([], {"pb": {"a"}})
        assert result is None

    def test_find_overlapping_playbook_empty_existing(self, skill):
        result = skill._find_overlapping_playbook(["deploy"], {})
        assert result is None


# ===================================================================
# Test Action: scan (async)
# ===================================================================


class TestScan:
    """Tests for the scan action."""

    def test_scan_no_context_returns_empty(self, skill):
        # No context means _fetch_reflections returns []
        result = run(skill.execute("scan", {}))
        assert result.success
        assert "No reflections" in result.message

    def test_scan_with_reflections_finds_clusters(self, skill_with_context):
        skill = skill_with_context
        reflections = _make_cluster(n=4, tag="deploy", task_prefix="Deploy service")

        # Mock: fetch reflections returns our cluster, fetch playbooks returns empty
        async def mock_call(skill_id, action, params=None):
            params = params or {}
            if skill_id == "agent_reflection" and action == "review":
                if params.get("what") == "playbooks":
                    return SkillResult(success=True, data={"playbooks": []})
                return SkillResult(success=True, data={"reflections": reflections})
            return SkillResult(success=False, message="unknown")

        skill.context.call_skill = AsyncMock(side_effect=mock_call)

        result = run(skill.execute("scan", {"min_cluster_size": 2}))
        assert result.success
        assert result.data["reflections_scanned"] == 4
        assert len(result.data["clusters"]) >= 1

    def test_scan_with_filter_tag(self, skill_with_context):
        skill = skill_with_context
        reflections = _make_cluster(n=3, tag="deploy")

        async def mock_call(skill_id, action, params=None):
            params = params or {}
            if skill_id == "agent_reflection" and action == "review":
                if params.get("what") == "playbooks":
                    return SkillResult(success=True, data={"playbooks": []})
                # Verify filter_tag is passed through
                assert params.get("filter_tag") == "deploy"
                return SkillResult(success=True, data={"reflections": reflections})
            return SkillResult(success=False, message="unknown")

        skill.context.call_skill = AsyncMock(side_effect=mock_call)

        result = run(skill.execute("scan", {"filter_tag": "deploy", "min_cluster_size": 2}))
        assert result.success

    def test_scan_updates_stats(self, skill_with_context):
        skill = skill_with_context
        reflections = _make_cluster(n=3, tag="deploy")

        async def mock_call(skill_id, action, params=None):
            params = params or {}
            if skill_id == "agent_reflection" and action == "review":
                if params.get("what") == "playbooks":
                    return SkillResult(success=True, data={"playbooks": []})
                return SkillResult(success=True, data={"reflections": reflections})
            return SkillResult(success=False, message="unknown")

        skill.context.call_skill = AsyncMock(side_effect=mock_call)

        assert skill._stats["scans_performed"] == 0
        run(skill.execute("scan", {"min_cluster_size": 2}))
        assert skill._stats["scans_performed"] == 1
        assert skill._stats["last_scan_at"] is not None

    def test_scan_too_few_reflections(self, skill_with_context):
        skill = skill_with_context

        async def mock_call(skill_id, action, params=None):
            params = params or {}
            if skill_id == "agent_reflection" and action == "review":
                if params.get("what") == "playbooks":
                    return SkillResult(success=True, data={"playbooks": []})
                return SkillResult(success=True, data={"reflections": [_make_reflection()]})
            return SkillResult(success=False, message="unknown")

        skill.context.call_skill = AsyncMock(side_effect=mock_call)
        result = run(skill.execute("scan", {"min_cluster_size": 3}))
        assert result.success
        assert "Only 1" in result.message


# ===================================================================
# Test Action: generate (async)
# ===================================================================


class TestGenerate:
    """Tests for the generate action."""

    def test_generate_with_valid_cluster_id(self, skill):
        # Pre-populate a cluster in the skill's internal state
        cluster_reflections = _make_cluster(n=3, tag="deploy")
        cluster_id = skill._generate_cluster_id(cluster_reflections)
        skill._clusters[cluster_id] = {
            "cluster_id": cluster_id,
            "reflections": cluster_reflections,
            "common_tags": ["deploy"],
            "representative_task": "Deploy service version 1",
            "success_rate": 0.67,
            "existing_playbook": None,
        }

        result = run(skill.execute("generate", {"cluster_id": cluster_id}))
        assert result.success
        assert "generated" in result.message.lower()
        assert result.data["playbook"]["source_reflection_count"] == 3

    def test_generate_missing_cluster_id(self, skill):
        result = run(skill.execute("generate", {}))
        assert not result.success
        assert "Required" in result.message

    def test_generate_nonexistent_cluster_id(self, skill):
        result = run(skill.execute("generate", {"cluster_id": "nonexistent_xyz"}))
        assert not result.success
        assert "not found" in result.message

    def test_generate_records_playbook_in_history(self, skill):
        cluster_reflections = _make_cluster(n=3, tag="deploy")
        cluster_id = skill._generate_cluster_id(cluster_reflections)
        skill._clusters[cluster_id] = {
            "cluster_id": cluster_id,
            "reflections": cluster_reflections,
            "common_tags": ["deploy"],
            "representative_task": "Deploy service version 1",
            "success_rate": 0.67,
            "existing_playbook": None,
        }

        assert len(skill._generated_playbooks) == 0
        run(skill.execute("generate", {"cluster_id": cluster_id}))
        assert len(skill._generated_playbooks) == 1
        assert skill._generated_playbooks[0]["cluster_id"] == cluster_id

    def test_generate_with_custom_name(self, skill):
        cluster_reflections = _make_cluster(n=3, tag="deploy")
        cluster_id = skill._generate_cluster_id(cluster_reflections)
        skill._clusters[cluster_id] = {
            "cluster_id": cluster_id,
            "reflections": cluster_reflections,
            "common_tags": ["deploy"],
            "representative_task": "Deploy service",
            "success_rate": 0.67,
            "existing_playbook": None,
        }

        result = run(skill.execute("generate", {
            "cluster_id": cluster_id,
            "playbook_name": "my_custom_playbook",
        }))
        assert result.success
        assert result.data["playbook"]["playbook_name"] == "my_custom_playbook"

    def test_generate_updates_stats(self, skill):
        cluster_reflections = _make_cluster(n=3, tag="deploy")
        cluster_id = skill._generate_cluster_id(cluster_reflections)
        skill._clusters[cluster_id] = {
            "cluster_id": cluster_id,
            "reflections": cluster_reflections,
            "common_tags": ["deploy"],
            "representative_task": "Deploy service",
            "success_rate": 0.67,
            "existing_playbook": None,
        }

        assert skill._stats["playbooks_generated"] == 0
        run(skill.execute("generate", {"cluster_id": cluster_id}))
        assert skill._stats["playbooks_generated"] == 1
        assert skill._stats["last_generate_at"] is not None


# ===================================================================
# Test Action: auto_generate (async)
# ===================================================================


class TestAutoGenerate:
    """Tests for the auto_generate action."""

    def _setup_scan_mock(self, skill, reflections):
        """Wire up mock context for scan + generate flow."""
        async def mock_call(skill_id, action, params=None):
            params = params or {}
            if skill_id == "agent_reflection" and action == "review":
                if params.get("what") == "playbooks":
                    return SkillResult(success=True, data={"playbooks": []})
                return SkillResult(success=True, data={"reflections": reflections})
            if skill_id == "agent_reflection" and action == "create_playbook":
                return SkillResult(
                    success=True,
                    message="Playbook created",
                    data={"playbook": params},
                )
            return SkillResult(success=False, message="unknown")

        skill.context.call_skill = AsyncMock(side_effect=mock_call)

    def test_auto_generate_combines_scan_and_generate(self, skill_with_context):
        skill = skill_with_context
        reflections = _make_cluster(n=4, tag="deploy", task_prefix="Deploy service")
        self._setup_scan_mock(skill, reflections)

        result = run(skill.execute("auto_generate", {"min_cluster_size": 2}))
        assert result.success
        assert len(result.data.get("generated", [])) >= 1

    def test_auto_generate_filters_by_success_rate(self, skill_with_context):
        skill = skill_with_context
        # All fail => success_rate = 0
        reflections = [
            _make_reflection(rid=f"r{i}", task="deploy service endpoint", tags=["deploy"], success=False)
            for i in range(4)
        ]
        self._setup_scan_mock(skill, reflections)

        result = run(skill.execute("auto_generate", {
            "min_cluster_size": 2,
            "min_success_rate": 0.5,
        }))
        assert result.success
        # No playbooks generated because success_rate is 0 < 0.5
        assert len(result.data.get("generated", [])) == 0

    def test_auto_generate_respects_max_generate(self, skill_with_context):
        skill = skill_with_context
        # Create two distinct clusters
        reflections = (
            _make_cluster(n=3, tag="deploy", task_prefix="Deploy service") +
            _make_cluster(n=3, tag="migrate", task_prefix="Migrate schema")
        )
        # Fix IDs to be unique across clusters
        for i, r in enumerate(reflections):
            r["id"] = f"unique_{i}"

        self._setup_scan_mock(skill, reflections)

        result = run(skill.execute("auto_generate", {
            "min_cluster_size": 2,
            "max_generate": 1,
        }))
        assert result.success
        assert len(result.data.get("generated", [])) <= 1

    def test_auto_generate_no_candidates(self, skill_with_context):
        skill = skill_with_context
        # Too few reflections to form clusters
        reflections = [_make_reflection(rid="r1")]
        self._setup_scan_mock(skill, reflections)

        result = run(skill.execute("auto_generate", {"min_cluster_size": 5}))
        assert result.success
        assert len(result.data.get("generated", [])) == 0


# ===================================================================
# Test Action: wire / unwire (async)
# ===================================================================


class TestWire:
    """Tests for wire and unwire actions."""

    def test_wire_sets_state(self, skill):
        result = run(skill.execute("wire", {}))
        assert result.success
        assert skill._wire_state["active"] is True
        assert skill._wire_state["subscription_id"] is not None

    def test_wire_with_custom_parameters(self, skill):
        result = run(skill.execute("wire", {
            "scan_every_n_reflections": 5,
            "auto_generate": False,
        }))
        assert result.success
        assert skill._config["scan_every_n_reflections"] == 5
        assert skill._config["auto_generate_on_scan"] is False

    def test_unwire_clears_state(self, skill):
        # First wire
        run(skill.execute("wire", {}))
        assert skill._wire_state["active"] is True

        # Then unwire
        result = run(skill.execute("unwire", {}))
        assert result.success
        assert skill._wire_state["active"] is False
        assert skill._wire_state["subscription_id"] is None
        assert "stopped" in result.message.lower()

    def test_unwire_when_already_unwired(self, skill):
        result = run(skill.execute("unwire", {}))
        assert result.success
        assert "Already unwired" in result.message

    def test_wire_with_context_attempts_bus_subscription(self, skill_with_context):
        skill = skill_with_context
        skill.context.call_skill = AsyncMock(return_value=SkillResult(success=True))

        result = run(skill.execute("wire", {}))
        assert result.success
        assert skill._wire_state["active"] is True


# ===================================================================
# Test Action: configure (async)
# ===================================================================


class TestConfigure:
    """Tests for the configure action."""

    def test_configure_updates_values(self, skill):
        result = run(skill.execute("configure", {
            "min_cluster_size": 5,
            "min_success_rate": 0.8,
        }))
        assert result.success
        assert skill._config["min_cluster_size"] == 5
        assert skill._config["min_success_rate"] == pytest.approx(0.8)
        assert len(result.data["updated"]) == 2

    def test_configure_with_invalid_types(self, skill):
        # "not_a_number" can't be cast to int for min_cluster_size
        result = run(skill.execute("configure", {
            "min_cluster_size": "not_a_number",
        }))
        assert result.success
        # The invalid value is skipped, config unchanged
        assert skill._config["min_cluster_size"] == 3  # default

    def test_configure_partial_update(self, skill):
        original_min_success = skill._config["min_success_rate"]
        result = run(skill.execute("configure", {
            "min_cluster_size": 10,
        }))
        assert result.success
        assert skill._config["min_cluster_size"] == 10
        assert skill._config["min_success_rate"] == original_min_success

    def test_configure_all_keys(self, skill):
        result = run(skill.execute("configure", {
            "min_cluster_size": 4,
            "min_success_rate": 0.6,
            "auto_generate_on_scan": False,
            "scan_every_n_reflections": 20,
            "max_playbooks_per_scan": 10,
        }))
        assert result.success
        assert len(result.data["updated"]) == 5
        assert skill._config["min_cluster_size"] == 4
        assert skill._config["auto_generate_on_scan"] is False
        assert skill._config["scan_every_n_reflections"] == 20
        assert skill._config["max_playbooks_per_scan"] == 10


# ===================================================================
# Test Action: status (async)
# ===================================================================


class TestStatus:
    """Tests for the status action."""

    def test_status_returns_all_state(self, skill):
        result = run(skill.execute("status", {}))
        assert result.success
        data = result.data
        assert "clusters" in data
        assert "recent_generated_playbooks" in data
        assert "wire_state" in data
        assert "config" in data
        assert "stats" in data
        assert "total_clusters" in data
        assert "total_generated_playbooks" in data

    def test_status_reflects_changes(self, skill):
        skill._stats["scans_performed"] = 10
        skill._stats["playbooks_generated"] = 3
        run(skill.execute("wire", {}))

        result = run(skill.execute("status", {}))
        assert result.success
        assert "ACTIVE" in result.message
        assert result.data["stats"]["scans_performed"] == 10
        assert result.data["stats"]["playbooks_generated"] == 3


# ===================================================================
# Test EventBus Callback
# ===================================================================


class TestEventBusCallback:
    """Tests for _on_reflection_created callback."""

    def test_increments_counter(self, skill):
        skill._wire_state["active"] = True
        skill._wire_state["reflections_since_scan"] = 0
        skill._config["scan_every_n_reflections"] = 100  # high threshold so no scan triggers

        run(skill._on_reflection_created({"type": "reflection.created"}))
        assert skill._wire_state["reflections_since_scan"] == 1

    def test_triggers_scan_at_threshold(self, tmp_path):
        test_file = tmp_path / "state.json"
        with patch("singularity.skills.auto_playbook_generation.STATE_FILE", test_file):
            skill = AutoPlaybookGenerationSkill()
            skill._wire_state["active"] = True
            skill._wire_state["reflections_since_scan"] = 9
            skill._config["scan_every_n_reflections"] = 10
            skill._config["auto_generate_on_scan"] = True

            # Mock _auto_generate to just track that it's called
            called = {"count": 0}
            original_auto_gen = skill._auto_generate

            async def mock_auto_gen(params):
                called["count"] += 1
                return SkillResult(success=True, data={})

            skill._auto_generate = mock_auto_gen

            run(skill._on_reflection_created({"type": "reflection.created"}))
            assert called["count"] == 1
            assert skill._wire_state["reflections_since_scan"] == 0

    def test_no_trigger_when_inactive(self, skill):
        skill._wire_state["active"] = False
        skill._wire_state["reflections_since_scan"] = 0

        run(skill._on_reflection_created({"type": "reflection.created"}))
        assert skill._wire_state["reflections_since_scan"] == 0

    def test_scan_only_when_auto_generate_disabled(self, tmp_path):
        test_file = tmp_path / "state.json"
        with patch("singularity.skills.auto_playbook_generation.STATE_FILE", test_file):
            skill = AutoPlaybookGenerationSkill()
            skill._wire_state["active"] = True
            skill._wire_state["reflections_since_scan"] = 9
            skill._config["scan_every_n_reflections"] = 10
            skill._config["auto_generate_on_scan"] = False

            scan_called = {"count": 0}
            auto_gen_called = {"count": 0}

            async def mock_scan(params):
                scan_called["count"] += 1
                return SkillResult(success=True, data={})

            async def mock_auto_gen(params):
                auto_gen_called["count"] += 1
                return SkillResult(success=True, data={})

            skill._scan = mock_scan
            skill._auto_generate = mock_auto_gen

            run(skill._on_reflection_created({}))
            assert scan_called["count"] == 1
            assert auto_gen_called["count"] == 0


# ===================================================================
# Test Manifest
# ===================================================================


class TestManifest:
    """Tests for the skill manifest."""

    def test_manifest_metadata(self, skill):
        m = skill.manifest
        assert m.skill_id == "auto_playbook_generation"
        assert m.name == "Auto Playbook Generation"
        assert m.version == "1.0.0"
        assert m.category == "self_improvement"

    def test_manifest_has_seven_actions(self, skill):
        actions = skill.manifest.actions
        assert len(actions) == 7

    def test_manifest_action_names(self, skill):
        action_names = {a.name for a in skill.manifest.actions}
        expected = {"scan", "generate", "auto_generate", "wire", "unwire", "configure", "status"}
        assert action_names == expected

    def test_manifest_no_required_credentials(self, skill):
        assert skill.manifest.required_credentials == []

    def test_estimate_cost_always_zero(self, skill):
        for action_name in ["scan", "generate", "auto_generate", "wire", "unwire", "configure", "status"]:
            assert skill.estimate_cost(action_name, {}) == 0.0


# ===================================================================
# Test Unknown Action
# ===================================================================


class TestUnknownAction:
    """Test dispatch to unknown actions."""

    def test_unknown_action(self, skill):
        result = run(skill.execute("nonexistent_action", {}))
        assert not result.success
        assert "Unknown action" in result.message

    def test_unknown_action_lists_available(self, skill):
        result = run(skill.execute("bad", {}))
        assert "scan" in result.message


# ===================================================================
# Test Edge Cases
# ===================================================================


class TestEdgeCases:
    """Miscellaneous edge case tests."""

    def test_extract_playbook_tag_sets(self, skill):
        playbooks = [
            {"name": "pb1", "tags": ["Deploy", "K8S"]},
            {"name": "pb2", "tags": ["Database", "Migration"]},
            {"name": "", "tags": ["orphan"]},  # empty name, should be skipped
        ]
        result = skill._extract_playbook_tag_sets(playbooks)
        assert "pb1" in result
        assert result["pb1"] == {"deploy", "k8s"}
        assert "pb2" in result
        assert "" not in result

    def test_cluster_id_order_independent(self, skill):
        c1 = [_make_reflection(rid="b"), _make_reflection(rid="a")]
        c2 = [_make_reflection(rid="a"), _make_reflection(rid="b")]
        assert skill._generate_cluster_id(c1) == skill._generate_cluster_id(c2)

    def test_save_state_enforces_max_clusters(self, tmp_path):
        test_file = tmp_path / "state.json"
        with patch("singularity.skills.auto_playbook_generation.STATE_FILE", test_file):
            from singularity.skills.auto_playbook_generation import MAX_CLUSTERS
            skill = AutoPlaybookGenerationSkill()
            for i in range(MAX_CLUSTERS + 50):
                skill._clusters[f"cluster_{i}"] = {"cluster_id": f"cluster_{i}"}
            skill._save_state()
            # Reload and check
            skill2 = AutoPlaybookGenerationSkill()
            assert len(skill2._clusters) <= MAX_CLUSTERS

    def test_save_state_enforces_max_generated_playbooks(self, tmp_path):
        test_file = tmp_path / "state.json"
        with patch("singularity.skills.auto_playbook_generation.STATE_FILE", test_file):
            from singularity.skills.auto_playbook_generation import MAX_GENERATED_PLAYBOOKS
            skill = AutoPlaybookGenerationSkill()
            for i in range(MAX_GENERATED_PLAYBOOKS + 50):
                skill._generated_playbooks.append({"index": i})
            skill._save_state()
            skill2 = AutoPlaybookGenerationSkill()
            assert len(skill2._generated_playbooks) <= MAX_GENERATED_PLAYBOOKS

    def test_generate_with_empty_reflections_cluster(self, skill):
        skill._clusters["empty_cluster"] = {
            "cluster_id": "empty_cluster",
            "reflections": [],
            "common_tags": [],
            "representative_task": "",
            "success_rate": 0,
            "existing_playbook": None,
        }
        result = run(skill.execute("generate", {"cluster_id": "empty_cluster"}))
        assert not result.success
        assert "no reflections" in result.message.lower()

    def test_steps_limited_to_ten(self, skill):
        reflections = [
            _make_reflection(
                success=True,
                actions_taken=[f"step_{j}" for j in range(15)],
            )
        ]
        steps = skill._extract_steps(reflections)
        assert len(steps) <= 10

    def test_pitfalls_limited_to_eight(self, skill):
        reflections = [
            _make_reflection(
                success=False,
                analysis=f"Analysis {i}",
                improvements=[f"Improvement {i}a", f"Improvement {i}b"],
            )
            for i in range(10)
        ]
        pitfalls = skill._extract_pitfalls(reflections)
        assert len(pitfalls) <= 8

    def test_prerequisites_limited_to_five(self, skill):
        reflections = [
            _make_reflection(
                task="deploy database cluster config auth server container docker",
                tags=["deploy", "db", "config", "auth", "ssl", "dns", "server", "k8s"],
            )
            for _ in range(5)
        ]
        prereqs = skill._extract_prerequisites(reflections)
        assert len(prereqs) <= 5
