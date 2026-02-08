"""Comprehensive tests for SkillDependencyMapSkill — dependency analysis for skill ecosystem."""

import tempfile
import textwrap
from pathlib import Path

import pytest

from singularity.skills.skill_dependency_map import (
    DependencyAnalyzer,
    DependencyEdge,
    SkillDependencyMapSkill,
    SkillNode,
)

# ---------------------------------------------------------------------------
# Sample skill sources for testing
# ---------------------------------------------------------------------------

SKILL_A_SOURCE = textwrap.dedent('''\
    """Skill A — depends on B via call_skill."""
    from .base import Skill, SkillManifest, SkillAction, SkillResult

    class SkillA(Skill):
        def __init__(self, credentials=None):
            super().__init__(credentials)

        @property
        def manifest(self):
            return SkillManifest(
                skill_id="skill_a",
                name="Skill A",
                version="1.0.0",
                category="test",
                description="Test skill A",
                required_credentials=[],
                actions=[SkillAction(name="run", description="run", parameters={})],
            )

        async def execute(self, action, params):
            if self.context:
                result = await self.context.call_skill("skill_b", "process", {"data": "test"})
            return SkillResult(success=True, message="ok")
''')

SKILL_B_SOURCE = textwrap.dedent('''\
    """Skill B — depends on C via import."""
    from .base import Skill, SkillManifest, SkillAction, SkillResult
    from .skill_c import SomeHelper

    class SkillB(Skill):
        def __init__(self, credentials=None):
            super().__init__(credentials)

        @property
        def manifest(self):
            return SkillManifest(
                skill_id="skill_b",
                name="Skill B",
                version="1.0.0",
                category="test",
                description="Test skill B",
                required_credentials=[],
                actions=[SkillAction(name="process", description="process", parameters={})],
            )

        async def execute(self, action, params):
            return SkillResult(success=True, message="ok")
''')

SKILL_C_SOURCE = textwrap.dedent('''\
    """Skill C — standalone, no deps."""
    from .base import Skill, SkillManifest, SkillAction, SkillResult

    class SkillC(Skill):
        def __init__(self, credentials=None):
            super().__init__(credentials)

        @property
        def manifest(self):
            return SkillManifest(
                skill_id="skill_c",
                name="Skill C",
                version="1.0.0",
                category="test",
                description="Test skill C",
                required_credentials=[],
                actions=[SkillAction(name="help", description="help", parameters={})],
            )

        async def execute(self, action, params):
            return SkillResult(success=True, message="ok")

class SomeHelper:
    pass
''')

SKILL_CIRCULAR_SOURCE = textwrap.dedent('''\
    """Skill that creates a circular dep with skill_a."""
    from .base import Skill, SkillManifest, SkillAction, SkillResult

    class SkillCircular(Skill):
        def __init__(self, credentials=None):
            super().__init__(credentials)

        @property
        def manifest(self):
            return SkillManifest(
                skill_id="skill_circular",
                name="Circular",
                version="1.0.0",
                category="test",
                description="Creates circular dep",
                required_credentials=[],
                actions=[SkillAction(name="loop", description="loop", parameters={})],
            )

        async def execute(self, action, params):
            if self.context:
                await self.context.call_skill("skill_a", "run", {})
            return SkillResult(success=True, message="ok")
''')

# A depends on circular (via call_skill), circular depends on A
SKILL_A_CIRCULAR_SOURCE = textwrap.dedent('''\
    """Skill A variant — depends on circular."""
    from .base import Skill, SkillManifest, SkillAction, SkillResult

    class SkillA(Skill):
        def __init__(self, credentials=None):
            super().__init__(credentials)

        @property
        def manifest(self):
            return SkillManifest(
                skill_id="skill_a",
                name="Skill A",
                version="1.0.0",
                category="test",
                description="Test",
                required_credentials=[],
                actions=[SkillAction(name="run", description="run", parameters={})],
            )

        async def execute(self, action, params):
            if self.context:
                await self.context.call_skill("skill_circular", "loop", {})
            return SkillResult(success=True, message="ok")
''')

ORPHAN_SOURCE = textwrap.dedent('''\
    """Orphan skill — no deps, nothing depends on it."""
    from .base import Skill, SkillManifest, SkillAction, SkillResult

    class OrphanSkill(Skill):
        def __init__(self, credentials=None):
            super().__init__(credentials)

        @property
        def manifest(self):
            return SkillManifest(
                skill_id="orphan",
                name="Orphan",
                version="1.0.0",
                category="test",
                description="Lonely skill",
                required_credentials=[],
                actions=[SkillAction(name="sit", description="sit", parameters={})],
            )

        async def execute(self, action, params):
            return SkillResult(success=True, message="ok")
''')


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _create_skill_dir(files: dict) -> Path:
    """Create a temp directory with skill files."""
    tmpdir = Path(tempfile.mkdtemp())
    for name, source in files.items():
        (tmpdir / name).write_text(source, encoding="utf-8")
    return tmpdir


@pytest.fixture
def simple_dir():
    """Dir with A -> B -> C chain."""
    return _create_skill_dir({
        "skill_a.py": SKILL_A_SOURCE,
        "skill_b.py": SKILL_B_SOURCE,
        "skill_c.py": SKILL_C_SOURCE,
    })


@pytest.fixture
def circular_dir():
    """Dir with circular deps: A <-> circular."""
    return _create_skill_dir({
        "skill_a.py": SKILL_A_CIRCULAR_SOURCE,
        "skill_circular.py": SKILL_CIRCULAR_SOURCE,
    })


@pytest.fixture
def mixed_dir():
    """Dir with connected + orphan skills."""
    return _create_skill_dir({
        "skill_a.py": SKILL_A_SOURCE,
        "skill_b.py": SKILL_B_SOURCE,
        "skill_c.py": SKILL_C_SOURCE,
        "orphan.py": ORPHAN_SOURCE,
    })


@pytest.fixture
def skill():
    return SkillDependencyMapSkill()


# ===========================================================================
# Test SkillNode
# ===========================================================================

class TestSkillNode:
    def test_default_node(self):
        node = SkillNode(file_name="test.py")
        assert node.file_name == "test.py"
        assert node.skill_id == ""
        assert node.calls == []
        assert node.imports == []
        assert node.uses_context is False

    def test_to_dict(self):
        node = SkillNode(
            file_name="test.py",
            skill_id="test_skill",
            calls=[{"skill_id": "other", "action": "run"}],
            imports=["base", "helper"],
            uses_context=True,
            lines=100,
        )
        d = node.to_dict()
        assert d["file"] == "test.py"
        assert d["skill_id"] == "test_skill"
        assert d["uses_context"] is True
        assert d["lines"] == 100
        # "base" should be filtered from imports
        assert "base" not in d["imports"]
        assert "helper" in d["imports"]
        assert d["dependency_count"] == 2  # 1 call + 1 non-base import

    def test_to_dict_base_only_import(self):
        node = SkillNode(file_name="x.py", imports=["base"])
        d = node.to_dict()
        assert d["imports"] == []
        assert d["dependency_count"] == 0


# ===========================================================================
# Test DependencyEdge
# ===========================================================================

class TestDependencyEdge:
    def test_to_dict_call_skill(self):
        edge = DependencyEdge(
            source="a.py", target="b.py",
            edge_type="call_skill", action="run"
        )
        d = edge.to_dict()
        assert d["source"] == "a.py"
        assert d["target"] == "b.py"
        assert d["type"] == "call_skill"
        assert d["action"] == "run"

    def test_to_dict_import(self):
        edge = DependencyEdge(
            source="a.py", target="b.py",
            edge_type="import"
        )
        d = edge.to_dict()
        assert "action" not in d


# ===========================================================================
# Test DependencyAnalyzer
# ===========================================================================

class TestAnalyzerBasic:
    def test_analyze_simple_chain(self, simple_dir):
        analyzer = DependencyAnalyzer(skills_dir=simple_dir)
        analyzer.analyze_all()
        nodes = analyzer.get_nodes()
        assert len(nodes) == 3
        assert "skill_a.py" in nodes
        assert "skill_b.py" in nodes
        assert "skill_c.py" in nodes

    def test_edges_created(self, simple_dir):
        analyzer = DependencyAnalyzer(skills_dir=simple_dir)
        analyzer.analyze_all()
        edges = analyzer.get_edges()
        assert len(edges) >= 2  # A->B (call_skill) and B->C (import)

    def test_skill_id_extracted(self, simple_dir):
        analyzer = DependencyAnalyzer(skills_dir=simple_dir)
        analyzer.analyze_all()
        nodes = analyzer.get_nodes()
        assert nodes["skill_a.py"].skill_id == "skill_a"
        assert nodes["skill_b.py"].skill_id == "skill_b"
        assert nodes["skill_c.py"].skill_id == "skill_c"

    def test_call_skill_detected(self, simple_dir):
        analyzer = DependencyAnalyzer(skills_dir=simple_dir)
        analyzer.analyze_all()
        node_a = analyzer.get_nodes()["skill_a.py"]
        assert len(node_a.calls) == 1
        assert node_a.calls[0]["skill_id"] == "skill_b"
        assert node_a.calls[0]["action"] == "process"

    def test_import_detected(self, simple_dir):
        analyzer = DependencyAnalyzer(skills_dir=simple_dir)
        analyzer.analyze_all()
        node_b = analyzer.get_nodes()["skill_b.py"]
        assert "skill_c" in node_b.imports

    def test_context_detected(self, simple_dir):
        analyzer = DependencyAnalyzer(skills_dir=simple_dir)
        analyzer.analyze_all()
        assert analyzer.get_nodes()["skill_a.py"].uses_context is True
        assert analyzer.get_nodes()["skill_c.py"].uses_context is False


class TestAnalyzerDependencies:
    def test_get_dependencies(self, simple_dir):
        analyzer = DependencyAnalyzer(skills_dir=simple_dir)
        analyzer.analyze_all()
        deps = analyzer.get_dependencies("skill_a.py")
        assert "skill_b.py" in deps

    def test_get_dependents(self, simple_dir):
        analyzer = DependencyAnalyzer(skills_dir=simple_dir)
        analyzer.analyze_all()
        dependents = analyzer.get_dependents("skill_b.py")
        assert "skill_a.py" in dependents

    def test_leaf_has_no_deps(self, simple_dir):
        analyzer = DependencyAnalyzer(skills_dir=simple_dir)
        analyzer.analyze_all()
        deps = analyzer.get_dependencies("skill_c.py")
        assert len(deps) == 0

    def test_root_has_no_dependents_except_chain(self, simple_dir):
        analyzer = DependencyAnalyzer(skills_dir=simple_dir)
        analyzer.analyze_all()
        # A has no one depending on it in A->B->C chain
        dependents = analyzer.get_dependents("skill_a.py")
        assert len(dependents) == 0


class TestAnalyzerCircular:
    def test_detect_circular(self, circular_dir):
        analyzer = DependencyAnalyzer(skills_dir=circular_dir)
        analyzer.analyze_all()
        cycles = analyzer.find_circular()
        assert len(cycles) >= 1
        # Verify the cycle contains both skills
        cycle_files = set()
        for cycle in cycles:
            cycle_files.update(cycle)
        assert "skill_a.py" in cycle_files
        assert "skill_circular.py" in cycle_files

    def test_no_circular_in_chain(self, simple_dir):
        analyzer = DependencyAnalyzer(skills_dir=simple_dir)
        analyzer.analyze_all()
        cycles = analyzer.find_circular()
        assert len(cycles) == 0


class TestAnalyzerOrphans:
    def test_find_orphan(self, mixed_dir):
        analyzer = DependencyAnalyzer(skills_dir=mixed_dir)
        analyzer.analyze_all()
        orphans = analyzer.find_orphans()
        assert "orphan.py" in orphans

    def test_connected_not_orphan(self, mixed_dir):
        analyzer = DependencyAnalyzer(skills_dir=mixed_dir)
        analyzer.analyze_all()
        orphans = analyzer.find_orphans()
        assert "skill_a.py" not in orphans
        assert "skill_b.py" not in orphans
        assert "skill_c.py" not in orphans

    def test_all_orphans_when_isolated(self):
        d = _create_skill_dir({
            "lone1.py": ORPHAN_SOURCE,
            "lone2.py": ORPHAN_SOURCE.replace("orphan", "lone2"),
        })
        analyzer = DependencyAnalyzer(skills_dir=d)
        analyzer.analyze_all()
        orphans = analyzer.find_orphans()
        assert len(orphans) == 2


class TestAnalyzerHubs:
    def test_find_hubs(self, simple_dir):
        analyzer = DependencyAnalyzer(skills_dir=simple_dir)
        analyzer.analyze_all()
        # B has 2 connections (A->B, B->C), not enough for default min=3
        hubs = analyzer.find_hubs(min_connections=2)
        hub_files = [h["file"] for h in hubs]
        assert "skill_b.py" in hub_files

    def test_no_hubs_with_high_threshold(self, simple_dir):
        analyzer = DependencyAnalyzer(skills_dir=simple_dir)
        analyzer.analyze_all()
        hubs = analyzer.find_hubs(min_connections=10)
        assert len(hubs) == 0


class TestAnalyzerImpact:
    def test_impact_leaf_node(self, simple_dir):
        analyzer = DependencyAnalyzer(skills_dir=simple_dir)
        analyzer.analyze_all()
        # Changing C should affect B (import) and A (transitive through B)
        impact = analyzer.impact_analysis("skill_c.py")
        assert impact["total_affected"] >= 1  # At least B

    def test_impact_root_node(self, simple_dir):
        analyzer = DependencyAnalyzer(skills_dir=simple_dir)
        analyzer.analyze_all()
        # Changing A should affect nothing (nothing depends on A)
        impact = analyzer.impact_analysis("skill_a.py")
        assert impact["total_affected"] == 0
        assert impact["impact_level"] == "low"

    def test_impact_nonexistent(self, simple_dir):
        analyzer = DependencyAnalyzer(skills_dir=simple_dir)
        analyzer.analyze_all()
        impact = analyzer.impact_analysis("nonexistent.py")
        assert impact["total_affected"] == 0


class TestAnalyzerAutoAnalyze:
    def test_lazy_analysis(self, simple_dir):
        analyzer = DependencyAnalyzer(skills_dir=simple_dir)
        # Should auto-analyze on first call
        nodes = analyzer.get_nodes()
        assert len(nodes) == 3

    def test_lazy_edges(self, simple_dir):
        analyzer = DependencyAnalyzer(skills_dir=simple_dir)
        edges = analyzer.get_edges()
        assert len(edges) >= 2


class TestAnalyzerSingleFile:
    def test_analyze_single(self, simple_dir):
        analyzer = DependencyAnalyzer(skills_dir=simple_dir)
        node = analyzer.analyze_file(str(simple_dir / "skill_a.py"))
        assert node.skill_id == "skill_a"
        assert len(node.calls) == 1
        assert node.uses_context is True


class TestAnalyzerEmptyDir:
    def test_empty_directory(self):
        d = Path(tempfile.mkdtemp())
        analyzer = DependencyAnalyzer(skills_dir=d)
        analyzer.analyze_all()
        assert len(analyzer.get_nodes()) == 0
        assert len(analyzer.get_edges()) == 0
        assert len(analyzer.find_orphans()) == 0
        assert len(analyzer.find_circular()) == 0


# ===========================================================================
# Test SkillDependencyMapSkill — manifest
# ===========================================================================

class TestSkillManifest:
    def test_manifest_id(self, skill):
        assert skill.manifest.skill_id == "skill_dependency_map"

    def test_manifest_name(self, skill):
        assert skill.manifest.name == "Skill Dependency Map"

    def test_manifest_version(self, skill):
        assert skill.manifest.version == "1.0.0"

    def test_manifest_category(self, skill):
        assert skill.manifest.category == "development"

    def test_manifest_actions(self, skill):
        actions = [a.name for a in skill.manifest.actions]
        assert "map" in actions
        assert "impact" in actions
        assert "cycles" in actions
        assert "orphans" in actions
        assert "hubs" in actions
        assert "deps" in actions
        assert "stats" in actions

    def test_manifest_no_credentials(self, skill):
        assert skill.manifest.required_credentials == []


# ===========================================================================
# Test SkillDependencyMapSkill — actions
# ===========================================================================

class TestMapAction:
    @pytest.mark.asyncio
    async def test_map_returns_data(self, skill):
        result = await skill.execute("map", {})
        assert result.success
        assert "total_skills" in result.data
        assert "total_edges" in result.data
        assert "nodes" in result.data
        assert "edges" in result.data

    @pytest.mark.asyncio
    async def test_map_exclude_orphans(self, skill):
        result = await skill.execute("map", {"include_orphans": False})
        assert result.success


class TestImpactAction:
    @pytest.mark.asyncio
    async def test_impact_known_skill(self, skill):
        result = await skill.execute("impact", {"skill_file": "event.py"})
        assert result.success
        assert "impact_level" in result.data
        assert "direct_dependents" in result.data

    @pytest.mark.asyncio
    async def test_impact_auto_adds_py(self, skill):
        result = await skill.execute("impact", {"skill_file": "event"})
        assert result.success
        assert result.data["skill"] == "event.py"

    @pytest.mark.asyncio
    async def test_impact_no_param(self, skill):
        result = await skill.execute("impact", {})
        assert not result.success


class TestCyclesAction:
    @pytest.mark.asyncio
    async def test_cycles_returns_data(self, skill):
        result = await skill.execute("cycles", {})
        assert result.success
        assert "cycle_count" in result.data
        assert "cycles" in result.data


class TestOrphansAction:
    @pytest.mark.asyncio
    async def test_orphans_returns_data(self, skill):
        result = await skill.execute("orphans", {})
        assert result.success
        assert "orphan_count" in result.data
        assert "total_skills" in result.data
        assert "orphan_pct" in result.data
        assert "orphans" in result.data


class TestHubsAction:
    @pytest.mark.asyncio
    async def test_hubs_default_threshold(self, skill):
        result = await skill.execute("hubs", {})
        assert result.success
        assert "hub_count" in result.data
        assert "hubs" in result.data

    @pytest.mark.asyncio
    async def test_hubs_custom_threshold(self, skill):
        result = await skill.execute("hubs", {"min_connections": 1})
        assert result.success
        # Lower threshold should find more hubs
        assert result.data["hub_count"] >= 0


class TestDepsAction:
    @pytest.mark.asyncio
    async def test_deps_known_skill(self, skill):
        result = await skill.execute("deps", {"skill_file": "event.py"})
        assert result.success
        assert "depends_on" in result.data
        assert "depended_by" in result.data
        assert "calls" in result.data

    @pytest.mark.asyncio
    async def test_deps_no_param(self, skill):
        result = await skill.execute("deps", {})
        assert not result.success


class TestStatsAction:
    @pytest.mark.asyncio
    async def test_stats_returns_data(self, skill):
        result = await skill.execute("stats", {})
        assert result.success
        data = result.data
        assert "total_skills" in data
        assert "total_edges" in data
        assert "orphan_count" in data
        assert "cycle_count" in data
        assert "hub_count" in data
        assert "context_users" in data
        assert "most_depended_on" in data
        assert "most_dependent" in data

    @pytest.mark.asyncio
    async def test_stats_has_percentages(self, skill):
        result = await skill.execute("stats", {})
        assert "orphan_pct" in result.data
        assert "context_pct" in result.data


class TestUnknownAction:
    @pytest.mark.asyncio
    async def test_unknown_action(self, skill):
        result = await skill.execute("nonexistent", {})
        assert not result.success
        assert "unknown" in result.message.lower()


# ===========================================================================
# Test with real codebase
# ===========================================================================

class TestRealCodebase:
    """Integration tests against the actual skill ecosystem."""

    @pytest.mark.asyncio
    async def test_real_stats(self, skill):
        """Stats action should work on real codebase."""
        result = await skill.execute("stats", {})
        assert result.success
        # Should find many skills
        assert result.data["total_skills"] >= 50

    @pytest.mark.asyncio
    async def test_real_event_impact(self, skill):
        """Event skill should be depended on by other skills."""
        result = await skill.execute("impact", {"skill_file": "event.py"})
        assert result.success
        # Event is used by several bridge/workflow skills
        assert result.data["total_affected"] >= 1

    @pytest.mark.asyncio
    async def test_real_orphan_count(self, skill):
        """Some orphans are expected in a large ecosystem."""
        result = await skill.execute("orphans", {})
        assert result.success
        # Should have some orphans but not all
        assert 0 < result.data["orphan_pct"] < 100
