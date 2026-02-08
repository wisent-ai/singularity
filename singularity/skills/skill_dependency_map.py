"""
SkillDependencyMapSkill - Dependency analysis and visualization for the skill ecosystem.

With 120+ skills, understanding how they connect is essential for:
- Impact analysis: which skills break if skill X changes?
- Orphan detection: skills that nothing depends on and that depend on nothing
- Circular dependency detection: mutual dependencies that complicate testing
- Hub identification: skills that are critical connectors
- Ecosystem health: is the dependency graph growing in a sustainable way?

This skill performs static analysis of skill source files to build a dependency
graph based on:
1. call_skill() invocations in SkillContext
2. Direct imports from sibling skill modules
3. SkillContext usage patterns (inter-skill communication capability)
"""

import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from .base import Skill, SkillAction, SkillManifest, SkillResult

SKILLS_DIR = Path(__file__).parent


@dataclass
class SkillNode:
    """A node in the dependency graph representing one skill."""
    file_name: str
    skill_id: str = ""
    calls: List[Dict[str, str]] = field(default_factory=list)  # [{skill_id, action}]
    imports: List[str] = field(default_factory=list)  # sibling module imports
    uses_context: bool = False
    lines: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file": self.file_name,
            "skill_id": self.skill_id,
            "calls": self.calls,
            "imports": [i for i in self.imports if i != "base"],
            "uses_context": self.uses_context,
            "lines": self.lines,
            "dependency_count": len(self.calls) + len([i for i in self.imports if i != "base"]),
        }


@dataclass
class DependencyEdge:
    """An edge in the dependency graph."""
    source: str  # skill file that depends
    target: str  # skill file being depended on
    edge_type: str  # "call_skill", "import", "context"
    action: str = ""  # action being called (for call_skill edges)

    def to_dict(self) -> Dict[str, Any]:
        d = {"source": self.source, "target": self.target, "type": self.edge_type}
        if self.action:
            d["action"] = self.action
        return d


class DependencyAnalyzer:
    """Static analysis engine for skill dependency mapping."""

    def __init__(self, skills_dir: Optional[Path] = None):
        self._skills_dir = skills_dir or SKILLS_DIR
        self._nodes: Dict[str, SkillNode] = {}
        self._edges: List[DependencyEdge] = []
        self._analyzed = False

    def analyze_all(self) -> None:
        """Analyze all skill files in the skills directory."""
        self._nodes = {}
        self._edges = []

        skill_files = sorted(self._skills_dir.glob("*.py"))
        skip = {"__init__.py", "base.py"}
        skill_files = [f for f in skill_files if f.name not in skip]

        for sf in skill_files:
            node = self._analyze_file(sf)
            self._nodes[node.file_name] = node

        # Build edges from node data
        self._build_edges()
        self._analyzed = True

    def analyze_file(self, filepath: str) -> SkillNode:
        """Analyze a single skill file."""
        return self._analyze_file(Path(filepath))

    def _analyze_file(self, path: Path) -> SkillNode:
        """Parse a skill file and extract dependency information."""
        node = SkillNode(file_name=path.name)

        try:
            source = path.read_text(encoding="utf-8")
        except Exception:
            return node

        lines = source.split("\n")
        node.lines = len(lines)

        # Extract skill_id from manifest
        skill_id_match = re.search(r'skill_id\s*=\s*["\']([^"\']+)["\']', source)
        if skill_id_match:
            node.skill_id = skill_id_match.group(1)

        # Find call_skill invocations
        call_pattern = re.compile(r'call_skill\(\s*["\'](\w+)["\']\s*,\s*["\'](\w+)["\']')
        for match in call_pattern.finditer(source):
            node.calls.append({"skill_id": match.group(1), "action": match.group(2)})

        # Find imports from sibling modules
        import_pattern = re.compile(r'from\s+\.(\w+)\s+import')
        for match in import_pattern.finditer(source):
            module_name = match.group(1)
            if module_name not in node.imports:
                node.imports.append(module_name)

        # Check for SkillContext usage
        node.uses_context = "self.context" in source

        return node

    def _build_edges(self) -> None:
        """Build dependency edges from analyzed nodes."""
        self._edges = []

        # Build skill_id -> file_name mapping
        id_to_file = {}
        for name, node in self._nodes.items():
            if node.skill_id:
                id_to_file[node.skill_id] = name

        # Also build module_name -> file_name mapping
        module_to_file = {}
        for name in self._nodes:
            module_name = name.replace(".py", "")
            module_to_file[module_name] = name

        for name, node in self._nodes.items():
            # call_skill edges
            for call in node.calls:
                target_file = id_to_file.get(call["skill_id"])
                if target_file:
                    self._edges.append(DependencyEdge(
                        source=name,
                        target=target_file,
                        edge_type="call_skill",
                        action=call["action"],
                    ))

            # import edges (skip "base" imports)
            for imp in node.imports:
                if imp == "base":
                    continue
                target_file = module_to_file.get(imp)
                if target_file:
                    self._edges.append(DependencyEdge(
                        source=name,
                        target=target_file,
                        edge_type="import",
                    ))

    def get_nodes(self) -> Dict[str, SkillNode]:
        if not self._analyzed:
            self.analyze_all()
        return self._nodes

    def get_edges(self) -> List[DependencyEdge]:
        if not self._analyzed:
            self.analyze_all()
        return self._edges

    def get_dependents(self, skill_file: str) -> List[str]:
        """Get skills that depend ON this skill (reverse deps)."""
        if not self._analyzed:
            self.analyze_all()
        return list(set(
            e.source for e in self._edges if e.target == skill_file
        ))

    def get_dependencies(self, skill_file: str) -> List[str]:
        """Get skills that this skill depends on."""
        if not self._analyzed:
            self.analyze_all()
        return list(set(
            e.target for e in self._edges if e.source == skill_file
        ))

    def find_circular(self) -> List[List[str]]:
        """Find circular dependency chains."""
        if not self._analyzed:
            self.analyze_all()

        # Build adjacency list
        adj: Dict[str, Set[str]] = defaultdict(set)
        for edge in self._edges:
            adj[edge.source].add(edge.target)

        # DFS cycle detection
        cycles = []
        visited: Set[str] = set()
        rec_stack: Set[str] = set()

        def _dfs(node: str, path: List[str]) -> None:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in adj.get(node, set()):
                if neighbor not in visited:
                    _dfs(neighbor, path)
                elif neighbor in rec_stack:
                    # Found cycle
                    cycle_start = path.index(neighbor)
                    cycle = path[cycle_start:] + [neighbor]
                    # Normalize cycle (start from smallest element)
                    min_idx = cycle.index(min(cycle[:-1]))
                    normalized = cycle[min_idx:-1] + cycle[:min_idx] + [cycle[min_idx]]
                    if normalized not in cycles:
                        cycles.append(normalized)

            path.pop()
            rec_stack.discard(node)

        for node in self._nodes:
            if node not in visited:
                _dfs(node, [])

        return cycles

    def find_orphans(self) -> List[str]:
        """Find skills with no incoming or outgoing dependency edges."""
        if not self._analyzed:
            self.analyze_all()

        connected = set()
        for edge in self._edges:
            connected.add(edge.source)
            connected.add(edge.target)

        return sorted(set(self._nodes.keys()) - connected)

    def find_hubs(self, min_connections: int = 3) -> List[Dict[str, Any]]:
        """Find skills that are dependency hubs (many connections)."""
        if not self._analyzed:
            self.analyze_all()

        in_degree: Dict[str, int] = defaultdict(int)
        out_degree: Dict[str, int] = defaultdict(int)

        for edge in self._edges:
            out_degree[edge.source] += 1
            in_degree[edge.target] += 1

        hubs = []
        for name in self._nodes:
            total = in_degree.get(name, 0) + out_degree.get(name, 0)
            if total >= min_connections:
                hubs.append({
                    "file": name,
                    "skill_id": self._nodes[name].skill_id,
                    "in_degree": in_degree.get(name, 0),
                    "out_degree": out_degree.get(name, 0),
                    "total_connections": total,
                })

        hubs.sort(key=lambda h: h["total_connections"], reverse=True)
        return hubs

    def impact_analysis(self, skill_file: str) -> Dict[str, Any]:
        """Analyze the impact of changing a skill (transitive dependents)."""
        if not self._analyzed:
            self.analyze_all()

        # BFS to find all transitive dependents
        adj_reverse: Dict[str, Set[str]] = defaultdict(set)
        for edge in self._edges:
            adj_reverse[edge.target].add(edge.source)

        affected: Set[str] = set()
        queue = [skill_file]
        while queue:
            current = queue.pop(0)
            for dependent in adj_reverse.get(current, set()):
                if dependent not in affected:
                    affected.add(dependent)
                    queue.append(dependent)

        # Direct vs transitive
        direct = set(adj_reverse.get(skill_file, set()))
        transitive = affected - direct

        return {
            "skill": skill_file,
            "direct_dependents": sorted(direct),
            "transitive_dependents": sorted(transitive),
            "total_affected": len(affected),
            "impact_level": "critical" if len(affected) >= 5 else "moderate" if len(affected) >= 2 else "low",
        }


class SkillDependencyMapSkill(Skill):
    """
    Dependency analysis and visualization for the Singularity skill ecosystem.

    Maps relationships between 120+ skills to understand impact, find orphans,
    detect circular dependencies, and identify critical hub skills.
    """

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self._analyzer = DependencyAnalyzer()

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="skill_dependency_map",
            name="Skill Dependency Map",
            version="1.0.0",
            category="development",
            description="Dependency analysis and visualization for the skill ecosystem",
            required_credentials=[],
            actions=[
                SkillAction(
                    name="map",
                    description="Generate complete dependency map of all skills",
                    parameters={
                        "include_orphans": {"type": "bool", "required": False,
                                            "description": "Include orphan skills (default: true)"},
                    },
                    estimated_cost=0,
                    estimated_duration_seconds=5,
                ),
                SkillAction(
                    name="impact",
                    description="Analyze impact of changing a specific skill",
                    parameters={
                        "skill_file": {"type": "string", "required": True,
                                       "description": "Skill filename (e.g., 'event.py')"},
                    },
                    estimated_cost=0,
                    estimated_duration_seconds=2,
                ),
                SkillAction(
                    name="cycles",
                    description="Detect circular dependencies in the skill graph",
                    parameters={},
                    estimated_cost=0,
                    estimated_duration_seconds=3,
                ),
                SkillAction(
                    name="orphans",
                    description="Find isolated skills with no dependencies",
                    parameters={},
                    estimated_cost=0,
                    estimated_duration_seconds=2,
                ),
                SkillAction(
                    name="hubs",
                    description="Find critical hub skills with many connections",
                    parameters={
                        "min_connections": {"type": "int", "required": False,
                                            "description": "Minimum connections to qualify (default: 3)"},
                    },
                    estimated_cost=0,
                    estimated_duration_seconds=2,
                ),
                SkillAction(
                    name="deps",
                    description="Get dependencies and dependents for a specific skill",
                    parameters={
                        "skill_file": {"type": "string", "required": True,
                                       "description": "Skill filename"},
                    },
                    estimated_cost=0,
                    estimated_duration_seconds=1,
                ),
                SkillAction(
                    name="stats",
                    description="Get aggregate dependency statistics for the ecosystem",
                    parameters={},
                    estimated_cost=0,
                    estimated_duration_seconds=3,
                ),
            ],
        )

    async def execute(self, action: str, params: Dict) -> SkillResult:
        try:
            if action == "map":
                return self._map(params)
            elif action == "impact":
                return self._impact(params)
            elif action == "cycles":
                return self._cycles(params)
            elif action == "orphans":
                return self._orphans(params)
            elif action == "hubs":
                return self._hubs(params)
            elif action == "deps":
                return self._deps(params)
            elif action == "stats":
                return self._stats(params)
            else:
                return SkillResult(success=False, message=f"Unknown action: {action}")
        except Exception as e:
            return SkillResult(success=False, message=f"Error: {e}")

    def _resolve_file(self, name: str) -> str:
        """Resolve skill name to filename."""
        if name.endswith(".py"):
            return name
        return f"{name}.py"

    def _map(self, params: Dict) -> SkillResult:
        """Generate complete dependency map."""
        self._analyzer.analyze_all()
        nodes = self._analyzer.get_nodes()
        edges = self._analyzer.get_edges()
        include_orphans = params.get("include_orphans", True)

        # Build node list
        node_list = []
        for name, node in sorted(nodes.items()):
            nd = node.to_dict()
            if not include_orphans and nd["dependency_count"] == 0:
                deps_of = self._analyzer.get_dependents(name)
                if not deps_of:
                    continue
            node_list.append(nd)

        # Build edge list
        edge_list = [e.to_dict() for e in edges]

        # Summary
        connected = set()
        for e in edges:
            connected.add(e.source)
            connected.add(e.target)

        return SkillResult(
            success=True,
            message=f"Dependency map: {len(nodes)} skills, {len(edges)} edges, "
                    f"{len(connected)} connected, {len(nodes) - len(connected)} orphans",
            data={
                "total_skills": len(nodes),
                "total_edges": len(edges),
                "connected_skills": len(connected),
                "orphan_count": len(nodes) - len(connected),
                "nodes": node_list,
                "edges": edge_list,
            },
        )

    def _impact(self, params: Dict) -> SkillResult:
        """Analyze impact of changing a skill."""
        skill_file = params.get("skill_file", "")
        if not skill_file:
            return SkillResult(success=False, message="skill_file parameter required")

        skill_file = self._resolve_file(skill_file)
        self._analyzer.analyze_all()

        result = self._analyzer.impact_analysis(skill_file)
        return SkillResult(
            success=True,
            message=f"Impact of {skill_file}: {result['total_affected']} affected skills "
                    f"({result['impact_level']}). Direct: {len(result['direct_dependents'])}, "
                    f"Transitive: {len(result['transitive_dependents'])}",
            data=result,
        )

    def _cycles(self, params: Dict) -> SkillResult:
        """Detect circular dependencies."""
        self._analyzer.analyze_all()
        cycles = self._analyzer.find_circular()

        return SkillResult(
            success=True,
            message=f"Found {len(cycles)} circular dependency chain(s)" if cycles
                    else "No circular dependencies found",
            data={
                "cycle_count": len(cycles),
                "cycles": [{"chain": c, "length": len(c) - 1} for c in cycles],
            },
        )

    def _orphans(self, params: Dict) -> SkillResult:
        """Find isolated skills."""
        self._analyzer.analyze_all()
        orphans = self._analyzer.find_orphans()
        nodes = self._analyzer.get_nodes()

        orphan_details = []
        for o in orphans:
            node = nodes.get(o)
            if node:
                orphan_details.append({
                    "file": o,
                    "skill_id": node.skill_id,
                    "lines": node.lines,
                    "uses_context": node.uses_context,
                })

        return SkillResult(
            success=True,
            message=f"Found {len(orphans)} orphan skills (no incoming or outgoing dependencies)",
            data={
                "orphan_count": len(orphans),
                "total_skills": len(nodes),
                "orphan_pct": round(len(orphans) / max(len(nodes), 1) * 100, 1),
                "orphans": orphan_details,
            },
        )

    def _hubs(self, params: Dict) -> SkillResult:
        """Find critical hub skills."""
        min_connections = params.get("min_connections", 3)
        self._analyzer.analyze_all()
        hubs = self._analyzer.find_hubs(min_connections=int(min_connections))

        return SkillResult(
            success=True,
            message=f"Found {len(hubs)} hub skills with {min_connections}+ connections"
                    if hubs else f"No skills with {min_connections}+ connections",
            data={
                "hub_count": len(hubs),
                "min_connections": min_connections,
                "hubs": hubs,
            },
        )

    def _deps(self, params: Dict) -> SkillResult:
        """Get deps and dependents for one skill."""
        skill_file = params.get("skill_file", "")
        if not skill_file:
            return SkillResult(success=False, message="skill_file parameter required")

        skill_file = self._resolve_file(skill_file)
        self._analyzer.analyze_all()

        dependencies = self._analyzer.get_dependencies(skill_file)
        dependents = self._analyzer.get_dependents(skill_file)
        node = self._analyzer.get_nodes().get(skill_file)

        return SkillResult(
            success=True,
            message=f"{skill_file}: depends on {len(dependencies)} skills, "
                    f"{len(dependents)} skills depend on it",
            data={
                "skill_file": skill_file,
                "skill_id": node.skill_id if node else "",
                "depends_on": sorted(dependencies),
                "depended_by": sorted(dependents),
                "calls": node.calls if node else [],
                "imports": [i for i in (node.imports if node else []) if i != "base"],
            },
        )

    def _stats(self, params: Dict) -> SkillResult:
        """Get aggregate dependency statistics."""
        self._analyzer.analyze_all()
        nodes = self._analyzer.get_nodes()
        edges = self._analyzer.get_edges()
        orphans = self._analyzer.find_orphans()
        cycles = self._analyzer.find_circular()
        hubs = self._analyzer.find_hubs(min_connections=3)

        # Edge type breakdown
        edge_types: Dict[str, int] = defaultdict(int)
        for e in edges:
            edge_types[e.edge_type] += 1

        # Connectivity stats
        in_degrees = defaultdict(int)
        out_degrees = defaultdict(int)
        for e in edges:
            in_degrees[e.target] += 1
            out_degrees[e.source] += 1

        context_users = sum(1 for n in nodes.values() if n.uses_context)

        # Most depended-on skills
        most_depended = sorted(in_degrees.items(), key=lambda x: -x[1])[:10]

        # Most dependent skills
        most_dependent = sorted(out_degrees.items(), key=lambda x: -x[1])[:10]

        return SkillResult(
            success=True,
            message=f"Ecosystem: {len(nodes)} skills, {len(edges)} edges, "
                    f"{len(orphans)} orphans, {len(cycles)} cycles, {len(hubs)} hubs. "
                    f"{context_users} skills use SkillContext.",
            data={
                "total_skills": len(nodes),
                "total_edges": len(edges),
                "edge_types": dict(edge_types),
                "orphan_count": len(orphans),
                "orphan_pct": round(len(orphans) / max(len(nodes), 1) * 100, 1),
                "cycle_count": len(cycles),
                "hub_count": len(hubs),
                "context_users": context_users,
                "context_pct": round(context_users / max(len(nodes), 1) * 100, 1),
                "most_depended_on": [{"file": f, "in_degree": d} for f, d in most_depended],
                "most_dependent": [{"file": f, "out_degree": d} for f, d in most_dependent],
                "avg_dependencies": round(
                    sum(out_degrees.values()) / max(len(nodes), 1), 2
                ),
            },
        )
