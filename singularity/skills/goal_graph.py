#!/usr/bin/env python3
"""
GoalDependencyGraphSkill - Understand goal relationships, ordering, and impact.

While GoalManager tracks goals with flat dependency lists, this skill provides
graph-level intelligence: topological ordering, critical path analysis, cycle
detection, impact analysis, and parallel path identification.

This enables agents to:
  1. Understand which goals unlock the most downstream work (impact analysis)
  2. Find the optimal execution order (topological sort with priority weighting)
  3. Detect and break circular dependencies before they block progress
  4. Identify independent goal chains that can run in parallel (replication!)
  5. Compute the critical path - the longest chain determining minimum completion time
  6. Auto-cascade status changes when goals complete or get blocked

Pillar: Goal Setting (primary), Replication (parallel path → delegate to replicas)
"""

import json
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple

from .base import Skill, SkillResult, SkillManifest, SkillAction

GOALS_FILE = Path(__file__).parent.parent / "data" / "goals.json"
GRAPH_DATA_FILE = Path(__file__).parent.parent / "data" / "goal_graph.json"


class GoalDependencyGraphSkill(Skill):
    """
    Graph-level intelligence over the agent's goal dependency structure.

    Provides topological ordering, critical path analysis, cycle detection,
    impact analysis, parallel path identification, and cascade propagation.
    """

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="goal_dependency_graph",
            name="Goal Dependency Graph",
            version="1.0.0",
            category="goal_setting",
            description="Graph analysis of goal dependencies: ordering, critical path, cycles, impact, parallel paths",
            actions=[
                SkillAction(
                    name="analyze",
                    description="Build and analyze the full goal dependency graph. Returns stats, layers, and any issues.",
                    parameters={},
                    estimated_cost=0.001,
                ),
                SkillAction(
                    name="topological_order",
                    description="Return goals in optimal execution order (respecting dependencies, weighted by priority).",
                    parameters={
                        "pillar": {"type": "string", "required": False, "description": "Filter to goals in this pillar"},
                    },
                    estimated_cost=0.001,
                ),
                SkillAction(
                    name="critical_path",
                    description="Find the critical path - the longest dependency chain determining minimum time to complete all goals.",
                    parameters={},
                    estimated_cost=0.001,
                ),
                SkillAction(
                    name="detect_cycles",
                    description="Find circular dependencies that would block goal completion.",
                    parameters={},
                    estimated_cost=0.001,
                ),
                SkillAction(
                    name="impact_analysis",
                    description="Analyze what gets unblocked if a specific goal is completed.",
                    parameters={
                        "goal_id": {"type": "string", "required": True, "description": "Goal ID to analyze impact for"},
                    },
                    estimated_cost=0.001,
                ),
                SkillAction(
                    name="parallel_paths",
                    description="Identify independent goal chains that can execute concurrently (useful for delegation to replicas).",
                    parameters={},
                    estimated_cost=0.001,
                ),
                SkillAction(
                    name="cascade_complete",
                    description="Mark a goal complete and propagate: unblock dependents, auto-activate newly ready goals.",
                    parameters={
                        "goal_id": {"type": "string", "required": True, "description": "Goal ID that was completed"},
                    },
                    estimated_cost=0.001,
                ),
                SkillAction(
                    name="suggest_next",
                    description="Recommend the highest-impact goal to work on next based on graph analysis.",
                    parameters={
                        "max_results": {"type": "integer", "required": False, "description": "Max suggestions (default 3)"},
                    },
                    estimated_cost=0.001,
                ),
            ],
            required_credentials=[],
        )

    def __init__(self):
        super().__init__()
        self._graph_cache = None
        self._cache_timestamp = None

    def _load_goals(self) -> Dict[str, Any]:
        """Load goals from GoalManager's data file."""
        if not GOALS_FILE.exists():
            return {}
        try:
            data = json.loads(GOALS_FILE.read_text())
            goals = data.get("goals", {})
            return {gid: g for gid, g in goals.items() if g.get("status") in ("active", "blocked")}
        except (json.JSONDecodeError, KeyError):
            return {}

    def _save_goals(self, all_goals: Dict[str, Any]):
        """Save updated goals back to GoalManager's data file."""
        if not GOALS_FILE.exists():
            return
        try:
            data = json.loads(GOALS_FILE.read_text())
            data["goals"].update(all_goals)
            GOALS_FILE.write_text(json.dumps(data, indent=2, default=str))
        except (json.JSONDecodeError, KeyError):
            pass

    def _save_graph_data(self, analysis: Dict[str, Any]):
        """Persist graph analysis results."""
        GRAPH_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
        history = []
        if GRAPH_DATA_FILE.exists():
            try:
                existing = json.loads(GRAPH_DATA_FILE.read_text())
                history = existing.get("history", [])[-49:]  # Keep last 50
            except (json.JSONDecodeError, KeyError):
                pass
        history.append({
            "timestamp": datetime.utcnow().isoformat(),
            "analysis": analysis,
        })
        GRAPH_DATA_FILE.write_text(json.dumps({"history": history}, indent=2, default=str))

    def _build_graph(self, goals: Dict[str, Any]) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
        """
        Build adjacency lists for the goal dependency graph.

        Returns:
            (forward_edges, reverse_edges)
            forward_edges[A] = [B, C] means A must be done before B and C
            reverse_edges[B] = [A] means B depends on A
        """
        forward = defaultdict(list)  # goal -> list of goals it unblocks
        reverse = defaultdict(list)  # goal -> list of goals it depends on

        goal_ids = set(goals.keys())
        for gid, goal in goals.items():
            deps = goal.get("depends_on", [])
            for dep in deps:
                if dep in goal_ids:
                    forward[dep].append(gid)
                    reverse[gid].append(dep)

        return dict(forward), dict(reverse)

    def _detect_cycles(self, goals: Dict[str, Any]) -> List[List[str]]:
        """Detect all cycles using DFS-based cycle detection."""
        forward, reverse = self._build_graph(goals)
        goal_ids = list(goals.keys())

        WHITE, GRAY, BLACK = 0, 1, 2
        color = {gid: WHITE for gid in goal_ids}
        parent = {}
        cycles = []

        def dfs(node, path):
            color[node] = GRAY
            for dep in goals.get(node, {}).get("depends_on", []):
                if dep not in color:
                    continue
                if color[dep] == GRAY:
                    # Found a cycle - extract it
                    cycle_start = path.index(dep) if dep in path else -1
                    if cycle_start >= 0:
                        cycle = path[cycle_start:] + [dep]
                        cycles.append(cycle)
                elif color[dep] == WHITE:
                    dfs(dep, path + [dep])
            color[node] = BLACK

        for gid in goal_ids:
            if color[gid] == WHITE:
                dfs(gid, [gid])

        return cycles

    def _topological_sort(self, goals: Dict[str, Any], pillar: Optional[str] = None) -> List[str]:
        """
        Kahn's algorithm for topological sort with priority-weighted tie-breaking.
        Returns goal IDs in optimal execution order.
        """
        if pillar:
            goals = {gid: g for gid, g in goals.items() if g.get("pillar") == pillar}

        goal_ids = set(goals.keys())
        in_degree = {gid: 0 for gid in goal_ids}

        for gid, goal in goals.items():
            for dep in goal.get("depends_on", []):
                if dep in goal_ids:
                    in_degree[gid] = in_degree.get(gid, 0) + 1

        priority_map = {"critical": 4, "high": 3, "medium": 2, "low": 1}

        # Start with nodes that have no dependencies
        queue = []
        for gid in goal_ids:
            if in_degree[gid] == 0:
                priority = priority_map.get(goals[gid].get("priority", "medium"), 2)
                queue.append((-priority, gid))
        queue.sort()

        result = []
        while queue:
            _, gid = queue.pop(0)
            result.append(gid)

            # Find goals that depend on this one
            for other_id, other_goal in goals.items():
                if gid in other_goal.get("depends_on", []):
                    in_degree[other_id] -= 1
                    if in_degree[other_id] == 0:
                        priority = priority_map.get(other_goal.get("priority", "medium"), 2)
                        queue.append((-priority, other_id))
                        queue.sort()

        return result

    def _find_critical_path(self, goals: Dict[str, Any]) -> List[str]:
        """
        Find the longest dependency chain (critical path).
        Uses dynamic programming on the DAG.
        """
        order = self._topological_sort(goals)
        if not order:
            return []

        goal_ids = set(goals.keys())
        # dist[gid] = (length of longest path ending at gid, predecessor)
        dist = {gid: (1, None) for gid in goal_ids}

        for gid in order:
            for other_id, other_goal in goals.items():
                if gid in other_goal.get("depends_on", []):
                    new_len = dist[gid][0] + 1
                    if new_len > dist[other_id][0]:
                        dist[other_id] = (new_len, gid)

        # Find the endpoint with longest path
        if not dist:
            return []

        end_id = max(dist, key=lambda x: dist[x][0])
        path = []
        current = end_id
        while current is not None:
            path.append(current)
            current = dist[current][1]

        path.reverse()
        return path

    def _find_parallel_paths(self, goals: Dict[str, Any]) -> List[List[str]]:
        """
        Find independent goal chains that can execute concurrently.
        Uses connected components on the undirected version of the dependency graph.
        """
        goal_ids = set(goals.keys())
        if not goal_ids:
            return []

        # Build undirected adjacency
        adj = defaultdict(set)
        for gid, goal in goals.items():
            for dep in goal.get("depends_on", []):
                if dep in goal_ids:
                    adj[gid].add(dep)
                    adj[dep].add(gid)

        # Find connected components via BFS
        visited = set()
        components = []

        for gid in goal_ids:
            if gid in visited:
                continue
            component = []
            queue = deque([gid])
            while queue:
                node = queue.popleft()
                if node in visited:
                    continue
                visited.add(node)
                component.append(node)
                for neighbor in adj.get(node, set()):
                    if neighbor not in visited:
                        queue.append(neighbor)
            if component:
                components.append(sorted(component))

        return sorted(components, key=len, reverse=True)

    def _compute_impact(self, goal_id: str, goals: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute the downstream impact of completing a goal.
        Returns all unique goals that would be directly or transitively unblocked.
        """
        forward, _ = self._build_graph(goals)
        goal_ids = set(goals.keys())

        # BFS from goal_id through forward edges
        directly_unblocked = []
        transitively_unblocked = []
        visited = set()
        recorded = set()  # Track goals already added to result lists
        queue = deque([goal_id])
        depth = {goal_id: 0}

        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)

            for dependent in forward.get(current, []):
                if dependent not in visited:
                    d = depth[current] + 1
                    if dependent not in depth:
                        depth[dependent] = d
                    else:
                        depth[dependent] = min(depth[dependent], d)
                    queue.append(dependent)

                    # Only record each dependent once
                    if dependent in recorded:
                        continue
                    recorded.add(dependent)

                    # Check if this goal would be fully unblocked
                    deps = set(goals.get(dependent, {}).get("depends_on", []))
                    remaining = deps.intersection(goal_ids) - {goal_id}
                    would_unblock = len(remaining) == 0

                    info = {
                        "goal_id": dependent,
                        "title": goals.get(dependent, {}).get("title", "Unknown"),
                        "depth": depth[dependent],
                        "fully_unblocked": would_unblock,
                        "remaining_blockers": list(remaining),
                    }
                    if depth[dependent] == 1:
                        directly_unblocked.append(info)
                    else:
                        transitively_unblocked.append(info)

        return {
            "goal_id": goal_id,
            "title": goals.get(goal_id, {}).get("title", "Unknown"),
            "directly_unblocked": directly_unblocked,
            "transitively_unblocked": transitively_unblocked,
            "total_downstream": len(directly_unblocked) + len(transitively_unblocked),
            "fully_unblocked_count": sum(
                1 for g in directly_unblocked + transitively_unblocked if g["fully_unblocked"]
            ),
        }

    def _suggest_next(self, goals: Dict[str, Any], max_results: int = 3) -> List[Dict[str, Any]]:
        """
        Recommend goals to work on based on graph analysis.
        Score = priority_weight * (1 + downstream_impact) * readiness
        """
        goal_ids = set(goals.keys())
        priority_map = {"critical": 4, "high": 3, "medium": 2, "low": 1}
        forward, reverse = self._build_graph(goals)

        suggestions = []
        for gid, goal in goals.items():
            # Check if all dependencies are met (goal is ready)
            deps = set(goal.get("depends_on", []))
            active_deps = deps.intersection(goal_ids)
            if active_deps:
                continue  # Still blocked

            # Compute score
            priority_weight = priority_map.get(goal.get("priority", "medium"), 2)
            impact = self._compute_impact(gid, goals)
            downstream_count = impact["total_downstream"]
            unblock_count = impact["fully_unblocked_count"]

            score = priority_weight * (1 + downstream_count) * (1 + unblock_count * 0.5)

            suggestions.append({
                "goal_id": gid,
                "title": goal.get("title", "Unknown"),
                "pillar": goal.get("pillar", "other"),
                "priority": goal.get("priority", "medium"),
                "score": round(score, 2),
                "downstream_goals": downstream_count,
                "would_fully_unblock": unblock_count,
                "reason": self._explain_suggestion(goal, downstream_count, unblock_count),
            })

        suggestions.sort(key=lambda x: x["score"], reverse=True)
        return suggestions[:max_results]

    def _explain_suggestion(self, goal: Dict, downstream: int, unblocks: int) -> str:
        """Generate a human-readable explanation for why a goal is suggested."""
        parts = []
        priority = goal.get("priority", "medium")
        if priority in ("critical", "high"):
            parts.append(f"{priority} priority")
        if downstream > 0:
            parts.append(f"unblocks {downstream} downstream goal{'s' if downstream != 1 else ''}")
        if unblocks > 0:
            parts.append(f"fully enables {unblocks} goal{'s' if unblocks != 1 else ''}")
        if not parts:
            parts.append("ready to execute with no blockers")
        return "; ".join(parts)

    async def execute(self, action: str, params: Dict[str, Any] = None) -> SkillResult:
        params = params or {}

        if action == "analyze":
            return self._action_analyze()
        elif action == "topological_order":
            return self._action_topological_order(params.get("pillar"))
        elif action == "critical_path":
            return self._action_critical_path()
        elif action == "detect_cycles":
            return self._action_detect_cycles()
        elif action == "impact_analysis":
            goal_id = params.get("goal_id")
            if not goal_id:
                return SkillResult(success=False, message="goal_id is required")
            return self._action_impact_analysis(goal_id)
        elif action == "parallel_paths":
            return self._action_parallel_paths()
        elif action == "cascade_complete":
            goal_id = params.get("goal_id")
            if not goal_id:
                return SkillResult(success=False, message="goal_id is required")
            return self._action_cascade_complete(goal_id)
        elif action == "suggest_next":
            max_results = params.get("max_results", 3)
            return self._action_suggest_next(max_results)
        else:
            return SkillResult(success=False, message=f"Unknown action: {action}")

    def _action_analyze(self) -> SkillResult:
        goals = self._load_goals()
        if not goals:
            return SkillResult(success=True, message="No active goals to analyze.", data={"total_goals": 0})

        forward, reverse = self._build_graph(goals)
        cycles = self._detect_cycles(goals)
        order = self._topological_sort(goals)
        critical = self._find_critical_path(goals)
        parallel = self._find_parallel_paths(goals)

        # Compute stats
        total_edges = sum(len(deps) for deps in forward.values())
        root_goals = [gid for gid in goals if not goals[gid].get("depends_on")]
        leaf_goals = [gid for gid in goals if gid not in forward]
        blocked = [gid for gid, g in goals.items() if g.get("status") == "blocked"]

        # Goals ready to execute (no unmet deps)
        goal_ids = set(goals.keys())
        ready = []
        for gid, g in goals.items():
            deps = set(g.get("depends_on", []))
            if not deps.intersection(goal_ids):
                ready.append(gid)

        analysis = {
            "total_goals": len(goals),
            "total_dependencies": total_edges,
            "root_goals": len(root_goals),
            "leaf_goals": len(leaf_goals),
            "ready_to_execute": len(ready),
            "blocked_goals": len(blocked),
            "has_cycles": len(cycles) > 0,
            "cycle_count": len(cycles),
            "critical_path_length": len(critical),
            "parallel_chains": len(parallel),
            "max_chain_length": max((len(c) for c in parallel), default=0),
        }

        self._save_graph_data(analysis)

        return SkillResult(
            success=True,
            message=f"Graph: {len(goals)} goals, {total_edges} deps, {len(ready)} ready, {len(parallel)} parallel chains, critical path={len(critical)}",
            data={
                "stats": analysis,
                "ready_goals": [{"id": gid, "title": goals[gid].get("title", "")} for gid in ready],
                "root_goals": [{"id": gid, "title": goals[gid].get("title", "")} for gid in root_goals],
                "cycles": cycles,
            },
        )

    def _action_topological_order(self, pillar: Optional[str] = None) -> SkillResult:
        goals = self._load_goals()
        if not goals:
            return SkillResult(success=True, message="No active goals.", data={"order": []})

        cycles = self._detect_cycles(goals)
        if cycles:
            return SkillResult(
                success=False,
                message=f"Cannot compute order: {len(cycles)} cycle(s) detected. Run detect_cycles for details.",
                data={"cycles": cycles},
            )

        order = self._topological_sort(goals, pillar)
        ordered_goals = []
        for i, gid in enumerate(order):
            g = goals.get(gid, {})
            ordered_goals.append({
                "position": i + 1,
                "goal_id": gid,
                "title": g.get("title", "Unknown"),
                "priority": g.get("priority", "medium"),
                "pillar": g.get("pillar", "other"),
                "depends_on": g.get("depends_on", []),
            })

        label = f" (pillar={pillar})" if pillar else ""
        return SkillResult(
            success=True,
            message=f"Execution order for {len(ordered_goals)} goals{label}",
            data={"order": ordered_goals},
        )

    def _action_critical_path(self) -> SkillResult:
        goals = self._load_goals()
        if not goals:
            return SkillResult(success=True, message="No active goals.", data={"path": []})

        path = self._find_critical_path(goals)
        path_goals = []
        for i, gid in enumerate(path):
            g = goals.get(gid, {})
            path_goals.append({
                "step": i + 1,
                "goal_id": gid,
                "title": g.get("title", "Unknown"),
                "priority": g.get("priority", "medium"),
                "pillar": g.get("pillar", "other"),
            })

        return SkillResult(
            success=True,
            message=f"Critical path: {len(path)} goals in the longest dependency chain",
            data={
                "path": path_goals,
                "length": len(path),
                "bottleneck": path_goals[0] if path_goals else None,
            },
        )

    def _action_detect_cycles(self) -> SkillResult:
        goals = self._load_goals()
        if not goals:
            return SkillResult(success=True, message="No active goals.", data={"cycles": []})

        cycles = self._detect_cycles(goals)
        if not cycles:
            return SkillResult(
                success=True,
                message="No dependency cycles found. Goal graph is a valid DAG.",
                data={"cycles": [], "is_dag": True},
            )

        cycle_details = []
        for cycle in cycles:
            cycle_details.append({
                "cycle": cycle,
                "goals": [
                    {"id": gid, "title": goals.get(gid, {}).get("title", "Unknown")}
                    for gid in cycle if gid in goals
                ],
                "suggested_break": cycle[-2] if len(cycle) >= 2 else cycle[0],
            })

        return SkillResult(
            success=False,
            message=f"Found {len(cycles)} dependency cycle(s). These must be broken for goals to be completable.",
            data={"cycles": cycle_details, "is_dag": False},
        )

    def _action_impact_analysis(self, goal_id: str) -> SkillResult:
        goals = self._load_goals()
        if goal_id not in goals:
            return SkillResult(success=False, message=f"Goal {goal_id} not found in active goals.")

        impact = self._compute_impact(goal_id, goals)
        return SkillResult(
            success=True,
            message=f"Completing '{impact['title']}' would affect {impact['total_downstream']} downstream goals, fully unblocking {impact['fully_unblocked_count']}",
            data=impact,
        )

    def _action_parallel_paths(self) -> SkillResult:
        goals = self._load_goals()
        if not goals:
            return SkillResult(success=True, message="No active goals.", data={"paths": []})

        components = self._find_parallel_paths(goals)
        paths = []
        for i, component in enumerate(components):
            path_goals = []
            for gid in component:
                g = goals.get(gid, {})
                path_goals.append({
                    "goal_id": gid,
                    "title": g.get("title", "Unknown"),
                    "priority": g.get("priority", "medium"),
                    "pillar": g.get("pillar", "other"),
                })
            paths.append({
                "chain_id": i + 1,
                "size": len(component),
                "goals": path_goals,
                "pillars": list(set(g.get("pillar", "other") for gid in component for g in [goals.get(gid, {})])),
            })

        delegatable = [p for p in paths if p["size"] >= 2]
        return SkillResult(
            success=True,
            message=f"Found {len(paths)} independent chains. {len(delegatable)} are large enough to delegate to replicas.",
            data={
                "paths": paths,
                "total_chains": len(paths),
                "delegatable_chains": len(delegatable),
                "concurrency_potential": len(paths),
            },
        )

    def _action_cascade_complete(self, goal_id: str) -> SkillResult:
        """Mark a goal complete and cascade effects through the graph."""
        if not GOALS_FILE.exists():
            return SkillResult(success=False, message="No goals file found.")

        try:
            data = json.loads(GOALS_FILE.read_text())
        except (json.JSONDecodeError, KeyError):
            return SkillResult(success=False, message="Could not read goals file.")

        all_goals = data.get("goals", {})
        if goal_id not in all_goals:
            return SkillResult(success=False, message=f"Goal {goal_id} not found.")

        goal = all_goals[goal_id]
        if goal.get("status") == "completed":
            return SkillResult(success=False, message=f"Goal {goal_id} is already completed.")

        # Mark complete
        goal["status"] = "completed"
        goal["completed_at"] = datetime.utcnow().isoformat()

        # Find goals that depended on this one
        unblocked = []
        activated = []
        active_ids = {gid for gid, g in all_goals.items() if g.get("status") in ("active", "blocked")}

        for gid, g in all_goals.items():
            if gid == goal_id:
                continue
            deps = g.get("depends_on", [])
            if goal_id not in deps:
                continue

            # Check if all dependencies are now met
            remaining_deps = [d for d in deps if d in active_ids and d != goal_id]
            if not remaining_deps and g.get("status") == "blocked":
                g["status"] = "active"
                activated.append({"goal_id": gid, "title": g.get("title", "Unknown")})
            elif goal_id in deps:
                unblocked.append({"goal_id": gid, "title": g.get("title", "Unknown"), "remaining_blockers": remaining_deps})

        # Save
        data["goals"] = all_goals
        GOALS_FILE.write_text(json.dumps(data, indent=2, default=str))

        return SkillResult(
            success=True,
            message=f"Completed '{goal.get('title', goal_id)}'. Activated {len(activated)} goals, partially unblocked {len(unblocked)} goals.",
            data={
                "completed": {"goal_id": goal_id, "title": goal.get("title", "Unknown")},
                "activated": activated,
                "partially_unblocked": unblocked,
                "total_affected": len(activated) + len(unblocked),
            },
        )

    def _action_suggest_next(self, max_results: int = 3) -> SkillResult:
        goals = self._load_goals()
        if not goals:
            return SkillResult(success=True, message="No active goals to suggest.", data={"suggestions": []})

        suggestions = self._suggest_next(goals, max_results)
        if not suggestions:
            return SkillResult(
                success=True,
                message="All active goals are blocked by dependencies. Consider breaking cycles or completing blocking goals.",
                data={"suggestions": [], "all_blocked": True},
            )

        top = suggestions[0]
        return SkillResult(
            success=True,
            message=f"Top suggestion: '{top['title']}' (score={top['score']}, unblocks {top['downstream_goals']} goals). {top['reason']}",
            data={"suggestions": suggestions},
        )
