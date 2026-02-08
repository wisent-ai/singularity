#!/usr/bin/env python3
"""
GoalDependencyGraphSkill - Understand goal relationships and execution ordering.

Strengthens the Goal Setting pillar by giving agents the ability to reason about
goal dependencies as a graph. Features:
- Build and visualize the dependency graph
- Find critical paths (longest dependency chains)
- Topological sort for optimal execution order
- Cycle detection to prevent deadlocks
- Impact analysis (what gets unblocked when a goal completes)
- Bottleneck detection (goals that block the most others)
- Suggest dependencies based on pillar/priority patterns
- Dependency health metrics

Integrates with GoalManagerSkill's existing depends_on field.
"""

import json
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple

from .base import Skill, SkillResult, SkillManifest, SkillAction


GOALS_FILE = Path(__file__).parent.parent / "data" / "goals.json"


class GoalDependencyGraphSkill(Skill):
    """
    Analyze goal dependency graphs to help agents understand relationships,
    find optimal execution order, and detect issues like cycles or bottlenecks.
    """

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)

    def _load_goals(self) -> Dict:
        """Load goals data."""
        try:
            with open(GOALS_FILE, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {"goals": [], "completed_goals": []}

    def _build_graph(self, data: Dict) -> Tuple[Dict[str, Dict], Dict[str, List[str]], Dict[str, List[str]]]:
        """
        Build adjacency lists from goals data.

        Returns:
            - goals_by_id: {goal_id: goal_dict}
            - forward_deps: {goal_id: [goals it depends ON]}
            - reverse_deps: {goal_id: [goals that depend ON it]}
        """
        all_goals = list(data.get("goals", []))
        completed = list(data.get("completed_goals", []))

        goals_by_id = {}
        for g in all_goals + completed:
            goals_by_id[g["id"]] = g

        # Forward: goal -> what it depends on
        forward_deps = defaultdict(list)
        # Reverse: goal -> what depends on it (dependents)
        reverse_deps = defaultdict(list)

        for g in all_goals:
            gid = g["id"]
            for dep in g.get("depends_on", []):
                if dep in goals_by_id:
                    forward_deps[gid].append(dep)
                    reverse_deps[dep].append(gid)

        return goals_by_id, dict(forward_deps), dict(reverse_deps)

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="goal_dep_graph",
            name="Goal Dependency Graph",
            version="1.0.0",
            category="meta",
            description="Analyze goal dependency graphs for execution ordering and bottleneck detection",
            actions=[
                SkillAction(
                    name="visualize",
                    description="Get a text visualization of the goal dependency graph",
                    parameters={
                        "pillar": {"type": "string", "required": False, "description": "Filter by pillar"},
                        "include_completed": {"type": "boolean", "required": False, "description": "Include completed goals (default: false)"},
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="critical_path",
                    description="Find the longest dependency chain (critical path) that determines minimum completion time",
                    parameters={},
                    estimated_cost=0,
                ),
                SkillAction(
                    name="execution_order",
                    description="Get topologically sorted execution order respecting all dependencies",
                    parameters={
                        "pillar": {"type": "string", "required": False, "description": "Filter by pillar"},
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="detect_cycles",
                    description="Find circular dependencies that would cause deadlocks",
                    parameters={},
                    estimated_cost=0,
                ),
                SkillAction(
                    name="impact",
                    description="Analyze what gets unblocked when a specific goal is completed",
                    parameters={
                        "goal_id": {"type": "string", "required": True, "description": "Goal ID to analyze impact for"},
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="bottlenecks",
                    description="Find goals that block the most other goals (highest-leverage to complete)",
                    parameters={
                        "top_n": {"type": "integer", "required": False, "description": "Number of top bottlenecks (default: 5)"},
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="suggest_dependencies",
                    description="Suggest missing dependencies based on goal patterns (same pillar, priority ordering)",
                    parameters={},
                    estimated_cost=0,
                ),
                SkillAction(
                    name="health",
                    description="Overall dependency graph health metrics",
                    parameters={},
                    estimated_cost=0,
                ),
            ],
            required_credentials=[],
        )

    def check_credentials(self) -> bool:
        return True

    async def execute(self, action: str, params: Dict) -> SkillResult:
        handlers = {
            "visualize": self._visualize,
            "critical_path": self._critical_path,
            "execution_order": self._execution_order,
            "detect_cycles": self._detect_cycles,
            "impact": self._impact,
            "bottlenecks": self._bottlenecks,
            "suggest_dependencies": self._suggest_dependencies,
            "health": self._health,
        }
        handler = handlers.get(action)
        if handler:
            return handler(params)
        return SkillResult(success=False, message=f"Unknown action: {action}")

    def _visualize(self, params: Dict) -> SkillResult:
        """Generate a text visualization of the dependency graph."""
        data = self._load_goals()
        pillar_filter = params.get("pillar", "").strip().lower()
        include_completed = params.get("include_completed", False)

        goals_by_id, forward_deps, reverse_deps = self._build_graph(data)

        active_goals = [g for g in data.get("goals", []) if g.get("status") == "active"]
        if pillar_filter:
            active_goals = [g for g in active_goals if g.get("pillar") == pillar_filter]

        if not active_goals and not include_completed:
            return SkillResult(
                success=True,
                message="No active goals to visualize.",
                data={"graph": [], "node_count": 0, "edge_count": 0},
            )

        # Build visual representation
        nodes = []
        edges = []
        completed_ids = {g["id"] for g in data.get("completed_goals", [])}

        for g in active_goals:
            gid = g["id"]
            deps = forward_deps.get(gid, [])
            dependents = reverse_deps.get(gid, [])

            status_marker = "O"  # active
            if deps:
                unmet = [d for d in deps if d not in completed_ids]
                if unmet:
                    status_marker = "X"  # blocked

            node_info = {
                "id": gid,
                "title": g["title"],
                "pillar": g["pillar"],
                "priority": g["priority"],
                "status": "blocked" if status_marker == "X" else "actionable",
                "depends_on": deps,
                "blocks": dependents,
                "depth": self._compute_depth(gid, forward_deps, completed_ids),
            }
            nodes.append(node_info)

            for dep in deps:
                dep_title = goals_by_id.get(dep, {}).get("title", dep)
                edges.append({
                    "from": dep,
                    "from_title": dep_title,
                    "to": gid,
                    "to_title": g["title"],
                    "satisfied": dep in completed_ids,
                })

        # Sort by depth for layered display
        nodes.sort(key=lambda n: (n["depth"], n["priority"]))

        # Text representation
        lines = []
        for node in nodes:
            indent = "  " * node["depth"]
            marker = "[X]" if node["status"] == "blocked" else "[O]"
            dep_str = ""
            if node["depends_on"]:
                dep_names = [goals_by_id.get(d, {}).get("title", d)[:20] for d in node["depends_on"]]
                dep_str = f" <- {', '.join(dep_names)}"
            block_str = ""
            if node["blocks"]:
                block_names = [goals_by_id.get(b, {}).get("title", b)[:20] for b in node["blocks"]]
                block_str = f" -> {', '.join(block_names)}"

            lines.append(
                f"{indent}{marker} [{node['priority'].upper()}] {node['title']} ({node['pillar']}){dep_str}{block_str}"
            )

        text_graph = "\n".join(lines) if lines else "Empty graph"

        return SkillResult(
            success=True,
            message=f"Dependency graph: {len(nodes)} goals, {len(edges)} edges",
            data={
                "text_graph": text_graph,
                "nodes": nodes,
                "edges": edges,
                "node_count": len(nodes),
                "edge_count": len(edges),
            },
        )

    def _compute_depth(self, goal_id: str, forward_deps: Dict[str, List[str]], completed_ids: Set[str]) -> int:
        """Compute the depth (longest dependency chain) to reach this goal."""
        visited = set()

        def dfs(gid: str) -> int:
            if gid in visited:
                return 0  # Cycle protection
            visited.add(gid)
            deps = forward_deps.get(gid, [])
            if not deps:
                return 0
            max_depth = 0
            for dep in deps:
                if dep not in completed_ids:
                    max_depth = max(max_depth, 1 + dfs(dep))
            return max_depth

        return dfs(goal_id)

    def _critical_path(self, params: Dict) -> SkillResult:
        """Find the longest dependency chain (critical path)."""
        data = self._load_goals()
        goals_by_id, forward_deps, reverse_deps = self._build_graph(data)

        active_goals = [g for g in data.get("goals", []) if g.get("status") == "active"]
        completed_ids = {g["id"] for g in data.get("completed_goals", [])}

        if not active_goals:
            return SkillResult(
                success=True,
                message="No active goals - no critical path.",
                data={"critical_path": [], "length": 0},
            )

        # Find the longest path from any root to any leaf
        best_path = []
        memo = {}

        def longest_path(gid: str, visited: Set[str]) -> List[str]:
            if gid in memo:
                return memo[gid]
            if gid in visited:
                return [gid]  # Cycle
            visited.add(gid)

            deps = forward_deps.get(gid, [])
            active_deps = [d for d in deps if d not in completed_ids and d in goals_by_id]

            if not active_deps:
                result = [gid]
            else:
                best_sub = []
                for dep in active_deps:
                    sub = longest_path(dep, visited)
                    if len(sub) > len(best_sub):
                        best_sub = sub
                result = best_sub + [gid]

            visited.discard(gid)
            memo[gid] = result
            return result

        for g in active_goals:
            path = longest_path(g["id"], set())
            if len(path) > len(best_path):
                best_path = path

        # Build annotated path
        annotated = []
        for i, gid in enumerate(best_path):
            goal = goals_by_id.get(gid, {})
            annotated.append({
                "step": i + 1,
                "goal_id": gid,
                "title": goal.get("title", "Unknown"),
                "pillar": goal.get("pillar", ""),
                "priority": goal.get("priority", ""),
                "is_completed": gid in completed_ids,
            })

        return SkillResult(
            success=True,
            message=f"Critical path: {len(best_path)} goals deep",
            data={
                "critical_path": annotated,
                "length": len(best_path),
                "start": annotated[0]["title"] if annotated else None,
                "end": annotated[-1]["title"] if annotated else None,
            },
        )

    def _execution_order(self, params: Dict) -> SkillResult:
        """Topological sort of goals respecting dependencies."""
        data = self._load_goals()
        pillar_filter = params.get("pillar", "").strip().lower()

        active_goals = [g for g in data.get("goals", []) if g.get("status") == "active"]
        if pillar_filter:
            active_goals = [g for g in active_goals if g.get("pillar") == pillar_filter]

        if not active_goals:
            return SkillResult(
                success=True,
                message="No active goals to order.",
                data={"order": [], "parallel_groups": []},
            )

        completed_ids = {g["id"] for g in data.get("completed_goals", [])}
        active_ids = {g["id"] for g in active_goals}

        # Build in-degree map (only considering active, unmet dependencies)
        in_degree = defaultdict(int)
        adj = defaultdict(list)  # dep -> [goals that depend on it]

        for g in active_goals:
            gid = g["id"]
            in_degree.setdefault(gid, 0)
            for dep in g.get("depends_on", []):
                if dep in active_ids and dep not in completed_ids:
                    in_degree[gid] += 1
                    adj[dep].append(gid)

        # Kahn's algorithm with parallel grouping
        queue = deque()
        for gid in active_ids:
            if in_degree[gid] == 0:
                queue.append(gid)

        order = []
        parallel_groups = []

        while queue:
            # All items currently in queue can run in parallel
            group = list(queue)
            queue.clear()

            group_goals = []
            for gid in group:
                goal = next((g for g in active_goals if g["id"] == gid), None)
                if goal:
                    group_goals.append({
                        "goal_id": gid,
                        "title": goal["title"],
                        "pillar": goal["pillar"],
                        "priority": goal["priority"],
                    })
                    order.append(gid)

            # Sort within group by priority (highest first)
            priority_map = {"critical": 4, "high": 3, "medium": 2, "low": 1}
            group_goals.sort(key=lambda g: priority_map.get(g["priority"], 0), reverse=True)

            parallel_groups.append({
                "wave": len(parallel_groups) + 1,
                "goals": group_goals,
                "can_parallel": len(group_goals) > 1,
            })

            # Process edges
            for gid in group:
                for dependent in adj.get(gid, []):
                    in_degree[dependent] -= 1
                    if in_degree[dependent] == 0:
                        queue.append(dependent)

        # Check for unprocessed (cyclic) goals
        unprocessed = active_ids - set(order)

        return SkillResult(
            success=True,
            message=f"Execution order: {len(order)} goals in {len(parallel_groups)} waves"
            + (f" ({len(unprocessed)} cyclic)" if unprocessed else ""),
            data={
                "parallel_groups": parallel_groups,
                "total_goals": len(order),
                "total_waves": len(parallel_groups),
                "cyclic_goals": list(unprocessed),
                "has_cycles": len(unprocessed) > 0,
            },
        )

    def _detect_cycles(self, params: Dict) -> SkillResult:
        """Detect circular dependencies."""
        data = self._load_goals()
        goals_by_id, forward_deps, reverse_deps = self._build_graph(data)

        active_goals = [g for g in data.get("goals", []) if g.get("status") == "active"]
        active_ids = {g["id"] for g in active_goals}

        cycles = []
        visited = set()
        rec_stack = set()

        def dfs(gid: str, path: List[str]):
            visited.add(gid)
            rec_stack.add(gid)
            path.append(gid)

            for dep in forward_deps.get(gid, []):
                if dep not in active_ids:
                    continue
                if dep not in visited:
                    dfs(dep, path)
                elif dep in rec_stack:
                    # Found a cycle
                    cycle_start = path.index(dep)
                    cycle = path[cycle_start:] + [dep]
                    cycle_titles = [
                        goals_by_id.get(cid, {}).get("title", cid)
                        for cid in cycle
                    ]
                    cycles.append({
                        "goal_ids": cycle,
                        "titles": cycle_titles,
                        "length": len(cycle) - 1,
                    })

            path.pop()
            rec_stack.discard(gid)

        for g in active_goals:
            if g["id"] not in visited:
                dfs(g["id"], [])

        if cycles:
            return SkillResult(
                success=False,
                message=f"Found {len(cycles)} circular dependencies! These will cause deadlocks.",
                data={
                    "cycles": cycles,
                    "count": len(cycles),
                    "recommendation": "Break cycles by removing one dependency from each cycle.",
                },
            )

        return SkillResult(
            success=True,
            message="No circular dependencies detected. Graph is a valid DAG.",
            data={"cycles": [], "count": 0},
        )

    def _impact(self, params: Dict) -> SkillResult:
        """Analyze what completing a goal would unblock."""
        goal_id = params.get("goal_id", "").strip()
        if not goal_id:
            return SkillResult(success=False, message="goal_id is required")

        data = self._load_goals()
        goals_by_id, forward_deps, reverse_deps = self._build_graph(data)

        if goal_id not in goals_by_id:
            return SkillResult(success=False, message=f"Goal not found: {goal_id}")

        target = goals_by_id[goal_id]
        completed_ids = {g["id"] for g in data.get("completed_goals", [])}

        # Direct dependents (goals that list this as a dependency)
        direct_dependents = reverse_deps.get(goal_id, [])

        # Check which would become unblocked
        directly_unblocked = []
        for dep_id in direct_dependents:
            dep_goal = goals_by_id.get(dep_id)
            if not dep_goal or dep_goal.get("status") != "active":
                continue
            # Would all deps be met if we complete this goal?
            all_deps = dep_goal.get("depends_on", [])
            simulated_completed = completed_ids | {goal_id}
            unmet = [d for d in all_deps if d not in simulated_completed]
            if not unmet:
                directly_unblocked.append({
                    "goal_id": dep_id,
                    "title": dep_goal.get("title", ""),
                    "pillar": dep_goal.get("pillar", ""),
                    "priority": dep_goal.get("priority", ""),
                })

        # Cascade: what would transitively become unblocked
        cascade_unblocked = []
        simulated_completed = completed_ids | {goal_id}
        newly_available = set(d["goal_id"] for d in directly_unblocked)

        # BFS cascade
        wave = 1
        queue = list(newly_available)
        while queue:
            next_queue = []
            for avail_id in queue:
                simulated_completed.add(avail_id)
                for downstream_id in reverse_deps.get(avail_id, []):
                    if downstream_id in simulated_completed or downstream_id in newly_available:
                        continue
                    ds_goal = goals_by_id.get(downstream_id)
                    if not ds_goal or ds_goal.get("status") != "active":
                        continue
                    all_deps = ds_goal.get("depends_on", [])
                    unmet = [d for d in all_deps if d not in simulated_completed]
                    if not unmet:
                        cascade_unblocked.append({
                            "goal_id": downstream_id,
                            "title": ds_goal.get("title", ""),
                            "pillar": ds_goal.get("pillar", ""),
                            "priority": ds_goal.get("priority", ""),
                            "cascade_wave": wave,
                        })
                        newly_available.add(downstream_id)
                        next_queue.append(downstream_id)
            queue = next_queue
            wave += 1

        total_impact = len(directly_unblocked) + len(cascade_unblocked)

        return SkillResult(
            success=True,
            message=f"Completing '{target.get('title', goal_id)}' unblocks {total_impact} goals ({len(directly_unblocked)} direct, {len(cascade_unblocked)} cascade)",
            data={
                "goal_id": goal_id,
                "title": target.get("title", ""),
                "directly_unblocked": directly_unblocked,
                "cascade_unblocked": cascade_unblocked,
                "total_unblocked": total_impact,
                "impact_score": total_impact,
            },
        )

    def _bottlenecks(self, params: Dict) -> SkillResult:
        """Find goals that block the most other goals."""
        top_n = params.get("top_n", 5)

        data = self._load_goals()
        goals_by_id, forward_deps, reverse_deps = self._build_graph(data)

        active_goals = [g for g in data.get("goals", []) if g.get("status") == "active"]
        completed_ids = {g["id"] for g in data.get("completed_goals", [])}

        if not active_goals:
            return SkillResult(
                success=True,
                message="No active goals - no bottlenecks.",
                data={"bottlenecks": []},
            )

        # For each active uncompleted goal, compute total impact
        impact_scores = []
        for g in active_goals:
            gid = g["id"]
            if gid in completed_ids:
                continue

            # Compute transitive impact
            direct_dependents = reverse_deps.get(gid, [])
            active_dependents = [d for d in direct_dependents if d not in completed_ids]

            # BFS for transitive count
            transitive_count = 0
            visited = {gid}
            queue = list(active_dependents)
            while queue:
                next_q = []
                for dep_id in queue:
                    if dep_id in visited:
                        continue
                    visited.add(dep_id)
                    transitive_count += 1
                    for downstream in reverse_deps.get(dep_id, []):
                        if downstream not in visited and downstream not in completed_ids:
                            next_q.append(downstream)
                queue = next_q

            impact_scores.append({
                "goal_id": gid,
                "title": g["title"],
                "pillar": g["pillar"],
                "priority": g["priority"],
                "direct_blockers": len(active_dependents),
                "transitive_blockers": transitive_count,
                "impact_score": len(active_dependents) * 2 + transitive_count,
            })

        # Sort by impact score descending
        impact_scores.sort(key=lambda x: x["impact_score"], reverse=True)
        bottlenecks = impact_scores[:top_n]

        return SkillResult(
            success=True,
            message=f"Top {len(bottlenecks)} bottlenecks (goals blocking the most work)",
            data={
                "bottlenecks": bottlenecks,
                "total_goals_with_dependents": len([s for s in impact_scores if s["impact_score"] > 0]),
            },
        )

    def _suggest_dependencies(self, params: Dict) -> SkillResult:
        """Suggest missing dependencies based on goal patterns."""
        data = self._load_goals()
        goals_by_id, forward_deps, reverse_deps = self._build_graph(data)

        active_goals = [g for g in data.get("goals", []) if g.get("status") == "active"]

        if len(active_goals) < 2:
            return SkillResult(
                success=True,
                message="Need at least 2 active goals to suggest dependencies.",
                data={"suggestions": []},
            )

        suggestions = []
        priority_rank = {"critical": 4, "high": 3, "medium": 2, "low": 1}

        # Pattern 1: Within same pillar, higher priority goals should likely come before lower
        by_pillar = defaultdict(list)
        for g in active_goals:
            by_pillar[g.get("pillar", "other")].append(g)

        for pillar, goals in by_pillar.items():
            if len(goals) < 2:
                continue

            # Sort by priority (high first)
            goals.sort(key=lambda g: priority_rank.get(g.get("priority", "medium"), 2), reverse=True)

            for i, high_g in enumerate(goals):
                for low_g in goals[i + 1:]:
                    high_pri = priority_rank.get(high_g.get("priority", "medium"), 2)
                    low_pri = priority_rank.get(low_g.get("priority", "medium"), 2)

                    if high_pri <= low_pri:
                        continue

                    # Check if dependency already exists
                    existing_deps = set(low_g.get("depends_on", []))
                    if high_g["id"] in existing_deps:
                        continue

                    # Only suggest if priority gap is >= 2
                    if high_pri - low_pri >= 2:
                        suggestions.append({
                            "from_id": high_g["id"],
                            "from_title": high_g["title"],
                            "to_id": low_g["id"],
                            "to_title": low_g["title"],
                            "reason": f"Same pillar ({pillar}): '{high_g['title']}' ({high_g['priority']}) should likely complete before '{low_g['title']}' ({low_g['priority']})",
                            "confidence": "medium",
                        })

        # Pattern 2: Goals with "foundation", "setup", "infrastructure" keywords
        # should be dependencies for others in same pillar
        foundation_keywords = {"foundation", "setup", "infrastructure", "base", "core", "framework", "init", "bootstrap"}
        for g in active_goals:
            title_lower = g.get("title", "").lower()
            is_foundational = any(kw in title_lower for kw in foundation_keywords)
            if not is_foundational:
                continue

            pillar = g.get("pillar", "other")
            for other in active_goals:
                if other["id"] == g["id"]:
                    continue
                if other.get("pillar") != pillar:
                    continue
                existing_deps = set(other.get("depends_on", []))
                if g["id"] in existing_deps:
                    continue

                suggestions.append({
                    "from_id": g["id"],
                    "from_title": g["title"],
                    "to_id": other["id"],
                    "to_title": other["title"],
                    "reason": f"'{g['title']}' appears foundational and should likely be a dependency for '{other['title']}'",
                    "confidence": "low",
                })

        # Deduplicate
        seen = set()
        unique_suggestions = []
        for s in suggestions:
            key = (s["from_id"], s["to_id"])
            if key not in seen:
                seen.add(key)
                unique_suggestions.append(s)

        return SkillResult(
            success=True,
            message=f"Found {len(unique_suggestions)} potential missing dependencies",
            data={
                "suggestions": unique_suggestions[:20],  # Cap at 20
                "count": len(unique_suggestions),
            },
        )

    def _health(self, params: Dict) -> SkillResult:
        """Overall dependency graph health metrics."""
        data = self._load_goals()
        goals_by_id, forward_deps, reverse_deps = self._build_graph(data)

        active_goals = [g for g in data.get("goals", []) if g.get("status") == "active"]
        completed_ids = {g["id"] for g in data.get("completed_goals", [])}

        if not active_goals:
            return SkillResult(
                success=True,
                message="No active goals - graph is empty.",
                data={"health": "empty", "metrics": {}},
            )

        total_active = len(active_goals)
        total_edges = sum(len(deps) for deps in forward_deps.values())

        # Count blocked goals
        blocked_count = 0
        actionable_count = 0
        for g in active_goals:
            deps = g.get("depends_on", [])
            unmet = [d for d in deps if d not in completed_ids]
            if unmet:
                blocked_count += 1
            else:
                actionable_count += 1

        # Isolated goals (no deps, no dependents)
        isolated = 0
        for g in active_goals:
            gid = g["id"]
            has_deps = len(forward_deps.get(gid, [])) > 0
            has_dependents = len(reverse_deps.get(gid, [])) > 0
            if not has_deps and not has_dependents:
                isolated += 1

        # Max depth
        max_depth = 0
        for g in active_goals:
            d = self._compute_depth(g["id"], forward_deps, completed_ids)
            max_depth = max(max_depth, d)

        # Detect cycles
        has_cycles = False
        visited = set()
        rec_stack = set()

        def has_cycle_dfs(gid):
            visited.add(gid)
            rec_stack.add(gid)
            for dep in forward_deps.get(gid, []):
                if dep not in visited:
                    if has_cycle_dfs(dep):
                        return True
                elif dep in rec_stack:
                    return True
            rec_stack.discard(gid)
            return False

        active_ids = {g["id"] for g in active_goals}
        for g in active_goals:
            if g["id"] not in visited:
                if has_cycle_dfs(g["id"]):
                    has_cycles = True
                    break

        # Health score (0-100)
        health_score = 100
        if has_cycles:
            health_score -= 30  # Cycles are serious
        if total_active > 0:
            blocked_ratio = blocked_count / total_active
            if blocked_ratio > 0.5:
                health_score -= 20  # Too many blocked goals
            elif blocked_ratio > 0.3:
                health_score -= 10
        if max_depth > 5:
            health_score -= 10  # Very deep chains are risky
        if isolated / max(total_active, 1) > 0.8:
            health_score -= 5  # Mostly disconnected, deps underused

        health_label = "healthy" if health_score >= 80 else "warning" if health_score >= 50 else "critical"

        return SkillResult(
            success=True,
            message=f"Graph health: {health_label} ({health_score}/100) - {total_active} goals, {total_edges} edges",
            data={
                "health": health_label,
                "health_score": health_score,
                "metrics": {
                    "total_active_goals": total_active,
                    "total_dependency_edges": total_edges,
                    "actionable_goals": actionable_count,
                    "blocked_goals": blocked_count,
                    "isolated_goals": isolated,
                    "max_chain_depth": max_depth,
                    "has_cycles": has_cycles,
                    "avg_dependencies": round(total_edges / max(total_active, 1), 2),
                },
                "issues": (
                    (["Circular dependencies detected - will cause deadlocks"] if has_cycles else [])
                    + ([f"{blocked_count}/{total_active} goals are blocked"] if blocked_count > total_active * 0.5 else [])
                    + ([f"Deep dependency chain ({max_depth} levels) - consider parallel paths"] if max_depth > 5 else [])
                ),
            },
        )
