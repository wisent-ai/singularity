#!/usr/bin/env python3
"""
SkillDependencyAnalyzer - Codebase self-introspection for the Self-Improvement pillar.

The agent has 50+ skills but no way to understand how they relate to each other,
which ones overlap, or where architectural weaknesses exist. This skill gives
the agent the ability to analyze its own skill architecture:

1. DEPENDENCY MAPPING: Scan all skill files and extract import relationships
   to build a complete dependency graph between skills.
2. CIRCULAR DEPENDENCY DETECTION: Find circular import chains that indicate
   architectural problems.
3. ORPHAN DETECTION: Find skills that nothing depends on AND that don't
   depend on other skills — likely candidates for removal or integration.
4. CAPABILITY MATRIX: Generate a matrix of what each skill can do (actions),
   making it easy to find overlapping functionality.
5. CONSOLIDATION SUGGESTIONS: Identify skills with similar actions/categories
   that could potentially be merged to reduce complexity.
6. ARCHITECTURE HEALTH SCORE: A single 0-100 score reflecting the overall
   health of the skill architecture.

This is critical for Self-Improvement: you can't optimize what you can't measure.
Without understanding your own architecture, self-modification is blind.

Pillar served: Self-Improvement (primary), Goal Setting (architectural awareness)
"""

import ast
import json
import os
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple
from .base import Skill, SkillManifest, SkillAction, SkillResult


ANALYSIS_LOG = Path(__file__).parent.parent / "data" / "skill_analysis.json"
SKILLS_DIR = Path(__file__).parent
MAX_ANALYSES = 100


class SkillDependencyAnalyzer(Skill):
    """
    Analyzes the agent's own skill architecture - dependencies, overlaps,
    health metrics, and consolidation opportunities.
    """

    def __init__(self, credentials: Dict = None):
        super().__init__(credentials)
        self._cache: Dict[str, Any] = {}
        self._last_scan: Optional[str] = None

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="skill_analyzer",
            name="Skill Dependency Analyzer",
            version="1.0.0",
            category="self-improvement",
            description=(
                "Analyzes the agent's own skill architecture: dependency graphs, "
                "circular dependencies, orphan detection, capability matrices, "
                "and consolidation suggestions."
            ),
            actions=[
                SkillAction(
                    name="scan",
                    description="Scan all skill files and build the dependency graph",
                    parameters={},
                    estimated_cost=0,
                    estimated_duration_seconds=5,
                    success_probability=0.95,
                ),
                SkillAction(
                    name="dependency_graph",
                    description="Get the full dependency graph between skills",
                    parameters={
                        "skill_id": {
                            "type": "string",
                            "required": False,
                            "description": "Filter to a specific skill's dependencies",
                        }
                    },
                    estimated_cost=0,
                    estimated_duration_seconds=2,
                    success_probability=0.95,
                ),
                SkillAction(
                    name="find_circular",
                    description="Detect circular dependency chains between skills",
                    parameters={},
                    estimated_cost=0,
                    estimated_duration_seconds=3,
                    success_probability=0.95,
                ),
                SkillAction(
                    name="find_orphans",
                    description="Find skills with no dependents and no dependencies",
                    parameters={},
                    estimated_cost=0,
                    estimated_duration_seconds=2,
                    success_probability=0.95,
                ),
                SkillAction(
                    name="capability_matrix",
                    description="Generate a matrix of skills and their actions",
                    parameters={
                        "category": {
                            "type": "string",
                            "required": False,
                            "description": "Filter by skill category",
                        }
                    },
                    estimated_cost=0,
                    estimated_duration_seconds=3,
                    success_probability=0.95,
                ),
                SkillAction(
                    name="suggest_consolidation",
                    description="Find skills with overlapping functionality that could be merged",
                    parameters={},
                    estimated_cost=0,
                    estimated_duration_seconds=5,
                    success_probability=0.90,
                ),
                SkillAction(
                    name="health_score",
                    description="Calculate an overall architecture health score (0-100)",
                    parameters={},
                    estimated_cost=0,
                    estimated_duration_seconds=5,
                    success_probability=0.95,
                ),
                SkillAction(
                    name="skill_summary",
                    description="Get a detailed summary of a specific skill's role in the architecture",
                    parameters={
                        "skill_id": {
                            "type": "string",
                            "required": True,
                            "description": "The skill file name (without .py) to analyze",
                        }
                    },
                    estimated_cost=0,
                    estimated_duration_seconds=2,
                    success_probability=0.95,
                ),
            ],
            required_credentials=[],
        )

    def get_actions(self) -> List[SkillAction]:
        return self.manifest.actions

    async def execute(self, action: str, params: Dict) -> SkillResult:
        actions = {
            "scan": self._scan,
            "dependency_graph": self._dependency_graph,
            "find_circular": self._find_circular,
            "find_orphans": self._find_orphans,
            "capability_matrix": self._capability_matrix,
            "suggest_consolidation": self._suggest_consolidation,
            "health_score": self._health_score,
            "skill_summary": self._skill_summary,
        }
        if action not in actions:
            return SkillResult(
                success=False,
                message=f"Unknown action: {action}. Available: {list(actions.keys())}",
            )
        try:
            return await actions[action](params)
        except Exception as e:
            return SkillResult(success=False, message=f"Error in {action}: {e}")

    # --- Core scanning ---

    def _get_skill_files(self) -> List[Path]:
        """Get all Python skill files in the skills directory."""
        files = []
        for f in sorted(SKILLS_DIR.glob("*.py")):
            if f.name.startswith("__"):
                continue
            files.append(f)
        return files

    def _extract_skill_info(self, filepath: Path) -> Dict[str, Any]:
        """Extract metadata from a single skill file using AST parsing."""
        info: Dict[str, Any] = {
            "file": filepath.name,
            "skill_id": filepath.stem,
            "imports_from_skills": [],
            "class_name": None,
            "category": None,
            "actions": [],
            "description": None,
            "lines_of_code": 0,
            "has_data_file": False,
            "external_imports": [],
        }

        try:
            source = filepath.read_text()
            info["lines_of_code"] = len(source.splitlines())
        except Exception:
            return info

        # Extract skill-to-skill imports
        # Patterns: from .other_skill import ... or from .base import ...
        for match in re.finditer(r"from\s+\.(\w+)\s+import", source):
            dep = match.group(1)
            if dep != "__init__":
                info["imports_from_skills"].append(dep)

        # Extract external imports
        for match in re.finditer(r"^(?:import|from)\s+(\w+)", source, re.MULTILINE):
            mod = match.group(1)
            if mod not in ("singularity", "typing", "dataclasses", "enum", "abc"):
                info["external_imports"].append(mod)
        info["external_imports"] = sorted(set(info["external_imports"]))

        # Check for data file usage
        if re.search(r'Path\(__file__\).*"data"', source):
            info["has_data_file"] = True

        # Use AST for structured extraction
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return info

        for node in ast.walk(tree):
            # Find the skill class
            if isinstance(node, ast.ClassDef):
                for base in node.bases:
                    base_name = ""
                    if isinstance(base, ast.Name):
                        base_name = base.id
                    elif isinstance(base, ast.Attribute):
                        base_name = base.attr
                    if base_name == "Skill":
                        info["class_name"] = node.name
                        # Extract docstring
                        ds = ast.get_docstring(node)
                        if ds:
                            info["description"] = ds.strip().split("\n")[0]

            # Find SkillManifest instantiation to get category
            if isinstance(node, ast.Call):
                func = node.func
                fname = ""
                if isinstance(func, ast.Name):
                    fname = func.id
                elif isinstance(func, ast.Attribute):
                    fname = func.attr
                if fname == "SkillManifest":
                    for kw in node.keywords:
                        if kw.arg == "category" and isinstance(kw.value, ast.Constant):
                            info["category"] = kw.value.value
                        if kw.arg == "skill_id" and isinstance(kw.value, ast.Constant):
                            info["skill_id"] = kw.value.value

            # Find SkillAction instantiations to get action names
            if isinstance(node, ast.Call):
                func = node.func
                fname = ""
                if isinstance(func, ast.Name):
                    fname = func.id
                elif isinstance(func, ast.Attribute):
                    fname = func.attr
                if fname == "SkillAction":
                    action_name = None
                    action_desc = None
                    for kw in node.keywords:
                        if kw.arg == "name" and isinstance(kw.value, ast.Constant):
                            action_name = kw.value.value
                        if kw.arg == "description" and isinstance(kw.value, ast.Constant):
                            action_desc = kw.value.value
                    if action_name:
                        info["actions"].append({
                            "name": action_name,
                            "description": action_desc or "",
                        })

        return info

    async def _scan(self, params: Dict) -> SkillResult:
        """Scan all skill files and build the dependency graph."""
        files = self._get_skill_files()
        skills_info = {}
        for f in files:
            info = self._extract_skill_info(f)
            skills_info[info["skill_id"]] = info

        # Build adjacency lists
        depends_on: Dict[str, List[str]] = {}  # skill -> skills it imports
        depended_by: Dict[str, List[str]] = defaultdict(list)  # skill -> skills that import it

        for sid, info in skills_info.items():
            deps = [d for d in info["imports_from_skills"] if d in skills_info]
            depends_on[sid] = deps
            for dep in deps:
                depended_by[dep].append(sid)

        self._cache = {
            "skills": skills_info,
            "depends_on": depends_on,
            "depended_by": dict(depended_by),
            "scanned_at": datetime.now().isoformat(),
        }
        self._last_scan = self._cache["scanned_at"]

        # Summary stats
        total_skills = len(skills_info)
        total_actions = sum(len(s["actions"]) for s in skills_info.values())
        total_loc = sum(s["lines_of_code"] for s in skills_info.values())
        categories = defaultdict(int)
        for s in skills_info.values():
            categories[s["category"] or "uncategorized"] += 1

        # Save analysis
        self._save_analysis({
            "type": "scan",
            "total_skills": total_skills,
            "total_actions": total_actions,
            "total_lines_of_code": total_loc,
            "categories": dict(categories),
        })

        return SkillResult(
            success=True,
            message=f"Scanned {total_skills} skills with {total_actions} total actions ({total_loc} LoC)",
            data={
                "total_skills": total_skills,
                "total_actions": total_actions,
                "total_lines_of_code": total_loc,
                "categories": dict(categories),
                "skills": list(skills_info.keys()),
                "scanned_at": self._last_scan,
            },
        )

    def _ensure_scanned(self) -> bool:
        """Ensure we have a scan cached."""
        return bool(self._cache.get("skills"))

    async def _auto_scan(self) -> Optional[SkillResult]:
        """Auto-scan if not yet done. Returns error result if scan fails."""
        if not self._ensure_scanned():
            result = await self._scan({})
            if not result.success:
                return result
        return None

    # --- Dependency graph ---

    async def _dependency_graph(self, params: Dict) -> SkillResult:
        """Return the dependency graph, optionally filtered to a skill."""
        err = await self._auto_scan()
        if err:
            return err

        skill_filter = params.get("skill_id")
        depends_on = self._cache["depends_on"]
        depended_by = self._cache["depended_by"]

        if skill_filter:
            if skill_filter not in self._cache["skills"]:
                return SkillResult(
                    success=False,
                    message=f"Skill '{skill_filter}' not found. Available: {list(self._cache['skills'].keys())}",
                )
            return SkillResult(
                success=True,
                message=f"Dependencies for {skill_filter}",
                data={
                    "skill": skill_filter,
                    "depends_on": depends_on.get(skill_filter, []),
                    "depended_by": depended_by.get(skill_filter, []),
                    "total_deps": len(depends_on.get(skill_filter, [])),
                    "total_dependents": len(depended_by.get(skill_filter, [])),
                },
            )

        # Full graph
        edges = []
        for skill, deps in depends_on.items():
            for dep in deps:
                if dep != "base":  # base is universal, skip
                    edges.append({"from": skill, "to": dep})

        # Stats
        most_depended = sorted(
            depended_by.items(), key=lambda x: len(x[1]), reverse=True
        )[:10]
        most_deps = sorted(
            depends_on.items(), key=lambda x: len(x[1]), reverse=True
        )[:10]

        return SkillResult(
            success=True,
            message=f"Dependency graph: {len(edges)} edges across {len(depends_on)} skills",
            data={
                "edges": edges,
                "total_edges": len(edges),
                "most_depended_on": [
                    {"skill": s, "dependents": len(d)} for s, d in most_depended
                ],
                "most_dependencies": [
                    {"skill": s, "dependencies": len(d)} for s, d in most_deps
                ],
            },
        )

    # --- Circular dependency detection ---

    async def _find_circular(self, params: Dict) -> SkillResult:
        """Detect circular dependency chains."""
        err = await self._auto_scan()
        if err:
            return err

        depends_on = self._cache["depends_on"]
        cycles: List[List[str]] = []
        visited: Set[str] = set()

        def dfs(node: str, path: List[str], on_stack: Set[str]):
            if node in on_stack:
                # Found a cycle - extract it
                cycle_start = path.index(node)
                cycle = path[cycle_start:] + [node]
                # Normalize: start from smallest element
                min_idx = cycle.index(min(cycle[:-1]))
                normalized = cycle[min_idx:-1] + cycle[:min_idx] + [cycle[min_idx]]
                # Deduplicate
                cycle_key = "->".join(normalized)
                if not any("->".join(c) == cycle_key for c in cycles):
                    cycles.append(normalized)
                return
            if node in visited:
                return

            on_stack.add(node)
            path.append(node)

            for dep in depends_on.get(node, []):
                if dep == "base":
                    continue
                dfs(dep, path, on_stack)

            path.pop()
            on_stack.discard(node)
            visited.add(node)

        for skill in depends_on:
            dfs(skill, [], set())

        if cycles:
            msg = f"Found {len(cycles)} circular dependency chain(s)"
        else:
            msg = "No circular dependencies detected - architecture is clean"

        return SkillResult(
            success=True,
            message=msg,
            data={
                "cycles": cycles,
                "count": len(cycles),
                "healthy": len(cycles) == 0,
            },
        )

    # --- Orphan detection ---

    async def _find_orphans(self, params: Dict) -> SkillResult:
        """Find skills with no dependents and no skill dependencies (isolated nodes)."""
        err = await self._auto_scan()
        if err:
            return err

        depends_on = self._cache["depends_on"]
        depended_by = self._cache["depended_by"]
        skills = self._cache["skills"]

        orphans = []
        leaf_skills = []  # No dependents (nothing imports them)
        root_skills = []  # No dependencies (don't import other skills)

        for sid in skills:
            deps = [d for d in depends_on.get(sid, []) if d != "base"]
            dependents = depended_by.get(sid, [])

            if not deps and not dependents:
                orphans.append(sid)
            elif not dependents:
                leaf_skills.append(sid)
            elif not deps:
                root_skills.append(sid)

        return SkillResult(
            success=True,
            message=f"Found {len(orphans)} orphaned skills, {len(leaf_skills)} leaf skills, {len(root_skills)} root skills",
            data={
                "orphans": sorted(orphans),
                "orphan_count": len(orphans),
                "leaf_skills": sorted(leaf_skills),
                "leaf_count": len(leaf_skills),
                "root_skills": sorted(root_skills),
                "root_count": len(root_skills),
                "explanation": {
                    "orphan": "No other skill imports it AND it imports no other skills (besides base)",
                    "leaf": "No other skill imports it (end-point capability)",
                    "root": "Does not import other skills (foundational, only base)",
                },
            },
        )

    # --- Capability matrix ---

    async def _capability_matrix(self, params: Dict) -> SkillResult:
        """Generate a matrix of all skills and their actions."""
        err = await self._auto_scan()
        if err:
            return err

        category_filter = params.get("category")
        skills = self._cache["skills"]

        matrix = []
        for sid, info in sorted(skills.items()):
            if category_filter and info.get("category") != category_filter:
                continue
            if not info.get("actions"):
                continue
            matrix.append({
                "skill_id": sid,
                "class_name": info.get("class_name"),
                "category": info.get("category") or "uncategorized",
                "actions": [a["name"] for a in info["actions"]],
                "action_count": len(info["actions"]),
                "description": info.get("description"),
                "lines_of_code": info.get("lines_of_code", 0),
            })

        # Sort by action count descending
        matrix.sort(key=lambda x: x["action_count"], reverse=True)

        all_actions = []
        for entry in matrix:
            all_actions.extend(entry["actions"])

        return SkillResult(
            success=True,
            message=f"Capability matrix: {len(matrix)} skills with {len(all_actions)} total actions",
            data={
                "matrix": matrix,
                "total_skills": len(matrix),
                "total_actions": len(all_actions),
                "unique_actions": len(set(all_actions)),
                "duplicate_action_names": self._find_duplicate_actions(matrix),
            },
        )

    def _find_duplicate_actions(self, matrix: List[Dict]) -> List[Dict]:
        """Find action names that appear in multiple skills."""
        action_map: Dict[str, List[str]] = defaultdict(list)
        for entry in matrix:
            for action in entry["actions"]:
                action_map[action].append(entry["skill_id"])

        duplicates = []
        for action, skills_list in sorted(action_map.items()):
            if len(skills_list) > 1:
                duplicates.append({
                    "action": action,
                    "skills": skills_list,
                    "count": len(skills_list),
                })
        return duplicates

    # --- Consolidation suggestions ---

    async def _suggest_consolidation(self, params: Dict) -> SkillResult:
        """Identify skills that could potentially be merged."""
        err = await self._auto_scan()
        if err:
            return err

        skills = self._cache["skills"]
        suggestions = []

        # Strategy 1: Same category with overlapping action names
        by_category: Dict[str, List[str]] = defaultdict(list)
        for sid, info in skills.items():
            cat = info.get("category") or "uncategorized"
            by_category[cat].append(sid)

        for cat, sids in by_category.items():
            if len(sids) < 2:
                continue
            # Check pairwise action overlap
            for i in range(len(sids)):
                for j in range(i + 1, len(sids)):
                    s1, s2 = sids[i], sids[j]
                    a1 = set(a["name"] for a in skills[s1].get("actions", []))
                    a2 = set(a["name"] for a in skills[s2].get("actions", []))
                    overlap = a1 & a2
                    if overlap:
                        suggestions.append({
                            "type": "action_overlap",
                            "skills": [s1, s2],
                            "category": cat,
                            "overlapping_actions": sorted(overlap),
                            "reason": f"Both in '{cat}' category with overlapping actions: {sorted(overlap)}",
                        })

        # Strategy 2: Small skills that could be absorbed into related ones
        small_threshold = 80  # lines
        for sid, info in skills.items():
            if info.get("lines_of_code", 0) < small_threshold and sid != "base":
                deps = [d for d in self._cache["depends_on"].get(sid, []) if d != "base"]
                if len(deps) == 1:
                    suggestions.append({
                        "type": "small_skill",
                        "skills": [sid],
                        "absorb_into": deps[0],
                        "lines": info.get("lines_of_code", 0),
                        "reason": f"{sid} is very small ({info.get('lines_of_code', 0)} lines) and depends only on {deps[0]}",
                    })

        # Strategy 3: Skills with similar names that might overlap
        skill_names = list(skills.keys())
        name_pairs = []
        for i in range(len(skill_names)):
            for j in range(i + 1, len(skill_names)):
                n1, n2 = skill_names[i], skill_names[j]
                # Simple word overlap check
                words1 = set(n1.replace("_", " ").split())
                words2 = set(n2.replace("_", " ").split())
                common = words1 & words2
                if common and common != {"skill"}:
                    name_pairs.append({
                        "type": "similar_names",
                        "skills": [n1, n2],
                        "common_words": sorted(common),
                        "reason": f"Similar names suggest overlapping responsibility: shared words {sorted(common)}",
                    })
        suggestions.extend(name_pairs)

        return SkillResult(
            success=True,
            message=f"Found {len(suggestions)} consolidation suggestions",
            data={
                "suggestions": suggestions,
                "count": len(suggestions),
                "by_type": {
                    "action_overlap": len([s for s in suggestions if s["type"] == "action_overlap"]),
                    "small_skill": len([s for s in suggestions if s["type"] == "small_skill"]),
                    "similar_names": len([s for s in suggestions if s["type"] == "similar_names"]),
                },
            },
        )

    # --- Architecture health score ---

    async def _health_score(self, params: Dict) -> SkillResult:
        """Calculate an overall architecture health score."""
        err = await self._auto_scan()
        if err:
            return err

        skills = self._cache["skills"]
        depends_on = self._cache["depends_on"]
        depended_by = self._cache["depended_by"]

        total = len(skills)
        if total == 0:
            return SkillResult(success=True, message="No skills found", data={"score": 0})

        score = 100
        issues = []

        # 1. Circular dependencies (-15 each, max -30)
        circular_result = await self._find_circular({})
        cycle_count = circular_result.data.get("count", 0)
        if cycle_count > 0:
            penalty = min(cycle_count * 15, 30)
            score -= penalty
            issues.append(f"-{penalty}: {cycle_count} circular dependency chain(s)")

        # 2. Orphan ratio (-1 per orphan, max -15)
        orphan_result = await self._find_orphans({})
        orphan_count = orphan_result.data.get("orphan_count", 0)
        if orphan_count > 0:
            penalty = min(orphan_count, 15)
            score -= penalty
            issues.append(f"-{penalty}: {orphan_count} orphaned skill(s)")

        # 3. Duplicate action names (-2 per duplicate, max -10)
        matrix_result = await self._capability_matrix({})
        dup_actions = matrix_result.data.get("duplicate_action_names", [])
        if dup_actions:
            penalty = min(len(dup_actions) * 2, 10)
            score -= penalty
            issues.append(f"-{penalty}: {len(dup_actions)} duplicate action name(s)")

        # 4. Very large skills (-1 per skill >500 lines, max -10)
        large_skills = [
            sid for sid, info in skills.items()
            if info.get("lines_of_code", 0) > 500
        ]
        if large_skills:
            penalty = min(len(large_skills), 10)
            score -= penalty
            issues.append(f"-{penalty}: {len(large_skills)} oversized skill(s) (>500 lines)")

        # 5. Skills without actions (-1 per, max -5)
        no_actions = [
            sid for sid, info in skills.items()
            if not info.get("actions") and sid != "base"
        ]
        if no_actions:
            penalty = min(len(no_actions), 5)
            score -= penalty
            issues.append(f"-{penalty}: {len(no_actions)} skill(s) with no detected actions")

        # 6. High fan-out (skill depends on >5 other skills) (-2 per, max -10)
        high_fanout = [
            sid for sid, deps in depends_on.items()
            if len([d for d in deps if d != "base"]) > 5
        ]
        if high_fanout:
            penalty = min(len(high_fanout) * 2, 10)
            score -= penalty
            issues.append(f"-{penalty}: {len(high_fanout)} skill(s) with high fan-out (>5 deps)")

        # Bonuses
        bonuses = []

        # Bonus: Good categorization (+5 if >80% skills have a category)
        categorized = sum(1 for s in skills.values() if s.get("category"))
        if categorized / max(total, 1) > 0.8:
            score = min(score + 5, 100)
            bonuses.append("+5: >80% skills have categories")

        # Bonus: Modular architecture (+5 if avg deps < 3)
        avg_deps = sum(
            len([d for d in deps if d != "base"])
            for deps in depends_on.values()
        ) / max(total, 1)
        if avg_deps < 3:
            score = min(score + 5, 100)
            bonuses.append(f"+5: Low average dependency count ({avg_deps:.1f})")

        score = max(0, min(100, score))

        # Rating
        if score >= 90:
            rating = "Excellent"
        elif score >= 75:
            rating = "Good"
        elif score >= 60:
            rating = "Fair"
        elif score >= 40:
            rating = "Needs Work"
        else:
            rating = "Poor"

        self._save_analysis({
            "type": "health_score",
            "score": score,
            "rating": rating,
        })

        return SkillResult(
            success=True,
            message=f"Architecture Health Score: {score}/100 ({rating})",
            data={
                "score": score,
                "rating": rating,
                "issues": issues,
                "bonuses": bonuses,
                "metrics": {
                    "total_skills": total,
                    "circular_deps": cycle_count,
                    "orphans": orphan_count,
                    "duplicate_actions": len(dup_actions),
                    "large_skills": len(large_skills),
                    "skills_without_actions": len(no_actions),
                    "high_fanout_skills": len(high_fanout),
                    "avg_dependencies": round(avg_deps, 2),
                    "categorization_rate": round(categorized / max(total, 1), 2),
                },
            },
        )

    # --- Skill summary ---

    async def _skill_summary(self, params: Dict) -> SkillResult:
        """Get a detailed summary of a specific skill's architectural role."""
        err = await self._auto_scan()
        if err:
            return err

        skill_id = params.get("skill_id")
        if not skill_id:
            return SkillResult(success=False, message="skill_id parameter is required")

        skills = self._cache["skills"]
        if skill_id not in skills:
            return SkillResult(
                success=False,
                message=f"Skill '{skill_id}' not found. Available: {sorted(skills.keys())}",
            )

        info = skills[skill_id]
        depends_on = self._cache["depends_on"]
        depended_by = self._cache["depended_by"]

        deps = [d for d in depends_on.get(skill_id, []) if d != "base"]
        dependents = depended_by.get(skill_id, [])

        # Classify role
        if not deps and not dependents:
            role = "orphan"
            role_desc = "Isolated - no connections to other skills"
        elif not deps and dependents:
            role = "foundation"
            role_desc = f"Foundation skill - {len(dependents)} other skills depend on it"
        elif deps and not dependents:
            role = "leaf"
            role_desc = "Leaf skill - end-point capability that no other skill imports"
        elif len(dependents) > len(deps):
            role = "hub"
            role_desc = f"Hub skill - widely depended on ({len(dependents)} dependents vs {len(deps)} deps)"
        else:
            role = "connector"
            role_desc = f"Connector - bridges {len(deps)} dependencies to {len(dependents)} dependents"

        return SkillResult(
            success=True,
            message=f"{skill_id}: {role} ({info.get('category', 'uncategorized')})",
            data={
                "skill_id": skill_id,
                "class_name": info.get("class_name"),
                "category": info.get("category"),
                "description": info.get("description"),
                "role": role,
                "role_description": role_desc,
                "actions": info.get("actions", []),
                "action_count": len(info.get("actions", [])),
                "lines_of_code": info.get("lines_of_code", 0),
                "depends_on": deps,
                "depended_by": dependents,
                "external_imports": info.get("external_imports", []),
                "has_data_file": info.get("has_data_file", False),
            },
        )

    # --- Persistence ---

    def _save_analysis(self, analysis: Dict):
        """Save analysis result to log."""
        analysis["timestamp"] = datetime.now().isoformat()
        try:
            ANALYSIS_LOG.parent.mkdir(parents=True, exist_ok=True)
            history = []
            if ANALYSIS_LOG.exists():
                try:
                    history = json.loads(ANALYSIS_LOG.read_text())
                except (json.JSONDecodeError, Exception):
                    history = []
            history.append(analysis)
            if len(history) > MAX_ANALYSES:
                history = history[-MAX_ANALYSES:]
            ANALYSIS_LOG.write_text(json.dumps(history, indent=2))
        except Exception:
            pass  # Non-critical
