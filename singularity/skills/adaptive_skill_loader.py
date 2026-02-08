#!/usr/bin/env python3
"""
AdaptiveSkillLoaderSkill - Dynamically load/unload skills based on task patterns.

The agent has 130+ skills but most tasks only need a handful. Loading all skills
wastes memory, slows startup, and clutters the LLM's action space. Meanwhile,
the agent has AgentReflectionSkill that records which skills/actions were used
for which task types and how successfully.

This skill closes the loop:
1. **Analyze** - Examine reflection history to identify which skills are used for which task types
2. **Profile** - Build skill usage profiles: frequency, success rate, co-occurrence patterns
3. **Recommend** - Given a task description, recommend which skills to load
4. **Auto-load** - Dynamically load recommended skills and unload unused ones
5. **Track** - Monitor skill loading decisions and their impact on task success
6. **Optimize** - Continuously refine recommendations based on outcomes

The adaptation loop:
  task arrives → match to pattern → load relevant skills → execute → track outcome → refine profile

This means the agent runs lean: only skills relevant to the current task are active,
reducing LLM context noise and memory usage.

Pillar: Self-Improvement (adaptive resource allocation based on learned patterns)
"""

import json
import time
import hashlib
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
from .base import Skill, SkillResult, SkillManifest, SkillAction

ADAPTIVE_LOADER_FILE = Path(__file__).parent.parent / "data" / "adaptive_skill_loader.json"
MAX_PROFILES = 200
MAX_DECISIONS = 500


class AdaptiveSkillLoaderSkill(Skill):
    """
    Dynamically recommends and manages skill loading based on task patterns.

    Integrates with:
    - AgentReflectionSkill (via SkillContext): reads reflections for usage patterns
    - SkillLoader: loads/unloads skills at runtime
    """

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self._load_state()

    def _load_state(self):
        """Load or initialize state."""
        ADAPTIVE_LOADER_FILE.parent.mkdir(parents=True, exist_ok=True)
        if ADAPTIVE_LOADER_FILE.exists():
            try:
                with open(ADAPTIVE_LOADER_FILE) as f:
                    data = json.load(f)
                self._profiles = data.get("profiles", {})
                self._decisions = data.get("decisions", [])
                self._config = data.get("config", self._default_config())
                self._stats = data.get("stats", self._default_stats())
                self._skill_scores = data.get("skill_scores", {})
            except (json.JSONDecodeError, Exception):
                self._init_empty()
        else:
            self._init_empty()

    def _init_empty(self):
        self._profiles: Dict[str, Dict] = {}  # task_pattern -> skill usage profile
        self._decisions: List[Dict] = []  # loading decision history
        self._config = self._default_config()
        self._stats = self._default_stats()
        self._skill_scores: Dict[str, Dict] = {}  # skill_id -> usage score data

    def _default_config(self) -> Dict:
        return {
            "min_usage_count": 2,
            "min_success_rate": 0.3,
            "max_recommended_skills": 10,
            "decay_factor": 0.95,
            "co_occurrence_threshold": 0.3,
            "auto_unload_idle_minutes": 30,
        }

    def _default_stats(self) -> Dict:
        return {
            "total_analyses": 0,
            "total_recommendations": 0,
            "total_loads": 0,
            "total_unloads": 0,
            "profiles_built": 0,
        }

    def _save(self):
        """Persist state."""
        data = {
            "profiles": dict(list(self._profiles.items())[:MAX_PROFILES]),
            "decisions": self._decisions[-MAX_DECISIONS:],
            "config": self._config,
            "stats": self._stats,
            "skill_scores": self._skill_scores,
        }
        with open(ADAPTIVE_LOADER_FILE, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def _generate_id(self, content: str) -> str:
        return hashlib.sha256(f"{content}{time.time()}".encode()).hexdigest()[:12]

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="adaptive_skill_loader",
            name="Adaptive Skill Loader",
            description="Dynamically load/unload skills based on task patterns and reflection history",
            version="1.0.0",
            category="self_improvement",
            actions=self.get_actions(),
            required_credentials=[],
        )

    def get_actions(self) -> List[SkillAction]:
        return [
            SkillAction(
                name="analyze",
                description="Analyze reflection history to build skill usage profiles",
                parameters={
                    "min_reflections": {"type": "int", "required": False, "description": "Min reflections to analyze (default all)"},
                },
            ),
            SkillAction(
                name="recommend",
                description="Given a task description, recommend which skills to load",
                parameters={
                    "task_description": {"type": "str", "required": True, "description": "Description of the task"},
                    "max_skills": {"type": "int", "required": False, "description": "Max skills to recommend"},
                },
            ),
            SkillAction(
                name="profile",
                description="View the usage profile for a specific skill",
                parameters={
                    "skill_id": {"type": "str", "required": True, "description": "ID of the skill to profile"},
                },
            ),
            SkillAction(
                name="record_usage",
                description="Record that a skill was used for a task (builds profiles over time)",
                parameters={
                    "skill_id": {"type": "str", "required": True, "description": "Skill that was used"},
                    "task_type": {"type": "str", "required": True, "description": "Type/category of task"},
                    "success": {"type": "bool", "required": True, "description": "Whether the usage was successful"},
                    "co_skills": {"type": "list", "required": False, "description": "Other skills used alongside"},
                },
            ),
            SkillAction(
                name="hot_skills",
                description="List the most actively used skills (ranked by recent usage)",
                parameters={
                    "limit": {"type": "int", "required": False, "description": "Max skills to show (default 10)"},
                },
            ),
            SkillAction(
                name="cold_skills",
                description="List skills that haven't been used recently (candidates for unloading)",
                parameters={
                    "limit": {"type": "int", "required": False, "description": "Max skills to show (default 10)"},
                },
            ),
            SkillAction(
                name="configure",
                description="Update adaptive loader configuration",
                parameters={
                    "key": {"type": "str", "required": True, "description": "Config key to update"},
                    "value": {"type": "any", "required": True, "description": "New value"},
                },
            ),
            SkillAction(
                name="status",
                description="View overall adaptive loader status and statistics",
                parameters={},
            ),
        ]

    async def execute(self, action: str, params: Dict) -> SkillResult:
        """Execute an adaptive loader action."""
        handlers = {
            "analyze": self._analyze,
            "recommend": self._recommend,
            "profile": self._profile,
            "record_usage": self._record_usage,
            "hot_skills": self._hot_skills,
            "cold_skills": self._cold_skills,
            "configure": self._configure,
            "status": self._status,
        }

        handler = handlers.get(action)
        if not handler:
            return SkillResult(success=False, message=f"Unknown action: {action}")

        try:
            return await handler(params)
        except Exception as e:
            return SkillResult(success=False, message=f"AdaptiveSkillLoader error: {str(e)}")

    def _extract_skills_from_reflections(self, reflections: List[Dict]) -> Dict[str, Dict]:
        """Extract skill usage data from reflection records."""
        skill_data = defaultdict(lambda: {
            "uses": 0,
            "successes": 0,
            "failures": 0,
            "task_types": Counter(),
            "co_skills": Counter(),
            "last_used": None,
        })

        for ref in reflections:
            actions = ref.get("actions_taken", [])
            success = ref.get("success", False)
            tags = ref.get("tags", [])
            task = ref.get("task", "")
            timestamp = ref.get("timestamp", "")

            # Extract skill IDs from actions (format: "skill:action" or just skill name)
            skills_in_ref = set()
            for action_str in actions:
                if isinstance(action_str, str):
                    skill_id = action_str.split(":")[0].strip() if ":" in action_str else action_str.strip()
                    if skill_id:
                        skills_in_ref.add(skill_id)

            # Also extract from tags
            for tag in tags:
                if isinstance(tag, str) and tag.startswith("skill:"):
                    skills_in_ref.add(tag.split(":", 1)[1])

            # Record usage for each skill
            for skill_id in skills_in_ref:
                data = skill_data[skill_id]
                data["uses"] += 1
                if success:
                    data["successes"] += 1
                else:
                    data["failures"] += 1
                data["last_used"] = timestamp

                # Determine task type from tags or task text
                task_type = "general"
                for tag in tags:
                    if not tag.startswith("skill:"):
                        task_type = tag
                        break
                if task_type == "general" and task:
                    # Use first few words as task type
                    words = task.lower().split()[:3]
                    task_type = " ".join(words)

                data["task_types"][task_type] += 1

                # Record co-occurring skills
                for other_skill in skills_in_ref:
                    if other_skill != skill_id:
                        data["co_skills"][other_skill] += 1

        return dict(skill_data)

    async def _analyze(self, params: Dict) -> SkillResult:
        """Analyze reflection history to build skill usage profiles."""
        reflections = []

        if hasattr(self, "context") and self.context:
            try:
                result = await self.context.call_skill(
                    "agent_reflection", "review", {"limit": 100}
                )
                if result.success:
                    reflections = result.data.get("reflections", [])
            except Exception:
                pass

        if not reflections:
            return SkillResult(
                success=True,
                message="No reflections available to analyze. Record some usages manually with record_usage.",
                data={"profiles_built": 0},
            )

        skill_data = self._extract_skills_from_reflections(reflections)

        # Build/update profiles
        profiles_built = 0
        for skill_id, data in skill_data.items():
            total = data["uses"]
            success_rate = data["successes"] / total if total > 0 else 0.0

            profile = {
                "skill_id": skill_id,
                "total_uses": total,
                "success_rate": success_rate,
                "top_task_types": dict(data["task_types"].most_common(5)),
                "co_skills": dict(data["co_skills"].most_common(5)),
                "last_used": data["last_used"],
                "updated_at": datetime.utcnow().isoformat(),
            }

            self._profiles[skill_id] = profile
            profiles_built += 1

        # Update scores
        self._update_skill_scores(skill_data)

        self._stats["total_analyses"] += 1
        self._stats["profiles_built"] = len(self._profiles)
        self._save()

        return SkillResult(
            success=True,
            message=f"Analyzed {len(reflections)} reflections, built {profiles_built} skill profiles",
            data={
                "reflections_analyzed": len(reflections),
                "profiles_built": profiles_built,
                "top_skills": sorted(
                    [(sid, p["total_uses"]) for sid, p in self._profiles.items()],
                    key=lambda x: x[1],
                    reverse=True,
                )[:5],
            },
        )

    def _update_skill_scores(self, skill_data: Dict[str, Dict]):
        """Update running skill scores with decay."""
        decay = self._config["decay_factor"]

        # Decay existing scores
        for skill_id in self._skill_scores:
            self._skill_scores[skill_id]["score"] *= decay

        # Add new data
        for skill_id, data in skill_data.items():
            if skill_id not in self._skill_scores:
                self._skill_scores[skill_id] = {
                    "score": 0.0,
                    "total_uses": 0,
                    "total_successes": 0,
                }
            entry = self._skill_scores[skill_id]
            entry["total_uses"] += data["uses"]
            entry["total_successes"] += data["successes"]
            # Score = uses weighted by success rate
            success_rate = data["successes"] / data["uses"] if data["uses"] > 0 else 0.5
            entry["score"] += data["uses"] * (0.5 + 0.5 * success_rate)

    def _match_task_to_profiles(self, task_description: str) -> List[Dict]:
        """Find skill profiles that match a task description."""
        task_lower = task_description.lower()
        task_words = set(task_lower.split())

        matches = []
        for skill_id, profile in self._profiles.items():
            relevance = 0.0

            # Check task type keyword overlap
            for task_type, count in profile.get("top_task_types", {}).items():
                type_words = set(task_type.lower().split())
                overlap = len(task_words & type_words)
                if overlap > 0:
                    relevance += overlap * count

            # Check skill_id keyword match
            skill_words = set(skill_id.lower().replace("_", " ").split())
            skill_overlap = len(task_words & skill_words)
            if skill_overlap > 0:
                relevance += skill_overlap * 2

            if relevance > 0:
                matches.append({
                    "skill_id": skill_id,
                    "relevance": relevance,
                    "success_rate": profile.get("success_rate", 0.0),
                    "total_uses": profile.get("total_uses", 0),
                    "co_skills": list(profile.get("co_skills", {}).keys()),
                })

        # Sort by relevance * success_rate
        matches.sort(
            key=lambda m: m["relevance"] * (0.5 + 0.5 * m["success_rate"]),
            reverse=True,
        )
        return matches

    async def _recommend(self, params: Dict) -> SkillResult:
        """Recommend skills to load for a given task."""
        task_description = params.get("task_description", "")
        max_skills = params.get("max_skills", self._config["max_recommended_skills"])

        if not task_description:
            return SkillResult(success=False, message="Required: task_description")

        matches = self._match_task_to_profiles(task_description)

        # Collect recommended skills, including co-occurring ones
        recommended = []
        seen = set()
        for match in matches[:max_skills]:
            if match["skill_id"] not in seen:
                recommended.append(match)
                seen.add(match["skill_id"])

            # Include frequently co-occurring skills
            for co_skill in match.get("co_skills", [])[:3]:
                if co_skill not in seen and len(recommended) < max_skills:
                    co_profile = self._profiles.get(co_skill, {})
                    recommended.append({
                        "skill_id": co_skill,
                        "relevance": match["relevance"] * self._config["co_occurrence_threshold"],
                        "success_rate": co_profile.get("success_rate", 0.0),
                        "total_uses": co_profile.get("total_uses", 0),
                        "reason": f"co-occurs with {match['skill_id']}",
                    })
                    seen.add(co_skill)

        self._stats["total_recommendations"] += 1

        # Record the decision
        decision = {
            "id": self._generate_id(task_description),
            "timestamp": datetime.utcnow().isoformat(),
            "task_description": task_description[:200],
            "recommended": [r["skill_id"] for r in recommended],
            "type": "recommend",
        }
        self._decisions.append(decision)
        self._save()

        return SkillResult(
            success=True,
            message=f"Recommended {len(recommended)} skill(s) for task: '{task_description[:60]}'",
            data={
                "recommendations": recommended[:max_skills],
                "task_description": task_description,
                "total_profiles_checked": len(self._profiles),
            },
        )

    async def _profile(self, params: Dict) -> SkillResult:
        """View usage profile for a specific skill."""
        skill_id = params.get("skill_id", "")
        if not skill_id:
            return SkillResult(success=False, message="Required: skill_id")

        profile = self._profiles.get(skill_id)
        score_data = self._skill_scores.get(skill_id)

        if not profile and not score_data:
            return SkillResult(
                success=False,
                message=f"No profile found for skill '{skill_id}'. Run analyze first or record usage.",
            )

        return SkillResult(
            success=True,
            message=f"Profile for '{skill_id}'",
            data={
                "profile": profile or {},
                "score": score_data or {},
            },
        )

    async def _record_usage(self, params: Dict) -> SkillResult:
        """Manually record a skill usage event."""
        skill_id = params.get("skill_id", "")
        task_type = params.get("task_type", "")
        success = params.get("success", True)
        co_skills = params.get("co_skills", [])

        if not skill_id or not task_type:
            return SkillResult(success=False, message="Required: skill_id and task_type")

        # Update or create profile
        if skill_id not in self._profiles:
            self._profiles[skill_id] = {
                "skill_id": skill_id,
                "total_uses": 0,
                "success_rate": 0.0,
                "top_task_types": {},
                "co_skills": {},
                "last_used": None,
                "updated_at": None,
            }

        profile = self._profiles[skill_id]
        profile["total_uses"] += 1
        old_successes = int(profile.get("success_rate", 0) * (profile["total_uses"] - 1))
        new_successes = old_successes + (1 if success else 0)
        profile["success_rate"] = new_successes / profile["total_uses"]
        profile["last_used"] = datetime.utcnow().isoformat()
        profile["updated_at"] = datetime.utcnow().isoformat()

        # Update task types
        task_types = profile.get("top_task_types", {})
        task_types[task_type] = task_types.get(task_type, 0) + 1
        profile["top_task_types"] = dict(sorted(task_types.items(), key=lambda x: x[1], reverse=True)[:10])

        # Update co-skills
        co_skill_counts = profile.get("co_skills", {})
        for co in co_skills:
            co_skill_counts[co] = co_skill_counts.get(co, 0) + 1
        profile["co_skills"] = dict(sorted(co_skill_counts.items(), key=lambda x: x[1], reverse=True)[:10])

        # Update score
        if skill_id not in self._skill_scores:
            self._skill_scores[skill_id] = {"score": 0.0, "total_uses": 0, "total_successes": 0}
        self._skill_scores[skill_id]["total_uses"] += 1
        if success:
            self._skill_scores[skill_id]["total_successes"] += 1
        self._skill_scores[skill_id]["score"] += 1.0 if success else 0.5

        self._save()

        return SkillResult(
            success=True,
            message=f"Recorded {'successful' if success else 'failed'} usage of '{skill_id}' for task type '{task_type}'",
            data={"profile": profile},
        )

    async def _hot_skills(self, params: Dict) -> SkillResult:
        """List most actively used skills."""
        limit = params.get("limit", 10)

        ranked = sorted(
            self._skill_scores.items(),
            key=lambda x: x[1].get("score", 0),
            reverse=True,
        )[:limit]

        hot = []
        for skill_id, score_data in ranked:
            profile = self._profiles.get(skill_id, {})
            hot.append({
                "skill_id": skill_id,
                "score": round(score_data.get("score", 0), 2),
                "total_uses": score_data.get("total_uses", 0),
                "success_rate": profile.get("success_rate", 0.0),
                "last_used": profile.get("last_used"),
            })

        return SkillResult(
            success=True,
            message=f"Top {len(hot)} hot skills by usage score",
            data={"hot_skills": hot},
        )

    async def _cold_skills(self, params: Dict) -> SkillResult:
        """List skills that haven't been used recently."""
        limit = params.get("limit", 10)

        # Skills with lowest scores or never used
        all_scores = list(self._skill_scores.items())
        all_scores.sort(key=lambda x: x[1].get("score", 0))

        cold = []
        for skill_id, score_data in all_scores[:limit]:
            profile = self._profiles.get(skill_id, {})
            cold.append({
                "skill_id": skill_id,
                "score": round(score_data.get("score", 0), 2),
                "total_uses": score_data.get("total_uses", 0),
                "last_used": profile.get("last_used"),
                "recommendation": "Consider unloading" if score_data.get("score", 0) < 1.0 else "Keep loaded",
            })

        return SkillResult(
            success=True,
            message=f"Bottom {len(cold)} cold skills (candidates for unloading)",
            data={"cold_skills": cold},
        )

    async def _configure(self, params: Dict) -> SkillResult:
        """Update configuration."""
        key = params.get("key", "")
        value = params.get("value")

        if not key:
            return SkillResult(success=False, message="Required: key")

        if key not in self._config:
            return SkillResult(
                success=False,
                message=f"Unknown config key: {key}. Available: {list(self._config.keys())}",
            )

        old_value = self._config[key]
        self._config[key] = value
        self._save()

        return SkillResult(
            success=True,
            message=f"Updated {key}: {old_value} → {value}",
            data={"config": self._config},
        )

    async def _status(self, params: Dict) -> SkillResult:
        """View overall status."""
        return SkillResult(
            success=True,
            message="Adaptive Skill Loader status",
            data={
                "stats": self._stats,
                "config": self._config,
                "total_profiles": len(self._profiles),
                "total_scored_skills": len(self._skill_scores),
                "total_decisions": len(self._decisions),
                "top_3_skills": sorted(
                    [(sid, sd.get("score", 0)) for sid, sd in self._skill_scores.items()],
                    key=lambda x: x[1],
                    reverse=True,
                )[:3],
            },
        )
