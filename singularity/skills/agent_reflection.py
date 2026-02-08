#!/usr/bin/env python3
"""
AgentReflectionSkill - Meta-cognitive reflection and playbook generation.

The critical gap in the self-improvement loop: the agent can execute tasks
and track outcomes (via PerformanceTracker, OutcomeTracker), but it cannot
*reflect* on patterns across executions to build reusable strategies.

This skill enables:
1. **Post-action reflection** - After each cycle, analyze what happened and why
2. **Pattern extraction** - Identify recurring success/failure patterns across reflections
3. **Playbook generation** - Create reusable step-by-step playbooks for task types
4. **Playbook matching** - Given a new task, find the best matching playbook
5. **Effectiveness tracking** - Track how well playbooks perform when applied
6. **Insight journaling** - Accumulate strategic insights across sessions

The reflection loop:
  execute → reflect → extract patterns → build playbook → match future tasks → execute better

Unlike LearnedBehavior (individual rules) or Experiment (controlled A/B tests),
this skill operates at the *strategic reasoning* level - understanding WHY
something worked and encoding that understanding into reusable knowledge.

Pillar: Self-Improvement (the "think about thinking" metacognitive loop)
"""

import json
import hashlib
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from .base import Skill, SkillResult, SkillManifest, SkillAction

REFLECTION_FILE = Path(__file__).parent.parent / "data" / "reflections.json"
MAX_REFLECTIONS = 500
MAX_PLAYBOOKS = 100
MAX_INSIGHTS = 200


class AgentReflectionSkill(Skill):
    """
    Meta-cognitive reflection skill for building reusable playbooks.

    Reflections capture:
    - What task was attempted
    - What actions were taken
    - What the outcome was
    - Why it succeeded/failed (agent's analysis)
    - What could be done differently

    Playbooks capture:
    - Task type pattern (what kind of tasks this applies to)
    - Step-by-step strategy
    - Common pitfalls to avoid
    - Expected outcomes
    - Effectiveness score based on usage history
    """

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self._ensure_data()

    def _ensure_data(self):
        """Load or initialize reflection data."""
        REFLECTION_FILE.parent.mkdir(parents=True, exist_ok=True)
        if REFLECTION_FILE.exists():
            try:
                with open(REFLECTION_FILE) as f:
                    data = json.load(f)
                self._reflections = data.get("reflections", [])
                self._playbooks = data.get("playbooks", {})
                self._insights = data.get("insights", [])
                self._stats = data.get("stats", {
                    "total_reflections": 0,
                    "total_playbooks": 0,
                    "playbook_uses": 0,
                    "playbook_successes": 0,
                })
            except (json.JSONDecodeError, Exception):
                self._init_empty()
        else:
            self._init_empty()

    def _init_empty(self):
        self._reflections: List[Dict] = []
        self._playbooks: Dict[str, Dict] = {}
        self._insights: List[Dict] = []
        self._stats = {
            "total_reflections": 0,
            "total_playbooks": 0,
            "playbook_uses": 0,
            "playbook_successes": 0,
        }

    def _save(self):
        """Persist reflection data."""
        data = {
            "reflections": self._reflections[-MAX_REFLECTIONS:],
            "playbooks": dict(list(self._playbooks.items())[:MAX_PLAYBOOKS]),
            "insights": self._insights[-MAX_INSIGHTS:],
            "stats": self._stats,
        }
        REFLECTION_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(REFLECTION_FILE, "w") as f:
            json.dump(data, f, indent=2, default=str)

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="agent_reflection",
            name="Agent Reflection",
            version="1.0.0",
            category="self_improvement",
            description="Meta-cognitive reflection, pattern extraction, and playbook generation for continuous self-improvement",
            actions=self.get_actions(),
            required_credentials=[],
        )

    def get_actions(self) -> List[SkillAction]:
        return [
            SkillAction(
                name="reflect",
                description="Record a post-action reflection analyzing what happened, why, and what to do differently",
                parameters={
                    "task": {"type": "str", "required": True, "description": "What task was attempted"},
                    "actions_taken": {"type": "list", "required": True, "description": "List of actions taken"},
                    "outcome": {"type": "str", "required": True, "description": "What happened (success/failure/partial)"},
                    "success": {"type": "bool", "required": True, "description": "Whether the task succeeded"},
                    "analysis": {"type": "str", "required": True, "description": "Why it succeeded or failed"},
                    "improvements": {"type": "list", "required": False, "description": "What could be done better next time"},
                    "tags": {"type": "list", "required": False, "description": "Tags for categorization (e.g., 'deployment', 'debugging')"},
                },
            ),
            SkillAction(
                name="create_playbook",
                description="Create a reusable playbook from reflections for a specific task type",
                parameters={
                    "name": {"type": "str", "required": True, "description": "Playbook name (e.g., 'deploy_service')"},
                    "task_pattern": {"type": "str", "required": True, "description": "Description of task types this applies to"},
                    "steps": {"type": "list", "required": True, "description": "Ordered list of step descriptions"},
                    "pitfalls": {"type": "list", "required": False, "description": "Common mistakes to avoid"},
                    "prerequisites": {"type": "list", "required": False, "description": "What must be true before starting"},
                    "expected_outcome": {"type": "str", "required": False, "description": "What success looks like"},
                    "tags": {"type": "list", "required": False, "description": "Tags for matching"},
                },
            ),
            SkillAction(
                name="find_playbook",
                description="Find the best matching playbook for a given task description",
                parameters={
                    "task_description": {"type": "str", "required": True, "description": "Description of the task to find a playbook for"},
                    "tags": {"type": "list", "required": False, "description": "Optional tags to filter by"},
                },
            ),
            SkillAction(
                name="record_playbook_use",
                description="Record that a playbook was used and whether it was successful",
                parameters={
                    "playbook_name": {"type": "str", "required": True, "description": "Name of the playbook used"},
                    "success": {"type": "bool", "required": True, "description": "Whether the playbook led to success"},
                    "notes": {"type": "str", "required": False, "description": "Notes on the usage"},
                },
            ),
            SkillAction(
                name="extract_patterns",
                description="Analyze recent reflections to extract recurring patterns",
                parameters={
                    "lookback": {"type": "int", "required": False, "description": "Number of recent reflections to analyze (default 20)"},
                    "filter_tag": {"type": "str", "required": False, "description": "Only analyze reflections with this tag"},
                },
            ),
            SkillAction(
                name="add_insight",
                description="Record a strategic insight or lesson learned",
                parameters={
                    "insight": {"type": "str", "required": True, "description": "The insight or lesson"},
                    "category": {"type": "str", "required": False, "description": "Category (e.g., 'performance', 'cost', 'reliability')"},
                    "confidence": {"type": "float", "required": False, "description": "Confidence level 0-1 (default 0.5)"},
                    "source_reflections": {"type": "list", "required": False, "description": "IDs of reflections that led to this insight"},
                },
            ),
            SkillAction(
                name="review",
                description="Review reflections, playbooks, and insights with optional filtering",
                parameters={
                    "what": {"type": "str", "required": False, "description": "'reflections', 'playbooks', 'insights', or 'all' (default 'all')"},
                    "limit": {"type": "int", "required": False, "description": "Max items to return (default 10)"},
                    "filter_tag": {"type": "str", "required": False, "description": "Filter by tag"},
                },
            ),
            SkillAction(
                name="evolve_playbook",
                description="Update a playbook based on new reflections and usage data",
                parameters={
                    "playbook_name": {"type": "str", "required": True, "description": "Name of playbook to evolve"},
                    "add_steps": {"type": "list", "required": False, "description": "New steps to add"},
                    "remove_steps": {"type": "list", "required": False, "description": "Step indices to remove"},
                    "add_pitfalls": {"type": "list", "required": False, "description": "New pitfalls to add"},
                    "update_pattern": {"type": "str", "required": False, "description": "Updated task pattern description"},
                },
            ),
        ]

    async def execute(self, action: str, params: Dict) -> SkillResult:
        """Execute a reflection action."""
        handlers = {
            "reflect": self._reflect,
            "create_playbook": self._create_playbook,
            "find_playbook": self._find_playbook,
            "record_playbook_use": self._record_playbook_use,
            "extract_patterns": self._extract_patterns,
            "add_insight": self._add_insight,
            "review": self._review,
            "evolve_playbook": self._evolve_playbook,
        }

        handler = handlers.get(action)
        if not handler:
            return SkillResult(success=False, message=f"Unknown action: {action}")

        try:
            return await handler(params)
        except Exception as e:
            return SkillResult(success=False, message=f"Reflection error: {str(e)}")

    def _generate_id(self, content: str) -> str:
        """Generate a short unique ID."""
        return hashlib.sha256(f"{content}{time.time()}".encode()).hexdigest()[:12]

    async def _reflect(self, params: Dict) -> SkillResult:
        """Record a post-action reflection."""
        task = params.get("task", "")
        actions_taken = params.get("actions_taken", [])
        outcome = params.get("outcome", "")
        success = params.get("success", False)
        analysis = params.get("analysis", "")
        improvements = params.get("improvements", [])
        tags = params.get("tags", [])

        if not task or not outcome or not analysis:
            return SkillResult(
                success=False,
                message="Required: task, outcome, and analysis",
            )

        reflection_id = self._generate_id(task)

        reflection = {
            "id": reflection_id,
            "timestamp": datetime.utcnow().isoformat(),
            "task": task,
            "actions_taken": actions_taken,
            "outcome": outcome,
            "success": success,
            "analysis": analysis,
            "improvements": improvements,
            "tags": tags,
        }

        self._reflections.append(reflection)
        self._stats["total_reflections"] += 1

        # Trim if over limit
        if len(self._reflections) > MAX_REFLECTIONS:
            self._reflections = self._reflections[-MAX_REFLECTIONS:]

        self._save()

        return SkillResult(
            success=True,
            message=f"Reflection recorded: {reflection_id}",
            data={
                "reflection_id": reflection_id,
                "total_reflections": self._stats["total_reflections"],
                "tags": tags,
            },
        )

    async def _create_playbook(self, params: Dict) -> SkillResult:
        """Create a reusable playbook from experience."""
        name = params.get("name", "")
        task_pattern = params.get("task_pattern", "")
        steps = params.get("steps", [])
        pitfalls = params.get("pitfalls", [])
        prerequisites = params.get("prerequisites", [])
        expected_outcome = params.get("expected_outcome", "")
        tags = params.get("tags", [])

        if not name or not task_pattern or not steps:
            return SkillResult(
                success=False,
                message="Required: name, task_pattern, and steps",
            )

        if name in self._playbooks:
            return SkillResult(
                success=False,
                message=f"Playbook '{name}' already exists. Use evolve_playbook to update it.",
            )

        playbook = {
            "name": name,
            "task_pattern": task_pattern,
            "steps": steps,
            "pitfalls": pitfalls,
            "prerequisites": prerequisites,
            "expected_outcome": expected_outcome,
            "tags": tags,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "version": 1,
            "uses": 0,
            "successes": 0,
            "effectiveness": 0.0,
        }

        self._playbooks[name] = playbook
        self._stats["total_playbooks"] += 1
        self._save()

        return SkillResult(
            success=True,
            message=f"Playbook '{name}' created with {len(steps)} steps",
            data={"playbook": playbook},
        )

    async def _find_playbook(self, params: Dict) -> SkillResult:
        """Find the best matching playbook for a task."""
        task_description = params.get("task_description", "").lower()
        filter_tags = params.get("tags", [])

        if not task_description:
            return SkillResult(
                success=False,
                message="Required: task_description",
            )

        if not self._playbooks:
            return SkillResult(
                success=True,
                message="No playbooks available yet. Create some after reflecting on past tasks.",
                data={"matches": []},
            )

        # Score each playbook by keyword overlap + tag match + effectiveness
        scored = []
        task_words = set(task_description.split())

        for name, pb in self._playbooks.items():
            score = 0.0

            # Keyword overlap with task_pattern
            pattern_words = set(pb["task_pattern"].lower().split())
            overlap = len(task_words & pattern_words)
            if pattern_words:
                score += (overlap / len(pattern_words)) * 50

            # Check if task description contains the pattern as substring
            if pb["task_pattern"].lower() in task_description:
                score += 30

            # Tag match
            pb_tags = set(t.lower() for t in pb.get("tags", []))
            filter_tags_lower = set(t.lower() for t in filter_tags)
            if filter_tags_lower and pb_tags:
                tag_overlap = len(filter_tags_lower & pb_tags)
                score += tag_overlap * 10

            # Effectiveness bonus (proven playbooks get preference)
            effectiveness = pb.get("effectiveness", 0)
            score += effectiveness * 20

            # Usage bonus (more used = more trusted)
            uses = pb.get("uses", 0)
            if uses > 0:
                score += min(uses, 10) * 2

            if score > 0:
                scored.append({
                    "name": name,
                    "score": round(score, 2),
                    "task_pattern": pb["task_pattern"],
                    "steps": pb["steps"],
                    "pitfalls": pb.get("pitfalls", []),
                    "effectiveness": effectiveness,
                    "uses": uses,
                })

        # Sort by score descending
        scored.sort(key=lambda x: x["score"], reverse=True)
        top_matches = scored[:5]

        if not top_matches:
            return SkillResult(
                success=True,
                message="No matching playbooks found for this task type.",
                data={"matches": []},
            )

        return SkillResult(
            success=True,
            message=f"Found {len(top_matches)} matching playbook(s). Best: '{top_matches[0]['name']}' (score: {top_matches[0]['score']})",
            data={"matches": top_matches},
        )

    async def _record_playbook_use(self, params: Dict) -> SkillResult:
        """Record that a playbook was used and its outcome."""
        name = params.get("playbook_name", "")
        success = params.get("success", False)
        notes = params.get("notes", "")

        if not name:
            return SkillResult(success=False, message="Required: playbook_name")

        if name not in self._playbooks:
            return SkillResult(
                success=False,
                message=f"Playbook '{name}' not found",
            )

        pb = self._playbooks[name]
        pb["uses"] += 1
        if success:
            pb["successes"] += 1
        pb["effectiveness"] = pb["successes"] / pb["uses"] if pb["uses"] > 0 else 0.0

        self._stats["playbook_uses"] += 1
        if success:
            self._stats["playbook_successes"] += 1

        # Record usage in history
        if "usage_history" not in pb:
            pb["usage_history"] = []
        pb["usage_history"].append({
            "timestamp": datetime.utcnow().isoformat(),
            "success": success,
            "notes": notes,
        })
        # Keep only last 20 uses
        pb["usage_history"] = pb["usage_history"][-20:]

        self._save()

        return SkillResult(
            success=True,
            message=f"Playbook '{name}' usage recorded. Effectiveness: {pb['effectiveness']:.0%} ({pb['successes']}/{pb['uses']})",
            data={
                "playbook_name": name,
                "uses": pb["uses"],
                "successes": pb["successes"],
                "effectiveness": pb["effectiveness"],
            },
        )

    async def _extract_patterns(self, params: Dict) -> SkillResult:
        """Analyze recent reflections to extract recurring patterns."""
        lookback = params.get("lookback", 20)
        filter_tag = params.get("filter_tag", "")

        recent = self._reflections[-lookback:]
        if filter_tag:
            recent = [r for r in recent if filter_tag in r.get("tags", [])]

        if len(recent) < 2:
            return SkillResult(
                success=True,
                message="Not enough reflections to extract patterns (need at least 2).",
                data={"patterns": []},
            )

        # Analyze success/failure rates
        total = len(recent)
        successes = sum(1 for r in recent if r.get("success"))
        failures = total - successes

        # Extract common tags
        tag_counts: Dict[str, Dict] = {}
        for r in recent:
            for tag in r.get("tags", []):
                if tag not in tag_counts:
                    tag_counts[tag] = {"total": 0, "successes": 0}
                tag_counts[tag]["total"] += 1
                if r.get("success"):
                    tag_counts[tag]["successes"] += 1

        # Tag success rates
        tag_patterns = []
        for tag, counts in sorted(tag_counts.items(), key=lambda x: x[1]["total"], reverse=True):
            rate = counts["successes"] / counts["total"] if counts["total"] > 0 else 0
            tag_patterns.append({
                "tag": tag,
                "total": counts["total"],
                "success_rate": round(rate, 2),
                "assessment": "strong" if rate >= 0.8 else "weak" if rate <= 0.3 else "moderate",
            })

        # Extract common improvement themes from failure reflections
        failure_reflections = [r for r in recent if not r.get("success")]
        improvement_themes: Dict[str, int] = {}
        for r in failure_reflections:
            for imp in r.get("improvements", []):
                # Simple word-level theme extraction
                key_words = [w.lower() for w in imp.split() if len(w) > 4]
                for word in key_words:
                    improvement_themes[word] = improvement_themes.get(word, 0) + 1

        # Top recurring improvement themes
        top_themes = sorted(improvement_themes.items(), key=lambda x: x[1], reverse=True)[:10]

        # Extract common action sequences from successes
        success_reflections = [r for r in recent if r.get("success")]
        action_frequency: Dict[str, int] = {}
        for r in success_reflections:
            for action in r.get("actions_taken", []):
                action_str = str(action) if not isinstance(action, str) else action
                action_frequency[action_str] = action_frequency.get(action_str, 0) + 1

        top_actions = sorted(action_frequency.items(), key=lambda x: x[1], reverse=True)[:10]

        patterns = {
            "summary": {
                "total_reflections": total,
                "success_rate": round(successes / total, 2) if total > 0 else 0,
                "successes": successes,
                "failures": failures,
            },
            "tag_patterns": tag_patterns,
            "recurring_improvement_themes": [
                {"theme": theme, "occurrences": count}
                for theme, count in top_themes
            ],
            "successful_action_patterns": [
                {"action": action, "frequency": count}
                for action, count in top_actions
            ],
        }

        return SkillResult(
            success=True,
            message=f"Extracted patterns from {total} reflections: {successes}/{total} success rate",
            data={"patterns": patterns},
        )

    async def _add_insight(self, params: Dict) -> SkillResult:
        """Record a strategic insight."""
        insight_text = params.get("insight", "")
        category = params.get("category", "general")
        confidence = params.get("confidence", 0.5)
        source_reflections = params.get("source_reflections", [])

        if not insight_text:
            return SkillResult(success=False, message="Required: insight")

        confidence = max(0.0, min(1.0, float(confidence)))

        insight = {
            "id": self._generate_id(insight_text),
            "timestamp": datetime.utcnow().isoformat(),
            "insight": insight_text,
            "category": category,
            "confidence": confidence,
            "source_reflections": source_reflections,
            "validated": False,
            "validation_count": 0,
        }

        self._insights.append(insight)

        if len(self._insights) > MAX_INSIGHTS:
            self._insights = self._insights[-MAX_INSIGHTS:]

        self._save()

        return SkillResult(
            success=True,
            message=f"Insight recorded (confidence: {confidence:.0%}): {insight_text[:80]}...",
            data={"insight": insight},
        )

    async def _review(self, params: Dict) -> SkillResult:
        """Review reflections, playbooks, and/or insights."""
        what = params.get("what", "all")
        limit = params.get("limit", 10)
        filter_tag = params.get("filter_tag", "")

        result_data: Dict[str, Any] = {}

        if what in ("reflections", "all"):
            reflections = self._reflections[-limit:]
            if filter_tag:
                reflections = [r for r in reflections if filter_tag in r.get("tags", [])]
            result_data["reflections"] = reflections[-limit:]

        if what in ("playbooks", "all"):
            playbooks = list(self._playbooks.values())
            if filter_tag:
                playbooks = [p for p in playbooks if filter_tag in p.get("tags", [])]
            # Sort by effectiveness
            playbooks.sort(key=lambda p: p.get("effectiveness", 0), reverse=True)
            result_data["playbooks"] = playbooks[:limit]

        if what in ("insights", "all"):
            insights = self._insights[-limit:]
            if filter_tag:
                insights = [i for i in insights if i.get("category") == filter_tag]
            result_data["insights"] = insights[-limit:]

        result_data["stats"] = self._stats

        return SkillResult(
            success=True,
            message=f"Review: {len(self._reflections)} reflections, {len(self._playbooks)} playbooks, {len(self._insights)} insights",
            data=result_data,
        )

    async def _evolve_playbook(self, params: Dict) -> SkillResult:
        """Evolve a playbook based on new experience."""
        name = params.get("playbook_name", "")
        add_steps = params.get("add_steps", [])
        remove_steps = params.get("remove_steps", [])
        add_pitfalls = params.get("add_pitfalls", [])
        update_pattern = params.get("update_pattern", "")

        if not name:
            return SkillResult(success=False, message="Required: playbook_name")

        if name not in self._playbooks:
            return SkillResult(
                success=False,
                message=f"Playbook '{name}' not found",
            )

        pb = self._playbooks[name]
        changes = []

        # Remove steps (process in reverse order to maintain indices)
        if remove_steps:
            for idx in sorted(remove_steps, reverse=True):
                if 0 <= idx < len(pb["steps"]):
                    removed = pb["steps"].pop(idx)
                    changes.append(f"Removed step {idx}: {removed[:50]}")

        # Add steps
        if add_steps:
            pb["steps"].extend(add_steps)
            changes.append(f"Added {len(add_steps)} new step(s)")

        # Add pitfalls
        if add_pitfalls:
            pb["pitfalls"] = pb.get("pitfalls", []) + add_pitfalls
            changes.append(f"Added {len(add_pitfalls)} new pitfall(s)")

        # Update pattern
        if update_pattern:
            pb["task_pattern"] = update_pattern
            changes.append(f"Updated task pattern to: {update_pattern[:60]}")

        pb["version"] = pb.get("version", 1) + 1
        pb["updated_at"] = datetime.utcnow().isoformat()

        self._save()

        return SkillResult(
            success=True,
            message=f"Playbook '{name}' evolved to v{pb['version']}: {', '.join(changes)}",
            data={"playbook": pb, "changes": changes},
        )

    def estimate_cost(self, action: str, params: Dict) -> float:
        """All reflection actions are free (local computation)."""
        return 0.0
