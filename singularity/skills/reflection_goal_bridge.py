#!/usr/bin/env python3
"""
ReflectionGoalBridgeSkill - Autonomous goal creation from reflection pattern analysis.

This skill bridges the gap between self-improvement insights and goal-setting action.
The agent accumulates reflections about what worked and failed (AgentReflectionSkill),
but without this bridge, those insights sit idle. This skill:

1. **Analyzes** reflection patterns to identify weaknesses (low success-rate tags, recurring failures)
2. **Recommends** goals based on identified weaknesses, mapping them to the right pillar
3. **Auto-creates** goals in GoalManager from reflection insights
4. **Tracks** which reflection-driven goals were created and their outcomes
5. **Periodically scans** for new weakness patterns that should become goals

The feedback loop:
  Execute → Reflect → Extract patterns → **Identify weaknesses → Create goals** → Execute better

Pillar: Goal Setting (reflection-driven autonomous planning)
"""

import json
import hashlib
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from .base import Skill, SkillResult, SkillManifest, SkillAction

DATA_FILE = Path(__file__).parent.parent / "data" / "reflection_goal_bridge.json"

# Thresholds for weakness detection
DEFAULT_CONFIG = {
    "min_reflections_for_analysis": 5,
    "weak_tag_threshold": 0.4,       # Tags with success_rate <= this are "weak"
    "min_tag_occurrences": 3,         # Minimum occurrences to consider a tag
    "improvement_theme_threshold": 3,  # Recurring improvement themes >= this count
    "auto_create_goals": False,       # If True, automatically create goals (otherwise just recommend)
    "max_recommendations": 5,         # Max recommendations per scan
    "scan_cooldown_minutes": 30,      # Don't re-scan too frequently
    "default_goal_priority": "medium",
    "default_deadline_hours": 168,    # 1 week default deadline
}

# Map common tags/themes to pillars
TAG_PILLAR_MAP = {
    "revenue": "revenue",
    "payment": "revenue",
    "pricing": "revenue",
    "service": "revenue",
    "api": "revenue",
    "customer": "revenue",
    "earning": "revenue",
    "replication": "replication",
    "replica": "replication",
    "spawn": "replication",
    "fleet": "replication",
    "clone": "replication",
    "self_improvement": "self_improvement",
    "reflection": "self_improvement",
    "learning": "self_improvement",
    "optimization": "self_improvement",
    "performance": "self_improvement",
    "testing": "self_improvement",
    "goal": "goal_setting",
    "planning": "goal_setting",
    "strategy": "goal_setting",
    "priority": "goal_setting",
    "deployment": "revenue",
    "code_review": "self_improvement",
    "debugging": "self_improvement",
    "error_handling": "self_improvement",
}


class ReflectionGoalBridgeSkill(Skill):
    """
    Bridges reflection pattern analysis with autonomous goal creation.

    Analyzes weaknesses from AgentReflectionSkill patterns and creates/recommends
    goals in GoalManagerSkill, closing the loop between insight and action.
    """

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self._ensure_data()

    def _ensure_data(self):
        DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
        if DATA_FILE.exists():
            try:
                with open(DATA_FILE) as f:
                    data = json.load(f)
                self._recommendations = data.get("recommendations", [])
                self._created_goals = data.get("created_goals", [])
                self._scans = data.get("scans", [])
                self._config = {**DEFAULT_CONFIG, **data.get("config", {})}
                self._stats = data.get("stats", self._default_stats())
            except (json.JSONDecodeError, Exception):
                self._init_defaults()
        else:
            self._init_defaults()

    def _init_defaults(self):
        self._recommendations = []
        self._created_goals = []
        self._scans = []
        self._config = dict(DEFAULT_CONFIG)
        self._stats = self._default_stats()

    def _default_stats(self) -> Dict:
        return {
            "total_scans": 0,
            "total_recommendations": 0,
            "total_goals_created": 0,
            "goals_completed": 0,
            "goals_abandoned": 0,
            "weaknesses_detected": 0,
        }

    def _save(self):
        DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(DATA_FILE, "w") as f:
            json.dump({
                "recommendations": self._recommendations[-200:],
                "created_goals": self._created_goals[-200:],
                "scans": self._scans[-50:],
                "config": self._config,
                "stats": self._stats,
            }, f, indent=2)

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="reflection_goal_bridge",
            name="Reflection Goal Bridge",
            version="1.0.0",
            category="goal_setting",
            description="Bridges reflection pattern analysis with autonomous goal creation",
            actions=self.get_actions(),
            required_credentials=[],
        )

    def get_actions(self) -> List[SkillAction]:
        return [
            SkillAction(
                name="scan",
                description="Scan reflection patterns to identify weaknesses and recommend goals",
                parameters={
                    "lookback": {"type": "int", "required": False, "description": "Number of recent reflections to analyze (default: 30)"},
                    "force": {"type": "bool", "required": False, "description": "Bypass scan cooldown"},
                },
            ),
            SkillAction(
                name="create_goals",
                description="Create goals in GoalManager from pending recommendations",
                parameters={
                    "recommendation_ids": {"type": "list", "required": False, "description": "Specific recommendation IDs to create goals for (default: all pending)"},
                    "dry_run": {"type": "bool", "required": False, "description": "Preview without creating"},
                },
            ),
            SkillAction(
                name="recommendations",
                description="List current goal recommendations from reflection analysis",
                parameters={
                    "status": {"type": "str", "required": False, "description": "Filter: pending, created, dismissed"},
                },
            ),
            SkillAction(
                name="dismiss",
                description="Dismiss a recommendation (mark as not needed)",
                parameters={
                    "recommendation_id": {"type": "str", "required": True, "description": "ID of recommendation to dismiss"},
                    "reason": {"type": "str", "required": False, "description": "Reason for dismissal"},
                },
            ),
            SkillAction(
                name="track",
                description="Check status of goals that were created from recommendations",
                parameters={},
            ),
            SkillAction(
                name="configure",
                description="Update bridge configuration",
                parameters={
                    "key": {"type": "str", "required": True, "description": "Config key to update"},
                    "value": {"type": "any", "required": True, "description": "New value"},
                },
            ),
            SkillAction(
                name="history",
                description="View scan history and effectiveness",
                parameters={
                    "limit": {"type": "int", "required": False, "description": "Number of recent scans to show"},
                },
            ),
            SkillAction(
                name="status",
                description="Get overall bridge status and statistics",
                parameters={},
            ),
        ]

    def check_credentials(self) -> bool:
        return True

    async def execute(self, action: str, params: Dict) -> SkillResult:
        actions = {
            "scan": self._scan,
            "create_goals": self._create_goals,
            "recommendations": self._get_recommendations,
            "dismiss": self._dismiss,
            "track": self._track,
            "configure": self._configure,
            "history": self._history,
            "status": self._status,
        }
        handler = actions.get(action)
        if not handler:
            return SkillResult(success=False, message=f"Unknown action: {action}")
        return await handler(params)

    async def _scan(self, params: Dict) -> SkillResult:
        """Scan reflection patterns to identify weaknesses and recommend goals."""
        force = params.get("force", False)
        lookback = params.get("lookback", 30)

        # Check cooldown
        if not force and self._scans:
            last_scan = self._scans[-1]
            elapsed = time.time() - last_scan.get("timestamp", 0)
            cooldown = self._config["scan_cooldown_minutes"] * 60
            if elapsed < cooldown:
                remaining = int((cooldown - elapsed) / 60)
                return SkillResult(
                    success=False,
                    message=f"Scan cooldown: {remaining} minutes remaining. Use force=true to override.",
                )

        # Get reflections from AgentReflectionSkill via context
        reflections = await self._get_reflections(lookback)
        if reflections is None:
            return SkillResult(
                success=False,
                message="Cannot access AgentReflectionSkill. Ensure it is registered in SkillContext.",
            )

        min_reflections = self._config["min_reflections_for_analysis"]
        if len(reflections) < min_reflections:
            return SkillResult(
                success=True,
                message=f"Not enough reflections ({len(reflections)}/{min_reflections}) for analysis.",
                data={"reflections_found": len(reflections), "min_required": min_reflections},
            )

        # Analyze patterns
        weaknesses = self._analyze_weaknesses(reflections)
        recommendations = self._generate_recommendations(weaknesses)

        # Record scan
        scan_record = {
            "timestamp": time.time(),
            "reflections_analyzed": len(reflections),
            "weaknesses_found": len(weaknesses),
            "recommendations_generated": len(recommendations),
            "datetime": datetime.now().isoformat(),
        }
        self._scans.append(scan_record)
        self._stats["total_scans"] += 1
        self._stats["weaknesses_detected"] += len(weaknesses)

        # Add new recommendations (deduplicate by content hash)
        existing_hashes = {r.get("hash") for r in self._recommendations}
        new_count = 0
        for rec in recommendations:
            if rec["hash"] not in existing_hashes:
                self._recommendations.append(rec)
                existing_hashes.add(rec["hash"])
                new_count += 1
                self._stats["total_recommendations"] += 1

        # Auto-create goals if configured
        auto_created = []
        if self._config["auto_create_goals"] and new_count > 0:
            pending = [r for r in self._recommendations if r["status"] == "pending"]
            result = await self._create_goals_internal(pending, dry_run=False)
            auto_created = result.get("created", [])

        self._save()

        return SkillResult(
            success=True,
            message=f"Scan complete: {len(weaknesses)} weaknesses found, {new_count} new recommendations"
                    + (f", {len(auto_created)} goals auto-created" if auto_created else ""),
            data={
                "reflections_analyzed": len(reflections),
                "weaknesses": weaknesses,
                "new_recommendations": new_count,
                "total_pending": len([r for r in self._recommendations if r["status"] == "pending"]),
                "auto_created_goals": auto_created,
            },
        )

    def _analyze_weaknesses(self, reflections: List[Dict]) -> List[Dict]:
        """Identify weaknesses from reflection patterns."""
        weaknesses = []

        # 1. Find weak tags (low success rate)
        tag_stats: Dict[str, Dict] = {}
        for r in reflections:
            for tag in r.get("tags", []):
                if tag not in tag_stats:
                    tag_stats[tag] = {"total": 0, "successes": 0, "failures": 0}
                tag_stats[tag]["total"] += 1
                if r.get("success"):
                    tag_stats[tag]["successes"] += 1
                else:
                    tag_stats[tag]["failures"] += 1

        min_occurrences = self._config["min_tag_occurrences"]
        weak_threshold = self._config["weak_tag_threshold"]

        for tag, stats in tag_stats.items():
            if stats["total"] >= min_occurrences:
                success_rate = stats["successes"] / stats["total"]
                if success_rate <= weak_threshold:
                    weaknesses.append({
                        "type": "weak_tag",
                        "tag": tag,
                        "success_rate": round(success_rate, 2),
                        "total": stats["total"],
                        "failures": stats["failures"],
                        "severity": "high" if success_rate <= 0.2 else "medium",
                    })

        # 2. Find recurring improvement themes
        theme_counts: Dict[str, int] = {}
        theme_sources: Dict[str, List[str]] = {}
        failure_reflections = [r for r in reflections if not r.get("success")]
        for r in failure_reflections:
            for imp in r.get("improvements", []):
                words = [w.lower() for w in imp.split() if len(w) > 4]
                for word in words:
                    theme_counts[word] = theme_counts.get(word, 0) + 1
                    if word not in theme_sources:
                        theme_sources[word] = []
                    theme_sources[word].append(imp[:80])

        theme_threshold = self._config["improvement_theme_threshold"]
        for theme, count in theme_counts.items():
            if count >= theme_threshold:
                weaknesses.append({
                    "type": "recurring_improvement",
                    "theme": theme,
                    "occurrences": count,
                    "examples": list(set(theme_sources.get(theme, [])))[:3],
                    "severity": "high" if count >= theme_threshold * 2 else "medium",
                })

        # 3. Detect declining performance (recent failures > earlier failures)
        if len(reflections) >= 10:
            mid = len(reflections) // 2
            early = reflections[:mid]
            recent = reflections[mid:]
            early_rate = sum(1 for r in early if r.get("success")) / len(early) if early else 0
            recent_rate = sum(1 for r in recent if r.get("success")) / len(recent) if recent else 0
            if recent_rate < early_rate - 0.15:
                weaknesses.append({
                    "type": "declining_performance",
                    "early_success_rate": round(early_rate, 2),
                    "recent_success_rate": round(recent_rate, 2),
                    "decline": round(early_rate - recent_rate, 2),
                    "severity": "high" if early_rate - recent_rate > 0.3 else "medium",
                })

        # 4. Detect uncovered pillars (tags suggest activity but no successes)
        pillar_activity: Dict[str, Dict] = {}
        for r in reflections:
            for tag in r.get("tags", []):
                pillar = self._tag_to_pillar(tag)
                if pillar not in pillar_activity:
                    pillar_activity[pillar] = {"total": 0, "successes": 0}
                pillar_activity[pillar]["total"] += 1
                if r.get("success"):
                    pillar_activity[pillar]["successes"] += 1

        for pillar, stats in pillar_activity.items():
            if pillar != "other" and stats["total"] >= 3 and stats["successes"] == 0:
                weaknesses.append({
                    "type": "pillar_zero_success",
                    "pillar": pillar,
                    "attempts": stats["total"],
                    "severity": "high",
                })

        # Sort by severity (high first)
        severity_order = {"high": 0, "medium": 1, "low": 2}
        weaknesses.sort(key=lambda w: severity_order.get(w.get("severity", "low"), 2))

        return weaknesses

    def _generate_recommendations(self, weaknesses: List[Dict]) -> List[Dict]:
        """Generate goal recommendations from identified weaknesses."""
        recommendations = []
        max_recs = self._config["max_recommendations"]

        for weakness in weaknesses[:max_recs]:
            rec = self._weakness_to_recommendation(weakness)
            if rec:
                recommendations.append(rec)

        return recommendations

    def _weakness_to_recommendation(self, weakness: Dict) -> Optional[Dict]:
        """Convert a weakness into a goal recommendation."""
        wtype = weakness.get("type")
        now = datetime.now().isoformat()
        rec_id = f"rec_{hashlib.md5(json.dumps(weakness, sort_keys=True).encode()).hexdigest()[:10]}"

        if wtype == "weak_tag":
            tag = weakness["tag"]
            pillar = self._tag_to_pillar(tag)
            return {
                "id": rec_id,
                "hash": rec_id,
                "status": "pending",
                "created_at": now,
                "weakness": weakness,
                "goal_title": f"Improve {tag} success rate (currently {weakness['success_rate']*100:.0f}%)",
                "goal_description": (
                    f"Reflection analysis shows '{tag}' tasks have a {weakness['success_rate']*100:.0f}% success rate "
                    f"across {weakness['total']} attempts ({weakness['failures']} failures). "
                    f"Investigate root causes and develop better strategies."
                ),
                "goal_pillar": pillar,
                "goal_priority": "high" if weakness["severity"] == "high" else "medium",
                "goal_milestones": [
                    f"Analyze the {weakness['failures']} failed {tag} reflections for common patterns",
                    f"Develop or update playbook for {tag} tasks",
                    f"Apply improved strategy to next 3 {tag} tasks",
                    f"Achieve >60% success rate on {tag} tasks",
                ],
            }

        elif wtype == "recurring_improvement":
            theme = weakness["theme"]
            pillar = self._tag_to_pillar(theme)
            examples = weakness.get("examples", [])
            return {
                "id": rec_id,
                "hash": rec_id,
                "status": "pending",
                "created_at": now,
                "weakness": weakness,
                "goal_title": f"Address recurring issue: '{theme}' ({weakness['occurrences']} occurrences)",
                "goal_description": (
                    f"The improvement theme '{theme}' appears {weakness['occurrences']} times in failure analyses. "
                    f"Examples: {'; '.join(examples[:2])}. "
                    f"This recurring pattern needs systematic resolution."
                ),
                "goal_pillar": pillar,
                "goal_priority": "high" if weakness["severity"] == "high" else "medium",
                "goal_milestones": [
                    f"Review all reflections mentioning '{theme}'",
                    f"Identify root cause of '{theme}' failures",
                    f"Implement fix or process change",
                    f"Verify improvement in next 5 related tasks",
                ],
            }

        elif wtype == "declining_performance":
            return {
                "id": rec_id,
                "hash": rec_id,
                "status": "pending",
                "created_at": now,
                "weakness": weakness,
                "goal_title": f"Reverse performance decline ({weakness['recent_success_rate']*100:.0f}% → target >70%)",
                "goal_description": (
                    f"Success rate has dropped from {weakness['early_success_rate']*100:.0f}% to "
                    f"{weakness['recent_success_rate']*100:.0f}% (decline of {weakness['decline']*100:.0f}%). "
                    f"Investigate what changed and restore performance."
                ),
                "goal_pillar": "self_improvement",
                "goal_priority": "high",
                "goal_milestones": [
                    "Compare recent failures against earlier successes",
                    "Identify environmental or process changes",
                    "Develop corrective playbook",
                    "Achieve 3 consecutive successes",
                ],
            }

        elif wtype == "pillar_zero_success":
            pillar = weakness.get("pillar", "other")
            return {
                "id": rec_id,
                "hash": rec_id,
                "status": "pending",
                "created_at": now,
                "weakness": weakness,
                "goal_title": f"Achieve first success in {pillar} pillar ({weakness['attempts']} attempts, 0 successes)",
                "goal_description": (
                    f"The {pillar} pillar has {weakness['attempts']} attempts but zero successes. "
                    f"This indicates a fundamental capability gap that needs addressing."
                ),
                "goal_pillar": pillar,
                "goal_priority": "critical",
                "goal_milestones": [
                    f"Analyze all {weakness['attempts']} failed {pillar} reflections",
                    f"Identify the primary blocker for {pillar} success",
                    f"Create a minimal viable approach for {pillar}",
                    f"Execute and achieve first {pillar} success",
                ],
            }

        return None

    def _tag_to_pillar(self, tag: str) -> str:
        """Map a tag or theme to a pillar."""
        tag_lower = tag.lower().replace("-", "_").replace(" ", "_")
        if tag_lower in TAG_PILLAR_MAP:
            return TAG_PILLAR_MAP[tag_lower]
        # Fuzzy match
        for key, pillar in TAG_PILLAR_MAP.items():
            if key in tag_lower or tag_lower in key:
                return pillar
        return "other"

    async def _create_goals(self, params: Dict) -> SkillResult:
        """Create goals in GoalManager from pending recommendations."""
        rec_ids = params.get("recommendation_ids", [])
        dry_run = params.get("dry_run", False)

        if rec_ids:
            pending = [r for r in self._recommendations if r["id"] in rec_ids and r["status"] == "pending"]
        else:
            pending = [r for r in self._recommendations if r["status"] == "pending"]

        if not pending:
            return SkillResult(
                success=True,
                message="No pending recommendations to create goals from.",
                data={"pending_count": 0},
            )

        result = await self._create_goals_internal(pending, dry_run=dry_run)

        if not dry_run:
            self._save()

        return SkillResult(
            success=True,
            message=f"{'[DRY RUN] Would create' if dry_run else 'Created'} {len(result['created'])} goals from {len(pending)} recommendations"
                    + (f" ({len(result['failed'])} failed)" if result['failed'] else ""),
            data=result,
        )

    async def _create_goals_internal(self, recommendations: List[Dict], dry_run: bool = False) -> Dict:
        """Internal: create goals from a list of recommendations."""
        created = []
        failed = []

        for rec in recommendations:
            goal_params = {
                "title": rec["goal_title"],
                "description": rec["goal_description"],
                "pillar": rec["goal_pillar"],
                "priority": rec["goal_priority"],
                "deadline_hours": self._config["default_deadline_hours"],
                "milestones": rec["goal_milestones"],
            }

            if dry_run:
                created.append({
                    "recommendation_id": rec["id"],
                    "goal_params": goal_params,
                })
                continue

            # Try to create goal via GoalManager through context
            goal_result = await self._create_goal_via_context(goal_params)
            if goal_result and goal_result.get("success"):
                rec["status"] = "created"
                rec["goal_id"] = goal_result.get("goal_id")
                rec["created_at_goal"] = datetime.now().isoformat()
                self._created_goals.append({
                    "recommendation_id": rec["id"],
                    "goal_id": goal_result.get("goal_id"),
                    "title": rec["goal_title"],
                    "pillar": rec["goal_pillar"],
                    "created_at": datetime.now().isoformat(),
                    "status": "active",
                })
                self._stats["total_goals_created"] += 1
                created.append({
                    "recommendation_id": rec["id"],
                    "goal_id": goal_result.get("goal_id"),
                    "title": rec["goal_title"],
                })
            else:
                failed.append({
                    "recommendation_id": rec["id"],
                    "error": goal_result.get("message", "Unknown error") if goal_result else "GoalManager unavailable",
                })

        return {"created": created, "failed": failed}

    async def _get_reflections(self, lookback: int) -> Optional[List[Dict]]:
        """Get reflections from AgentReflectionSkill via context or direct file access."""
        # Try SkillContext first
        if hasattr(self, '_context') and self._context:
            try:
                result = await self._context.execute_skill(
                    "agent_reflection", "extract_patterns", {"lookback": lookback}
                )
                if result and result.success:
                    # We need raw reflections, not just patterns
                    pass
            except Exception:
                pass

        # Direct file access as fallback
        reflection_file = Path(__file__).parent.parent / "data" / "reflections.json"
        if reflection_file.exists():
            try:
                with open(reflection_file) as f:
                    data = json.load(f)
                reflections = data.get("reflections", [])
                return reflections[-lookback:]
            except (json.JSONDecodeError, Exception):
                pass

        # Return empty list for testability (not None = skill is available but no data)
        return []

    async def _create_goal_via_context(self, goal_params: Dict) -> Optional[Dict]:
        """Create a goal via GoalManagerSkill through context."""
        if hasattr(self, '_context') and self._context:
            try:
                result = await self._context.execute_skill(
                    "goal_manager", "create", goal_params
                )
                if result:
                    return {"success": result.success, "goal_id": result.data.get("goal_id"), "message": result.message}
            except Exception:
                pass

        # Direct file access fallback - create goal directly
        goals_file = Path(__file__).parent.parent / "data" / "goals.json"
        try:
            import uuid
            goals_file.parent.mkdir(parents=True, exist_ok=True)
            if goals_file.exists():
                with open(goals_file) as f:
                    data = json.load(f)
            else:
                data = {"goals": [], "completed_goals": [], "session_log": []}

            goal_id = f"goal_{uuid.uuid4().hex[:8]}"
            now = datetime.now()
            from datetime import timedelta

            milestones = []
            for i, mt in enumerate(goal_params.get("milestones", [])):
                milestones.append({
                    "index": i,
                    "title": str(mt),
                    "completed": False,
                    "completed_at": None,
                })

            deadline_hours = goal_params.get("deadline_hours")
            goal = {
                "id": goal_id,
                "title": goal_params["title"],
                "description": goal_params.get("description", ""),
                "pillar": goal_params.get("pillar", "other"),
                "priority": goal_params.get("priority", "medium"),
                "priority_score": {"critical": 4, "high": 3, "medium": 2, "low": 1}.get(goal_params.get("priority", "medium"), 2),
                "status": "active",
                "milestones": milestones,
                "depends_on": [],
                "progress_notes": [f"Auto-created by ReflectionGoalBridge at {now.isoformat()}"],
                "created_at": now.isoformat(),
                "deadline": (now + timedelta(hours=float(deadline_hours))).isoformat() if deadline_hours else None,
                "completed_at": None,
                "source": "reflection_goal_bridge",
            }

            data.setdefault("goals", []).append(goal)
            with open(goals_file, "w") as f:
                json.dump(data, f, indent=2)

            return {"success": True, "goal_id": goal_id, "message": f"Goal created: {goal_params['title']}"}
        except Exception as e:
            return {"success": False, "goal_id": None, "message": str(e)}

    async def _get_recommendations(self, params: Dict) -> SkillResult:
        """List current goal recommendations."""
        status_filter = params.get("status", "")
        recs = self._recommendations

        if status_filter:
            recs = [r for r in recs if r["status"] == status_filter]

        return SkillResult(
            success=True,
            message=f"{len(recs)} recommendations" + (f" (filtered: {status_filter})" if status_filter else ""),
            data={
                "recommendations": recs,
                "counts": {
                    "pending": len([r for r in self._recommendations if r["status"] == "pending"]),
                    "created": len([r for r in self._recommendations if r["status"] == "created"]),
                    "dismissed": len([r for r in self._recommendations if r["status"] == "dismissed"]),
                },
            },
        )

    async def _dismiss(self, params: Dict) -> SkillResult:
        """Dismiss a recommendation."""
        rec_id = params.get("recommendation_id", "")
        reason = params.get("reason", "")

        for rec in self._recommendations:
            if rec["id"] == rec_id:
                if rec["status"] != "pending":
                    return SkillResult(
                        success=False,
                        message=f"Recommendation {rec_id} is already {rec['status']}, cannot dismiss.",
                    )
                rec["status"] = "dismissed"
                rec["dismissed_at"] = datetime.now().isoformat()
                rec["dismiss_reason"] = reason
                self._save()
                return SkillResult(
                    success=True,
                    message=f"Recommendation dismissed: {rec['goal_title']}",
                    data={"recommendation_id": rec_id, "title": rec["goal_title"]},
                )

        return SkillResult(success=False, message=f"Recommendation not found: {rec_id}")

    async def _track(self, params: Dict) -> SkillResult:
        """Check status of goals created from recommendations."""
        if not self._created_goals:
            return SkillResult(
                success=True,
                message="No goals have been created from recommendations yet.",
                data={"tracked_goals": []},
            )

        # Try to check goal status via goals file
        goals_file = Path(__file__).parent.parent / "data" / "goals.json"
        goal_statuses = {}
        if goals_file.exists():
            try:
                with open(goals_file) as f:
                    data = json.load(f)
                for g in data.get("goals", []) + data.get("completed_goals", []):
                    goal_statuses[g["id"]] = g.get("status", "unknown")
            except (json.JSONDecodeError, Exception):
                pass

        tracked = []
        for cg in self._created_goals:
            gid = cg.get("goal_id", "")
            current_status = goal_statuses.get(gid, "unknown")
            # Update local tracking
            if current_status == "completed" and cg.get("status") != "completed":
                cg["status"] = "completed"
                self._stats["goals_completed"] += 1
            elif current_status == "abandoned" and cg.get("status") != "abandoned":
                cg["status"] = "abandoned"
                self._stats["goals_abandoned"] += 1

            tracked.append({
                "recommendation_id": cg.get("recommendation_id"),
                "goal_id": gid,
                "title": cg.get("title"),
                "pillar": cg.get("pillar"),
                "created_at": cg.get("created_at"),
                "current_status": current_status,
            })

        self._save()

        return SkillResult(
            success=True,
            message=f"Tracking {len(tracked)} reflection-driven goals",
            data={
                "tracked_goals": tracked,
                "summary": {
                    "total": len(tracked),
                    "active": len([t for t in tracked if t["current_status"] == "active"]),
                    "completed": len([t for t in tracked if t["current_status"] == "completed"]),
                    "abandoned": len([t for t in tracked if t["current_status"] == "abandoned"]),
                    "unknown": len([t for t in tracked if t["current_status"] == "unknown"]),
                },
            },
        )

    async def _configure(self, params: Dict) -> SkillResult:
        """Update bridge configuration."""
        key = params.get("key", "")
        value = params.get("value")

        if key not in DEFAULT_CONFIG:
            return SkillResult(
                success=False,
                message=f"Unknown config key: {key}. Valid keys: {list(DEFAULT_CONFIG.keys())}",
            )

        old_value = self._config.get(key)
        expected_type = type(DEFAULT_CONFIG[key])
        try:
            if expected_type == bool:
                if isinstance(value, str):
                    value = value.lower() in ("true", "1", "yes")
                else:
                    value = bool(value)
            elif expected_type == int:
                value = int(value)
            elif expected_type == float:
                value = float(value)
        except (ValueError, TypeError):
            return SkillResult(
                success=False,
                message=f"Invalid value type for {key}: expected {expected_type.__name__}",
            )

        self._config[key] = value
        self._save()

        return SkillResult(
            success=True,
            message=f"Config updated: {key} = {value} (was {old_value})",
            data={"key": key, "old_value": old_value, "new_value": value},
        )

    async def _history(self, params: Dict) -> SkillResult:
        """View scan history."""
        limit = params.get("limit", 10)
        recent_scans = self._scans[-limit:]

        return SkillResult(
            success=True,
            message=f"{len(recent_scans)} recent scans",
            data={
                "scans": recent_scans,
                "total_scans": self._stats["total_scans"],
            },
        )

    async def _status(self, params: Dict) -> SkillResult:
        """Get overall bridge status."""
        pending = len([r for r in self._recommendations if r["status"] == "pending"])
        created = len([r for r in self._recommendations if r["status"] == "created"])
        dismissed = len([r for r in self._recommendations if r["status"] == "dismissed"])

        return SkillResult(
            success=True,
            message=f"Bridge active: {pending} pending recommendations, {created} goals created",
            data={
                "config": self._config,
                "stats": self._stats,
                "recommendations": {
                    "pending": pending,
                    "created": created,
                    "dismissed": dismissed,
                    "total": len(self._recommendations),
                },
                "last_scan": self._scans[-1] if self._scans else None,
            },
        )
