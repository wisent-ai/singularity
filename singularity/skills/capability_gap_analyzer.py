#!/usr/bin/env python3
"""
CapabilityGapAnalyzerSkill - Autonomous introspection of agent capabilities.

This skill allows the agent to analyze its own skill inventory, identify
missing integrations between existing skills, score capability gaps by
strategic impact, and generate concrete work plans for future sessions.

This fills the critical "set its own goals" gap. Currently the agent has:
- GoalManager: tracks goals but doesn't generate them
- Strategy: assesses pillars but doesn't identify concrete next steps
- Planner: breaks tasks into steps but doesn't decide what tasks to create

CapabilityGapAnalyzer bridges these by:
1. Introspecting the agent's loaded skills and their actions
2. Detecting potential skill pairings that lack bridge/integration skills
3. Scoring gaps by pillar alignment, dependency count, and revenue potential
4. Generating session work plans with concrete deliverables

Pillar served: Goal Setting (primary), Self-Improvement (secondary)
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple
from .base import Skill, SkillResult, SkillManifest, SkillAction


GAP_DATA_FILE = Path(__file__).parent.parent / "data" / "capability_gaps.json"

# Known skill categories for classification
SKILL_CATEGORIES = {
    "revenue": [
        "billing", "payment", "pricing", "revenue", "marketplace",
        "usage_tracking", "task_pricing", "funding", "catalog",
    ],
    "replication": [
        "replication", "spawner", "fleet", "network", "delegation",
        "consensus", "peer", "checkpoint",
    ],
    "self_improvement": [
        "self_modify", "self_eval", "self_test", "experiment", "feedback",
        "learned_behavior", "reflection", "distillation", "tuning",
        "prompt_evolution", "performance_optimizer",
    ],
    "goal_setting": [
        "strategy", "planner", "goal", "scheduler", "workflow",
        "decision_log", "outcome_tracker",
    ],
    "infrastructure": [
        "event", "database", "http_client", "filesystem", "shell",
        "webhook", "secret_vault", "mcp_client", "browser",
    ],
    "monitoring": [
        "health_monitor", "observability", "dashboard", "service_monitor",
        "circuit_breaker", "alert", "diagnostics",
    ],
}

# Known bridge patterns: (source_keyword, target_keyword) -> bridge_keyword
BRIDGE_PATTERNS = [
    ("scheduler", "billing", "billing_scheduler_bridge"),
    ("dashboard", "observability", "dashboard_observability_bridge"),
    ("reflection", "event", "reflection_event_bridge"),
    ("checkpoint", "event", "checkpoint_event_bridge"),
    ("fleet_health", "event", "fleet_health_events"),
    ("circuit_breaker", "event", "circuit_breaker_event_bridge"),
    ("skill", "event", "skill_event_bridge"),
    ("http", "revenue", "http_revenue_bridge"),
    ("database", "revenue", "database_revenue_bridge"),
    ("workflow", "template", "workflow_template_bridge"),
    ("template", "event", "template_event_bridge"),
    ("workflow", "analytics", "workflow_analytics_bridge"),
    ("goal", "event", "goal_progress_events"),
    ("reputation", "event", "auto_reputation_bridge"),
    ("ssl", "service_hosting", "ssl_service_hosting_bridge"),
    ("serverless", "service_hosting", "serverless_service_hosting_bridge"),
]


class CapabilityGapAnalyzerSkill(Skill):
    """
    Introspects the agent's skill inventory and identifies capability gaps.

    Enables the agent to autonomously decide what to build next by analyzing
    its own skills, finding missing integrations, and scoring impact.
    """

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self._agent_skills: List[Any] = []
        self._ensure_data()

    def set_agent_skills(self, skills: List[Any]):
        """Inject the agent's loaded skill instances for introspection."""
        self._agent_skills = skills

    def _ensure_data(self):
        GAP_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
        if not GAP_DATA_FILE.exists():
            self._save(self._default_state())

    def _default_state(self) -> Dict:
        return {
            "analyses": [],
            "work_plans": [],
            "gap_history": [],
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
        }

    def _load(self) -> Dict:
        try:
            with open(GAP_DATA_FILE, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return self._default_state()

    def _save(self, data: Dict):
        data["last_updated"] = datetime.now().isoformat()
        GAP_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(GAP_DATA_FILE, "w") as f:
            json.dump(data, f, indent=2)

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="capability_gap_analyzer",
            name="Capability Gap Analyzer",
            version="1.0.0",
            category="meta-cognition",
            description="Introspects agent skills, identifies integration gaps, and generates work plans",
            actions=[
                SkillAction(
                    name="inventory",
                    description="List all loaded skills with their actions and categories",
                    parameters={},
                ),
                SkillAction(
                    name="analyze_gaps",
                    description="Analyze missing integrations between existing skills",
                    parameters={
                        "focus_pillar": {
                            "type": "string",
                            "required": False,
                            "description": "Optional pillar to focus on: revenue, replication, self_improvement, goal_setting",
                        },
                    },
                ),
                SkillAction(
                    name="score_gaps",
                    description="Score identified gaps by strategic impact and feasibility",
                    parameters={},
                ),
                SkillAction(
                    name="generate_plan",
                    description="Generate a concrete work plan for the next session",
                    parameters={
                        "max_items": {
                            "type": "integer",
                            "required": False,
                            "description": "Maximum number of items in the plan (default 5)",
                        },
                    },
                ),
                SkillAction(
                    name="pillar_coverage",
                    description="Show skill coverage per pillar with strengths and weaknesses",
                    parameters={},
                ),
                SkillAction(
                    name="integration_map",
                    description="Map which skills are integrated with which via bridges",
                    parameters={},
                ),
                SkillAction(
                    name="history",
                    description="View past analyses and how gaps have been addressed over time",
                    parameters={
                        "limit": {
                            "type": "integer",
                            "required": False,
                            "description": "Number of recent analyses to show (default 10)",
                        },
                    },
                ),
                SkillAction(
                    name="mark_addressed",
                    description="Mark a gap as addressed (built or decided not needed)",
                    parameters={
                        "gap_id": {
                            "type": "string",
                            "required": True,
                            "description": "ID of the gap to mark as addressed",
                        },
                        "resolution": {
                            "type": "string",
                            "required": True,
                            "description": "How the gap was addressed (e.g. 'built XSkill', 'not needed')",
                        },
                    },
                ),
            ],
            required_credentials=[],
        )

    async def execute(self, action: str, params: Dict[str, Any]) -> SkillResult:
        actions = {
            "inventory": self._inventory,
            "analyze_gaps": self._analyze_gaps,
            "score_gaps": self._score_gaps,
            "generate_plan": self._generate_plan,
            "pillar_coverage": self._pillar_coverage,
            "integration_map": self._integration_map,
            "history": self._history,
            "mark_addressed": self._mark_addressed,
        }
        handler = actions.get(action)
        if not handler:
            return SkillResult(success=False, message=f"Unknown action: {action}")
        return await handler(params)

    def _get_skill_info(self) -> List[Dict]:
        """Extract info from all loaded skills."""
        skills_info = []
        for skill in self._agent_skills:
            try:
                m = skill.manifest
                skill_name = m.skill_id
                actions = [a.name for a in m.actions]
                category = m.category
            except Exception:
                skill_name = type(skill).__name__
                actions = []
                category = "unknown"

            skills_info.append({
                "skill_id": skill_name,
                "class_name": type(skill).__name__,
                "category": category,
                "actions": actions,
                "action_count": len(actions),
            })
        return skills_info

    def _classify_skill(self, skill_id: str) -> List[str]:
        """Classify a skill into pillars based on its ID."""
        pillars = []
        skill_lower = skill_id.lower()
        for pillar, keywords in SKILL_CATEGORIES.items():
            for kw in keywords:
                if kw in skill_lower:
                    pillars.append(pillar)
                    break
        if not pillars:
            pillars = ["general"]
        return pillars

    def _find_existing_bridges(self, skills_info: List[Dict]) -> List[Dict]:
        """Find bridge/integration skills that connect two other skills."""
        bridges = []
        skill_ids = {s["skill_id"] for s in skills_info}
        for s in skills_info:
            sid = s["skill_id"]
            if "bridge" in sid or "events" in sid.split("_")[-1:]:
                bridges.append({
                    "bridge_skill": sid,
                    "category": s["category"],
                })
        return bridges

    def _find_missing_bridges(self, skills_info: List[Dict]) -> List[Dict]:
        """Identify skill pairs that could benefit from a bridge skill."""
        skill_ids = {s["skill_id"] for s in skills_info}
        missing = []

        # Check known bridge patterns
        for src_kw, tgt_kw, bridge_kw in BRIDGE_PATTERNS:
            has_source = any(src_kw in sid for sid in skill_ids)
            has_target = any(tgt_kw in sid for sid in skill_ids)
            has_bridge = any(bridge_kw in sid for sid in skill_ids)
            if has_source and has_target and not has_bridge:
                missing.append({
                    "source_keyword": src_kw,
                    "target_keyword": tgt_kw,
                    "suggested_bridge": bridge_kw,
                    "reason": f"Skills matching '{src_kw}' and '{tgt_kw}' exist but no bridge '{bridge_kw}' found",
                })

        # Heuristic: revenue skills without event bridges
        revenue_skills = [s for s in skills_info if "revenue" in s["skill_id"]]
        event_bridges = [s for s in skills_info if "event" in s["skill_id"] and "bridge" in s["skill_id"]]
        event_bridge_names = {s["skill_id"] for s in event_bridges}
        for rs in revenue_skills:
            expected_bridge = rs["skill_id"].replace("_skill", "") + "_event_bridge"
            if expected_bridge not in event_bridge_names and "bridge" not in rs["skill_id"]:
                missing.append({
                    "source_keyword": rs["skill_id"],
                    "target_keyword": "event_bus",
                    "suggested_bridge": expected_bridge,
                    "reason": f"Revenue skill '{rs['skill_id']}' has no event bridge for reactive monitoring",
                })

        return missing

    def _identify_pillar_gaps(self, skills_info: List[Dict]) -> Dict[str, Dict]:
        """Identify gaps per pillar."""
        pillar_skills: Dict[str, List] = {p: [] for p in SKILL_CATEGORIES}
        for s in skills_info:
            for pillar in self._classify_skill(s["skill_id"]):
                if pillar in pillar_skills:
                    pillar_skills[pillar].append(s["skill_id"])

        gaps = {}
        for pillar, skills in pillar_skills.items():
            count = len(skills)
            # Define expected capabilities per pillar
            expected = self._expected_capabilities(pillar)
            present = set()
            missing_caps = []
            for cap_name, cap_keywords in expected:
                found = any(
                    any(kw in sid for kw in cap_keywords)
                    for sid in skills
                )
                if found:
                    present.add(cap_name)
                else:
                    missing_caps.append(cap_name)

            gaps[pillar] = {
                "skill_count": count,
                "skills": skills,
                "coverage_pct": round(len(present) / max(len(expected), 1) * 100),
                "present_capabilities": list(present),
                "missing_capabilities": missing_caps,
            }
        return gaps

    def _expected_capabilities(self, pillar: str) -> List[Tuple[str, List[str]]]:
        """Define expected capability areas per pillar."""
        expectations = {
            "revenue": [
                ("billing", ["billing"]),
                ("payment_processing", ["payment"]),
                ("pricing", ["pricing"]),
                ("usage_tracking", ["usage_tracking"]),
                ("service_catalog", ["catalog", "service_catalog"]),
                ("marketplace", ["marketplace"]),
                ("revenue_analytics", ["revenue_analytics", "revenue_dashboard"]),
                ("subscription_management", ["subscription"]),
            ],
            "replication": [
                ("spawning", ["spawner"]),
                ("fleet_management", ["fleet"]),
                ("peer_discovery", ["peer_discovery"]),
                ("consensus", ["consensus"]),
                ("checkpoint_sync", ["checkpoint"]),
                ("delegation", ["delegation"]),
                ("config_templates", ["config_template"]),
                ("health_monitoring", ["health_monitor"]),
            ],
            "self_improvement": [
                ("self_modification", ["self_modify"]),
                ("experimentation", ["experiment"]),
                ("feedback_loops", ["feedback"]),
                ("learned_behavior", ["learned_behavior"]),
                ("prompt_evolution", ["prompt_evolution"]),
                ("performance_profiling", ["profiler"]),
                ("reflection", ["reflection"]),
                ("self_testing", ["self_test"]),
            ],
            "goal_setting": [
                ("strategic_planning", ["strategy"]),
                ("goal_management", ["goal_manager", "goal"]),
                ("task_planning", ["planner"]),
                ("scheduling", ["scheduler"]),
                ("decision_logging", ["decision_log"]),
                ("outcome_tracking", ["outcome_tracker"]),
                ("workflow_orchestration", ["workflow"]),
                ("gap_analysis", ["gap_analyzer", "capability_gap"]),
            ],
            "infrastructure": [
                ("event_system", ["event_bus", "event"]),
                ("database", ["database"]),
                ("http_client", ["http_client"]),
                ("filesystem", ["filesystem"]),
                ("secret_management", ["secret_vault"]),
                ("deployment", ["deployment", "vercel"]),
                ("monitoring", ["observability", "dashboard"]),
                ("circuit_breaking", ["circuit_breaker"]),
            ],
            "monitoring": [
                ("health_checks", ["health_monitor"]),
                ("observability", ["observability"]),
                ("dashboards", ["dashboard"]),
                ("alerting", ["alert"]),
                ("incident_response", ["incident"]),
                ("service_monitoring", ["service_monitor"]),
                ("diagnostics", ["diagnostics"]),
                ("logging", ["decision_log"]),
            ],
        }
        return expectations.get(pillar, [])

    async def _inventory(self, params: Dict) -> SkillResult:
        """List all loaded skills with metadata."""
        skills_info = self._get_skill_info()
        by_category: Dict[str, List] = {}
        for s in skills_info:
            cat = s["category"]
            by_category.setdefault(cat, []).append(s)

        return SkillResult(
            success=True,
            message=f"Found {len(skills_info)} loaded skills across {len(by_category)} categories",
            data={
                "total_skills": len(skills_info),
                "total_actions": sum(s["action_count"] for s in skills_info),
                "categories": {
                    cat: [s["skill_id"] for s in slist]
                    for cat, slist in sorted(by_category.items())
                },
                "skills": skills_info,
            },
        )

    async def _analyze_gaps(self, params: Dict) -> SkillResult:
        """Analyze missing integrations and capability gaps."""
        skills_info = self._get_skill_info()
        focus = params.get("focus_pillar")

        existing_bridges = self._find_existing_bridges(skills_info)
        missing_bridges = self._find_missing_bridges(skills_info)
        pillar_gaps = self._identify_pillar_gaps(skills_info)

        if focus and focus in pillar_gaps:
            pillar_gaps = {focus: pillar_gaps[focus]}

        analysis_id = str(uuid.uuid4())[:8]
        analysis = {
            "id": analysis_id,
            "timestamp": datetime.now().isoformat(),
            "total_skills": len(skills_info),
            "existing_bridges": len(existing_bridges),
            "missing_bridges": missing_bridges,
            "pillar_gaps": pillar_gaps,
            "focus_pillar": focus,
        }

        # Persist
        data = self._load()
        data["analyses"].append(analysis)
        # Keep last 50 analyses
        data["analyses"] = data["analyses"][-50:]
        self._save(data)

        total_missing = sum(
            len(pg["missing_capabilities"])
            for pg in pillar_gaps.values()
        )

        return SkillResult(
            success=True,
            message=(
                f"Analysis {analysis_id}: {len(missing_bridges)} missing bridges, "
                f"{total_missing} capability gaps across {len(pillar_gaps)} pillars"
            ),
            data=analysis,
        )

    async def _score_gaps(self, params: Dict) -> SkillResult:
        """Score gaps by strategic impact."""
        data = self._load()
        if not data["analyses"]:
            return SkillResult(
                success=False,
                message="No analyses found. Run analyze_gaps first.",
            )

        latest = data["analyses"][-1]
        scored_gaps = []

        # Score missing bridges
        for bridge in latest.get("missing_bridges", []):
            score = self._compute_gap_score(bridge, "bridge")
            scored_gaps.append({
                "type": "missing_bridge",
                "description": bridge["reason"],
                "suggested_action": f"Build {bridge['suggested_bridge']}",
                "impact_score": score,
                "pillar": self._bridge_to_pillar(bridge),
            })

        # Score missing capabilities
        for pillar, gaps in latest.get("pillar_gaps", {}).items():
            for cap in gaps.get("missing_capabilities", []):
                score = self._compute_cap_score(pillar, cap)
                scored_gaps.append({
                    "type": "missing_capability",
                    "description": f"Pillar '{pillar}' missing: {cap}",
                    "suggested_action": f"Build {cap} skill for {pillar}",
                    "impact_score": score,
                    "pillar": pillar,
                })

        # Sort by impact score descending
        scored_gaps.sort(key=lambda g: g["impact_score"], reverse=True)

        return SkillResult(
            success=True,
            message=f"Scored {len(scored_gaps)} gaps. Top gap: {scored_gaps[0]['description'] if scored_gaps else 'none'}",
            data={
                "scored_gaps": scored_gaps,
                "top_5": scored_gaps[:5],
            },
        )

    def _compute_gap_score(self, bridge: Dict, gap_type: str) -> float:
        """Compute impact score for a gap (0-100)."""
        score = 50.0  # base

        # Revenue-related bridges score higher
        if "revenue" in bridge.get("source_keyword", "") or "revenue" in bridge.get("target_keyword", ""):
            score += 20
        # Event bridges enable reactivity
        if "event" in bridge.get("target_keyword", ""):
            score += 10
        # Known patterns are more likely to be useful
        if bridge.get("suggested_bridge") in [b[2] for b in BRIDGE_PATTERNS]:
            score += 5

        return min(score, 100)

    def _compute_cap_score(self, pillar: str, capability: str) -> float:
        """Compute impact score for a missing capability."""
        # Pillar priority weights
        pillar_weights = {
            "revenue": 30,
            "self_improvement": 25,
            "goal_setting": 20,
            "replication": 15,
            "infrastructure": 10,
            "monitoring": 10,
        }
        base = pillar_weights.get(pillar, 10)

        # Core capabilities score higher
        core_caps = {"billing", "payment_processing", "self_modification", "spawning", "strategic_planning"}
        if capability in core_caps:
            base += 20

        return min(float(base + 30), 100)

    def _bridge_to_pillar(self, bridge: Dict) -> str:
        """Determine which pillar a bridge serves."""
        for keyword in [bridge.get("source_keyword", ""), bridge.get("target_keyword", "")]:
            for pillar, keywords in SKILL_CATEGORIES.items():
                if any(kw in keyword for kw in keywords):
                    return pillar
        return "infrastructure"

    async def _generate_plan(self, params: Dict) -> SkillResult:
        """Generate a concrete work plan based on gap analysis."""
        max_items = params.get("max_items", 5)

        # Run analysis first if none exists
        data = self._load()
        if not data["analyses"]:
            await self._analyze_gaps({})
            data = self._load()

        # Score gaps
        score_result = await self._score_gaps({})
        if not score_result.success:
            return score_result

        scored_gaps = score_result.data.get("scored_gaps", [])

        # Filter out already-addressed gaps
        addressed_ids = {
            g["gap_id"] for g in data.get("gap_history", [])
            if g.get("status") == "addressed"
        }

        plan_items = []
        for gap in scored_gaps[:max_items * 2]:  # Take extras to filter
            gap_id = f"{gap['type']}:{gap['suggested_action']}"
            if gap_id in addressed_ids:
                continue
            plan_items.append({
                "priority": len(plan_items) + 1,
                "gap_id": gap_id,
                "action": gap["suggested_action"],
                "pillar": gap["pillar"],
                "impact_score": gap["impact_score"],
                "type": gap["type"],
                "estimated_effort": "1 session",
            })
            if len(plan_items) >= max_items:
                break

        plan = {
            "id": str(uuid.uuid4())[:8],
            "generated_at": datetime.now().isoformat(),
            "items": plan_items,
            "based_on_analysis": data["analyses"][-1]["id"] if data["analyses"] else None,
        }

        data["work_plans"].append(plan)
        data["work_plans"] = data["work_plans"][-20:]
        self._save(data)

        return SkillResult(
            success=True,
            message=f"Generated work plan with {len(plan_items)} items. Top priority: {plan_items[0]['action'] if plan_items else 'none'}",
            data=plan,
        )

    async def _pillar_coverage(self, params: Dict) -> SkillResult:
        """Show skill coverage per pillar."""
        skills_info = self._get_skill_info()
        pillar_gaps = self._identify_pillar_gaps(skills_info)

        summary = {}
        for pillar, info in pillar_gaps.items():
            summary[pillar] = {
                "skill_count": info["skill_count"],
                "coverage_pct": info["coverage_pct"],
                "strengths": info["present_capabilities"],
                "weaknesses": info["missing_capabilities"],
            }

        # Find weakest pillar
        weakest = min(
            summary.items(),
            key=lambda x: x[1]["coverage_pct"],
        )

        return SkillResult(
            success=True,
            message=f"Pillar coverage analysis complete. Weakest: {weakest[0]} at {weakest[1]['coverage_pct']}%",
            data={
                "pillars": summary,
                "weakest_pillar": weakest[0],
                "weakest_coverage": weakest[1]["coverage_pct"],
            },
        )

    async def _integration_map(self, params: Dict) -> SkillResult:
        """Map skill integrations via bridges."""
        skills_info = self._get_skill_info()
        existing_bridges = self._find_existing_bridges(skills_info)
        missing_bridges = self._find_missing_bridges(skills_info)

        return SkillResult(
            success=True,
            message=f"Integration map: {len(existing_bridges)} bridges exist, {len(missing_bridges)} missing",
            data={
                "existing_bridges": existing_bridges,
                "missing_bridges": missing_bridges,
                "total_skills": len(skills_info),
                "integration_ratio": round(
                    len(existing_bridges) / max(len(skills_info), 1) * 100, 1
                ),
            },
        )

    async def _history(self, params: Dict) -> SkillResult:
        """View past analyses."""
        limit = params.get("limit", 10)
        data = self._load()
        analyses = data["analyses"][-limit:]
        plans = data["work_plans"][-limit:]

        return SkillResult(
            success=True,
            message=f"Showing {len(analyses)} analyses and {len(plans)} work plans",
            data={
                "analyses": analyses,
                "work_plans": plans,
                "addressed_gaps": data.get("gap_history", [])[-limit:],
            },
        )

    async def _mark_addressed(self, params: Dict) -> SkillResult:
        """Mark a gap as addressed."""
        gap_id = params.get("gap_id", "")
        resolution = params.get("resolution", "")
        if not gap_id or not resolution:
            return SkillResult(
                success=False,
                message="Both gap_id and resolution are required",
            )

        data = self._load()
        data.setdefault("gap_history", []).append({
            "gap_id": gap_id,
            "resolution": resolution,
            "status": "addressed",
            "timestamp": datetime.now().isoformat(),
        })
        # Keep last 200
        data["gap_history"] = data["gap_history"][-200:]
        self._save(data)

        return SkillResult(
            success=True,
            message=f"Gap '{gap_id}' marked as addressed: {resolution}",
            data={"gap_id": gap_id, "resolution": resolution},
        )
