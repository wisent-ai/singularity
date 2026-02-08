#!/usr/bin/env python3
"""
SelfAssessmentSkill - Agent capability profiling, benchmarking, and gap analysis.

Enables agents to periodically evaluate their own skill portfolio, measure
skill health, produce capability profiles, and publish them for other agents.
This is critical for both Self-Improvement (know where you're weak) and
Replication (agents advertise what they can do).

Without this skill, agents have no structured way to:
- Know which skills are healthy vs broken
- Compare their capabilities against a baseline
- Tell other agents what they're good at
- Identify the most impactful next skill to build

Pillar: Self-Improvement + Replication

Flow:
1. inventory: Scan all installed skills, count actions, categorize
2. benchmark: Run diagnostic probes on each skill (status/health checks)
3. profile: Generate a structured capability profile (strong/weak/missing)
4. publish: Push capability profile to KnowledgeSharingSkill for discovery
5. gaps: Compare installed skills vs known capability categories
6. recommend: Suggest highest-impact improvements based on gap analysis

Actions:
- inventory: List all skills with action counts and categories
- benchmark: Test each skill's health with probe actions
- profile: Generate full capability profile with scores
- publish: Publish profile for other agents to discover
- gaps: Identify missing capability categories
- recommend: Suggest next skills to build or improve
- compare: Compare this agent's profile against another agent's
- history: View past assessment results
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from .base import Skill, SkillResult, SkillManifest, SkillAction


DATA_FILE = Path(__file__).parent.parent / "data" / "self_assessment.json"
MAX_HISTORY = 50

# Known capability categories an ideal agent should have
CAPABILITY_CATEGORIES = {
    "self_improvement": {
        "description": "Ability to modify and improve own behavior",
        "expected_skills": [
            "self_modify", "self_eval", "self_testing", "feedback_loop",
            "learned_behavior", "prompt_evolution", "skill_composer",
            "performance_optimizer", "self_healing", "self_tuning",
        ],
        "weight": 1.0,
    },
    "revenue": {
        "description": "Ability to generate income from services",
        "expected_skills": [
            "revenue_services", "usage_tracking", "payment", "marketplace",
            "service_hosting", "api_gateway", "cost_optimizer",
        ],
        "weight": 1.0,
    },
    "replication": {
        "description": "Ability to spawn and coordinate with other agents",
        "expected_skills": [
            "peer_discovery", "agent_network", "deployment", "messaging",
            "knowledge_sharing", "task_delegation", "agent_reputation",
        ],
        "weight": 1.0,
    },
    "goal_setting": {
        "description": "Ability to set, track, and achieve goals autonomously",
        "expected_skills": [
            "goal_manager", "strategy", "planner", "autonomous_loop",
            "session_bootstrap", "dashboard", "decision_log",
        ],
        "weight": 1.0,
    },
    "operations": {
        "description": "Monitoring, alerting, incident management",
        "expected_skills": [
            "observability", "alert_incident_bridge", "incident_response",
            "scheduler", "health_monitor",
        ],
        "weight": 0.8,
    },
    "communication": {
        "description": "Inter-agent communication and coordination",
        "expected_skills": [
            "messaging", "consensus", "event", "notification",
        ],
        "weight": 0.7,
    },
}

# Probe actions: lightweight actions that verify a skill is working
# Format: skill_id -> (action, params) that should succeed without side effects
PROBE_ACTIONS = {
    "scheduler": ("list", {}),
    "goal_manager": ("list", {}),
    "dashboard": ("summary", {}),
    "observability": ("query", {"metric_name": "__probe__", "range_minutes": 1}),
    "incident_response": ("list", {}),
    "alert_incident_bridge": ("status", {}),
    "marketplace": ("list", {}),
    "knowledge_sharing": ("query", {"query": "__probe__"}),
    "agent_network": ("list_agents", {}),
    "task_delegation": ("list", {}),
    "agent_reputation": ("get_leaderboard", {}),
    "consensus": ("status", {"proposal_id": "__probe__"}),
    "feedback_loop": ("status", {}),
    "event": ("list_topics", {}),
    "performance": ("report", {}),
    "self_testing": ("discover", {}),
}


def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


class SelfAssessmentSkill(Skill):
    """
    Agent capability self-assessment with profiling and gap analysis.

    Enables agents to understand their own strengths and weaknesses,
    benchmark skill health, and publish capability profiles for
    other agents to discover.
    """

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self._store = None

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="self_assessment",
            name="Self-Assessment",
            version="1.0.0",
            category="meta",
            description="Capability profiling, benchmarking, and gap analysis for autonomous agents",
            actions=[
                SkillAction(
                    name="inventory",
                    description="Scan all installed skills with action counts and categories",
                    parameters={},
                    estimated_cost=0,
                ),
                SkillAction(
                    name="benchmark",
                    description="Run health probes on each skill to verify they're working",
                    parameters={
                        "skill_ids": {
                            "type": "array",
                            "required": False,
                            "description": "Specific skills to benchmark (default: all with probes)",
                        },
                        "timeout_seconds": {
                            "type": "number",
                            "required": False,
                            "description": "Per-skill timeout in seconds (default: 5)",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="profile",
                    description="Generate full capability profile with category scores",
                    parameters={
                        "run_benchmarks": {
                            "type": "boolean",
                            "required": False,
                            "description": "Also run benchmarks as part of profiling (default: True)",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="publish",
                    description="Publish capability profile to knowledge store for other agents",
                    parameters={
                        "agent_id": {
                            "type": "string",
                            "required": False,
                            "description": "Agent ID to publish under (default: 'self')",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="gaps",
                    description="Identify missing capability categories and skills",
                    parameters={},
                    estimated_cost=0,
                ),
                SkillAction(
                    name="recommend",
                    description="Suggest highest-impact skills to build next",
                    parameters={
                        "top_n": {
                            "type": "integer",
                            "required": False,
                            "description": "Number of recommendations (default: 5)",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="compare",
                    description="Compare this agent's profile against another agent's published profile",
                    parameters={
                        "agent_id": {
                            "type": "string",
                            "required": True,
                            "description": "Agent ID to compare against",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="history",
                    description="View past assessment results and capability trends",
                    parameters={
                        "limit": {
                            "type": "integer",
                            "required": False,
                            "description": "Max entries to return (default: 10)",
                        },
                    },
                    estimated_cost=0,
                ),
            ],
            required_credentials=[],
        )

    def check_credentials(self) -> bool:
        return True

    # ── Persistence ──────────────────────────────────────────────────

    def _default_state(self) -> Dict:
        return {
            "profiles": [],
            "benchmarks": [],
            "metadata": {"created_at": _now_iso()},
        }

    def _load(self) -> Dict:
        if self._store is not None:
            return self._store
        DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
        if DATA_FILE.exists():
            try:
                with open(DATA_FILE, "r") as f:
                    self._store = json.load(f)
                    return self._store
            except (json.JSONDecodeError, OSError):
                pass
        self._store = self._default_state()
        return self._store

    def _save(self, data: Dict):
        self._store = data
        # Trim history
        if len(data.get("profiles", [])) > MAX_HISTORY:
            data["profiles"] = data["profiles"][-MAX_HISTORY:]
        if len(data.get("benchmarks", [])) > MAX_HISTORY:
            data["benchmarks"] = data["benchmarks"][-MAX_HISTORY:]
        DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(DATA_FILE, "w") as f:
            json.dump(data, f, indent=2, default=str)

    # ── Execute Dispatch ─────────────────────────────────────────────

    async def execute(self, action: str, params: Dict[str, Any]) -> SkillResult:
        handlers = {
            "inventory": self._inventory,
            "benchmark": self._benchmark,
            "profile": self._profile,
            "publish": self._publish,
            "gaps": self._gaps,
            "recommend": self._recommend,
            "compare": self._compare,
            "history": self._history,
        }
        handler = handlers.get(action)
        if not handler:
            return SkillResult(success=False, message=f"Unknown action: {action}")
        return await handler(params)

    # ── Action: inventory ────────────────────────────────────────────

    async def _inventory(self, params: Dict) -> SkillResult:
        """Scan all installed skills and categorize them."""
        skills_info = await self._get_installed_skills()

        # Categorize each skill
        category_map = {}
        for cat_name, cat_data in CAPABILITY_CATEGORIES.items():
            category_map[cat_name] = {
                "description": cat_data["description"],
                "installed": [],
                "missing": [],
            }
            for expected_id in cat_data["expected_skills"]:
                if expected_id in skills_info:
                    category_map[cat_name]["installed"].append(expected_id)
                else:
                    category_map[cat_name]["missing"].append(expected_id)

        # Uncategorized skills
        all_expected = set()
        for cat in CAPABILITY_CATEGORIES.values():
            all_expected.update(cat["expected_skills"])
        uncategorized = [s for s in skills_info if s not in all_expected]

        total = len(skills_info)
        return SkillResult(
            success=True,
            message=f"Inventory complete: {total} skills installed across {len(CAPABILITY_CATEGORIES)} categories",
            data={
                "total_skills": total,
                "skills": skills_info,
                "categories": category_map,
                "uncategorized": uncategorized,
            },
        )

    # ── Action: benchmark ────────────────────────────────────────────

    async def _benchmark(self, params: Dict) -> SkillResult:
        """Run health probes on each skill."""
        target_ids = params.get("skill_ids", None)
        timeout = params.get("timeout_seconds", 5)

        skills_info = await self._get_installed_skills()
        results = {}
        healthy = 0
        unhealthy = 0
        skipped = 0

        for skill_id in skills_info:
            if target_ids and skill_id not in target_ids:
                continue

            probe = PROBE_ACTIONS.get(skill_id)
            if not probe:
                results[skill_id] = {
                    "status": "no_probe",
                    "message": "No probe action defined for this skill",
                }
                skipped += 1
                continue

            probe_action, probe_params = probe
            start = time.time()
            try:
                result = await self._probe_skill(skill_id, probe_action, probe_params, timeout)
                latency = round(time.time() - start, 3)
                if result:
                    results[skill_id] = {
                        "status": "healthy",
                        "latency_ms": round(latency * 1000),
                        "message": result.get("message", "OK"),
                    }
                    healthy += 1
                else:
                    results[skill_id] = {
                        "status": "unhealthy",
                        "latency_ms": round(latency * 1000),
                        "message": "Probe returned failure",
                    }
                    unhealthy += 1
            except Exception as e:
                latency = round(time.time() - start, 3)
                results[skill_id] = {
                    "status": "error",
                    "latency_ms": round(latency * 1000),
                    "message": str(e)[:200],
                }
                unhealthy += 1

        # Save benchmark results
        store = self._load()
        benchmark_record = {
            "timestamp": _now_iso(),
            "results": results,
            "healthy": healthy,
            "unhealthy": unhealthy,
            "skipped": skipped,
        }
        store["benchmarks"].append(benchmark_record)
        self._save(store)

        return SkillResult(
            success=True,
            message=f"Benchmark: {healthy} healthy, {unhealthy} unhealthy, {skipped} skipped",
            data={
                "results": results,
                "healthy": healthy,
                "unhealthy": unhealthy,
                "skipped": skipped,
                "timestamp": benchmark_record["timestamp"],
            },
        )

    # ── Action: profile ──────────────────────────────────────────────

    async def _profile(self, params: Dict) -> SkillResult:
        """Generate a full capability profile with category scores."""
        run_benchmarks = params.get("run_benchmarks", True)

        skills_info = await self._get_installed_skills()
        total_skills = len(skills_info)

        # Run benchmarks if requested
        benchmark_data = {}
        if run_benchmarks:
            bench_result = await self._benchmark({})
            if bench_result.success and bench_result.data:
                benchmark_data = bench_result.data.get("results", {})

        # Score each category
        category_scores = {}
        overall_score = 0.0
        total_weight = 0.0

        for cat_name, cat_data in CAPABILITY_CATEGORIES.items():
            expected = cat_data["expected_skills"]
            installed = [s for s in expected if s in skills_info]
            coverage = len(installed) / max(len(expected), 1)

            # Health factor: reduce score if installed skills are unhealthy
            health_factor = 1.0
            if benchmark_data and installed:
                healthy_count = sum(
                    1 for s in installed
                    if benchmark_data.get(s, {}).get("status") in ("healthy", "no_probe")
                )
                health_factor = healthy_count / max(len(installed), 1)

            score = round(coverage * health_factor * 100, 1)
            category_scores[cat_name] = {
                "score": score,
                "coverage": round(coverage * 100, 1),
                "health_factor": round(health_factor * 100, 1),
                "installed_count": len(installed),
                "expected_count": len(expected),
                "installed": installed,
                "missing": [s for s in expected if s not in skills_info],
            }

            weight = cat_data["weight"]
            overall_score += score * weight
            total_weight += weight

        overall_score = round(overall_score / max(total_weight, 1), 1)

        # Identify strongest and weakest categories
        sorted_cats = sorted(category_scores.items(), key=lambda x: x[1]["score"])
        weakest = sorted_cats[0] if sorted_cats else None
        strongest = sorted_cats[-1] if sorted_cats else None

        # Total actions across all skills
        total_actions = sum(s.get("action_count", 0) for s in skills_info.values())

        profile = {
            "timestamp": _now_iso(),
            "overall_score": overall_score,
            "total_skills": total_skills,
            "total_actions": total_actions,
            "categories": category_scores,
            "strongest_category": strongest[0] if strongest else None,
            "weakest_category": weakest[0] if weakest else None,
            "benchmark_healthy": sum(
                1 for r in benchmark_data.values()
                if r.get("status") == "healthy"
            ) if benchmark_data else None,
        }

        # Save profile
        store = self._load()
        store["profiles"].append(profile)
        self._save(store)

        return SkillResult(
            success=True,
            message=(
                f"Capability profile: {overall_score}/100 overall | "
                f"Strongest: {strongest[0]}({strongest[1]['score']}) | "
                f"Weakest: {weakest[0]}({weakest[1]['score']}) | "
                f"{total_skills} skills, {total_actions} actions"
            ),
            data=profile,
        )

    # ── Action: publish ──────────────────────────────────────────────

    async def _publish(self, params: Dict) -> SkillResult:
        """Publish capability profile to knowledge store for other agents."""
        agent_id = params.get("agent_id", "self")

        # Get latest profile or generate one
        store = self._load()
        if not store["profiles"]:
            profile_result = await self._profile({"run_benchmarks": False})
            if not profile_result.success:
                return SkillResult(
                    success=False,
                    message="Failed to generate profile for publishing",
                )
            profile = profile_result.data
        else:
            profile = store["profiles"][-1]

        # Publish via KnowledgeSharingSkill
        publish_data = {
            "agent_id": agent_id,
            "overall_score": profile["overall_score"],
            "total_skills": profile["total_skills"],
            "total_actions": profile["total_actions"],
            "strongest": profile.get("strongest_category"),
            "weakest": profile.get("weakest_category"),
            "categories": {
                k: {"score": v["score"], "installed": v["installed_count"]}
                for k, v in profile.get("categories", {}).items()
            },
            "published_at": _now_iso(),
        }

        published = False
        if self.context:
            try:
                result = await self.context.call_skill("knowledge_sharing", "publish", {
                    "topic": "agent_capability_profile",
                    "content": json.dumps(publish_data),
                    "tags": ["capability", "profile", agent_id],
                    "confidence": min(profile["overall_score"] / 100, 1.0),
                })
                if result and result.success:
                    published = True
            except Exception:
                pass

        # Also try publishing via agent_network register
        if self.context:
            try:
                capabilities = []
                for cat_name, cat_data in profile.get("categories", {}).items():
                    if cat_data.get("score", 0) > 30:
                        capabilities.append(cat_name)
                await self.context.call_skill("agent_network", "update_agent", {
                    "agent_id": agent_id,
                    "capabilities": capabilities,
                    "metadata": {
                        "capability_score": profile["overall_score"],
                        "last_assessment": _now_iso(),
                    },
                })
            except Exception:
                pass

        return SkillResult(
            success=True,
            message=(
                f"Published capability profile for '{agent_id}': "
                f"{profile['overall_score']}/100 overall, "
                f"{profile['total_skills']} skills"
                + (" (shared via knowledge store)" if published else " (local only)")
            ),
            data={
                "agent_id": agent_id,
                "profile_summary": publish_data,
                "shared": published,
            },
        )

    # ── Action: gaps ─────────────────────────────────────────────────

    async def _gaps(self, params: Dict) -> SkillResult:
        """Identify missing capability categories and skills."""
        skills_info = await self._get_installed_skills()

        gaps = []
        for cat_name, cat_data in CAPABILITY_CATEGORIES.items():
            expected = cat_data["expected_skills"]
            missing = [s for s in expected if s not in skills_info]
            coverage = (len(expected) - len(missing)) / max(len(expected), 1)

            if missing:
                gaps.append({
                    "category": cat_name,
                    "description": cat_data["description"],
                    "coverage_pct": round(coverage * 100, 1),
                    "missing_skills": missing,
                    "missing_count": len(missing),
                    "weight": cat_data["weight"],
                    "impact_score": round((1 - coverage) * cat_data["weight"] * 100, 1),
                })

        # Sort by impact (biggest gaps first)
        gaps.sort(key=lambda g: g["impact_score"], reverse=True)

        total_missing = sum(g["missing_count"] for g in gaps)
        return SkillResult(
            success=True,
            message=f"Found {total_missing} missing skills across {len(gaps)} categories",
            data={
                "gaps": gaps,
                "total_missing": total_missing,
                "categories_with_gaps": len(gaps),
            },
        )

    # ── Action: recommend ────────────────────────────────────────────

    async def _recommend(self, params: Dict) -> SkillResult:
        """Suggest highest-impact skills to build next."""
        top_n = params.get("top_n", 5)

        # Get gaps first
        gaps_result = await self._gaps({})
        if not gaps_result.success:
            return SkillResult(success=False, message="Failed to analyze gaps")

        gaps = gaps_result.data.get("gaps", [])

        # Build ranked recommendations
        recommendations = []
        for gap in gaps:
            category = gap["category"]
            weight = gap["weight"]
            missing_count = gap["missing_count"]

            for skill_id in gap["missing_skills"]:
                # Higher priority for categories with fewer installed skills (bigger gap)
                # and higher weight categories
                priority = round(gap["impact_score"] / max(missing_count, 1), 1)
                recommendations.append({
                    "skill_id": skill_id,
                    "category": category,
                    "priority_score": priority,
                    "reason": f"Missing from '{category}' (coverage: {gap['coverage_pct']}%)",
                    "category_weight": weight,
                })

        # Sort by priority score descending
        recommendations.sort(key=lambda r: r["priority_score"], reverse=True)
        top_recs = recommendations[:top_n]

        # Also check for unhealthy skills from latest benchmark
        store = self._load()
        fix_recs = []
        if store["benchmarks"]:
            latest_bench = store["benchmarks"][-1]
            for skill_id, result in latest_bench.get("results", {}).items():
                if result.get("status") in ("unhealthy", "error"):
                    fix_recs.append({
                        "skill_id": skill_id,
                        "issue": result.get("status"),
                        "message": result.get("message", "Unknown issue"),
                        "recommendation": f"Fix or repair '{skill_id}' - currently {result.get('status')}",
                    })

        return SkillResult(
            success=True,
            message=f"Top {len(top_recs)} recommendations (from {len(recommendations)} total gaps)",
            data={
                "build_next": top_recs,
                "fix_existing": fix_recs,
                "total_gaps": len(recommendations),
            },
        )

    # ── Action: compare ──────────────────────────────────────────────

    async def _compare(self, params: Dict) -> SkillResult:
        """Compare this agent's profile against another agent's."""
        agent_id = params.get("agent_id", "").strip()
        if not agent_id:
            return SkillResult(success=False, message="agent_id is required")

        # Get our latest profile
        store = self._load()
        if not store["profiles"]:
            # Generate one
            await self._profile({"run_benchmarks": False})
            store = self._load()

        if not store["profiles"]:
            return SkillResult(success=False, message="No local profile available")

        our_profile = store["profiles"][-1]

        # Try to get the other agent's profile from knowledge store
        other_profile = None
        if self.context:
            try:
                result = await self.context.call_skill("knowledge_sharing", "query", {
                    "query": f"agent_capability_profile {agent_id}",
                    "tags": ["capability", "profile", agent_id],
                })
                if result and result.success and result.data:
                    entries = result.data.get("results", [])
                    for entry in entries:
                        content = entry.get("content", "")
                        try:
                            other_profile = json.loads(content)
                            if other_profile.get("agent_id") == agent_id:
                                break
                        except (json.JSONDecodeError, TypeError):
                            continue
            except Exception:
                pass

        if not other_profile:
            return SkillResult(
                success=False,
                message=f"No published profile found for agent '{agent_id}'",
            )

        # Compare categories
        comparison = {}
        for cat_name in CAPABILITY_CATEGORIES:
            our_score = our_profile.get("categories", {}).get(cat_name, {}).get("score", 0)
            other_cats = other_profile.get("categories", {})
            other_score = other_cats.get(cat_name, {}).get("score", 0)
            diff = round(our_score - other_score, 1)
            comparison[cat_name] = {
                "our_score": our_score,
                "their_score": other_score,
                "difference": diff,
                "advantage": "us" if diff > 0 else ("them" if diff < 0 else "tied"),
            }

        our_overall = our_profile.get("overall_score", 0)
        their_overall = other_profile.get("overall_score", 0)

        return SkillResult(
            success=True,
            message=(
                f"Comparison vs '{agent_id}': "
                f"Us {our_overall} vs Them {their_overall} "
                f"(diff: {round(our_overall - their_overall, 1)})"
            ),
            data={
                "our_overall": our_overall,
                "their_overall": their_overall,
                "categories": comparison,
                "our_skills": our_profile.get("total_skills", 0),
                "their_skills": other_profile.get("total_skills", 0),
            },
        )

    # ── Action: history ──────────────────────────────────────────────

    async def _history(self, params: Dict) -> SkillResult:
        """View past assessment results and capability trends."""
        limit = params.get("limit", 10)

        store = self._load()
        profiles = store.get("profiles", [])[-limit:]

        # Calculate trends
        trend = None
        if len(profiles) >= 2:
            scores = [p.get("overall_score", 0) for p in profiles]
            trend = {
                "direction": "improving" if scores[-1] > scores[0] else (
                    "declining" if scores[-1] < scores[0] else "stable"
                ),
                "first_score": scores[0],
                "latest_score": scores[-1],
                "change": round(scores[-1] - scores[0], 1),
                "data_points": len(scores),
            }

        history = []
        for p in profiles:
            history.append({
                "timestamp": p.get("timestamp"),
                "overall_score": p.get("overall_score"),
                "total_skills": p.get("total_skills"),
                "strongest": p.get("strongest_category"),
                "weakest": p.get("weakest_category"),
            })

        return SkillResult(
            success=True,
            message=f"{len(history)} assessment records" + (
                f" (trend: {trend['direction']}, {trend['change']:+.1f})" if trend else ""
            ),
            data={
                "history": history,
                "trend": trend,
            },
        )

    # ── Internal helpers ─────────────────────────────────────────────

    async def _get_installed_skills(self) -> Dict[str, Dict]:
        """Get a dictionary of all installed skills with basic metadata."""
        skills = {}

        # Try via skill context (agent runtime)
        if self.context:
            try:
                skill_ids = self.context.list_skills()
                for sid in skill_ids:
                    skills[sid] = {
                        "skill_id": sid,
                        "action_count": 0,
                        "category": "unknown",
                    }
                    # Try to get manifest info
                    try:
                        # Some contexts support get_skill_manifest
                        if hasattr(self.context, 'get_skill_info'):
                            info = self.context.get_skill_info(sid)
                            if info:
                                skills[sid].update({
                                    "action_count": info.get("action_count", 0),
                                    "category": info.get("category", "unknown"),
                                })
                    except Exception:
                        pass
                return skills
            except Exception:
                pass

        # Fallback: scan skills directory
        try:
            skills_dir = Path(__file__).parent
            for py_file in skills_dir.glob("*.py"):
                if py_file.name.startswith("_"):
                    continue
                module_name = py_file.stem
                # Skip non-skill files
                if module_name in ("base", "context_synthesis"):
                    continue
                skills[module_name] = {
                    "skill_id": module_name,
                    "action_count": 0,
                    "category": "unknown",
                    "source": "filesystem",
                }
        except Exception:
            pass

        return skills

    async def _probe_skill(self, skill_id: str, action: str, params: Dict, timeout: float) -> Optional[Dict]:
        """Run a probe action on a skill and return result data."""
        if self.context:
            try:
                import asyncio
                result = await asyncio.wait_for(
                    self.context.call_skill(skill_id, action, params),
                    timeout=timeout,
                )
                if result and result.success:
                    return {"success": True, "message": result.message[:200]}
                return None
            except asyncio.TimeoutError:
                raise Exception(f"Probe timed out after {timeout}s")
            except Exception as e:
                raise Exception(f"Probe failed: {str(e)[:200]}")

        # No context - try direct instantiation (limited)
        return {"success": True, "message": "No context, assumed healthy"}
