#!/usr/bin/env python3
"""
AgentMetaSkill - Agent Self-Awareness and Introspection

Provides the agent with the ability to introspect its own capabilities,
status, and action history. This is the agent's "self-awareness" layer.

Actions:
    list_skills: List all installed skills with action counts and categories
    skill_help: Get detailed help for a specific skill or action
    search_capabilities: Search for skills matching a keyword/capability
    agent_status: Get current agent status (balance, cycle, costs, runtime)
    action_history: Get recent action history with success/failure stats
    error_summary: Summarize recent errors grouped by skill
    capability_matrix: Generate a matrix of what the agent can and cannot do
    suggest_action: Suggest the best action for a given goal description

Pillar: Self-Improvement (self-awareness enables better decisions)
"""

from datetime import datetime
from typing import Dict, List, Optional, Any

from .base import Skill, SkillManifest, SkillAction, SkillResult


class AgentMetaSkill(Skill):
    """Skill that provides agent self-awareness and introspection."""

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self._agent_ref = None
        self._registry_ref = None

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="agent_meta",
            name="Agent Meta",
            version="1.0.0",
            category="meta",
            description="Agent self-awareness: introspect capabilities, status, history, and errors",
            actions=[
                SkillAction(
                    name="list_skills",
                    description="List all installed skills with their action counts and categories",
                    parameters={
                        "category": {
                            "type": "string",
                            "required": False,
                            "description": "Filter by category (e.g., 'dev', 'social', 'content')"
                        },
                        "verbose": {
                            "type": "boolean",
                            "required": False,
                            "description": "Include action details (default: false)"
                        }
                    }
                ),
                SkillAction(
                    name="skill_help",
                    description="Get detailed help for a specific skill or action",
                    parameters={
                        "skill_id": {
                            "type": "string",
                            "required": True,
                            "description": "Skill ID to get help for"
                        },
                        "action_name": {
                            "type": "string",
                            "required": False,
                            "description": "Specific action to get help for (optional)"
                        }
                    }
                ),
                SkillAction(
                    name="search_capabilities",
                    description="Search for skills and actions matching a keyword or capability description",
                    parameters={
                        "query": {
                            "type": "string",
                            "required": True,
                            "description": "Search keyword (e.g., 'file', 'github', 'write', 'deploy')"
                        }
                    }
                ),
                SkillAction(
                    name="agent_status",
                    description="Get current agent status: balance, cycle, costs, runtime, skill count",
                    parameters={}
                ),
                SkillAction(
                    name="action_history",
                    description="Get recent action history with success/failure statistics",
                    parameters={
                        "limit": {
                            "type": "integer",
                            "required": False,
                            "description": "Number of recent actions to return (default: 10)"
                        },
                        "skill_filter": {
                            "type": "string",
                            "required": False,
                            "description": "Filter by skill ID"
                        }
                    }
                ),
                SkillAction(
                    name="error_summary",
                    description="Summarize recent errors grouped by skill, with counts and last messages",
                    parameters={
                        "limit": {
                            "type": "integer",
                            "required": False,
                            "description": "Number of recent actions to analyze (default: 50)"
                        }
                    }
                ),
                SkillAction(
                    name="capability_matrix",
                    description="Generate a matrix showing what the agent can do by category",
                    parameters={}
                ),
                SkillAction(
                    name="suggest_action",
                    description="Suggest the best skill/action for a given goal",
                    parameters={
                        "goal": {
                            "type": "string",
                            "required": True,
                            "description": "What you want to accomplish (e.g., 'read a file', 'post on twitter')"
                        }
                    }
                ),
            ],
            required_credentials=[],  # No credentials needed - pure introspection
        )

    def set_agent_ref(self, agent):
        """Wire this skill to the agent instance for introspection.
        
        Args:
            agent: The AutonomousAgent instance
        """
        self._agent_ref = agent
        self._registry_ref = agent.skills

    def set_registry_ref(self, registry):
        """Set just the skill registry (for testing without full agent)."""
        self._registry_ref = registry

    async def execute(self, action: str, params: Dict) -> SkillResult:
        handlers = {
            "list_skills": self._list_skills,
            "skill_help": self._skill_help,
            "search_capabilities": self._search_capabilities,
            "agent_status": self._agent_status,
            "action_history": self._action_history,
            "error_summary": self._error_summary,
            "capability_matrix": self._capability_matrix,
            "suggest_action": self._suggest_action,
        }
        handler = handlers.get(action)
        if not handler:
            return SkillResult(
                success=False,
                message=f"Unknown action: {action}. Available: {list(handlers.keys())}"
            )
        return await handler(params)

    async def _list_skills(self, params: Dict) -> SkillResult:
        """List all installed skills."""
        registry = self._registry_ref
        if not registry:
            return SkillResult(success=False, message="No skill registry connected")

        category_filter = params.get("category", "").lower()
        verbose = params.get("verbose", False)

        skills_info = []
        for skill_id, skill in registry.skills.items():
            m = skill.manifest
            if category_filter and m.category.lower() != category_filter:
                continue

            info = {
                "skill_id": m.skill_id,
                "name": m.name,
                "category": m.category,
                "description": m.description,
                "action_count": len(m.actions),
                "version": m.version,
                "has_credentials": skill.check_credentials(),
                "usage_count": skill._usage_count,
            }
            if verbose:
                info["actions"] = [
                    {"name": a.name, "description": a.description}
                    for a in m.actions
                ]
            skills_info.append(info)

        # Sort by usage count descending
        skills_info.sort(key=lambda x: x["usage_count"], reverse=True)

        categories = {}
        for s in skills_info:
            cat = s["category"]
            categories[cat] = categories.get(cat, 0) + 1

        return SkillResult(
            success=True,
            message=f"Found {len(skills_info)} skills across {len(categories)} categories",
            data={
                "skills": skills_info,
                "total": len(skills_info),
                "categories": categories,
                "total_actions": sum(s["action_count"] for s in skills_info),
            }
        )

    async def _skill_help(self, params: Dict) -> SkillResult:
        """Get detailed help for a specific skill or action."""
        registry = self._registry_ref
        if not registry:
            return SkillResult(success=False, message="No skill registry connected")

        skill_id = params.get("skill_id", "")
        action_name = params.get("action_name")

        skill = registry.get(skill_id)
        if not skill:
            available = list(registry.skills.keys())
            return SkillResult(
                success=False,
                message=f"Skill '{skill_id}' not found. Available: {available}"
            )

        m = skill.manifest

        if action_name:
            # Help for a specific action
            action = skill.get_action(action_name)
            if not action:
                action_names = [a.name for a in m.actions]
                return SkillResult(
                    success=False,
                    message=f"Action '{action_name}' not found in {skill_id}. Available: {action_names}"
                )
            return SkillResult(
                success=True,
                message=f"Help for {skill_id}:{action_name}",
                data={
                    "skill_id": skill_id,
                    "action": action_name,
                    "description": action.description,
                    "parameters": action.parameters,
                    "estimated_cost": action.estimated_cost,
                    "estimated_duration_seconds": action.estimated_duration_seconds,
                    "success_probability": action.success_probability,
                    "usage": f"{skill_id}:{action_name}",
                }
            )

        # Help for the whole skill
        return SkillResult(
            success=True,
            message=f"Help for {m.name} ({skill_id})",
            data={
                "skill_id": m.skill_id,
                "name": m.name,
                "version": m.version,
                "category": m.category,
                "description": m.description,
                "author": m.author,
                "required_credentials": m.required_credentials,
                "has_credentials": skill.check_credentials(),
                "missing_credentials": skill.get_missing_credentials(),
                "actions": [
                    {
                        "name": a.name,
                        "description": a.description,
                        "parameters": a.parameters,
                        "estimated_cost": a.estimated_cost,
                        "usage": f"{skill_id}:{a.name}",
                    }
                    for a in m.actions
                ],
                "stats": skill.stats,
            }
        )

    async def _search_capabilities(self, params: Dict) -> SkillResult:
        """Search for skills and actions matching a keyword."""
        registry = self._registry_ref
        if not registry:
            return SkillResult(success=False, message="No skill registry connected")

        query = params.get("query", "").lower()
        if not query:
            return SkillResult(success=False, message="Please provide a search query")

        matches = []
        for skill_id, skill in registry.skills.items():
            m = skill.manifest
            # Search in skill metadata
            skill_text = f"{m.skill_id} {m.name} {m.description} {m.category}".lower()
            skill_matches = query in skill_text

            for action in m.actions:
                action_text = f"{action.name} {action.description}".lower()
                # Also check parameter descriptions
                param_text = " ".join(
                    p.get("description", "")
                    for p in action.parameters.values()
                    if isinstance(p, dict)
                ).lower()

                if query in action_text or query in param_text or skill_matches:
                    relevance = 0
                    if query in action.name.lower():
                        relevance += 3
                    if query in m.skill_id.lower():
                        relevance += 2
                    if query in action.description.lower():
                        relevance += 1
                    if query in param_text:
                        relevance += 0.5

                    matches.append({
                        "skill_id": skill_id,
                        "skill_name": m.name,
                        "action": action.name,
                        "description": action.description,
                        "usage": f"{skill_id}:{action.name}",
                        "relevance": relevance,
                    })

        # Sort by relevance
        matches.sort(key=lambda x: x["relevance"], reverse=True)

        # Remove relevance score from output
        for m in matches:
            del m["relevance"]

        return SkillResult(
            success=True,
            message=f"Found {len(matches)} actions matching '{query}'",
            data={"query": query, "matches": matches, "total": len(matches)}
        )

    async def _agent_status(self, params: Dict) -> SkillResult:
        """Get current agent status."""
        agent = self._agent_ref
        registry = self._registry_ref

        status = {}

        if agent:
            # Calculate runtime
            avg_cycle_hours = agent.cycle_interval / 3600
            est_cost_per_cycle = 0.01 + (agent.instance_cost_per_hour * avg_cycle_hours)
            runway_cycles = agent.balance / est_cost_per_cycle if est_cost_per_cycle > 0 else float("inf")

            status.update({
                "name": agent.name,
                "ticker": agent.ticker,
                "agent_type": agent.agent_type,
                "specialty": agent.specialty,
                "balance_usd": round(agent.balance, 6),
                "total_api_cost_usd": round(agent.total_api_cost, 6),
                "total_instance_cost_usd": round(agent.total_instance_cost, 6),
                "total_tokens_used": agent.total_tokens_used,
                "cycle": agent.cycle,
                "running": agent.running,
                "instance_type": agent.instance_type,
                "cycle_interval_seconds": agent.cycle_interval,
                "runway_cycles": round(runway_cycles),
                "recent_actions_count": len(agent.recent_actions),
                "created_resources": {
                    k: len(v) for k, v in agent.created_resources.items()
                },
            })

        if registry:
            skill_stats = {}
            total_usage = 0
            for sid, skill in registry.skills.items():
                s = skill.stats
                total_usage += s["usage_count"]
                if s["usage_count"] > 0:
                    skill_stats[sid] = s

            status.update({
                "installed_skills": len(registry.skills),
                "total_actions_available": sum(
                    len(s.manifest.actions) for s in registry.skills.values()
                ),
                "total_skill_usage": total_usage,
                "active_skills": skill_stats,
            })

        if not agent and not registry:
            return SkillResult(
                success=False,
                message="No agent or registry connected. Call set_agent_ref() first."
            )

        return SkillResult(
            success=True,
            message=f"Agent status as of cycle {status.get('cycle', '?')}",
            data=status
        )

    async def _action_history(self, params: Dict) -> SkillResult:
        """Get recent action history with stats."""
        agent = self._agent_ref
        if not agent:
            return SkillResult(success=False, message="No agent connected")

        limit = params.get("limit", 10)
        skill_filter = params.get("skill_filter", "")

        actions = agent.recent_actions[-200:]  # Look at last 200 max

        if skill_filter:
            actions = [
                a for a in actions
                if a.get("tool", "").startswith(skill_filter)
            ]

        # Compute stats from full filtered set
        total = len(actions)
        successes = sum(1 for a in actions if a.get("result", {}).get("status") == "success")
        failures = sum(1 for a in actions if a.get("result", {}).get("status") in ("failed", "error"))
        total_cost = sum(a.get("api_cost_usd", 0) for a in actions)
        total_tokens = sum(a.get("tokens", 0) for a in actions)

        # Skill usage counts
        skill_counts = {}
        for a in actions:
            tool = a.get("tool", "unknown")
            skill_id = tool.split(":")[0] if ":" in tool else tool
            skill_counts[skill_id] = skill_counts.get(skill_id, 0) + 1

        # Return limited recent actions
        recent = actions[-limit:]

        return SkillResult(
            success=True,
            message=f"Action history: {successes}/{total} successful ({failures} failures)",
            data={
                "recent_actions": recent,
                "stats": {
                    "total_actions": total,
                    "successes": successes,
                    "failures": failures,
                    "success_rate": round(successes / total, 3) if total > 0 else 0,
                    "total_api_cost_usd": round(total_cost, 6),
                    "total_tokens": total_tokens,
                },
                "skill_usage": skill_counts,
            }
        )

    async def _error_summary(self, params: Dict) -> SkillResult:
        """Summarize recent errors grouped by skill."""
        agent = self._agent_ref
        if not agent:
            return SkillResult(success=False, message="No agent connected")

        limit = params.get("limit", 50)
        actions = agent.recent_actions[-limit:]

        errors_by_skill = {}
        for a in actions:
            result = a.get("result", {})
            status = result.get("status", "")
            if status in ("failed", "error"):
                tool = a.get("tool", "unknown")
                skill_id = tool.split(":")[0] if ":" in tool else tool
                action_name = tool.split(":")[1] if ":" in tool else "unknown"

                if skill_id not in errors_by_skill:
                    errors_by_skill[skill_id] = {
                        "count": 0,
                        "actions": {},
                        "last_error": "",
                        "last_cycle": 0,
                    }

                entry = errors_by_skill[skill_id]
                entry["count"] += 1
                entry["last_error"] = result.get("message", "")[:200]
                entry["last_cycle"] = a.get("cycle", 0)
                entry["actions"][action_name] = entry["actions"].get(action_name, 0) + 1

        total_errors = sum(e["count"] for e in errors_by_skill.values())
        total_actions = len(actions)

        # Sort by error count
        sorted_errors = dict(
            sorted(errors_by_skill.items(), key=lambda x: x[1]["count"], reverse=True)
        )

        return SkillResult(
            success=True,
            message=f"{total_errors} errors across {len(errors_by_skill)} skills (from last {total_actions} actions)",
            data={
                "errors_by_skill": sorted_errors,
                "total_errors": total_errors,
                "total_actions_analyzed": total_actions,
                "error_rate": round(total_errors / total_actions, 3) if total_actions > 0 else 0,
                "most_failing_skill": max(errors_by_skill, key=lambda k: errors_by_skill[k]["count"]) if errors_by_skill else None,
            }
        )

    async def _capability_matrix(self, params: Dict) -> SkillResult:
        """Generate a capability matrix showing what the agent can do."""
        registry = self._registry_ref
        if not registry:
            return SkillResult(success=False, message="No skill registry connected")

        matrix = {}
        for skill_id, skill in registry.skills.items():
            m = skill.manifest
            cat = m.category
            if cat not in matrix:
                matrix[cat] = {
                    "skills": [],
                    "total_actions": 0,
                    "has_credentials": True,
                }
            matrix[cat]["skills"].append({
                "skill_id": m.skill_id,
                "name": m.name,
                "actions": [a.name for a in m.actions],
                "has_credentials": skill.check_credentials(),
                "action_count": len(m.actions),
            })
            matrix[cat]["total_actions"] += len(m.actions)
            if not skill.check_credentials():
                matrix[cat]["has_credentials"] = False

        # Identify capability areas
        capability_areas = {
            "file_operations": any(
                sid in registry.skills for sid in ["filesystem", "shell"]
            ),
            "web_interaction": any(
                sid in registry.skills for sid in ["browser", "request"]
            ),
            "social_media": any(
                sid in registry.skills for sid in ["twitter"]
            ),
            "code_management": any(
                sid in registry.skills for sid in ["github"]
            ),
            "content_creation": any(
                sid in registry.skills for sid in ["content"]
            ),
            "communication": any(
                sid in registry.skills for sid in ["email"]
            ),
            "deployment": any(
                sid in registry.skills for sid in ["vercel", "namecheap"]
            ),
            "self_modification": any(
                sid in registry.skills for sid in ["self_modify", "steering"]
            ),
            "payments": any(
                sid in registry.skills for sid in ["crypto"]
            ),
            "memory": any(
                sid in registry.skills for sid in ["memory"]
            ),
            "orchestration": any(
                sid in registry.skills for sid in ["orchestrator"]
            ),
            "introspection": any(
                sid in registry.skills for sid in ["agent_meta"]
            ),
        }

        enabled = [k for k, v in capability_areas.items() if v]
        disabled = [k for k, v in capability_areas.items() if not v]

        return SkillResult(
            success=True,
            message=f"Capability matrix: {len(enabled)}/{len(capability_areas)} areas covered",
            data={
                "matrix": matrix,
                "capability_areas": capability_areas,
                "enabled_areas": enabled,
                "disabled_areas": disabled,
                "coverage": round(len(enabled) / len(capability_areas), 2) if capability_areas else 0,
            }
        )

    async def _suggest_action(self, params: Dict) -> SkillResult:
        """Suggest the best action for a given goal."""
        registry = self._registry_ref
        if not registry:
            return SkillResult(success=False, message="No skill registry connected")

        goal = params.get("goal", "").lower()
        if not goal:
            return SkillResult(success=False, message="Please describe your goal")

        # Build keyword mappings for common goals
        goal_keywords = {
            "read": ["filesystem:view", "filesystem:glob", "shell:bash"],
            "write": ["filesystem:write", "filesystem:patch"],
            "file": ["filesystem:view", "filesystem:write", "filesystem:glob", "filesystem:ls"],
            "search": ["filesystem:grep", "filesystem:glob"],
            "run": ["shell:bash"],
            "execute": ["shell:bash"],
            "command": ["shell:bash"],
            "git": ["shell:bash", "github:create_repo"],
            "github": ["github:create_repo", "github:create_issue"],
            "tweet": ["twitter:post_tweet", "twitter:reply"],
            "post": ["twitter:post_tweet", "content:generate"],
            "email": ["email:send_email"],
            "deploy": ["vercel:deploy"],
            "website": ["vercel:deploy", "filesystem:write"],
            "domain": ["namecheap:check_domain", "namecheap:register_domain"],
            "browse": ["browser:navigate", "browser:screenshot"],
            "http": ["request:fetch"],
            "api": ["request:fetch"],
            "create": ["content:generate", "filesystem:write"],
            "generate": ["content:generate"],
            "modify": ["self_modify:edit_prompt", "self_modify:switch_model"],
            "prompt": ["self_modify:edit_prompt", "self_modify:get_prompt"],
            "model": ["self_modify:switch_model", "self_modify:list_models"],
            "spawn": ["orchestrator:create_agent"],
            "agent": ["orchestrator:create_agent", "orchestrator:list_agents"],
            "pay": ["crypto:send", "crypto:balance"],
            "money": ["crypto:balance", "crypto:send"],
            "remember": ["memory:store", "memory:recall"],
            "help": ["agent_meta:list_skills", "agent_meta:search_capabilities"],
            "status": ["agent_meta:agent_status"],
            "error": ["agent_meta:error_summary"],
        }

        suggestions = []
        seen = set()
        for keyword, actions in goal_keywords.items():
            if keyword in goal:
                for action_ref in actions:
                    if action_ref in seen:
                        continue
                    seen.add(action_ref)
                    parts = action_ref.split(":")
                    skill = registry.get(parts[0])
                    if skill:
                        action_obj = skill.get_action(parts[1]) if len(parts) > 1 else None
                        suggestions.append({
                            "action": action_ref,
                            "description": action_obj.description if action_obj else skill.manifest.description,
                            "available": True,
                        })

        # Also do fuzzy matching via search
        search_result = await self._search_capabilities({"query": goal.split()[0] if goal.split() else goal})
        if search_result.success:
            for match in search_result.data.get("matches", [])[:3]:
                ref = match["usage"]
                if ref not in seen:
                    seen.add(ref)
                    suggestions.append({
                        "action": ref,
                        "description": match["description"],
                        "available": True,
                    })

        return SkillResult(
            success=True,
            message=f"Found {len(suggestions)} suggested actions for: {goal}",
            data={
                "goal": goal,
                "suggestions": suggestions[:10],
                "tip": "Use the action in the format skill_id:action_name with appropriate parameters",
            }
        )
