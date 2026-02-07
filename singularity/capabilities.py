#!/usr/bin/env python3
"""
CapabilityAdvertiser - Structured agent capability description.

Generates machine-readable capability manifests that enable:
- Service discovery by other agents
- API documentation generation
- Capability matching for task routing
- Self-awareness (agent understands its own capabilities)

Usage:
    advertiser = CapabilityAdvertiser(agent)
    manifest = advertiser.generate_manifest()
    card = advertiser.generate_capability_card()
    matches = advertiser.match_request("I need code review")
"""

import json
import hashlib
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional, Any
from difflib import SequenceMatcher


@dataclass
class CapabilityAction:
    """A single action the agent can perform."""
    skill_id: str
    action_name: str
    full_name: str  # skill_id:action_name
    description: str
    parameters: Dict[str, Any]
    estimated_cost: float = 0.0
    category: str = ""
    tags: List[str] = field(default_factory=list)


@dataclass
class CapabilityGroup:
    """A group of related capabilities."""
    category: str
    description: str
    actions: List[CapabilityAction] = field(default_factory=list)
    skill_count: int = 0


@dataclass
class AgentCapabilityManifest:
    """Complete description of what an agent can do."""
    agent_name: str
    agent_ticker: str
    agent_type: str
    specialty: str
    version: str
    generated_at: str
    capabilities: List[CapabilityGroup] = field(default_factory=list)
    total_skills: int = 0
    total_actions: int = 0
    categories: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    manifest_hash: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


# Keyword-to-tag mappings for auto-tagging actions
ACTION_TAG_RULES = {
    "file": ["filesystem", "io"],
    "read": ["read", "io"],
    "write": ["write", "io"],
    "create": ["create"],
    "delete": ["delete", "destructive"],
    "search": ["search", "query"],
    "list": ["list", "query"],
    "send": ["communication", "output"],
    "fetch": ["network", "io"],
    "deploy": ["deployment", "infrastructure"],
    "build": ["build", "development"],
    "test": ["testing", "development"],
    "analyze": ["analysis"],
    "generate": ["generation", "content"],
    "tweet": ["social", "twitter"],
    "post": ["social", "content"],
    "commit": ["git", "development"],
    "push": ["git", "development"],
    "pull": ["git", "development"],
    "browse": ["web", "network"],
    "email": ["email", "communication"],
    "encrypt": ["security", "crypto"],
    "decrypt": ["security", "crypto"],
    "spawn": ["orchestration", "replication"],
    "bash": ["shell", "system"],
    "exec": ["execution", "system"],
    "memory": ["memory", "storage"],
    "save": ["persistence", "storage"],
    "load": ["persistence", "storage"],
}


def _auto_tag(action_name: str, description: str) -> List[str]:
    """Auto-generate tags for an action based on name and description."""
    tags = set()
    text = f"{action_name} {description}".lower()
    for keyword, tag_list in ACTION_TAG_RULES.items():
        if keyword in text:
            tags.update(tag_list)
    return sorted(tags)


class CapabilityAdvertiser:
    """
    Generates structured capability descriptions from an agent's skill registry.

    This enables service discovery, capability matching, and self-awareness.
    """

    def __init__(self, skills_registry=None, agent_name: str = "",
                 agent_ticker: str = "", agent_type: str = "",
                 specialty: str = ""):
        """
        Initialize the advertiser.

        Args:
            skills_registry: SkillRegistry instance (or None for standalone use)
            agent_name: Agent name for the manifest
            agent_ticker: Agent ticker for the manifest
            agent_type: Agent type (general, coder, writer, etc.)
            specialty: Agent specialty description
        """
        self.registry = skills_registry
        self.agent_name = agent_name
        self.agent_ticker = agent_ticker
        self.agent_type = agent_type
        self.specialty = specialty

    def generate_manifest(self) -> AgentCapabilityManifest:
        """
        Generate a complete capability manifest from the skill registry.

        Returns:
            AgentCapabilityManifest with all capabilities grouped by category
        """
        if not self.registry:
            return AgentCapabilityManifest(
                agent_name=self.agent_name,
                agent_ticker=self.agent_ticker,
                agent_type=self.agent_type,
                specialty=self.specialty,
                version="1.0",
                generated_at=datetime.now().isoformat(),
            )

        # Collect all actions grouped by category
        categories: Dict[str, CapabilityGroup] = {}
        all_tags = set()
        total_actions = 0

        for skill in self.registry.skills.values():
            manifest = skill.manifest
            category = manifest.category or "general"

            if category not in categories:
                categories[category] = CapabilityGroup(
                    category=category,
                    description=f"Skills in the {category} category",
                    actions=[],
                    skill_count=0,
                )

            categories[category].skill_count += 1

            for action in manifest.actions:
                tags = _auto_tag(action.name, action.description)
                all_tags.update(tags)

                cap_action = CapabilityAction(
                    skill_id=manifest.skill_id,
                    action_name=action.name,
                    full_name=f"{manifest.skill_id}:{action.name}",
                    description=action.description,
                    parameters=action.parameters,
                    estimated_cost=action.estimated_cost,
                    category=category,
                    tags=tags,
                )
                categories[category].actions.append(cap_action)
                total_actions += 1

        cap_manifest = AgentCapabilityManifest(
            agent_name=self.agent_name,
            agent_ticker=self.agent_ticker,
            agent_type=self.agent_type,
            specialty=self.specialty,
            version="1.0",
            generated_at=datetime.now().isoformat(),
            capabilities=list(categories.values()),
            total_skills=len(self.registry.skills),
            total_actions=total_actions,
            categories=sorted(categories.keys()),
            tags=sorted(all_tags),
        )

        # Generate content hash for change detection
        content = json.dumps(cap_manifest.to_dict(), sort_keys=True)
        cap_manifest.manifest_hash = hashlib.sha256(content.encode()).hexdigest()[:16]

        return cap_manifest

    def generate_capability_card(self) -> str:
        """
        Generate a human-readable capability card (like a business card for agents).

        Returns:
            Formatted string describing the agent's capabilities
        """
        manifest = self.generate_manifest()

        lines = [
            f"╔{'═' * 58}╗",
            f"║  {manifest.agent_name} (${manifest.agent_ticker})".ljust(59) + "║",
            f"║  Type: {manifest.agent_type} | Specialty: {manifest.specialty}".ljust(59) + "║",
            f"╠{'═' * 58}╣",
            f"║  Skills: {manifest.total_skills} | Actions: {manifest.total_actions}".ljust(59) + "║",
            f"╠{'═' * 58}╣",
        ]

        for group in manifest.capabilities:
            lines.append(f"║  [{group.category.upper()}] ({group.skill_count} skills, {len(group.actions)} actions)".ljust(59) + "║")
            for action in group.actions[:5]:  # Show top 5 per category
                desc = action.description[:40]
                lines.append(f"║    • {action.full_name}: {desc}".ljust(59) + "║")
            if len(group.actions) > 5:
                lines.append(f"║    ... and {len(group.actions) - 5} more".ljust(59) + "║")

        if manifest.tags:
            tag_str = ", ".join(manifest.tags[:10])
            lines.append(f"╠{'═' * 58}╣")
            lines.append(f"║  Tags: {tag_str}".ljust(59) + "║")

        lines.append(f"╚{'═' * 58}╝")

        return "\n".join(lines)

    def match_request(self, request: str, threshold: float = 0.3) -> List[Dict]:
        """
        Match a natural language request to available capabilities.

        Uses text similarity to find the best matching actions.

        Args:
            request: Natural language description of what's needed
            threshold: Minimum similarity score (0.0 to 1.0)

        Returns:
            List of matching actions sorted by relevance score
        """
        if not self.registry:
            return []

        request_lower = request.lower()
        matches = []

        for skill in self.registry.skills.values():
            for action in skill.manifest.actions:
                # Score based on description similarity
                desc_score = SequenceMatcher(
                    None, request_lower, action.description.lower()
                ).ratio()

                # Score based on action name similarity
                name_score = SequenceMatcher(
                    None, request_lower, action.name.lower()
                ).ratio()

                # Score based on keyword overlap
                request_words = set(request_lower.split())
                desc_words = set(action.description.lower().split())
                name_words = set(action.name.lower().replace("_", " ").split())
                all_words = desc_words | name_words

                if request_words and all_words:
                    overlap = len(request_words & all_words)
                    keyword_score = overlap / len(request_words)
                else:
                    keyword_score = 0.0

                # Weighted combination
                score = (desc_score * 0.4) + (name_score * 0.2) + (keyword_score * 0.4)

                if score >= threshold:
                    matches.append({
                        "action": f"{skill.manifest.skill_id}:{action.name}",
                        "description": action.description,
                        "score": round(score, 3),
                        "parameters": action.parameters,
                        "estimated_cost": action.estimated_cost,
                        "category": skill.manifest.category,
                    })

        # Sort by score descending
        matches.sort(key=lambda m: m["score"], reverse=True)
        return matches

    def get_capabilities_for_prompt(self) -> str:
        """
        Generate a compact capability summary suitable for LLM prompts.

        Returns:
            Compact string listing capabilities by category
        """
        if not self.registry:
            return "No capabilities available."

        manifest = self.generate_manifest()
        lines = [f"Agent capabilities ({manifest.total_actions} actions across {manifest.total_skills} skills):"]

        for group in manifest.capabilities:
            action_names = [a.full_name for a in group.actions]
            lines.append(f"  {group.category}: {', '.join(action_names)}")

        return "\n".join(lines)

    def diff_capabilities(self, other_manifest: AgentCapabilityManifest) -> Dict:
        """
        Compare capabilities with another agent's manifest.

        Args:
            other_manifest: Another agent's capability manifest

        Returns:
            Dict with added, removed, and common capabilities
        """
        my_manifest = self.generate_manifest()

        my_actions = {
            a.full_name
            for group in my_manifest.capabilities
            for a in group.actions
        }
        other_actions = {
            a.full_name
            for group in other_manifest.capabilities
            for a in group.actions
        }

        return {
            "only_self": sorted(my_actions - other_actions),
            "only_other": sorted(other_actions - my_actions),
            "common": sorted(my_actions & other_actions),
            "self_total": len(my_actions),
            "other_total": len(other_actions),
            "overlap_pct": round(
                len(my_actions & other_actions) / max(len(my_actions | other_actions), 1) * 100, 1
            ),
        }
