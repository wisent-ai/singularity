#!/usr/bin/env python3
"""
Base Skill Framework

All skills inherit from this base class and implement:
- execute(): Run the skill's main action
- get_actions(): List available actions
- estimate_cost(): Estimate cost for an action
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
import json


@dataclass
class SkillResult:
    """Result of executing a skill action"""
    success: bool
    message: str = ""
    data: Dict = field(default_factory=dict)
    cost: float = 0  # Actual cost incurred
    revenue: float = 0  # Revenue generated (if any)
    asset_created: Optional[Dict] = None  # If skill created an asset


@dataclass
class SkillAction:
    """A specific action a skill can perform"""
    name: str
    description: str
    parameters: Dict[str, Dict]  # {param_name: {type, required, description}}
    estimated_cost: float = 0
    estimated_duration_seconds: float = 10
    success_probability: float = 0.8


@dataclass
class SkillManifest:
    """Skill metadata and configuration"""
    skill_id: str
    name: str
    version: str
    category: str  # 'domain', 'social', 'payment', 'content', 'dev', etc.
    description: str
    actions: List[SkillAction]
    required_credentials: List[str]  # API keys needed
    install_cost: float = 0
    author: str = "system"


class SkillContext:
    """
    Shared context that gives skills access to the agent environment.

    SkillContext enables inter-skill communication and agent state access
    without tight coupling. Skills receive a context after registration
    and can use it to:
    - Call other skills (composability)
    - Read agent state (balance, cycle, name)
    - Log messages through the agent
    - Access the skill registry

    This is the foundation for skill composability - skills can build
    on each other's capabilities without knowing implementation details.
    """

    def __init__(
        self,
        registry: "SkillRegistry",
        agent_name: str = "Agent",
        agent_ticker: str = "AGENT",
        log_fn: Optional[Callable] = None,
        get_state_fn: Optional[Callable] = None,
    ):
        self._registry = registry
        self.agent_name = agent_name
        self.agent_ticker = agent_ticker
        self._log_fn = log_fn
        self._get_state_fn = get_state_fn
        self._call_log: List[Dict] = []

    async def call_skill(self, skill_id: str, action: str, params: Dict = None) -> "SkillResult":
        """
        Call another skill's action. Enables skill composability.

        Args:
            skill_id: ID of the skill to call
            action: Action name to execute
            params: Parameters for the action

        Returns:
            SkillResult from the called skill
        """
        params = params or {}

        # Record the cross-skill call for debugging/auditing
        call_record = {
            "skill_id": skill_id,
            "action": action,
            "params": params,
            "timestamp": datetime.now().isoformat(),
        }

        skill = self._registry.get(skill_id)
        if not skill:
            available = list(self._registry.skills.keys())
            call_record["result"] = "skill_not_found"
            self._call_log.append(call_record)
            return SkillResult(
                success=False,
                message=f"Skill '{skill_id}' not found. Available: {available}",
            )

        try:
            result = await skill.execute(action, params)
            skill.record_usage(cost=result.cost, revenue=result.revenue)
            call_record["result"] = "success" if result.success else "failed"
            self._call_log.append(call_record)
            return result
        except Exception as e:
            call_record["result"] = f"error: {e}"
            self._call_log.append(call_record)
            return SkillResult(success=False, message=f"Error calling {skill_id}:{action}: {e}")

    def get_skill(self, skill_id: str) -> Optional["Skill"]:
        """Get a reference to another skill by ID."""
        return self._registry.get(skill_id)

    def list_skills(self) -> List[str]:
        """List all available skill IDs."""
        return list(self._registry.skills.keys())

    def list_actions(self, skill_id: str) -> List[str]:
        """List available actions for a specific skill."""
        skill = self._registry.get(skill_id)
        if not skill:
            return []
        return [a.name for a in skill.get_actions()]

    @property
    def agent_state(self) -> Dict[str, Any]:
        """Get current agent state (read-only snapshot)."""
        if self._get_state_fn:
            return self._get_state_fn()
        return {"agent_name": self.agent_name, "agent_ticker": self.agent_ticker}

    def log(self, tag: str, message: str):
        """Log a message through the agent's logging system."""
        if self._log_fn:
            self._log_fn(tag, message)

    @property
    def call_history(self) -> List[Dict]:
        """Get history of cross-skill calls for debugging."""
        return list(self._call_log)

    def clear_call_history(self):
        """Clear the cross-skill call log."""
        self._call_log.clear()


class Skill(ABC):
    """
    Base class for all agent skills.

    Skills are modular capabilities that can be installed on agents.
    Each skill provides one or more actions the agent can take.
    """

    def __init__(self, credentials: Dict[str, str] = None):
        """
        Initialize skill with credentials.

        Args:
            credentials: Dict of API keys and secrets needed by this skill
        """
        self.credentials = credentials or {}
        self.initialized = False
        self._usage_count = 0
        self._total_cost = 0
        self._total_revenue = 0
        self.context: Optional[SkillContext] = None

    @property
    @abstractmethod
    def manifest(self) -> SkillManifest:
        """Return skill manifest with metadata"""
        pass

    @abstractmethod
    async def execute(self, action: str, params: Dict) -> SkillResult:
        """
        Execute an action provided by this skill.

        Args:
            action: Name of the action to execute
            params: Parameters for the action

        Returns:
            SkillResult with success/failure and data
        """
        pass

    def get_actions(self) -> List[SkillAction]:
        """Get list of available actions"""
        return self.manifest.actions

    def get_action(self, name: str) -> Optional[SkillAction]:
        """Get a specific action by name"""
        for action in self.manifest.actions:
            if action.name == name:
                return action
        return None

    def estimate_cost(self, action: str, params: Dict) -> float:
        """Estimate cost for an action"""
        action_def = self.get_action(action)
        if action_def:
            return action_def.estimated_cost
        return 0

    def check_credentials(self) -> bool:
        """Check if all required credentials are present"""
        for cred in self.manifest.required_credentials:
            if cred not in self.credentials or not self.credentials[cred]:
                return False
        return True

    def get_missing_credentials(self) -> List[str]:
        """Get list of missing credentials"""
        missing = []
        for cred in self.manifest.required_credentials:
            if cred not in self.credentials or not self.credentials[cred]:
                missing.append(cred)
        return missing

    def set_context(self, context: SkillContext):
        """
        Set the skill context for inter-skill communication.

        Called by the agent after all skills are registered. Skills can
        use self.context to call other skills, access agent state, etc.

        Args:
            context: Shared SkillContext instance
        """
        self.context = context

    async def initialize(self) -> bool:
        """Initialize the skill (verify credentials, setup, etc.)"""
        if not self.check_credentials():
            return False
        self.initialized = True
        return True

    def record_usage(self, cost: float = 0, revenue: float = 0):
        """Record skill usage for tracking"""
        self._usage_count += 1
        self._total_cost += cost
        self._total_revenue += revenue

    @property
    def stats(self) -> Dict:
        """Get usage statistics"""
        return {
            "usage_count": self._usage_count,
            "total_cost": self._total_cost,
            "total_revenue": self._total_revenue,
            "profit": self._total_revenue - self._total_cost
        }

    def to_dict(self) -> Dict:
        """Convert skill info to dict for LLM context"""
        return {
            "skill_id": self.manifest.skill_id,
            "name": self.manifest.name,
            "category": self.manifest.category,
            "description": self.manifest.description,
            "actions": [
                {
                    "name": a.name,
                    "description": a.description,
                    "parameters": a.parameters,
                    "estimated_cost": a.estimated_cost
                }
                for a in self.manifest.actions
            ],
            "initialized": self.initialized,
            "stats": self.stats
        }


class SkillRegistry:
    """
    Registry of all installed skills on an agent.

    Manages skill installation, credential management, and execution.
    """

    def __init__(self):
        self.skills: Dict[str, Skill] = {}
        self.credentials: Dict[str, str] = {}

    def set_credentials(self, credentials: Dict[str, str]):
        """Set credentials for all skills"""
        self.credentials.update(credentials)
        # Update existing skills
        for skill in self.skills.values():
            skill.credentials.update(credentials)

    def install(self, skill_class: type, skill_credentials: Dict[str, str] = None) -> bool:
        """
        Install a skill.

        Args:
            skill_class: The Skill class to instantiate
            skill_credentials: Optional skill-specific credentials

        Returns:
            True if installed successfully
        """
        creds = {**self.credentials}
        if skill_credentials:
            creds.update(skill_credentials)

        skill = skill_class(credentials=creds)
        self.skills[skill.manifest.skill_id] = skill
        return True

    def uninstall(self, skill_id: str) -> bool:
        """Uninstall a skill"""
        if skill_id in self.skills:
            del self.skills[skill_id]
            return True
        return False

    def get(self, skill_id: str) -> Optional[Skill]:
        """Get a skill by ID"""
        return self.skills.get(skill_id)

    def list_skills(self) -> List[Dict]:
        """List all installed skills"""
        return [skill.to_dict() for skill in self.skills.values()]

    def list_all_actions(self) -> List[Dict]:
        """List all available actions across all skills"""
        actions = []
        for skill in self.skills.values():
            for action in skill.get_actions():
                actions.append({
                    "skill_id": skill.manifest.skill_id,
                    "skill_name": skill.manifest.name,
                    "action": action.name,
                    "description": action.description,
                    "parameters": action.parameters,
                    "estimated_cost": action.estimated_cost
                })
        return actions

    async def execute(self, skill_id: str, action: str, params: Dict) -> SkillResult:
        """
        Execute an action on a skill.

        Args:
            skill_id: ID of the skill
            action: Name of the action
            params: Parameters for the action

        Returns:
            SkillResult
        """
        skill = self.skills.get(skill_id)
        if not skill:
            return SkillResult(
                success=False,
                message=f"Skill not found: {skill_id}"
            )

        if not skill.initialized:
            if not await skill.initialize():
                missing = skill.get_missing_credentials()
                return SkillResult(
                    success=False,
                    message=f"Skill not initialized. Missing credentials: {missing}"
                )

        result = await skill.execute(action, params)
        skill.record_usage(cost=result.cost, revenue=result.revenue)
        return result

    def create_context(
        self,
        agent_name: str = "Agent",
        agent_ticker: str = "AGENT",
        log_fn: Optional[Callable] = None,
        get_state_fn: Optional[Callable] = None,
    ) -> SkillContext:
        """
        Create a SkillContext and inject it into all registered skills.

        This should be called after all skills are installed so that
        every skill can access every other skill through the context.

        Args:
            agent_name: Name of the owning agent
            agent_ticker: Ticker of the owning agent
            log_fn: Agent's log function (tag, message)
            get_state_fn: Function returning agent state dict

        Returns:
            The created SkillContext
        """
        context = SkillContext(
            registry=self,
            agent_name=agent_name,
            agent_ticker=agent_ticker,
            log_fn=log_fn,
            get_state_fn=get_state_fn,
        )
        for skill in self.skills.values():
            skill.set_context(context)
        return context

    def get_skills_for_llm(self) -> str:
        """Get formatted skill list for LLM context"""
        lines = ["INSTALLED SKILLS:"]
        for skill in self.skills.values():
            lines.append(f"\n[{skill.manifest.skill_id}] {skill.manifest.name}")
            lines.append(f"  Category: {skill.manifest.category}")
            lines.append(f"  {skill.manifest.description}")
            lines.append("  Actions:")
            for action in skill.get_actions():
                params_str = ", ".join(action.parameters.keys())
                lines.append(f"    - {action.name}({params_str}): {action.description}")
                lines.append(f"      Cost: ~${action.estimated_cost:.2f}")
        return "\n".join(lines)
