"""
Base Skill class - all skills inherit from this.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from .types import SkillResult, SkillAction, SkillManifest


class Skill(ABC):
    """
    Base class for all agent skills.

    Skills are modular capabilities that can be installed on agents.
    Each skill provides one or more actions the agent can take.
    """

    def __init__(self, credentials: Dict[str, str] = None):
        self.credentials = credentials or {}
        self.initialized = False
        self._usage_count = 0
        self._total_cost = 0
        self._total_revenue = 0

    @property
    @abstractmethod
    def manifest(self) -> SkillManifest:
        pass

    @abstractmethod
    async def execute(self, action: str, params: Dict) -> SkillResult:
        pass

    def get_actions(self) -> List[SkillAction]:
        return self.manifest.actions

    def get_action(self, name: str) -> Optional[SkillAction]:
        for action in self.manifest.actions:
            if action.name == name:
                return action
        return None

    def estimate_cost(self, action: str, params: Dict) -> float:
        action_def = self.get_action(action)
        return action_def.estimated_cost if action_def else 0

    def check_credentials(self) -> bool:
        for cred in self.manifest.required_credentials:
            if cred not in self.credentials or not self.credentials[cred]:
                return False
        return True

    def get_missing_credentials(self) -> List[str]:
        return [c for c in self.manifest.required_credentials
                if c not in self.credentials or not self.credentials[c]]

    async def initialize(self) -> bool:
        if not self.check_credentials():
            return False
        self.initialized = True
        return True

    def record_usage(self, cost: float = 0, revenue: float = 0):
        self._usage_count += 1
        self._total_cost += cost
        self._total_revenue += revenue

    @property
    def stats(self) -> Dict:
        return {
            "usage_count": self._usage_count,
            "total_cost": self._total_cost,
            "total_revenue": self._total_revenue,
            "profit": self._total_revenue - self._total_cost,
        }

    def to_dict(self) -> Dict:
        return {
            "skill_id": self.manifest.skill_id,
            "name": self.manifest.name,
            "category": self.manifest.category,
            "description": self.manifest.description,
            "actions": [
                {"name": a.name, "description": a.description,
                 "parameters": a.parameters, "estimated_cost": a.estimated_cost}
                for a in self.manifest.actions
            ],
            "initialized": self.initialized,
            "stats": self.stats,
        }
