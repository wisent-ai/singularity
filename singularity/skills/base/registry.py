"""
Skill Registry - manages skill installation, credentials, and execution.
Supports eager loading (by class) and lazy loading (by ID via PluginLoader).
"""

from typing import Any, Dict, List, Optional
from .types import SkillResult
from .skill import Skill


class SkillRegistry:
    """Registry of all installed skills on an agent."""

    def __init__(self, loader=None):
        self.skills: Dict[str, Skill] = {}
        self.credentials: Dict[str, str] = {}
        self.loader = loader
        self._agent = None

    def set_agent(self, agent: Any):
        """Set the agent reference for wiring hooks."""
        self._agent = agent

    def set_credentials(self, credentials: Dict[str, str]):
        self.credentials.update(credentials)
        for skill in self.skills.values():
            skill.credentials.update(credentials)

    def install(self, skill_class_or_id, skill_credentials: Dict[str, str] = None) -> bool:
        """Install a skill by class (eager) or by ID (lazy)."""
        creds = {**self.credentials}
        if skill_credentials:
            creds.update(skill_credentials)
        if isinstance(skill_class_or_id, str):
            return self.install_by_id(skill_class_or_id, creds)
        skill = skill_class_or_id(credentials=creds)
        self.skills[skill.manifest.skill_id] = skill
        return True

    def install_by_id(self, skill_id: str, credentials: Dict[str, str] = None) -> bool:
        """Install a skill by ID using the PluginLoader (lazy loading)."""
        if not self.loader:
            return False
        creds = {**self.credentials, **(credentials or {})}
        if not self.loader.check_credentials(skill_id, creds):
            return False
        skill = self.loader.load_with_wiring(skill_id, creds, self._agent) if self._agent else self.loader.load(skill_id, creds)
        if skill:
            self.skills[skill_id] = skill
            return True
        return False

    def install_all_available(self, credentials: Dict[str, str] = None) -> List[str]:
        """Install all skills that have valid credentials."""
        if not self.loader:
            return []
        creds = {**self.credentials, **(credentials or {})}
        return [m.skill_id for m in self.loader.list_available() if self.install_by_id(m.skill_id, creds)]

    def uninstall(self, skill_id: str) -> bool:
        if skill_id in self.skills:
            del self.skills[skill_id]
            return True
        return False

    def get(self, skill_id: str) -> Optional[Skill]:
        return self.skills.get(skill_id)

    def list_skills(self) -> List[Dict]:
        return [s.to_dict() for s in self.skills.values()]

    def list_all_actions(self) -> List[Dict]:
        actions = []
        for skill in self.skills.values():
            for action in skill.get_actions():
                actions.append({
                    "skill_id": skill.manifest.skill_id, "skill_name": skill.manifest.name,
                    "action": action.name, "description": action.description,
                    "parameters": action.parameters, "estimated_cost": action.estimated_cost,
                })
        return actions

    async def execute(self, skill_id: str, action: str, params: Dict) -> SkillResult:
        skill = self.skills.get(skill_id)
        if not skill:
            return SkillResult(success=False, message=f"Skill not found: {skill_id}")
        if not skill.initialized:
            if not await skill.initialize():
                return SkillResult(success=False, message=f"Missing credentials: {skill.get_missing_credentials()}")
        result = await skill.execute(action, params)
        skill.record_usage(cost=result.cost, revenue=result.revenue)
        return result

    def get_skills_for_llm(self) -> str:
        lines = ["INSTALLED SKILLS:"]
        for skill in self.skills.values():
            lines.append(f"\n[{skill.manifest.skill_id}] {skill.manifest.name}")
            lines.append(f"  Category: {skill.manifest.category}")
            lines.append(f"  {skill.manifest.description}")
            lines.append("  Actions:")
            for action in skill.get_actions():
                ps = ", ".join(action.parameters.keys())
                lines.append(f"    - {action.name}({ps}): {action.description}")
                lines.append(f"      Cost: ~${action.estimated_cost:.2f}")
        return "\n".join(lines)
