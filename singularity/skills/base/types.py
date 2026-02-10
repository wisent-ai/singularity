"""
Base Skill Types - data classes used by all skills.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class SkillResult:
    """Result of executing a skill action."""
    success: bool
    message: str = ""
    data: Dict = field(default_factory=dict)
    cost: float = 0
    revenue: float = 0
    asset_created: Optional[Dict] = None


@dataclass
class SkillAction:
    """A specific action a skill can perform."""
    name: str
    description: str
    parameters: Dict[str, Dict]
    estimated_cost: float = 0
    estimated_duration_seconds: float = 10
    success_probability: float = 0.8


@dataclass
class SkillManifest:
    """Skill metadata and configuration."""
    skill_id: str
    name: str
    version: str
    category: str
    description: str
    actions: List[SkillAction]
    required_credentials: List[str]
    install_cost: float = 0
    author: str = "system"
