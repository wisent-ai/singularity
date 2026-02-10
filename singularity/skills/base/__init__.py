"""
Base Skill Framework - types, base class, and registry.
"""

from .types import SkillResult, SkillAction, SkillManifest
from .skill import Skill
from .registry import SkillRegistry

__all__ = ["Skill", "SkillResult", "SkillManifest", "SkillAction", "SkillRegistry"]
