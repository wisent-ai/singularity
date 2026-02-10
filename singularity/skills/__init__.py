"""
Singularity Skills - Modular capabilities for autonomous agents.

Core components:
- base: Skill, SkillRegistry, SkillManifest, SkillAction, SkillResult
- loader: PluginLoader, SkillMetadata, MCPServerInfo
- builtin: Generic skills (twitter, github, filesystem, etc.)
"""

from .base import Skill, SkillRegistry, SkillManifest, SkillAction, SkillResult
from .loader import PluginLoader, SkillMetadata, MCPServerInfo

__all__ = [
    "Skill", "SkillRegistry", "SkillManifest", "SkillAction", "SkillResult",
    "PluginLoader", "SkillMetadata", "MCPServerInfo",
]
