"""
WisentBot Skills - Modular capabilities for autonomous agents.

Skills provide specific capabilities that agents can use to interact
with the world. Each skill has a manifest describing its actions.
"""

from .base import Skill, SkillRegistry, SkillManifest, SkillAction, SkillResult
from .browser import BrowserSkill
from .content import ContentCreationSkill
from .email import EmailSkill
from .filesystem import FilesystemSkill
from .github import GitHubSkill
from .mcp_client import MCPClientSkill
from .namecheap import NamecheapSkill
from .request import RequestSkill
from .shell import ShellSkill
from .twitter import TwitterSkill
from .vercel import VercelSkill
from .self_modify import SelfModifySkill
from .steering import SteeringSkill
from .memory import MemorySkill

__all__ = [
    # Base
    "Skill",
    "SkillRegistry",
    "SkillManifest",
    "SkillAction",
    "SkillResult",
    # Skills
    "BrowserSkill",
    "ContentCreationSkill",
    "EmailSkill",
    "FilesystemSkill",
    "GitHubSkill",
    "MCPClientSkill",
    "NamecheapSkill",
    "RequestSkill",
    "ShellSkill",
    "TwitterSkill",
    "VercelSkill",
    "SelfModifySkill",
    "SteeringSkill",
    "MemorySkill",
]
