"""
Singularity - Autonomous AI Agent Framework

An open-source framework for building autonomous AI agents that can
execute tasks, manage resources, and interact with the real world.
"""

__version__ = "0.2.0"

from .autonomous_agent import AutonomousAgent
from .cognition import (
    CognitionEngine, AgentState, Decision, Action, TokenUsage,
    calculate_api_cost, UNIFIED_AGENT_PROMPT, MESSAGE_FROM_CREATOR,
    build_result_message,
)
from .skills.base import Skill, SkillRegistry, SkillManifest, SkillAction, SkillResult
from .skills.loader import PluginLoader, SkillMetadata, MCPServerInfo

__all__ = [
    "AutonomousAgent",
    "CognitionEngine", "AgentState", "Decision", "Action", "TokenUsage",
    "calculate_api_cost", "UNIFIED_AGENT_PROMPT", "MESSAGE_FROM_CREATOR",
    "build_result_message",
    "Skill", "SkillRegistry", "SkillManifest", "SkillAction", "SkillResult",
    "PluginLoader", "SkillMetadata", "MCPServerInfo",
]
