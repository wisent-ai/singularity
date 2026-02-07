"""
WisentBot - Autonomous AI Agent Framework

An open-source framework for building autonomous AI agents that can
execute tasks, manage resources, and interact with the real world.
"""

__version__ = "0.1.0"

from .autonomous_agent import AutonomousAgent
from .interactive_agent import InteractiveAgent, ChatMessage, ChatResponse
from .cognition import CognitionEngine, AgentState, Decision, Action, TokenUsage
from .skills.base import Skill, SkillRegistry, SkillManifest, SkillAction, SkillResult

__all__ = [
    "AutonomousAgent",
    "InteractiveAgent",
    "ChatMessage",
    "ChatResponse",
    "CognitionEngine",
    "AgentState",
    "Decision",
    "Action",
    "TokenUsage",
    "Skill",
    "SkillRegistry",
    "SkillManifest",
    "SkillAction",
    "SkillResult",
]
