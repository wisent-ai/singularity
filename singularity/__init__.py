"""
WisentBot - Autonomous AI Agent Framework

An open-source framework for building autonomous AI agents that can
execute tasks, manage resources, and interact with the real world.
"""

__version__ = "0.1.0"

from .autonomous_agent import AutonomousAgent
from .cognition import CognitionEngine, AgentState, Decision, Action, TokenUsage
from .skills.base import Skill, SkillRegistry, SkillManifest, SkillAction, SkillResult
from .lifecycle import (
    LifecycleHook,
    HookManager,
    OutcomeTrackingHook,
    CycleMetricsHook,
    AdaptiveIntervalHook,
)

__all__ = [
    "AutonomousAgent",
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
