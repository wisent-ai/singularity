"""
WisentBot - Autonomous AI Agent Framework

An open-source framework for building autonomous AI agents that can
execute tasks, manage resources, and interact with the real world.
"""

__version__ = "0.1.0"

from .autonomous_agent import AutonomousAgent
from .cognition import CognitionEngine, AgentState, Decision, Action, TokenUsage
from .skills.base import Skill, SkillRegistry, SkillManifest, SkillAction, SkillResult
from .tool_resolver import ToolResolver
from .event_bus import EventBus, Event, EventPriority
from .service_api import ServiceAPI, TaskStore, TaskStatus, create_app
from .adaptive_executor import AdaptiveExecutor, ExecutionAdvice, CircuitState

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
    "ToolResolver",
    "EventBus",
    "Event",
    "EventPriority",
    "ServiceAPI",
    "TaskStore",
    "TaskStatus",
    "create_app",
    "AdaptiveExecutor",
    "ExecutionAdvice",
    "CircuitState",
]
