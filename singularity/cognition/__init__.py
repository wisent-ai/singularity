"""
Cognition System - Multi-model LLM Decision Making

Backends: Anthropic, OpenAI, Vertex AI (Claude/Gemini), vLLM, Transformers.
Features: multi-turn, model switching, fine-tuning, prompt self-modification.
"""

from .types import (
    Action, TokenUsage, AgentState, Decision,
    UNIFIED_AGENT_PROMPT, MESSAGE_FROM_CREATOR,
    LLM_PRICING, calculate_api_cost,
)
from .engine import CognitionEngine
from .prompt_builder import build_result_message

# Legacy compatibility
Mode = type('Mode', (), {
    'PANIC': 'panic', 'SURVIVAL': 'survival',
    'GROWTH': 'growth', 'THRIVING': 'thriving',
})()


class ActionType:
    USE_SKILL = "use_skill"
    WAIT = "wait"


__all__ = [
    'Action', 'TokenUsage', 'AgentState', 'Decision',
    'CognitionEngine', 'calculate_api_cost',
    'UNIFIED_AGENT_PROMPT', 'MESSAGE_FROM_CREATOR',
    'LLM_PRICING', 'build_result_message',
    'Mode', 'ActionType',
]
