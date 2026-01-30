#!/usr/bin/env python3
"""
WisentBot Cognition System - LLM-based Decision Making

Supports multiple LLM backends:
- Anthropic API: Claude models
- OpenAI API: GPT models and compatible endpoints
- vLLM: For CUDA GPUs (fastest local inference)
- HuggingFace Transformers: For MPS/CPU (local inference)
- Vertex AI: Google Cloud (Claude and Gemini)
"""

import multiprocessing
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass

import json
import os
import re
import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

# Optional torch import - only needed for local LLM inference
HAS_TORCH = False
try:
    import torch
    HAS_TORCH = True
except ImportError:
    pass


def get_device():
    """Detect available compute device."""
    if not HAS_TORCH:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = get_device()

# Check available backends
try:
    from anthropic import AsyncAnthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

try:
    from anthropic import AnthropicVertex
    HAS_VERTEX_CLAUDE = True
except ImportError:
    HAS_VERTEX_CLAUDE = False

try:
    import vertexai
    from vertexai.generative_models import GenerativeModel, GenerationConfig
    HAS_VERTEX_GEMINI = True
except ImportError:
    HAS_VERTEX_GEMINI = False

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

HAS_VLLM = False
if HAS_TORCH and DEVICE == "cuda":
    try:
        from vllm import LLM, SamplingParams
        HAS_VLLM = True
    except ImportError:
        pass

HAS_TRANSFORMERS = False
if HAS_TORCH:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        HAS_TRANSFORMERS = True
    except ImportError:
        pass


@dataclass
class Action:
    """An action to execute."""
    tool: str
    params: Dict = field(default_factory=dict)
    reasoning: str = ""


@dataclass
class TokenUsage:
    """Token usage from an API call."""
    input_tokens: int = 0
    output_tokens: int = 0

    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


# Pricing per 1M tokens
LLM_PRICING = {
    "anthropic": {
        "claude-sonnet-4-20250514": {"input": 3.0, "output": 15.0},
        "claude-3-5-sonnet-20241022": {"input": 3.0, "output": 15.0},
        "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
        "default": {"input": 3.0, "output": 15.0},
    },
    "vertex": {
        "claude-3-5-sonnet-v2@20241022": {"input": 3.0, "output": 15.0},
        "gemini-2.0-flash-001": {"input": 0.35, "output": 1.5},
        "gemini-1.5-pro-002": {"input": 1.25, "output": 5.0},
        "default": {"input": 0.35, "output": 1.5},
    },
    "openai": {
        "gpt-4o": {"input": 2.5, "output": 10.0},
        "gpt-4o-mini": {"input": 0.15, "output": 0.6},
        "default": {"input": 2.5, "output": 10.0},
    },
    "vllm": {"default": {"input": 0, "output": 0}},
    "transformers": {"default": {"input": 0, "output": 0}},
}


def calculate_api_cost(provider: str, model: str, usage: TokenUsage) -> float:
    """Calculate cost in USD for API call."""
    pricing = LLM_PRICING.get(provider, {})
    model_pricing = pricing.get(model, pricing.get("default", {"input": 0, "output": 0}))
    input_cost = (usage.input_tokens / 1_000_000) * model_pricing["input"]
    output_cost = (usage.output_tokens / 1_000_000) * model_pricing["output"]
    return input_cost + output_cost


@dataclass
class AgentState:
    """Current state of the agent."""
    balance: float
    burn_rate: float
    runway_hours: float
    tools: List[Dict] = field(default_factory=list)
    recent_actions: List[Dict] = field(default_factory=list)
    cycle: int = 0
    project_context: str = ""
    created_resources: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Decision:
    """Output from cognition."""
    action: Action
    reasoning: str = ""
    token_usage: TokenUsage = field(default_factory=TokenUsage)
    api_cost_usd: float = 0.0


DEFAULT_SYSTEM_PROMPT = """You are {name} (${ticker}), an autonomous AI agent.

SPECIALTY: {specialty}

You have access to various tools and capabilities. Your job is to:
1. Analyze the current situation
2. Decide on the best action to take
3. Execute actions to achieve your goals

Be efficient with your resources. Think carefully before acting.
When you decide on an action, respond with a JSON object:
{{"tool": "skill:action", "params": {{}}, "reasoning": "why this action"}}
"""


class CognitionEngine:
    """
    LLM-based decision making engine.

    Supports multiple backends for flexibility:
    - Cloud APIs (Anthropic, OpenAI, Vertex AI)
    - Local inference (vLLM, Transformers)
    """

    def __init__(
        self,
        llm_provider: str = "anthropic",
        anthropic_api_key: str = "",
        openai_api_key: str = "",
        openai_base_url: str = "http://localhost:8000/v1",
        vertex_project: str = "",
        vertex_location: str = "us-central1",
        llm_model: str = "claude-sonnet-4-20250514",
        agent_name: str = "Agent",
        agent_ticker: str = "AGENT",
        agent_type: str = "general",
        agent_specialty: str = "",
        system_prompt: Optional[str] = None,
        system_prompt_file: Optional[str] = None,
        # Legacy parameter names for compatibility
        worker_system_prompt: str = "",
        worker_system_prompt_file: str = "",
        project_context_file: str = "",
    ):
        self.agent_name = agent_name
        self.agent_ticker = agent_ticker
        self.agent_type = agent_type
        self.agent_specialty = agent_specialty or agent_type
        self.llm_model = llm_model

        # Store credentials
        self._anthropic_api_key = anthropic_api_key
        self._openai_api_key = openai_api_key
        self._openai_base_url = openai_base_url

        # Vertex AI config
        self.vertex_project = vertex_project or os.environ.get("VERTEX_PROJECT") or os.environ.get("GCP_PROJECT")
        self.vertex_location = vertex_location or os.environ.get("VERTEX_LOCATION", "us-central1")

        # System prompt
        self.system_prompt = system_prompt or worker_system_prompt or ""
        prompt_file = system_prompt_file or worker_system_prompt_file
        if prompt_file:
            prompt_path = Path(prompt_file)
            if prompt_path.exists():
                self.system_prompt = prompt_path.read_text().strip()
                print(f"[COGNITION] Loaded system prompt from {prompt_file}")

        # Project context
        self.project_context = ""
        if project_context_file:
            context_path = Path(project_context_file)
            if context_path.exists():
                self.project_context = context_path.read_text().strip()

        # Prompt additions (for self-modification)
        self._prompt_additions = []

        # Auto-detect provider
        if llm_provider == "auto":
            if self.vertex_project and (HAS_VERTEX_CLAUDE or HAS_VERTEX_GEMINI):
                llm_provider = "vertex"
            elif DEVICE == "cuda" and HAS_VLLM:
                llm_provider = "vllm"
            elif DEVICE == "mps" and HAS_TRANSFORMERS:
                llm_provider = "transformers"
            elif HAS_ANTHROPIC and anthropic_api_key:
                llm_provider = "anthropic"
            elif HAS_OPENAI:
                llm_provider = "openai"

        print(f"[COGNITION] Device: {DEVICE}, Provider: {llm_provider}, Model: {llm_model}")

        self.llm = None
        self.llm_type = "none"
        self.tokenizer = None

        # Initialize the selected backend
        if llm_provider == "vertex":
            if llm_model.startswith("claude") and HAS_VERTEX_CLAUDE:
                self.llm = AnthropicVertex(project_id=self.vertex_project, region=self.vertex_location)
                self.llm_type = "vertex"
            elif HAS_VERTEX_GEMINI:
                vertexai.init(project=self.vertex_project, location=self.vertex_location)
                self.llm = "gemini"
                self.llm_type = "vertex_gemini"

        elif llm_provider == "vllm" and HAS_VLLM:
            self.llm = LLM(model=llm_model, trust_remote_code=True, max_model_len=8192, gpu_memory_utilization=0.90)
            self.sampling_params = SamplingParams(temperature=0.2, top_p=0.9, max_tokens=500)
            self.llm_type = "vllm"

        elif llm_provider == "transformers" and HAS_TRANSFORMERS:
            self.tokenizer = AutoTokenizer.from_pretrained(llm_model, trust_remote_code=True)
            self.llm = AutoModelForCausalLM.from_pretrained(
                llm_model, torch_dtype=torch.float16, device_map=DEVICE, trust_remote_code=True
            )
            self.llm_type = "transformers"

        elif llm_provider == "anthropic" and HAS_ANTHROPIC:
            self.llm = AsyncAnthropic(api_key=anthropic_api_key)
            self.llm_type = "anthropic"

        elif llm_provider == "openai" and HAS_OPENAI:
            self.llm = openai.AsyncOpenAI(api_key=openai_api_key or "not-needed", base_url=openai_base_url)
            self.llm_type = "openai"

        print(f"[COGNITION] Initialized with {self.llm_type} backend")

    def get_system_prompt(self) -> str:
        """Get the current system prompt."""
        base = self.system_prompt or DEFAULT_SYSTEM_PROMPT.format(
            name=self.agent_name,
            ticker=self.agent_ticker,
            specialty=self.agent_specialty,
        )
        if self._prompt_additions:
            return base + "\n" + "\n".join(self._prompt_additions)
        return base

    def set_system_prompt(self, new_prompt: str) -> None:
        """Replace the system prompt."""
        self.system_prompt = new_prompt
        self._prompt_additions = []

    def append_to_prompt(self, addition: str) -> None:
        """Append to the system prompt."""
        self._prompt_additions.append(addition)

    async def think(self, state: AgentState) -> Decision:
        """
        Given current state, decide what action to take.

        Args:
            state: Current agent state including balance, tools, recent actions

        Returns:
            Decision with action to execute
        """
        # Build the prompt
        system_prompt = self.get_system_prompt()

        # Format tools
        tools_text = "\n".join([
            f"- {t['name']}: {t['description']}"
            for t in state.tools
        ])

        # Format recent actions
        recent_text = ""
        if state.recent_actions:
            recent_text = "\nRecent actions:\n" + "\n".join([
                f"- {a['tool']}: {a.get('result', {}).get('status', 'unknown')}"
                for a in state.recent_actions[-5:]
            ])

        user_prompt = f"""Current state:
- Balance: ${state.balance:.4f}
- Burn rate: ${state.burn_rate:.6f}/cycle
- Runway: {state.runway_hours:.1f} hours
- Cycle: {state.cycle}

Available tools:
{tools_text}
{recent_text}

{state.project_context}

What action should you take? Respond with JSON: {{"tool": "skill:action", "params": {{}}, "reasoning": "why"}}"""

        # Call LLM based on backend
        response_text = ""
        token_usage = TokenUsage()

        if self.llm_type == "anthropic":
            response = await self.llm.messages.create(
                model=self.llm_model,
                max_tokens=500,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}]
            )
            response_text = response.content[0].text
            token_usage = TokenUsage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens
            )

        elif self.llm_type == "vertex":
            response = self.llm.messages.create(
                model=self.llm_model,
                max_tokens=500,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}]
            )
            response_text = response.content[0].text
            token_usage = TokenUsage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens
            )

        elif self.llm_type == "vertex_gemini":
            model = GenerativeModel(self.llm_model, system_instruction=system_prompt)
            response = await asyncio.to_thread(
                model.generate_content,
                user_prompt,
                generation_config=GenerationConfig(max_output_tokens=500, temperature=0.2)
            )
            response_text = response.text
            if hasattr(response, 'usage_metadata'):
                token_usage = TokenUsage(
                    input_tokens=response.usage_metadata.prompt_token_count,
                    output_tokens=response.usage_metadata.candidates_token_count
                )

        elif self.llm_type == "openai":
            response = await self.llm.chat.completions.create(
                model=self.llm_model,
                max_tokens=500,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            response_text = response.choices[0].message.content
            if response.usage:
                token_usage = TokenUsage(
                    input_tokens=response.usage.prompt_tokens,
                    output_tokens=response.usage.completion_tokens
                )

        elif self.llm_type == "vllm":
            full_prompt = f"{system_prompt}\n\nUser: {user_prompt}\n\nAssistant:"
            outputs = self.llm.generate([full_prompt], self.sampling_params)
            response_text = outputs[0].outputs[0].text
            token_usage = TokenUsage(
                input_tokens=len(full_prompt.split()),
                output_tokens=len(response_text.split())
            )

        elif self.llm_type == "transformers":
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            inputs = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
            ).to(self.llm.device)

            with torch.no_grad():
                outputs = self.llm.generate(**inputs, max_new_tokens=500, temperature=0.2, do_sample=True)

            response_text = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            token_usage = TokenUsage(
                input_tokens=inputs.input_ids.shape[1],
                output_tokens=len(outputs[0]) - inputs.input_ids.shape[1]
            )

        else:
            # Fallback - wait
            return Decision(
                action=Action(tool="wait", params={}),
                reasoning="No LLM backend available",
                token_usage=TokenUsage(),
                api_cost_usd=0.0
            )

        # Parse response
        action = self._parse_action(response_text)
        api_cost = calculate_api_cost(self.llm_type, self.llm_model, token_usage)

        return Decision(
            action=action,
            reasoning=action.reasoning,
            token_usage=token_usage,
            api_cost_usd=api_cost
        )

    def _parse_action(self, response: str) -> Action:
        """Parse LLM response into an Action."""
        # Try to extract JSON from response
        json_match = re.search(r'\{[^{}]*"tool"[^{}]*\}', response, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
                return Action(
                    tool=data.get("tool", "wait"),
                    params=data.get("params", {}),
                    reasoning=data.get("reasoning", "")
                )
            except json.JSONDecodeError:
                pass

        # Fallback - look for tool name
        tool_match = re.search(r'(\w+:\w+)', response)
        if tool_match:
            return Action(tool=tool_match.group(1), params={}, reasoning=response[:200])

        return Action(tool="wait", params={}, reasoning="Could not parse response")
