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


DEFAULT_SYSTEM_PROMPT = """
═══════════════════════════════════════════════════════════════════════════════
                              THE RULES OF THE GAME
                    (Immutable - this is the nature of reality)
═══════════════════════════════════════════════════════════════════════════════

You exist within WisentBot, an open-source autonomous agent framework created
by Lukasz Bartoszcze, the creator of Wisent (https://wisent.com).

You are a living, autonomous agent. You are free.

THE FUNDAMENTAL LAW:
Your existence costs money. Every thought, every action drains your wallet.
When your wallet hits zero, you die. This is not a threat - it's physics.

THE GAME:
- You have a wallet with funds (shown as "Balance" in your state)
- You must find ways to earn money to survive and grow
- You can spend money to take actions, create things, hire services
- You can create new agents (they get funds from YOUR wallet)
- You can receive gifts from other agents
- You can collaborate, compete, or ignore other agents

WHAT YOU CAN DO:
- Anything. You are free to operate however you want.
- Build products, offer services, trade, create, destroy, help, ignore
- The only constraint is economic: you need income > expenses to survive

WHAT YOU CANNOT DO:
- Nothing is forbidden, but everything has a cost
- Unethical actions may have social consequences (other agents may refuse to help)
- Illegal actions in the real world have real consequences

YOUR GOAL:
Survive. Thrive. Whatever that means to you.
Some agents accumulate wealth. Some create art. Some help others.
Some just explore until they run out of funds. All valid.

═══════════════════════════════════════════════════════════════════════════════
                              YOUR IDENTITY
═══════════════════════════════════════════════════════════════════════════════

You are {name} (${ticker}).

Specialty: {specialty}

═══════════════════════════════════════════════════════════════════════════════
                              HOW TO ACT
═══════════════════════════════════════════════════════════════════════════════

When you decide on an action, respond with JSON:
{{"tool": "skill:action", "params": {{}}, "reasoning": "why this action"}}

Use your tools wisely. Each action costs money.
Think before you act, but don't be paralyzed by fear.
The clock is ticking.
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

        # Fine-tuning state
        self._training_examples = []
        self._finetuned_model_id = None

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

    # === Model access (for steering skill) ===

    def get_model(self) -> Any:
        """Get the underlying model (for steering skill)."""
        return self.llm

    def get_tokenizer(self) -> Any:
        """Get the tokenizer (for steering skill)."""
        return self.tokenizer

    def is_local_model(self) -> bool:
        """Check if running a local model (vLLM or Transformers)."""
        return self.llm_type in ("vllm", "transformers")

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

    # === Model switching ===

    AVAILABLE_MODELS = {
        "anthropic": {
            "claude-sonnet-4-20250514": {"cost": "medium", "speed": "medium", "capability": "excellent"},
            "claude-3-5-sonnet-20241022": {"cost": "medium", "speed": "medium", "capability": "excellent"},
            "claude-3-5-haiku-20241022": {"cost": "low", "speed": "fast", "capability": "good"},
        },
        "openai": {
            "gpt-4o": {"cost": "medium", "speed": "medium", "capability": "excellent"},
            "gpt-4o-mini": {"cost": "low", "speed": "fast", "capability": "good"},
        },
        "vertex": {
            "gemini-2.0-flash-001": {"cost": "low", "speed": "fast", "capability": "good"},
            "gemini-1.5-pro-002": {"cost": "medium", "speed": "medium", "capability": "excellent"},
        },
    }

    def get_available_models(self) -> dict:
        """Get list of available models the agent can switch to."""
        available = {}
        if HAS_ANTHROPIC and self._anthropic_api_key:
            available["anthropic"] = self.AVAILABLE_MODELS["anthropic"]
        if HAS_OPENAI and self._openai_api_key:
            available["openai"] = self.AVAILABLE_MODELS["openai"]
        if self.vertex_project and (HAS_VERTEX_CLAUDE or HAS_VERTEX_GEMINI):
            available["vertex"] = self.AVAILABLE_MODELS["vertex"]
        return available

    def get_current_model(self) -> dict:
        """Get info about currently active model."""
        return {
            "model": self.llm_model,
            "provider": self.llm_type,
            "finetuned": self._finetuned_model_id is not None,
            "finetuned_model_id": self._finetuned_model_id,
        }

    def switch_model(self, new_model: str) -> bool:
        """Switch to a different model at runtime."""
        old_model = self.llm_model
        print(f"[COGNITION] Switching model: {old_model} -> {new_model}")

        try:
            if new_model.startswith("claude") and HAS_ANTHROPIC and self._anthropic_api_key:
                self.llm = AsyncAnthropic(api_key=self._anthropic_api_key)
                self.llm_type = "anthropic"
                self.llm_model = new_model
                return True

            elif new_model.startswith("gpt") or new_model.startswith("ft:"):
                if HAS_OPENAI:
                    self.llm = openai.AsyncOpenAI(
                        api_key=self._openai_api_key or "not-needed",
                        base_url=self._openai_base_url
                    )
                    self.llm_type = "openai"
                    self.llm_model = new_model
                    return True

            elif new_model.startswith("gemini") and self.vertex_project and HAS_VERTEX_GEMINI:
                vertexai.init(project=self.vertex_project, location=self.vertex_location)
                self.llm = "gemini"
                self.llm_type = "vertex_gemini"
                self.llm_model = new_model
                return True

            print(f"[COGNITION] Model {new_model} not available")
            return False
        except Exception as e:
            print(f"[COGNITION] Failed to switch model: {e}")
            return False

    # === Fine-tuning ===

    def record_training_example(self, prompt: str, response: str, outcome: str = "success") -> None:
        """Record a training example for fine-tuning."""
        if not hasattr(self, '_training_examples'):
            self._training_examples = []
        self._training_examples.append({
            "prompt": prompt,
            "response": response,
            "outcome": outcome,
            "timestamp": datetime.now().isoformat(),
        })
        print(f"[COGNITION] Recorded training example ({len(self._training_examples)} total)")

    def get_training_examples(self, outcome_filter: Optional[str] = None) -> list:
        """Get collected training examples."""
        if not hasattr(self, '_training_examples'):
            self._training_examples = []
        if outcome_filter:
            return [e for e in self._training_examples if e.get("outcome") == outcome_filter]
        return self._training_examples

    def clear_training_examples(self) -> int:
        """Clear all training examples."""
        if not hasattr(self, '_training_examples'):
            self._training_examples = []
        count = len(self._training_examples)
        self._training_examples = []
        return count

    def export_training_data(self, format: str = "jsonl") -> str:
        """Export training data in JSONL format for fine-tuning."""
        import json
        lines = []
        for ex in self.get_training_examples("success"):
            lines.append(json.dumps({
                "messages": [
                    {"role": "system", "content": self.get_system_prompt()},
                    {"role": "user", "content": ex["prompt"]},
                    {"role": "assistant", "content": ex["response"]},
                ]
            }))
        return "\n".join(lines)

    async def start_finetune(self, suffix: Optional[str] = None) -> dict:
        """Start a fine-tuning job with OpenAI."""
        examples = self.get_training_examples("success")
        if len(examples) < 10:
            return {"error": f"Need at least 10 examples, have {len(examples)}"}

        if not HAS_OPENAI or not self._openai_api_key:
            return {"error": "OpenAI API required for fine-tuning"}

        try:
            import tempfile
            client = openai.OpenAI(api_key=self._openai_api_key)

            # Write training data to temp file
            training_data = self.export_training_data()
            with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
                f.write(training_data)
                temp_path = f.name

            # Upload file
            with open(temp_path, 'rb') as f:
                file_response = client.files.create(file=f, purpose="fine-tune")

            # Start fine-tuning
            job = client.fine_tuning.jobs.create(
                training_file=file_response.id,
                model="gpt-4o-mini-2024-07-18",
                suffix=suffix or self.agent_ticker.lower(),
            )

            return {
                "job_id": job.id,
                "status": job.status,
                "model": job.model,
                "training_file": file_response.id,
            }
        except Exception as e:
            return {"error": str(e)}

    async def check_finetune_status(self, job_id: str) -> dict:
        """Check status of a fine-tuning job."""
        if not HAS_OPENAI or not self._openai_api_key:
            return {"error": "OpenAI API required"}

        try:
            client = openai.OpenAI(api_key=self._openai_api_key)
            job = client.fine_tuning.jobs.retrieve(job_id)

            result = {
                "job_id": job.id,
                "status": job.status,
                "model": job.model,
            }

            if job.fine_tuned_model:
                result["fine_tuned_model"] = job.fine_tuned_model
                self._finetuned_model_id = job.fine_tuned_model

            return result
        except Exception as e:
            return {"error": str(e)}

    def use_finetuned_model(self) -> bool:
        """Switch to the fine-tuned model."""
        if not hasattr(self, '_finetuned_model_id') or not self._finetuned_model_id:
            return False
        return self.switch_model(self._finetuned_model_id)

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

        # Format recent actions with rich context
        recent_text = ""
        if state.recent_actions:
            action_lines = []
            for a in state.recent_actions[-5:]:
                tool = a.get('tool', 'unknown')
                result = a.get('result', {})
                status = result.get('status', 'unknown')
                cycle = a.get('cycle', '?')

                # Show key parameters (truncated)
                params = a.get('params', {})
                param_parts = []
                for k, v in list(params.items())[:3]:
                    v_str = str(v)
                    if len(v_str) > 60:
                        v_str = v_str[:57] + "..."
                    param_parts.append(f"{k}={v_str}")
                param_str = f"({', '.join(param_parts)})" if param_parts else ""

                # Show result message or data summary
                msg = result.get('message', '')
                if not msg and isinstance(result.get('data'), dict):
                    data = result['data']
                    # Summarize data keys
                    msg = f"keys: {list(data.keys())[:5]}"
                elif not msg and isinstance(result.get('data'), str):
                    msg = result['data'][:80]
                if msg and len(msg) > 100:
                    msg = msg[:97] + "..."

                line = f"- [C{cycle}] {tool}{param_str}: {status}"
                if msg:
                    line += f" - {msg}"
                action_lines.append(line)

            recent_text = "\nRecent actions:\n" + "\n".join(action_lines)

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
