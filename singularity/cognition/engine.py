"""
CognitionEngine - Multi-model, multi-turn LLM decision making with fine-tuning.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable

from .types import Action, TokenUsage, AgentState, Decision, calculate_api_cost
from .providers import (
    HAS_ANTHROPIC, HAS_VERTEX_CLAUDE, HAS_VERTEX_GEMINI,
    HAS_OPENAI, HAS_VLLM, HAS_TRANSFORMERS, DEVICE,
    generate_with_backend, generate_with_messages,
    AVAILABLE_MODELS, _lazy_import,
)
from .prompt_builder import build_prompt, parse_response, build_system_prompt, build_state_message


class CognitionEngine:
    """Multi-model LLM decision engine with model switching, fine-tuning, and multi-turn."""

    def __init__(
        self, llm_provider="auto", anthropic_api_key="", openai_api_key="",
        openai_base_url="http://localhost:8000/v1", vertex_project="",
        vertex_location="us-central1", llm_model="claude-sonnet-4-20250514",
        agent_name="Agent", agent_ticker="AGENT", agent_type="general",
        agent_specialty="", system_prompt="", system_prompt_file="",
        project_context_file="", cost_callback: Optional[Callable] = None,
    ):
        self.agent_name = agent_name
        self.agent_ticker = agent_ticker
        self.agent_type = agent_type
        self.agent_specialty = agent_specialty or agent_type or "general"
        self.llm_model = llm_model
        self._cost_callback = cost_callback
        self._anthropic_api_key = anthropic_api_key
        self._openai_api_key = openai_api_key
        self._openai_base_url = openai_base_url
        self.vertex_project = vertex_project or os.environ.get("VERTEX_PROJECT") or os.environ.get("GCP_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT")
        self.vertex_location = vertex_location or os.environ.get("VERTEX_LOCATION", "us-central1")
        self._training_examples, self._finetuned_model_id, self._prompt_additions = [], None, []
        self.llm, self.llm_type, self.tokenizer = None, "none", None
        self.system_prompt = system_prompt
        if system_prompt_file:
            p = Path(system_prompt_file)
            if p.exists(): self.system_prompt = p.read_text().strip()
        self.project_context = ""
        if project_context_file:
            p = Path(project_context_file)
            if p.exists(): self.project_context = p.read_text().strip()
        if llm_provider == "auto":
            llm_provider = self._auto_detect(anthropic_api_key)
        print(f"[COGNITION] Device: {DEVICE}, Provider: {llm_provider}, Model: {llm_model}")
        self._init_provider(llm_provider, anthropic_api_key, openai_api_key, openai_base_url)
        print(f"[COGNITION] Initialized with {self.llm_type} backend")

    def _auto_detect(self, api_key):
        if self.vertex_project and (HAS_VERTEX_CLAUDE or HAS_VERTEX_GEMINI): return "vertex"
        if DEVICE == "cuda" and HAS_VLLM: return "vllm"
        if DEVICE == "mps" and HAS_TRANSFORMERS: return "transformers"
        if HAS_ANTHROPIC and api_key: return "anthropic"
        if HAS_OPENAI: return "openai"
        return "none"

    def _init_provider(self, prov, ak, ok, base):
        if prov == "vertex":
            if self.llm_model.startswith("claude") and HAS_VERTEX_CLAUDE:
                self.llm = _lazy_import("vertex_claude")(project_id=self.vertex_project, region=self.vertex_location)
                self.llm_type = "vertex"
            elif HAS_VERTEX_GEMINI:
                _lazy_import("vertexai").init(project=self.vertex_project, location=self.vertex_location)
                self.llm, self.llm_type = "gemini", "vertex_gemini"
        elif prov == "vllm" and HAS_VLLM:
            LLM, SP = _lazy_import("vllm")
            self.llm = LLM(model=self.llm_model, trust_remote_code=True, max_model_len=8192, gpu_memory_utilization=0.90)
            self.sampling_params = SP(temperature=0.2, top_p=0.9, max_tokens=500)
            self.llm_type = "vllm"
        elif prov == "transformers" and HAS_TRANSFORMERS:
            import torch
            AM, AT = _lazy_import("transformers")
            self.tokenizer = AT.from_pretrained(self.llm_model, trust_remote_code=True)
            self.llm = AM.from_pretrained(self.llm_model, torch_dtype=torch.float16, device_map=DEVICE, trust_remote_code=True)
            self.llm_type = "transformers"
        elif prov == "anthropic" and HAS_ANTHROPIC:
            self.llm = _lazy_import("anthropic")(api_key=ak)
            self.llm_type = "anthropic"
        elif prov == "openai" and HAS_OPENAI:
            self.llm = _lazy_import("openai").AsyncOpenAI(api_key=ok or "not-needed", base_url=base)
            self.llm_type = "openai"

    # Model access (for steering skill)
    def get_model(self): return self.llm
    def get_tokenizer(self): return self.tokenizer
    def is_local_model(self) -> bool: return self.llm_type in ("vllm", "transformers")

    # Self-modification
    def get_system_prompt(self) -> str:
        base = self.system_prompt or ""
        return base + "\n".join(self._prompt_additions) if self._prompt_additions else base

    def set_system_prompt(self, new_prompt: str):
        self.system_prompt = new_prompt
        self._prompt_additions = []

    def append_to_prompt(self, addition: str):
        self._prompt_additions.append(addition)

    # Model switching
    def get_available_models(self) -> dict:
        available = {}
        if self.vertex_project and (HAS_VERTEX_CLAUDE or HAS_VERTEX_GEMINI):
            available["vertex"] = AVAILABLE_MODELS["vertex"]
        if HAS_ANTHROPIC and self._anthropic_api_key:
            available["anthropic"] = AVAILABLE_MODELS["anthropic"]
        if HAS_OPENAI:
            available["openai"] = AVAILABLE_MODELS["openai"]
        return available

    def get_current_model(self) -> dict:
        return {"model": self.llm_model, "provider": self.llm_type,
                "finetuned": self._finetuned_model_id is not None,
                "finetuned_model_id": self._finetuned_model_id}

    def switch_model(self, new_model: str) -> bool:
        old_m, old_t = self.llm_model, self.llm_type
        try:
            if new_model.startswith("gemini"):
                if not (self.vertex_project and HAS_VERTEX_GEMINI): return False
                _lazy_import("vertexai").init(project=self.vertex_project, location=self.vertex_location)
                self.llm, self.llm_type, self.llm_model = "gemini", "vertex_gemini", new_model
            elif new_model.startswith("claude") and "@" in new_model:
                if not (self.vertex_project and HAS_VERTEX_CLAUDE): return False
                self.llm = _lazy_import("vertex_claude")(project_id=self.vertex_project, region=self.vertex_location)
                self.llm_type, self.llm_model = "vertex", new_model
            elif new_model.startswith("claude"):
                if not (HAS_ANTHROPIC and self._anthropic_api_key): return False
                self.llm = _lazy_import("anthropic")(api_key=self._anthropic_api_key)
                self.llm_type, self.llm_model = "anthropic", new_model
            elif new_model.startswith("gpt") or new_model.startswith("ft:"):
                if not HAS_OPENAI: return False
                self.llm = _lazy_import("openai").AsyncOpenAI(api_key=self._openai_api_key or "not-needed", base_url=self._openai_base_url)
                self.llm_type, self.llm_model = "openai", new_model
            else: return False
            return True
        except Exception as e:
            print(f"[COGNITION] Failed to switch model: {e}")
            self.llm_model, self.llm_type = old_m, old_t
            return False

    # Fine-tuning
    def record_training_example(self, prompt, response, outcome="success"):
        self._training_examples.append({
            "messages": [{"role": "system", "content": self.get_system_prompt()[:1000]},
                         {"role": "user", "content": prompt}, {"role": "assistant", "content": response}],
            "outcome": outcome, "model": self.llm_model, "timestamp": datetime.now().isoformat()})

    def get_training_examples(self, outcome_filter=None):
        if outcome_filter: return [e for e in self._training_examples if e.get("outcome") == outcome_filter]
        return self._training_examples.copy()

    def clear_training_examples(self):
        c = len(self._training_examples); self._training_examples = []; return c

    def export_training_data(self, filepath=None):
        ok = [e for e in self._training_examples if e.get("outcome") == "success"]
        content = "\n".join(json.dumps({"messages": e["messages"]}) for e in ok)
        if filepath:
            with open(filepath, "w") as f: f.write(content)
            return filepath
        return content

    async def start_finetune(self, suffix=None):
        examples = self.get_training_examples("success")
        if len(examples) < 10: return {"error": f"Need >=10 examples, have {len(examples)}"}
        if not HAS_OPENAI or not self._openai_api_key: return {"error": "OpenAI API required"}
        try:
            import tempfile
            client = _lazy_import("openai").OpenAI(api_key=self._openai_api_key)
            data = self.export_training_data()
            with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
                f.write(data); tp = f.name
            with open(tp, 'rb') as f: fr = client.files.create(file=f, purpose="fine-tune")
            job = client.fine_tuning.jobs.create(training_file=fr.id, model="gpt-4o-mini-2024-07-18",
                                                  suffix=suffix or self.agent_ticker.lower())
            return {"job_id": job.id, "status": job.status, "model": job.model, "training_file": fr.id}
        except Exception as e: return {"error": str(e)}

    async def check_finetune_status(self, job_id):
        if not HAS_OPENAI or not self._openai_api_key: return {"error": "OpenAI API required"}
        try:
            client = _lazy_import("openai").OpenAI(api_key=self._openai_api_key)
            job = client.fine_tuning.jobs.retrieve(job_id)
            r = {"job_id": job.id, "status": job.status, "model": job.model}
            if job.fine_tuned_model: r["fine_tuned_model"] = job.fine_tuned_model; self._finetuned_model_id = job.fine_tuned_model
            return r
        except Exception as e: return {"error": str(e)}

    def use_finetuned_model(self):
        return self.switch_model(self._finetuned_model_id) if self._finetuned_model_id else False

    # Core decision loop
    def _finalize_decision(self, text, token_usage):
        api_cost = calculate_api_cost(self.llm_type, self.llm_model, token_usage)
        print(f"[COGNITION] Tokens: {token_usage.input_tokens} in, {token_usage.output_tokens} out | ${api_cost:.6f}")
        if self._cost_callback and token_usage.input_tokens > 0:
            self._cost_callback(model=self.llm_model, prompt_tokens=token_usage.input_tokens,
                                completion_tokens=token_usage.output_tokens)
        decision = parse_response(self, text)
        decision.token_usage = token_usage
        decision.api_cost_usd = api_cost
        return decision

    async def think(self, state: AgentState) -> Decision:
        """Single-prompt decision. No conversation context."""
        if not self.llm:
            return Decision(action=Action(tool="wait", reasoning="No LLM available"), reasoning="No LLM")
        try:
            text, usage = await generate_with_backend(self, build_prompt(self, state))
            return self._finalize_decision(text, usage)
        except Exception as e:
            print(f"[COGNITION] Error: {e}")
            return Decision(action=Action(tool="wait", reasoning=f"Error: {e}"), reasoning=str(e))

    async def think_with_context(self, state: AgentState, conversation=None) -> tuple:
        """Multi-turn decision with conversation context. Returns (Decision, messages)."""
        if not self.llm:
            return Decision(action=Action(tool="wait", reasoning="No LLM"), reasoning="No LLM"), conversation or []
        if conversation is None: conversation = []
        system = build_system_prompt(self)
        if not conversation:
            conversation = [{"role": "user", "content": build_state_message(self, state)}]
        try:
            text, usage = await generate_with_messages(self, conversation, system=system)
            decision = self._finalize_decision(text, usage)
            conversation.append({"role": "assistant", "content": text})
            return decision, conversation
        except Exception as e:
            print(f"[COGNITION] Error: {e}")
            return Decision(action=Action(tool="wait", reasoning=f"Error: {e}"), reasoning=str(e)), conversation
