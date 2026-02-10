"""
Humanizer Skill

Humanize AI-generated text using the AuthorMist model to make it
undetectable by AI content detectors.
"""

import os
from typing import Dict, Optional

import httpx

from singularity.skills.base import Skill, SkillResult, SkillManifest, SkillAction


def _a(n, d, p, cost=0.0, dur=30, prob=0.85):
    return SkillAction(name=n, description=d, parameters=p, estimated_cost=cost,
                       estimated_duration_seconds=dur, success_probability=prob)


def _p(n, t, r, d):
    return {n: {"type": t, "required": r, "description": d}}


MODEL_ID = "authormist/authormist-originality"


class HumanizerSkill(Skill):
    """
    Humanize AI-generated text using AuthorMist (Qwen2.5-3B fine-tuned with RL).

    Runs locally via transformers (GPU recommended) or via a custom HF Inference Endpoint.

    Optional credentials:
    - HF_INFERENCE_ENDPOINT: Custom HuggingFace Inference Endpoint URL (skips local model)
    - HF_TOKEN: Auth token for HF Inference Endpoint
    """

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="humanizer",
            name="AI Text Humanizer",
            version="1.0.0",
            category="content",
            description="Humanize AI-generated text using AuthorMist to evade AI detection",
            required_credentials=[],
            install_cost=0,
            actions=[
                _a("humanize", "Paraphrase text through AuthorMist to make it undetectable as AI", {
                    **_p("text", "string", True, "AI-generated text to humanize"),
                    **_p("temperature", "string", False, "Generation temperature 0.1-1.0 (default: 0.7)"),
                    **_p("max_tokens", "integer", False, "Max new tokens to generate (default: 1024)"),
                }, cost=0.0, dur=30, prob=0.90),
                _a("humanize_chunks", "Humanize long text by splitting into chunks and processing each", {
                    **_p("text", "string", True, "Long AI-generated text to humanize"),
                    **_p("chunk_size", "integer", False, "Words per chunk (default: 400)"),
                    **_p("temperature", "string", False, "Generation temperature 0.1-1.0 (default: 0.7)"),
                }, cost=0.0, dur=120, prob=0.85),
                _a("check_model", "Check if AuthorMist model is loaded and ready", {
                }, cost=0.0, dur=2, prob=0.99),
            ]
        )

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self.http = httpx.AsyncClient()
        self._model = None
        self._tokenizer = None
        self._device = None
        self._endpoint = (
            self.credentials.get("HF_INFERENCE_ENDPOINT")
            or os.environ.get("HF_INFERENCE_ENDPOINT")
        )
        self._hf_token = (
            self.credentials.get("HF_TOKEN")
            or os.environ.get("HF_TOKEN")
        )

    def _load_model(self):
        """Lazy-load AuthorMist model and tokenizer."""
        if self._model is not None:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self._device == "cuda" else torch.float32

        self._model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, torch_dtype=dtype
        ).to(self._device)
        self._model.eval()

    async def execute(self, action: str, params: Dict) -> SkillResult:
        from . import actions

        try:
            dispatch = {
                "humanize": lambda: actions.humanize(self, params),
                "humanize_chunks": lambda: actions.humanize_chunks(self, params),
                "check_model": lambda: actions.check_model(self, params),
            }
            handler = dispatch.get(action)
            if not handler:
                return SkillResult(success=False, message=f"Unknown action: {action}")
            return await handler()
        except Exception as e:
            return SkillResult(success=False, message=f"humanizer error: {str(e)}")

    async def close(self):
        await self.http.aclose()
