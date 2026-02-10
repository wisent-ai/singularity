"""
Create Benchmark Skill

Orchestrates the full pipeline: design a benchmark, generate examples,
validate quality, publish to HuggingFace Hub, generate papers, submit to arXiv.
"""

import json
import os
from pathlib import Path
from typing import Dict

import httpx

from singularity.skills.base import Skill, SkillResult, SkillManifest, SkillAction

try:
    from anthropic import AsyncAnthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False


def _a(n, d, p, cost=0.0, dur=30, prob=0.85):
    return SkillAction(name=n, description=d, parameters=p, estimated_cost=cost,
                       estimated_duration_seconds=dur, success_probability=prob)


def _p(n, t, r, d):
    return {n: {"type": t, "required": r, "description": d}}


SYSTEM_PROMPT = (
    "You are an expert AI researcher specializing in evaluation benchmark design. "
    "You create rigorous, well-structured evaluation datasets with clear metrics, "
    "balanced categories, and high-quality examples. Always output valid JSON when asked."
)


class CreateBenchmarkSkill(Skill):
    """
    Create evaluation benchmarks: design → generate examples → validate →
    publish to HuggingFace → generate paper → submit to arXiv.

    Required credentials:
    - HF_TOKEN: HuggingFace Hub API token (for publishing datasets)

    Optional:
    - ARXIV_USERNAME + ARXIV_PASSWORD: For arXiv SWORD submission
    - ANTHROPIC_API_KEY: If LLM not injected via set_llm()
    """

    HF_API = "https://huggingface.co/api"
    ARXIV_SWORD = "https://arxiv.org/sword-app"

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="create_benchmark",
            name="Create Benchmark",
            version="1.0.0",
            category="research",
            description="Design evaluation benchmarks, generate examples, validate, publish to HuggingFace, generate papers, submit to arXiv",
            required_credentials=[],
            install_cost=0,
            actions=[
                _a("design_benchmark", "Design a benchmark spec: task type, metrics, example schema, categories", {
                    **_p("topic", "string", True, "Topic or domain for the benchmark"),
                    **_p("num_examples", "integer", False, "Target number of examples (default: 100)"),
                    **_p("difficulty", "string", False, "Difficulty distribution: uniform, easy_heavy, hard_heavy (default: uniform)")
                }, cost=0.05, dur=30),
                _a("generate_examples", "Generate evaluation examples in batches following the benchmark spec", {
                    **_p("project_name", "string", True, "Benchmark project name"),
                    **_p("batch_size", "integer", False, "Examples per batch (default: 10)"),
                    **_p("split", "string", False, "Dataset split: train, test, val (default: test)")
                }, cost=0.08, dur=45),
                _a("validate_benchmark", "Validate benchmark quality: duplicates, balance, format, coverage", {
                    **_p("project_name", "string", True, "Benchmark project name")
                }, cost=0.0, dur=5, prob=0.95),
                _a("publish_to_huggingface", "Create HF dataset repo, upload JSONL splits + dataset card", {
                    **_p("project_name", "string", True, "Benchmark project name"),
                    **_p("hf_repo_name", "string", False, "HuggingFace repo name (default: project_name)"),
                    **_p("private", "boolean", False, "Private dataset? (default: false)")
                }, cost=0.0, dur=20),
                _a("generate_paper", "Generate a LaTeX academic paper describing the benchmark", {
                    **_p("project_name", "string", True, "Benchmark project name"),
                    **_p("authors", "string", False, "Comma-separated author names"),
                    **_p("abstract_focus", "string", False, "Key points to emphasize in the abstract")
                }, cost=0.10, dur=60),
                _a("submit_to_arxiv", "Package LaTeX and submit via arXiv SWORD API", {
                    **_p("project_name", "string", True, "Benchmark project name"),
                    **_p("categories", "string", False, "arXiv categories, e.g. cs.CL cs.AI (default: cs.CL)"),
                    **_p("comments", "string", False, "Submission comments")
                }, cost=0.0, dur=30),
                _a("get_status", "Get pipeline progress and next step suggestion", {
                    **_p("project_name", "string", True, "Benchmark project name")
                }, cost=0.0, dur=2, prob=0.99),
            ]
        )

    def __init__(self, credentials: Dict[str, str] = None, llm=None, llm_type: str = None, model: str = None):
        super().__init__(credentials)
        self.http = httpx.AsyncClient()
        self._workspace = Path(os.environ.get("AGENT_WORKSPACE", "/tmp/agent_workspace")) / "create_benchmark"

        if llm:
            self.llm = llm
            self.llm_type = llm_type or "anthropic"
            self.model = model or "claude-sonnet-4-20250514"
        else:
            self._init_llm()

    def _init_llm(self):
        api_key = self.credentials.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
        if api_key and HAS_ANTHROPIC:
            self.llm = AsyncAnthropic(api_key=api_key)
            self.llm_type = "anthropic"
            self.model = "claude-sonnet-4-20250514"
        else:
            self.llm = None
            self.llm_type = "none"

    def set_llm(self, llm, llm_type: str, model: str):
        self.llm = llm
        self.llm_type = llm_type
        self.model = model

    async def _generate(self, prompt: str, system: str = None, max_tokens: int = 8000) -> str:
        if self.llm_type == "anthropic":
            response = await self.llm.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system or SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        raise RuntimeError("No LLM available — inject one via set_llm() or provide ANTHROPIC_API_KEY")

    # ── State management ──────────────────────────────────────────────

    def _state_path(self, project_name: str) -> Path:
        return self._workspace / project_name / "state.json"

    def _load_state(self, project_name: str) -> Dict:
        path = self._state_path(project_name)
        if path.exists():
            return json.loads(path.read_text())
        return {
            "project_name": project_name,
            "spec": None,
            "examples": {"train": [], "test": [], "val": []},
            "validation": None,
            "huggingface": None,
            "paper_latex": None,
            "arxiv": None,
        }

    def _save_state(self, project_name: str, state: Dict):
        path = self._state_path(project_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=2))
        tmp.rename(path)

    # ── Execute dispatch ──────────────────────────────────────────────

    async def execute(self, action: str, params: Dict) -> SkillResult:
        from . import benchmark, huggingface, arxiv

        try:
            dispatch = {
                "design_benchmark": lambda: benchmark.design_benchmark(self, params),
                "generate_examples": lambda: benchmark.generate_examples(self, params),
                "validate_benchmark": lambda: benchmark.validate_benchmark(self, params),
                "publish_to_huggingface": lambda: huggingface.publish_to_huggingface(self, params),
                "generate_paper": lambda: arxiv.generate_paper(self, params),
                "submit_to_arxiv": lambda: arxiv.submit_to_arxiv(self, params),
                "get_status": lambda: arxiv.get_status(self, params),
            }
            handler = dispatch.get(action)
            if not handler:
                return SkillResult(success=False, message=f"Unknown action: {action}")
            return await handler()
        except Exception as e:
            return SkillResult(success=False, message=f"create_benchmark error: {str(e)}")

    async def close(self):
        await self.http.aclose()
