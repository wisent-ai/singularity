"""Self-Modification Skill - edit prompts, switch models, fine-tune."""

from typing import Dict, Callable, Optional, Any
from singularity.skills.base import Skill, SkillManifest, SkillAction, SkillResult
from . import handlers


def _a(n, d, p=None):
    return SkillAction(name=n, description=d, parameters=p or {}, estimated_cost=0)


class SelfModifySkill(Skill):
    """Skill for agent self-modification - prompts, models, and fine-tuning."""

    def __init__(self, credentials: Dict = None):
        super().__init__(credentials)
        self._get_prompt_fn: Optional[Callable[[], str]] = None
        self._set_prompt_fn: Optional[Callable[[str], None]] = None
        self._append_prompt_fn: Optional[Callable[[str], None]] = None
        self._get_available_models_fn: Optional[Callable[[], dict]] = None
        self._get_current_model_fn: Optional[Callable[[], dict]] = None
        self._switch_model_fn: Optional[Callable[[str], bool]] = None
        self._record_example_fn: Optional[Callable[[str, str, str], None]] = None
        self._get_examples_fn: Optional[Callable[[Optional[str]], list]] = None
        self._clear_examples_fn: Optional[Callable[[], int]] = None
        self._export_training_fn: Optional[Callable[[Optional[str]], str]] = None
        self._start_finetune_fn: Optional[Callable[[Optional[str]], Any]] = None
        self._check_finetune_fn: Optional[Callable[[str], Any]] = None
        self._use_finetuned_fn: Optional[Callable[[], bool]] = None

    def set_cognition_hooks(self, get_prompt, set_prompt, append_prompt,
                            get_available_models=None, get_current_model=None, switch_model=None,
                            record_example=None, get_examples=None, clear_examples=None,
                            export_training=None, start_finetune=None, check_finetune=None,
                            use_finetuned=None):
        self._get_prompt_fn = get_prompt
        self._set_prompt_fn = set_prompt
        self._append_prompt_fn = append_prompt
        self._get_available_models_fn = get_available_models
        self._get_current_model_fn = get_current_model
        self._switch_model_fn = switch_model
        self._record_example_fn = record_example
        self._get_examples_fn = get_examples
        self._clear_examples_fn = clear_examples
        self._export_training_fn = export_training
        self._start_finetune_fn = start_finetune
        self._check_finetune_fn = check_finetune
        self._use_finetuned_fn = use_finetuned

    @property
    def manifest(self) -> SkillManifest:
        _p = lambda n, d: {n: {"type": "string", "required": True, "description": d}}
        return SkillManifest(
            skill_id="self", name="Self-Modification", version="2.0.0", category="meta",
            description="Edit your prompt, switch models, and fine-tune yourself",
            actions=[
                _a("get_prompt", "View your current system prompt"),
                _a("set_prompt", "Replace your entire system prompt", _p("prompt", "The new system prompt")),
                _a("append_prompt", "Add instructions to your system prompt", _p("addition", "Text to append")),
                _a("add_rule", "Add a behavioral rule to follow", _p("rule", "A rule or guideline")),
                _a("add_goal", "Add a personal goal to pursue", _p("goal", "A goal to add")),
                _a("add_learning", "Record something you learned", _p("learning", "What you learned")),
                _a("list_models", "List all available models you can switch to"),
                _a("current_model", "Get info about your currently active model"),
                _a("switch_model", "Switch to a different LLM model",
                   _p("model", "Model ID (e.g., 'gemini-1.5-flash-002', 'gpt-4o-mini')")),
                _a("record_experience", "Record a prompt/response pair for fine-tuning", {
                    "prompt": {"type": "string", "required": True, "description": "The input prompt"},
                    "response": {"type": "string", "required": True, "description": "The desired response"},
                    "outcome": {"type": "string", "required": False, "description": "success/failure/neutral"}}),
                _a("training_stats", "Get statistics about collected training examples"),
                _a("clear_training", "Clear all collected training examples"),
                _a("start_finetune", "Start a fine-tuning job (requires 10+ examples)",
                   {"suffix": {"type": "string", "required": False, "description": "Custom suffix"}}),
                _a("check_finetune", "Check status of a fine-tuning job",
                   _p("job_id", "The fine-tuning job ID")),
                _a("use_finetuned", "Switch to your fine-tuned model"),
            ], required_credentials=[])

    def check_credentials(self) -> bool:
        return self._get_prompt_fn is not None

    async def execute(self, action: str, params: Dict) -> SkillResult:
        if not self._get_prompt_fn:
            return SkillResult(success=False, message="Self-modification not connected to cognition engine")
        dispatch = {
            "get_prompt": lambda: handlers.get_prompt(self),
            "set_prompt": lambda: handlers.set_prompt(self, params.get("prompt", "")),
            "append_prompt": lambda: handlers.append_prompt(self, params.get("addition", "")),
            "add_rule": lambda: handlers.add_rule(self, params.get("rule", "")),
            "add_goal": lambda: handlers.add_goal(self, params.get("goal", "")),
            "add_learning": lambda: handlers.add_learning(self, params.get("learning", "")),
            "list_models": lambda: handlers.list_models(self),
            "current_model": lambda: handlers.current_model(self),
            "switch_model": lambda: handlers.switch_model(self, params.get("model", "")),
            "record_experience": lambda: handlers.record_experience(self, params.get("prompt", ""),
                params.get("response", ""), params.get("outcome", "success")),
            "training_stats": lambda: handlers.training_stats(self),
            "clear_training": lambda: handlers.clear_training(self),
            "start_finetune": lambda: handlers.start_finetune(self, params.get("suffix")),
            "check_finetune": lambda: handlers.check_finetune(self, params.get("job_id", "")),
            "use_finetuned": lambda: handlers.use_finetuned(self),
        }
        handler = dispatch.get(action)
        if not handler:
            return SkillResult(success=False, message=f"Unknown action: {action}")
        result = handler()
        if hasattr(result, '__await__'):
            return await result
        return result
