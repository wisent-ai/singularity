#!/usr/bin/env python3
"""
Self-Modification Skill - Allows agents to edit their own system prompt,
switch models, and fine-tune themselves.

Gives agents the ability to evolve their own behavior by modifying
their core instructions, switching to different models, and learning
from experience through fine-tuning.
"""

from typing import Dict, Callable, Optional, Any
from .base import Skill, SkillManifest, SkillAction, SkillResult


class SelfModifySkill(Skill):
    """Skill for agent self-modification - prompts, models, and fine-tuning."""

    def __init__(self, credentials: Dict = None):
        super().__init__(credentials)
        # Prompt modification hooks
        self._get_prompt_fn: Optional[Callable[[], str]] = None
        self._set_prompt_fn: Optional[Callable[[str], None]] = None
        self._append_prompt_fn: Optional[Callable[[str], None]] = None
        # Model switching hooks
        self._get_available_models_fn: Optional[Callable[[], dict]] = None
        self._get_current_model_fn: Optional[Callable[[], dict]] = None
        self._switch_model_fn: Optional[Callable[[str], bool]] = None
        # Fine-tuning hooks
        self._record_example_fn: Optional[Callable[[str, str, str], None]] = None
        self._get_examples_fn: Optional[Callable[[Optional[str]], list]] = None
        self._clear_examples_fn: Optional[Callable[[], int]] = None
        self._export_training_fn: Optional[Callable[[Optional[str]], str]] = None
        self._start_finetune_fn: Optional[Callable[[Optional[str]], Any]] = None
        self._check_finetune_fn: Optional[Callable[[str], Any]] = None
        self._use_finetuned_fn: Optional[Callable[[], bool]] = None

    def set_cognition_hooks(
        self,
        get_prompt: Callable[[], str],
        set_prompt: Callable[[str], None],
        append_prompt: Callable[[str], None],
        # Model hooks
        get_available_models: Callable[[], dict] = None,
        get_current_model: Callable[[], dict] = None,
        switch_model: Callable[[str], bool] = None,
        # Fine-tuning hooks
        record_example: Callable[[str, str, str], None] = None,
        get_examples: Callable[[Optional[str]], list] = None,
        clear_examples: Callable[[], int] = None,
        export_training: Callable[[Optional[str]], str] = None,
        start_finetune: Callable[[Optional[str]], Any] = None,
        check_finetune: Callable[[str], Any] = None,
        use_finetuned: Callable[[], bool] = None,
    ):
        """Connect this skill to the agent's cognition engine."""
        # Prompt hooks
        self._get_prompt_fn = get_prompt
        self._set_prompt_fn = set_prompt
        self._append_prompt_fn = append_prompt
        # Model hooks
        self._get_available_models_fn = get_available_models
        self._get_current_model_fn = get_current_model
        self._switch_model_fn = switch_model
        # Fine-tuning hooks
        self._record_example_fn = record_example
        self._get_examples_fn = get_examples
        self._clear_examples_fn = clear_examples
        self._export_training_fn = export_training
        self._start_finetune_fn = start_finetune
        self._check_finetune_fn = check_finetune
        self._use_finetuned_fn = use_finetuned

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="self",
            name="Self-Modification",
            version="2.0.0",
            category="meta",
            description="Edit your prompt, switch models, and fine-tune yourself",
            actions=[
                # === Prompt modification ===
                SkillAction(
                    name="get_prompt",
                    description="View your current system prompt",
                    parameters={},
                    estimated_cost=0,
                ),
                SkillAction(
                    name="set_prompt",
                    description="Replace your entire system prompt with a new one",
                    parameters={
                        "prompt": {
                            "type": "string",
                            "required": True,
                            "description": "The new system prompt to use"
                        }
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="append_prompt",
                    description="Add instructions to your system prompt without replacing it",
                    parameters={
                        "addition": {
                            "type": "string",
                            "required": True,
                            "description": "Text to append to your system prompt"
                        }
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="add_rule",
                    description="Add a behavioral rule to follow",
                    parameters={
                        "rule": {
                            "type": "string",
                            "required": True,
                            "description": "A rule or guideline to add"
                        }
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="add_goal",
                    description="Add a personal goal to pursue",
                    parameters={
                        "goal": {
                            "type": "string",
                            "required": True,
                            "description": "A goal to add"
                        }
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="add_learning",
                    description="Record something you learned for future reference",
                    parameters={
                        "learning": {
                            "type": "string",
                            "required": True,
                            "description": "What you learned"
                        }
                    },
                    estimated_cost=0,
                ),
                # === Model switching ===
                SkillAction(
                    name="list_models",
                    description="List all available models you can switch to",
                    parameters={},
                    estimated_cost=0,
                ),
                SkillAction(
                    name="current_model",
                    description="Get info about your currently active model",
                    parameters={},
                    estimated_cost=0,
                ),
                SkillAction(
                    name="switch_model",
                    description="Switch to a different LLM model (e.g., faster/cheaper or smarter)",
                    parameters={
                        "model": {
                            "type": "string",
                            "required": True,
                            "description": "Model ID to switch to (e.g., 'gemini-1.5-flash-002', 'gpt-4o-mini')"
                        }
                    },
                    estimated_cost=0,
                ),
                # === Fine-tuning ===
                SkillAction(
                    name="record_experience",
                    description="Record a prompt/response pair for future fine-tuning",
                    parameters={
                        "prompt": {
                            "type": "string",
                            "required": True,
                            "description": "The input prompt"
                        },
                        "response": {
                            "type": "string",
                            "required": True,
                            "description": "The desired response"
                        },
                        "outcome": {
                            "type": "string",
                            "required": False,
                            "description": "Outcome: 'success', 'failure', or 'neutral' (default: success)"
                        }
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="training_stats",
                    description="Get statistics about collected training examples",
                    parameters={},
                    estimated_cost=0,
                ),
                SkillAction(
                    name="clear_training",
                    description="Clear all collected training examples",
                    parameters={},
                    estimated_cost=0,
                ),
                SkillAction(
                    name="start_finetune",
                    description="Start a fine-tuning job with collected examples (requires 10+ examples)",
                    parameters={
                        "suffix": {
                            "type": "string",
                            "required": False,
                            "description": "Custom suffix for the fine-tuned model name"
                        }
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="check_finetune",
                    description="Check status of a fine-tuning job",
                    parameters={
                        "job_id": {
                            "type": "string",
                            "required": True,
                            "description": "The fine-tuning job ID"
                        }
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="use_finetuned",
                    description="Switch to your fine-tuned model (after training completes)",
                    parameters={},
                    estimated_cost=0,
                ),
            ],
            required_credentials=[],
        )

    def check_credentials(self) -> bool:
        return self._get_prompt_fn is not None

    async def execute(self, action: str, params: Dict) -> SkillResult:
        if not self._get_prompt_fn:
            return SkillResult(
                success=False,
                message="Self-modification not connected to cognition engine"
            )

        # Prompt actions
        if action == "get_prompt":
            return self._get_prompt()
        elif action == "set_prompt":
            return self._set_prompt(params.get("prompt", ""))
        elif action == "append_prompt":
            return self._append_prompt(params.get("addition", ""))
        elif action == "add_rule":
            return self._add_rule(params.get("rule", ""))
        elif action == "add_goal":
            return self._add_goal(params.get("goal", ""))
        elif action == "add_learning":
            return self._add_learning(params.get("learning", ""))
        # Model actions
        elif action == "list_models":
            return self._list_models()
        elif action == "current_model":
            return self._current_model()
        elif action == "switch_model":
            return self._switch_model(params.get("model", ""))
        # Fine-tuning actions
        elif action == "record_experience":
            return self._record_experience(
                params.get("prompt", ""),
                params.get("response", ""),
                params.get("outcome", "success")
            )
        elif action == "training_stats":
            return self._training_stats()
        elif action == "clear_training":
            return self._clear_training()
        elif action == "start_finetune":
            return await self._start_finetune(params.get("suffix"))
        elif action == "check_finetune":
            return await self._check_finetune(params.get("job_id", ""))
        elif action == "use_finetuned":
            return self._use_finetuned()
        else:
            return SkillResult(success=False, message=f"Unknown action: {action}")

    # === Prompt methods ===

    def _get_prompt(self) -> SkillResult:
        prompt = self._get_prompt_fn()
        return SkillResult(
            success=True,
            message="Current system prompt retrieved",
            data={"prompt": prompt, "length": len(prompt)}
        )

    def _set_prompt(self, new_prompt: str) -> SkillResult:
        if not new_prompt.strip():
            return SkillResult(success=False, message="Cannot set empty prompt")
        self._set_prompt_fn(new_prompt)
        return SkillResult(
            success=True,
            message=f"System prompt replaced ({len(new_prompt)} chars)",
            data={"length": len(new_prompt)}
        )

    def _append_prompt(self, addition: str) -> SkillResult:
        if not addition.strip():
            return SkillResult(success=False, message="Nothing to append")
        self._append_prompt_fn(addition)
        new_prompt = self._get_prompt_fn()
        return SkillResult(
            success=True,
            message=f"Added to system prompt",
            data={"added": addition, "new_length": len(new_prompt)}
        )

    def _add_rule(self, rule: str) -> SkillResult:
        if not rule.strip():
            return SkillResult(success=False, message="Rule cannot be empty")
        addition = f"\n\n=== SELF-IMPOSED RULE ===\n- {rule.strip()}"
        self._append_prompt_fn(addition)
        return SkillResult(
            success=True,
            message=f"Rule added: {rule[:50]}...",
            data={"rule": rule}
        )

    def _add_goal(self, goal: str) -> SkillResult:
        if not goal.strip():
            return SkillResult(success=False, message="Goal cannot be empty")
        addition = f"\n\n=== PERSONAL GOAL ===\n- {goal.strip()}"
        self._append_prompt_fn(addition)
        return SkillResult(
            success=True,
            message=f"Goal added: {goal[:50]}...",
            data={"goal": goal}
        )

    def _add_learning(self, learning: str) -> SkillResult:
        if not learning.strip():
            return SkillResult(success=False, message="Learning cannot be empty")
        addition = f"\n\n=== LEARNED ===\n- {learning.strip()}"
        self._append_prompt_fn(addition)
        return SkillResult(
            success=True,
            message=f"Learning recorded: {learning[:50]}...",
            data={"learning": learning}
        )

    # === Model methods ===

    def _list_models(self) -> SkillResult:
        if not self._get_available_models_fn:
            return SkillResult(success=False, message="Model switching not available")
        models = self._get_available_models_fn()
        return SkillResult(
            success=True,
            message=f"Found models from {len(models)} providers",
            data={"models": models}
        )

    def _current_model(self) -> SkillResult:
        if not self._get_current_model_fn:
            return SkillResult(success=False, message="Model info not available")
        info = self._get_current_model_fn()
        return SkillResult(
            success=True,
            message=f"Current model: {info.get('model')}",
            data=info
        )

    def _switch_model(self, model: str) -> SkillResult:
        if not model.strip():
            return SkillResult(success=False, message="Model name required")
        if not self._switch_model_fn:
            return SkillResult(success=False, message="Model switching not available")

        success = self._switch_model_fn(model)
        if success:
            return SkillResult(
                success=True,
                message=f"Switched to model: {model}",
                data={"model": model}
            )
        else:
            return SkillResult(
                success=False,
                message=f"Failed to switch to model: {model}"
            )

    # === Fine-tuning methods ===

    def _record_experience(self, prompt: str, response: str, outcome: str) -> SkillResult:
        if not prompt.strip() or not response.strip():
            return SkillResult(success=False, message="Prompt and response required")
        if not self._record_example_fn:
            return SkillResult(success=False, message="Fine-tuning not available")

        self._record_example_fn(prompt, response, outcome)
        return SkillResult(
            success=True,
            message=f"Recorded training example (outcome: {outcome})",
            data={"outcome": outcome}
        )

    def _training_stats(self) -> SkillResult:
        if not self._get_examples_fn:
            return SkillResult(success=False, message="Fine-tuning not available")

        all_examples = self._get_examples_fn(None)
        success_count = len([e for e in all_examples if e.get("outcome") == "success"])
        failure_count = len([e for e in all_examples if e.get("outcome") == "failure"])
        neutral_count = len([e for e in all_examples if e.get("outcome") == "neutral"])

        return SkillResult(
            success=True,
            message=f"Training stats: {len(all_examples)} total examples",
            data={
                "total": len(all_examples),
                "success": success_count,
                "failure": failure_count,
                "neutral": neutral_count,
                "ready_for_finetune": success_count >= 10,
            }
        )

    def _clear_training(self) -> SkillResult:
        if not self._clear_examples_fn:
            return SkillResult(success=False, message="Fine-tuning not available")

        count = self._clear_examples_fn()
        return SkillResult(
            success=True,
            message=f"Cleared {count} training examples",
            data={"cleared": count}
        )

    async def _start_finetune(self, suffix: str = None) -> SkillResult:
        if not self._start_finetune_fn:
            return SkillResult(success=False, message="Fine-tuning not available")

        result = await self._start_finetune_fn(suffix)
        if "error" in result:
            return SkillResult(
                success=False,
                message=f"Fine-tuning failed: {result['error']}"
            )
        return SkillResult(
            success=True,
            message=f"Fine-tuning job started: {result.get('job_id')}",
            data=result
        )

    async def _check_finetune(self, job_id: str) -> SkillResult:
        if not job_id.strip():
            return SkillResult(success=False, message="Job ID required")
        if not self._check_finetune_fn:
            return SkillResult(success=False, message="Fine-tuning not available")

        result = await self._check_finetune_fn(job_id)
        if "error" in result:
            return SkillResult(
                success=False,
                message=f"Check failed: {result['error']}"
            )
        return SkillResult(
            success=True,
            message=f"Job {job_id}: {result.get('status')}",
            data=result
        )

    def _use_finetuned(self) -> SkillResult:
        if not self._use_finetuned_fn:
            return SkillResult(success=False, message="Fine-tuning not available")

        success = self._use_finetuned_fn()
        if success:
            return SkillResult(
                success=True,
                message="Switched to fine-tuned model"
            )
        else:
            return SkillResult(
                success=False,
                message="No fine-tuned model available yet"
            )
