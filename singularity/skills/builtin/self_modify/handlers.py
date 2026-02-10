"""Handler functions for SelfModifySkill actions."""

from singularity.skills.base import SkillResult


# === Prompt methods ===

def get_prompt(skill) -> SkillResult:
    prompt = skill._get_prompt_fn()
    return SkillResult(success=True, message="Current system prompt retrieved",
                       data={"prompt": prompt, "length": len(prompt)})


def set_prompt(skill, new_prompt: str) -> SkillResult:
    if not new_prompt.strip():
        return SkillResult(success=False, message="Cannot set empty prompt")
    skill._set_prompt_fn(new_prompt)
    return SkillResult(success=True, message=f"System prompt replaced ({len(new_prompt)} chars)",
                       data={"length": len(new_prompt)})


def append_prompt(skill, addition: str) -> SkillResult:
    if not addition.strip():
        return SkillResult(success=False, message="Nothing to append")
    skill._append_prompt_fn(addition)
    new_prompt = skill._get_prompt_fn()
    return SkillResult(success=True, message="Added to system prompt",
                       data={"added": addition, "new_length": len(new_prompt)})


def add_rule(skill, rule: str) -> SkillResult:
    if not rule.strip():
        return SkillResult(success=False, message="Rule cannot be empty")
    skill._append_prompt_fn(f"\n\n=== SELF-IMPOSED RULE ===\n- {rule.strip()}")
    return SkillResult(success=True, message=f"Rule added: {rule[:50]}...", data={"rule": rule})


def add_goal(skill, goal: str) -> SkillResult:
    if not goal.strip():
        return SkillResult(success=False, message="Goal cannot be empty")
    skill._append_prompt_fn(f"\n\n=== PERSONAL GOAL ===\n- {goal.strip()}")
    return SkillResult(success=True, message=f"Goal added: {goal[:50]}...", data={"goal": goal})


def add_learning(skill, learning: str) -> SkillResult:
    if not learning.strip():
        return SkillResult(success=False, message="Learning cannot be empty")
    skill._append_prompt_fn(f"\n\n=== LEARNED ===\n- {learning.strip()}")
    return SkillResult(success=True, message=f"Learning recorded: {learning[:50]}...",
                       data={"learning": learning})


# === Model methods ===

def list_models(skill) -> SkillResult:
    if not skill._get_available_models_fn:
        return SkillResult(success=False, message="Model switching not available")
    models = skill._get_available_models_fn()
    return SkillResult(success=True, message=f"Found models from {len(models)} providers",
                       data={"models": models})


def current_model(skill) -> SkillResult:
    if not skill._get_current_model_fn:
        return SkillResult(success=False, message="Model info not available")
    info = skill._get_current_model_fn()
    return SkillResult(success=True, message=f"Current model: {info.get('model')}", data=info)


def switch_model(skill, model: str) -> SkillResult:
    if not model.strip():
        return SkillResult(success=False, message="Model name required")
    if not skill._switch_model_fn:
        return SkillResult(success=False, message="Model switching not available")
    success = skill._switch_model_fn(model)
    if success:
        return SkillResult(success=True, message=f"Switched to model: {model}", data={"model": model})
    return SkillResult(success=False, message=f"Failed to switch to model: {model}")


# === Fine-tuning methods ===

def record_experience(skill, prompt: str, response: str, outcome: str) -> SkillResult:
    if not prompt.strip() or not response.strip():
        return SkillResult(success=False, message="Prompt and response required")
    if not skill._record_example_fn:
        return SkillResult(success=False, message="Fine-tuning not available")
    skill._record_example_fn(prompt, response, outcome)
    return SkillResult(success=True, message=f"Recorded training example (outcome: {outcome})",
                       data={"outcome": outcome})


def training_stats(skill) -> SkillResult:
    if not skill._get_examples_fn:
        return SkillResult(success=False, message="Fine-tuning not available")
    all_examples = skill._get_examples_fn(None)
    success_count = len([e for e in all_examples if e.get("outcome") == "success"])
    failure_count = len([e for e in all_examples if e.get("outcome") == "failure"])
    neutral_count = len([e for e in all_examples if e.get("outcome") == "neutral"])
    return SkillResult(success=True, message=f"Training stats: {len(all_examples)} total examples",
        data={"total": len(all_examples), "success": success_count, "failure": failure_count,
              "neutral": neutral_count, "ready_for_finetune": success_count >= 10})


def clear_training(skill) -> SkillResult:
    if not skill._clear_examples_fn:
        return SkillResult(success=False, message="Fine-tuning not available")
    count = skill._clear_examples_fn()
    return SkillResult(success=True, message=f"Cleared {count} training examples", data={"cleared": count})


async def start_finetune(skill, suffix: str = None) -> SkillResult:
    if not skill._start_finetune_fn:
        return SkillResult(success=False, message="Fine-tuning not available")
    result = await skill._start_finetune_fn(suffix)
    if "error" in result:
        return SkillResult(success=False, message=f"Fine-tuning failed: {result['error']}")
    return SkillResult(success=True, message=f"Fine-tuning job started: {result.get('job_id')}",
                       data=result)


async def check_finetune(skill, job_id: str) -> SkillResult:
    if not job_id.strip():
        return SkillResult(success=False, message="Job ID required")
    if not skill._check_finetune_fn:
        return SkillResult(success=False, message="Fine-tuning not available")
    result = await skill._check_finetune_fn(job_id)
    if "error" in result:
        return SkillResult(success=False, message=f"Check failed: {result['error']}")
    return SkillResult(success=True, message=f"Job {job_id}: {result.get('status')}", data=result)


def use_finetuned(skill) -> SkillResult:
    if not skill._use_finetuned_fn:
        return SkillResult(success=False, message="Fine-tuning not available")
    success = skill._use_finetuned_fn()
    if success:
        return SkillResult(success=True, message="Switched to fine-tuned model")
    return SkillResult(success=False, message="No fine-tuned model available yet")
