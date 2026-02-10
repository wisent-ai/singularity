"""
Benchmark design, example generation, and validation.
"""

import hashlib
import json
from collections import Counter
from typing import Dict, List

from singularity.skills.base import SkillResult


def hash_example(example: Dict) -> str:
    """Content hash for duplicate detection."""
    return hashlib.sha256(json.dumps(example, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]


def parse_json(text: str):
    """Extract and parse JSON from LLM response, handling markdown fences."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for sc, ec in [("{", "}"), ("[", "]")]:
        start = text.find(sc)
        if start == -1:
            continue
        depth = 0
        for i in range(start, len(text)):
            if text[i] == sc:
                depth += 1
            elif text[i] == ec:
                depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    break
    return None


async def design_benchmark(skill, params: Dict) -> SkillResult:
    """LLM designs a benchmark spec: task type, metrics, schema, categories."""
    topic = params.get("topic")
    if not topic:
        return SkillResult(success=False, message="Missing required parameter: topic")

    num_examples = params.get("num_examples", 100)
    difficulty = params.get("difficulty", "uniform")

    prompt = f"""Design an evaluation benchmark for the following topic:

Topic: {topic}
Target examples: {num_examples}
Difficulty distribution: {difficulty}

Return a JSON object with exactly these fields:
{{
  "name": "short_snake_case_name",
  "description": "One paragraph describing what this benchmark evaluates",
  "task_type": "multiple_choice | open_ended | classification | ranking | generation | extraction",
  "metrics": ["list of evaluation metrics, e.g. accuracy, f1, bleu, human_eval"],
  "example_schema": {{
    "field_name": "description of this field and its type"
  }},
  "categories": ["list of 4-8 topic categories for balanced coverage"],
  "splits": {{"train": 0.0, "test": 1.0, "val": 0.0}},
  "difficulty_levels": ["easy", "medium", "hard"],
  "num_examples": {num_examples}
}}

The example_schema should define all fields each example must have.
For multiple_choice include: question, choices, correct_answer, explanation, difficulty, category.
For open_ended include: prompt, reference_answer, evaluation_criteria, difficulty, category.
Adapt the schema to best fit the topic.

Return ONLY valid JSON, no markdown fences or explanation."""

    raw = await skill._generate(prompt)
    spec = parse_json(raw)
    if not spec:
        return SkillResult(success=False, message="Failed to parse benchmark spec",
                           data={"raw_response": raw[:500]})

    required = ["name", "description", "task_type", "metrics", "example_schema", "categories"]
    missing = [f for f in required if f not in spec]
    if missing:
        return SkillResult(success=False, message=f"Spec missing fields: {missing}")

    if "splits" not in spec:
        spec["splits"] = {"train": 0.0, "test": 1.0, "val": 0.0}
    spec["num_examples"] = num_examples

    project_name = spec["name"]
    state = skill._load_state(project_name)
    state["spec"] = spec
    skill._save_state(project_name, state)

    return SkillResult(
        success=True,
        message=f"Designed benchmark '{spec['name']}': {spec['task_type']} with {len(spec['categories'])} categories",
        data={"project_name": project_name, "spec": spec}
    )


async def generate_examples(skill, params: Dict) -> SkillResult:
    """LLM generates evaluation examples in batches following the spec."""
    project_name = params.get("project_name")
    if not project_name:
        return SkillResult(success=False, message="Missing required parameter: project_name")

    state = skill._load_state(project_name)
    spec = state.get("spec")
    if not spec:
        return SkillResult(success=False, message=f"No spec for '{project_name}'. Run design_benchmark first.")

    batch_size = min(params.get("batch_size", 10), 25)
    split = params.get("split", "test")
    if split not in ("train", "test", "val"):
        return SkillResult(success=False, message=f"Invalid split '{split}'.")

    existing = state["examples"].get(split, [])
    existing_hashes = {hash_example(e) for e in existing}

    # Focus on least-covered categories
    cat_counts = Counter(e.get("category", "") for e in existing)
    categories = spec.get("categories", ["general"])
    focus_cats = sorted(categories, key=lambda c: cat_counts.get(c, 0))[:max(3, len(categories) // 2)]

    prompt = f"""Generate exactly {batch_size} evaluation examples for this benchmark:

Benchmark: {spec['name']}
Task type: {spec['task_type']}
Description: {spec['description']}

Each example MUST follow this schema:
{json.dumps(spec['example_schema'], indent=2)}

Focus categories: {json.dumps(focus_cats)}
Difficulty levels: {json.dumps(spec.get('difficulty_levels', ['easy', 'medium', 'hard']))}
Existing count: {len(existing)}, Target: {spec.get('num_examples', 100)}

Requirements:
- Unique, non-trivial examples
- Cover the focus categories
- Distribute difficulty levels evenly
- For multiple choice, ensure plausible distractors

Return ONLY a JSON array of {batch_size} objects, no markdown fences."""

    raw = await skill._generate(prompt, max_tokens=8000)
    examples = parse_json(raw)
    if not examples or not isinstance(examples, list):
        return SkillResult(success=False, message="Failed to parse examples",
                           data={"raw_response": raw[:500]})

    new_examples, duplicates = [], 0
    for ex in examples:
        h = hash_example(ex)
        if h not in existing_hashes:
            existing_hashes.add(h)
            new_examples.append(ex)
        else:
            duplicates += 1

    existing.extend(new_examples)
    state["examples"][split] = existing
    state["validation"] = None
    skill._save_state(project_name, state)

    total = sum(len(state["examples"].get(s, [])) for s in ("train", "test", "val"))
    return SkillResult(
        success=True,
        message=f"Generated {len(new_examples)} for '{split}' ({duplicates} dupes). Split: {len(existing)}. Total: {total}/{spec.get('num_examples', 100)}",
        data={"project_name": project_name, "split": split, "new_count": len(new_examples),
              "duplicates": duplicates, "split_total": len(existing), "overall_total": total}
    )


async def validate_benchmark(skill, params: Dict) -> SkillResult:
    """Check quality: schema conformance, duplicates, category balance, split sizes."""
    project_name = params.get("project_name")
    if not project_name:
        return SkillResult(success=False, message="Missing required parameter: project_name")

    state = skill._load_state(project_name)
    spec = state.get("spec")
    if not spec:
        return SkillResult(success=False, message=f"No spec for '{project_name}'.")

    schema_fields = set(spec.get("example_schema", {}).keys())
    categories = set(spec.get("categories", []))
    target = spec.get("num_examples", 100)

    errors: List[str] = []
    warnings: List[str] = []
    stats: Dict = {}
    total_examples, all_hashes = 0, set()

    for split in ("train", "test", "val"):
        examples = state["examples"].get(split, [])
        total_examples += len(examples)
        split_stats = _validate_split(examples, schema_fields, categories, split, all_hashes, errors, warnings)
        stats[split] = split_stats

    if total_examples == 0:
        errors.append("No examples generated yet")
    elif total_examples < target:
        warnings.append(f"Only {total_examples}/{target} examples generated")

    passed = len(errors) == 0
    validation = {"passed": passed, "errors": errors, "warnings": warnings,
                  "stats": stats, "total_examples": total_examples, "target": target}
    state["validation"] = validation
    skill._save_state(project_name, state)

    status = "PASSED" if passed else "FAILED"
    return SkillResult(
        success=True,
        message=f"Validation {status}: {total_examples}/{target} examples, {len(errors)} errors, {len(warnings)} warnings",
        data={"project_name": project_name, "validation": validation}
    )


def _validate_split(examples, schema_fields, categories, split, all_hashes, errors, warnings):
    """Validate a single split, mutating errors/warnings/all_hashes."""
    split_hashes, dup_count = set(), 0
    schema_violations = 0

    for i, ex in enumerate(examples):
        h = hash_example(ex)
        if h in split_hashes:
            dup_count += 1
        split_hashes.add(h)

        if not isinstance(ex, dict):
            errors.append(f"{split}[{i}]: not a dict")
            schema_violations += 1
        else:
            missing = schema_fields - set(ex.keys())
            if missing:
                schema_violations += 1
                if schema_violations <= 3:
                    warnings.append(f"{split}[{i}]: missing fields {missing}")

    cross_dups = len(split_hashes & all_hashes)
    all_hashes |= split_hashes

    if dup_count:
        warnings.append(f"{split}: {dup_count} in-split duplicates")
    if cross_dups:
        warnings.append(f"{split}: {cross_dups} cross-split duplicates")
    if schema_violations > 3:
        warnings.append(f"{split}: {schema_violations} total schema violations")

    cat_counts = Counter(ex.get("category", "unknown") for ex in examples if isinstance(ex, dict))
    diff_counts = Counter(ex.get("difficulty", "unknown") for ex in examples if isinstance(ex, dict))

    if len(cat_counts) > 1:
        counts = list(cat_counts.values())
        balance = max(counts) / max(min(counts), 1)
        if balance > 3:
            warnings.append(f"{split}: category imbalance {balance:.1f}x")
    else:
        balance = 1.0

    return {"count": len(examples), "category_distribution": dict(cat_counts),
            "difficulty_distribution": dict(diff_counts), "balance_ratio": round(balance, 2)}
