#!/usr/bin/env python3
"""
PromptEvolutionSkill - Systematic prompt versioning, A/B testing, and evolution.

Provides the intelligence layer on top of SelfModifySkill for autonomous
prompt improvement. Tracks prompt versions, records outcomes per version,
compares performance, and recommends/rolls back prompt changes.

Pillar: Self-Improvement (act → measure → adapt cycle for prompts)
"""

import json
import hashlib
import os
import time
from typing import Dict, List, Optional, Any, Callable
from .base import Skill, SkillManifest, SkillAction, SkillResult


class PromptEvolutionSkill(Skill):
    """Systematic prompt evolution with version tracking and A/B testing."""

    def __init__(self, credentials: Dict = None):
        super().__init__(credentials)
        self._storage_dir = os.path.join(
            os.path.expanduser("~"), ".singularity", "prompt_evolution"
        )
        os.makedirs(self._storage_dir, exist_ok=True)
        self._versions_file = os.path.join(self._storage_dir, "versions.json")
        self._outcomes_file = os.path.join(self._storage_dir, "outcomes.json")
        self._config_file = os.path.join(self._storage_dir, "config.json")
        # Hooks for reading/writing the active prompt
        self._get_prompt_fn: Optional[Callable[[], str]] = None
        self._set_prompt_fn: Optional[Callable[[str], None]] = None

    def set_prompt_hooks(
        self,
        get_prompt: Callable[[], str],
        set_prompt: Callable[[str], None],
    ):
        """Connect to the agent's prompt read/write functions."""
        self._get_prompt_fn = get_prompt
        self._set_prompt_fn = set_prompt

    # --- Persistence helpers ---

    def _load_json(self, path: str, default: Any = None) -> Any:
        if default is None:
            default = []
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
        return default

    def _save_json(self, path: str, data: Any):
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def _load_versions(self) -> List[Dict]:
        return self._load_json(self._versions_file, [])

    def _save_versions(self, versions: List[Dict]):
        self._save_json(self._versions_file, versions)

    def _load_outcomes(self) -> List[Dict]:
        return self._load_json(self._outcomes_file, [])

    def _save_outcomes(self, outcomes: List[Dict]):
        self._save_json(self._outcomes_file, outcomes)

    def _load_config(self) -> Dict:
        return self._load_json(self._config_file, {
            "active_version": None,
            "auto_rollback_threshold": 0.3,
            "min_outcomes_for_comparison": 5,
        })

    def _save_config(self, config: Dict):
        self._save_json(self._config_file, config)

    @staticmethod
    def _prompt_hash(prompt: str) -> str:
        return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]

    # --- Manifest ---

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="prompt_evolution",
            name="Prompt Evolution",
            version="1.0.0",
            category="meta",
            description="Systematic prompt versioning, A/B testing, and evolution",
            actions=[
                SkillAction(
                    name="snapshot",
                    description="Save the current prompt as a named version",
                    parameters={
                        "label": {
                            "type": "string",
                            "required": True,
                            "description": "Short label for this version (e.g. 'baseline', 'concise-v2')",
                        },
                        "notes": {
                            "type": "string",
                            "required": False,
                            "description": "Why this version was created",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="record_outcome",
                    description="Record a success/failure outcome for the active prompt version",
                    parameters={
                        "outcome": {
                            "type": "string",
                            "required": True,
                            "description": "'success' or 'failure'",
                        },
                        "context": {
                            "type": "string",
                            "required": False,
                            "description": "What task was attempted",
                        },
                        "score": {
                            "type": "number",
                            "required": False,
                            "description": "Numeric quality score 0.0-1.0 (optional)",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="compare",
                    description="Compare performance of two prompt versions",
                    parameters={
                        "version_a": {
                            "type": "string",
                            "required": False,
                            "description": "Label or hash of first version (default: active)",
                        },
                        "version_b": {
                            "type": "string",
                            "required": False,
                            "description": "Label or hash of second version (default: previous)",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="rollback",
                    description="Revert to a previous prompt version",
                    parameters={
                        "version": {
                            "type": "string",
                            "required": True,
                            "description": "Label or hash of the version to restore",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="list_versions",
                    description="List all saved prompt versions with performance stats",
                    parameters={},
                    estimated_cost=0,
                ),
                SkillAction(
                    name="best_version",
                    description="Find the best-performing prompt version based on outcomes",
                    parameters={
                        "min_outcomes": {
                            "type": "integer",
                            "required": False,
                            "description": "Minimum outcomes required to rank (default: 3)",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="mutate",
                    description="Create a new prompt variant by modifying a section",
                    parameters={
                        "section": {
                            "type": "string",
                            "required": True,
                            "description": "The text section to replace in current prompt",
                        },
                        "replacement": {
                            "type": "string",
                            "required": True,
                            "description": "The new text to use instead",
                        },
                        "label": {
                            "type": "string",
                            "required": True,
                            "description": "Label for the new variant",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="active_version",
                    description="Show info about the currently active prompt version",
                    parameters={},
                    estimated_cost=0,
                ),
            ],
            required_credentials=[],
        )

    def check_credentials(self) -> bool:
        return True

    async def execute(self, action: str, params: Dict) -> SkillResult:
        handlers = {
            "snapshot": self._snapshot,
            "record_outcome": self._record_outcome,
            "compare": self._compare,
            "rollback": self._rollback,
            "list_versions": self._list_versions,
            "best_version": self._best_version,
            "mutate": self._mutate,
            "active_version": self._active_version,
        }
        handler = handlers.get(action)
        if not handler:
            return SkillResult(success=False, message=f"Unknown action: {action}")
        return handler(params)

    # --- Version lookup helper ---

    def _find_version(self, identifier: str, versions: List[Dict]) -> Optional[Dict]:
        """Find a version by label or hash prefix."""
        for v in versions:
            if v["label"] == identifier or v["hash"].startswith(identifier):
                return v
        return None

    # --- Actions ---

    def _snapshot(self, params: Dict) -> SkillResult:
        label = params.get("label", "").strip()
        if not label:
            return SkillResult(success=False, message="Label is required")
        notes = params.get("notes", "")

        # Get current prompt
        prompt = None
        if self._get_prompt_fn:
            prompt = self._get_prompt_fn()
        if not prompt:
            return SkillResult(
                success=False,
                message="No prompt available. Connect prompt hooks first.",
            )

        versions = self._load_versions()

        # Check for duplicate label
        if self._find_version(label, versions):
            return SkillResult(
                success=False,
                message=f"Version '{label}' already exists. Use a different label.",
            )

        prompt_hash = self._prompt_hash(prompt)
        version = {
            "label": label,
            "hash": prompt_hash,
            "prompt": prompt,
            "notes": notes,
            "created_at": time.time(),
            "length": len(prompt),
        }
        versions.append(version)
        self._save_versions(versions)

        # Set as active
        config = self._load_config()
        config["active_version"] = label
        self._save_config(config)

        return SkillResult(
            success=True,
            message=f"Saved prompt version '{label}' (hash: {prompt_hash})",
            data={
                "label": label,
                "hash": prompt_hash,
                "length": len(prompt),
                "total_versions": len(versions),
            },
        )

    def _record_outcome(self, params: Dict) -> SkillResult:
        outcome = params.get("outcome", "").strip().lower()
        if outcome not in ("success", "failure"):
            return SkillResult(
                success=False,
                message="Outcome must be 'success' or 'failure'",
            )

        config = self._load_config()
        active = config.get("active_version")

        # If no active version, try to identify from current prompt
        if not active:
            if self._get_prompt_fn:
                current_hash = self._prompt_hash(self._get_prompt_fn())
                versions = self._load_versions()
                for v in versions:
                    if v["hash"] == current_hash:
                        active = v["label"]
                        break
            if not active:
                active = "__untracked__"

        context = params.get("context", "")
        score = params.get("score")
        if score is not None:
            try:
                score = float(score)
                score = max(0.0, min(1.0, score))
            except (ValueError, TypeError):
                score = None

        outcomes = self._load_outcomes()
        record = {
            "version": active,
            "outcome": outcome,
            "context": context,
            "score": score,
            "timestamp": time.time(),
        }
        outcomes.append(record)
        self._save_outcomes(outcomes)

        return SkillResult(
            success=True,
            message=f"Recorded {outcome} for version '{active}'",
            data={"version": active, "outcome": outcome, "total_outcomes": len(outcomes)},
        )

    def _version_stats(self, version_label: str, outcomes: List[Dict]) -> Dict:
        """Compute performance stats for a version."""
        relevant = [o for o in outcomes if o["version"] == version_label]
        if not relevant:
            return {
                "total": 0,
                "successes": 0,
                "failures": 0,
                "success_rate": 0.0,
                "avg_score": None,
            }
        successes = sum(1 for o in relevant if o["outcome"] == "success")
        failures = sum(1 for o in relevant if o["outcome"] == "failure")
        scores = [o["score"] for o in relevant if o.get("score") is not None]
        avg_score = sum(scores) / len(scores) if scores else None
        return {
            "total": len(relevant),
            "successes": successes,
            "failures": failures,
            "success_rate": successes / len(relevant) if relevant else 0.0,
            "avg_score": round(avg_score, 3) if avg_score is not None else None,
        }

    def _compare(self, params: Dict) -> SkillResult:
        versions = self._load_versions()
        outcomes = self._load_outcomes()
        config = self._load_config()

        # Determine version A
        version_a_id = params.get("version_a", "").strip()
        if not version_a_id:
            version_a_id = config.get("active_version", "")
        if not version_a_id and versions:
            version_a_id = versions[-1]["label"]

        # Determine version B
        version_b_id = params.get("version_b", "").strip()
        if not version_b_id and len(versions) >= 2:
            # Find the version before A
            a_idx = None
            for i, v in enumerate(versions):
                if v["label"] == version_a_id:
                    a_idx = i
                    break
            if a_idx is not None and a_idx > 0:
                version_b_id = versions[a_idx - 1]["label"]
            else:
                version_b_id = versions[-2]["label"] if len(versions) >= 2 else ""

        if not version_a_id or not version_b_id:
            return SkillResult(
                success=False,
                message="Need at least 2 versions to compare",
            )

        stats_a = self._version_stats(version_a_id, outcomes)
        stats_b = self._version_stats(version_b_id, outcomes)

        # Determine winner
        winner = None
        if stats_a["total"] > 0 and stats_b["total"] > 0:
            # Prefer avg_score if both have it
            if stats_a["avg_score"] is not None and stats_b["avg_score"] is not None:
                if stats_a["avg_score"] > stats_b["avg_score"]:
                    winner = version_a_id
                elif stats_b["avg_score"] > stats_a["avg_score"]:
                    winner = version_b_id
                else:
                    winner = "tie"
            else:
                if stats_a["success_rate"] > stats_b["success_rate"]:
                    winner = version_a_id
                elif stats_b["success_rate"] > stats_a["success_rate"]:
                    winner = version_b_id
                else:
                    winner = "tie"

        return SkillResult(
            success=True,
            message=f"Comparison: {version_a_id} vs {version_b_id} → winner: {winner or 'insufficient data'}",
            data={
                "version_a": {"label": version_a_id, **stats_a},
                "version_b": {"label": version_b_id, **stats_b},
                "winner": winner,
            },
        )

    def _rollback(self, params: Dict) -> SkillResult:
        version_id = params.get("version", "").strip()
        if not version_id:
            return SkillResult(success=False, message="Version label or hash required")

        if not self._set_prompt_fn:
            return SkillResult(
                success=False,
                message="Prompt hooks not connected. Cannot rollback.",
            )

        versions = self._load_versions()
        target = self._find_version(version_id, versions)
        if not target:
            return SkillResult(
                success=False,
                message=f"Version '{version_id}' not found",
            )

        self._set_prompt_fn(target["prompt"])

        config = self._load_config()
        config["active_version"] = target["label"]
        self._save_config(config)

        return SkillResult(
            success=True,
            message=f"Rolled back to version '{target['label']}' (hash: {target['hash']})",
            data={"label": target["label"], "hash": target["hash"], "length": target["length"]},
        )

    def _list_versions(self, params: Dict) -> SkillResult:
        versions = self._load_versions()
        outcomes = self._load_outcomes()
        config = self._load_config()
        active = config.get("active_version")

        summary = []
        for v in versions:
            stats = self._version_stats(v["label"], outcomes)
            summary.append({
                "label": v["label"],
                "hash": v["hash"],
                "length": v["length"],
                "created_at": v["created_at"],
                "notes": v.get("notes", ""),
                "active": v["label"] == active,
                **stats,
            })

        return SkillResult(
            success=True,
            message=f"{len(versions)} prompt version(s) tracked",
            data={"versions": summary, "active_version": active},
        )

    def _best_version(self, params: Dict) -> SkillResult:
        min_outcomes = int(params.get("min_outcomes", 3))
        versions = self._load_versions()
        outcomes = self._load_outcomes()

        candidates = []
        for v in versions:
            stats = self._version_stats(v["label"], outcomes)
            if stats["total"] >= min_outcomes:
                # Rank by avg_score if available, else success_rate
                rank_score = stats["avg_score"] if stats["avg_score"] is not None else stats["success_rate"]
                candidates.append({
                    "label": v["label"],
                    "hash": v["hash"],
                    "rank_score": rank_score,
                    **stats,
                })

        if not candidates:
            return SkillResult(
                success=True,
                message=f"No versions have {min_outcomes}+ outcomes yet",
                data={"best": None, "candidates": []},
            )

        candidates.sort(key=lambda c: c["rank_score"], reverse=True)
        best = candidates[0]

        return SkillResult(
            success=True,
            message=f"Best version: '{best['label']}' (score: {best['rank_score']:.3f})",
            data={"best": best, "candidates": candidates},
        )

    def _mutate(self, params: Dict) -> SkillResult:
        section = params.get("section", "")
        replacement = params.get("replacement", "")
        label = params.get("label", "").strip()

        if not section or not replacement or not label:
            return SkillResult(
                success=False,
                message="section, replacement, and label are all required",
            )

        if not self._get_prompt_fn or not self._set_prompt_fn:
            return SkillResult(
                success=False,
                message="Prompt hooks not connected",
            )

        current_prompt = self._get_prompt_fn()
        if section not in current_prompt:
            return SkillResult(
                success=False,
                message="Section not found in current prompt",
                data={"section_preview": section[:100]},
            )

        # Check for duplicate label
        versions = self._load_versions()
        if self._find_version(label, versions):
            return SkillResult(
                success=False,
                message=f"Version '{label}' already exists",
            )

        # Snapshot current first (auto-label if needed)
        config = self._load_config()
        current_active = config.get("active_version")
        if current_active and not self._find_version(current_active, versions):
            # Current active version not saved - snapshot it
            pre_hash = self._prompt_hash(current_prompt)
            versions.append({
                "label": current_active,
                "hash": pre_hash,
                "prompt": current_prompt,
                "notes": "Auto-snapshot before mutation",
                "created_at": time.time(),
                "length": len(current_prompt),
            })

        # Apply mutation
        new_prompt = current_prompt.replace(section, replacement, 1)
        new_hash = self._prompt_hash(new_prompt)

        # Save new version
        versions.append({
            "label": label,
            "hash": new_hash,
            "prompt": new_prompt,
            "notes": f"Mutation of '{current_active or 'unknown'}': replaced section",
            "created_at": time.time(),
            "length": len(new_prompt),
        })
        self._save_versions(versions)

        # Apply the new prompt
        self._set_prompt_fn(new_prompt)
        config["active_version"] = label
        self._save_config(config)

        return SkillResult(
            success=True,
            message=f"Created mutation '{label}' and activated it",
            data={
                "label": label,
                "hash": new_hash,
                "length": len(new_prompt),
                "section_replaced": section[:80],
                "replacement_preview": replacement[:80],
            },
        )

    def _active_version(self, params: Dict) -> SkillResult:
        config = self._load_config()
        active = config.get("active_version")

        if not active:
            return SkillResult(
                success=True,
                message="No active version tracked. Use 'snapshot' to start tracking.",
                data={"active_version": None},
            )

        versions = self._load_versions()
        outcomes = self._load_outcomes()
        version = self._find_version(active, versions)
        stats = self._version_stats(active, outcomes)

        data = {
            "label": active,
            **stats,
        }
        if version:
            data["hash"] = version["hash"]
            data["length"] = version["length"]
            data["created_at"] = version["created_at"]

        return SkillResult(
            success=True,
            message=f"Active version: '{active}' (success rate: {stats['success_rate']:.1%})",
            data=data,
        )
