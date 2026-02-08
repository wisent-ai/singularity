#!/usr/bin/env python3
"""
AutoPlaybookGenerationSkill - Automatically generate playbooks from clusters of similar reflections.

This skill bridges AgentReflection and ReflectionEventBridge to close the
self-improvement loop:

1. **Scan** reflections for clusters of similar tasks (by tag overlap and keyword similarity)
2. **Detect** when clusters are "ripe" for a playbook (enough reflections, sufficient success data)
3. **Generate** playbooks automatically by extracting patterns from clustered reflections:
   - Common task pattern from task descriptions
   - Steps from successful reflections' actions_taken (ordered by frequency)
   - Pitfalls from failed reflections' analysis and improvements
   - Prerequisites from common context across reflections
   - Expected outcome from successful outcomes
   - Tags as union of all reflection tags
4. **Wire** to EventBus for fully automatic playbook generation on a cadence

The self-improvement flywheel:
  reflections accumulate -> clusters form -> playbooks auto-generated -> future tasks use playbooks -> better outcomes

Pillar: Self-Improvement (automated knowledge distillation from experience)
"""

import hashlib
import json
import os
import re
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .base import Skill, SkillAction, SkillManifest, SkillResult

STATE_FILE = Path(__file__).parent.parent / "data" / "auto_playbook_generation.json"
MAX_CLUSTERS = 200
MAX_GENERATED_PLAYBOOKS = 500

# Words to ignore when computing keyword similarity in task descriptions
STOP_WORDS: Set[str] = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through", "during",
    "before", "after", "above", "below", "between", "out", "off", "over",
    "under", "again", "further", "then", "once", "here", "there", "when",
    "where", "why", "how", "all", "each", "every", "both", "few", "more",
    "most", "other", "some", "such", "no", "nor", "not", "only", "own",
    "same", "so", "than", "too", "very", "just", "because", "but", "and",
    "or", "if", "while", "about", "up", "its", "it", "this", "that",
    "these", "those", "i", "me", "my", "we", "our", "you", "your",
    "he", "him", "his", "she", "her", "they", "them", "their", "what",
    "which", "who", "whom", "auto", "action", "failed", "successful",
}


def _tokenize(text: str) -> Set[str]:
    """Extract significant lowercase word tokens from text, filtering stop words."""
    words = set(re.findall(r"[a-z][a-z0-9_]+", text.lower()))
    return words - STOP_WORDS


def _jaccard(set_a: Set[str], set_b: Set[str]) -> float:
    """Compute Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def _shared_words_ratio(set_a: Set[str], set_b: Set[str]) -> float:
    """Compute shared words ratio: |intersection| / min(|a|, |b|)."""
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    minimum = min(len(set_a), len(set_b))
    return intersection / minimum if minimum > 0 else 0.0


class AutoPlaybookGenerationSkill(Skill):
    """
    Automatically generate playbooks from clusters of similar reflections.

    Scans the reflection history for groups of related reflections (based on
    tag overlap and task keyword similarity), then synthesizes playbooks by
    extracting common patterns, steps, pitfalls, and expected outcomes from
    each cluster.

    Can be wired to the EventBus to run automatically whenever enough new
    reflections have accumulated.
    """

    def __init__(self, credentials: Dict[str, str] = None):
        """Initialize the skill and load persisted state."""
        super().__init__(credentials)
        self._load_state()

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def _load_state(self) -> None:
        """Load persisted state from disk, or initialize fresh state."""
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        if STATE_FILE.exists():
            try:
                with open(STATE_FILE) as f:
                    data = json.load(f)
                self._clusters: Dict[str, Dict] = data.get("clusters", {})
                self._generated_playbooks: List[Dict] = data.get("generated_playbooks", [])
                self._wire_state: Dict[str, Any] = data.get("wire_state", self._default_wire_state())
                self._config: Dict[str, Any] = data.get("config", self._default_config())
                self._stats: Dict[str, Any] = data.get("stats", self._default_stats())
            except (json.JSONDecodeError, Exception):
                self._init_empty()
        else:
            self._init_empty()

    def _init_empty(self) -> None:
        """Initialize all state to defaults."""
        self._clusters: Dict[str, Dict] = {}
        self._generated_playbooks: List[Dict] = []
        self._wire_state: Dict[str, Any] = self._default_wire_state()
        self._config: Dict[str, Any] = self._default_config()
        self._stats: Dict[str, Any] = self._default_stats()

    def _default_wire_state(self) -> Dict[str, Any]:
        """Return default wire state."""
        return {
            "active": False,
            "subscription_id": None,
            "reflections_since_scan": 0,
        }

    def _default_config(self) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            "min_cluster_size": 3,
            "min_success_rate": 0.3,
            "auto_generate_on_scan": True,
            "scan_every_n_reflections": 10,
            "max_playbooks_per_scan": 5,
        }

    def _default_stats(self) -> Dict[str, Any]:
        """Return default statistics."""
        return {
            "scans_performed": 0,
            "clusters_found": 0,
            "playbooks_generated": 0,
            "last_scan_at": None,
            "last_generate_at": None,
        }

    def _save_state(self) -> None:
        """Persist current state to disk."""
        # Enforce size limits
        cluster_items = list(self._clusters.items())
        if len(cluster_items) > MAX_CLUSTERS:
            self._clusters = dict(cluster_items[-MAX_CLUSTERS:])
        if len(self._generated_playbooks) > MAX_GENERATED_PLAYBOOKS:
            self._generated_playbooks = self._generated_playbooks[-MAX_GENERATED_PLAYBOOKS:]

        data = {
            "clusters": self._clusters,
            "generated_playbooks": self._generated_playbooks,
            "wire_state": self._wire_state,
            "config": self._config,
            "stats": self._stats,
        }
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(STATE_FILE, "w") as f:
            json.dump(data, f, indent=2, default=str)

    # ------------------------------------------------------------------
    # Manifest
    # ------------------------------------------------------------------

    @property
    def manifest(self) -> SkillManifest:
        """Return skill manifest with metadata and action definitions."""
        return SkillManifest(
            skill_id="auto_playbook_generation",
            name="Auto Playbook Generation",
            version="1.0.0",
            category="self_improvement",
            description=(
                "Automatically generate playbooks from clusters of similar reflections. "
                "Scans reflection history, clusters by tag and keyword similarity, "
                "and synthesizes reusable playbooks from patterns in the data."
            ),
            actions=self.get_actions(),
            required_credentials=[],
        )

    def get_actions(self) -> List[SkillAction]:
        """Return all available actions for this skill."""
        return [
            SkillAction(
                name="scan",
                description=(
                    "Scan reflections for playbook-worthy clusters. Groups reflections "
                    "by tag overlap and task keyword similarity, identifies clusters "
                    "that don't yet have a corresponding playbook."
                ),
                parameters={
                    "min_cluster_size": {
                        "type": "integer", "required": False,
                        "description": "Minimum reflections to form a cluster (default: 3)",
                    },
                    "lookback": {
                        "type": "integer", "required": False,
                        "description": "Number of recent reflections to analyze (default: 50)",
                    },
                    "filter_tag": {
                        "type": "string", "required": False,
                        "description": "Only include reflections with this tag",
                    },
                },
                estimated_cost=0.0,
            ),
            SkillAction(
                name="generate",
                description=(
                    "Generate a playbook from a specific reflection cluster. "
                    "Extracts common patterns, steps, pitfalls, and expected outcomes "
                    "from the cluster's reflections and creates a new playbook."
                ),
                parameters={
                    "cluster_id": {
                        "type": "string", "required": True,
                        "description": "Cluster ID from scan results to generate a playbook from",
                    },
                    "playbook_name": {
                        "type": "string", "required": False,
                        "description": "Custom name for the playbook (auto-generated if not provided)",
                    },
                    "tags": {
                        "type": "array", "required": False,
                        "description": "Additional tags for the generated playbook",
                    },
                },
                estimated_cost=0.0,
            ),
            SkillAction(
                name="auto_generate",
                description=(
                    "Scan reflections and automatically generate playbooks for all "
                    "ripe clusters that meet the minimum success rate threshold."
                ),
                parameters={
                    "min_cluster_size": {
                        "type": "integer", "required": False,
                        "description": "Minimum reflections to form a cluster (default: 3)",
                    },
                    "min_success_rate": {
                        "type": "float", "required": False,
                        "description": "Minimum success rate for a cluster to be eligible (default: 0.3)",
                    },
                    "max_generate": {
                        "type": "integer", "required": False,
                        "description": "Maximum number of playbooks to generate in one run (default: 5)",
                    },
                },
                estimated_cost=0.0,
            ),
            SkillAction(
                name="wire",
                description=(
                    "Wire to EventBus for automatic playbook generation. "
                    "Subscribes to reflection.created events and triggers "
                    "scan + auto_generate after every N reflections."
                ),
                parameters={
                    "scan_every_n_reflections": {
                        "type": "integer", "required": False,
                        "description": "Trigger scan after this many new reflections (default: 10)",
                    },
                    "auto_generate": {
                        "type": "boolean", "required": False,
                        "description": "Automatically generate playbooks on scan (default: True)",
                    },
                },
                estimated_cost=0.0,
            ),
            SkillAction(
                name="unwire",
                description="Remove EventBus subscription and stop automatic generation.",
                parameters={},
                estimated_cost=0.0,
            ),
            SkillAction(
                name="configure",
                description="Update configuration parameters without rewiring.",
                parameters={
                    "min_cluster_size": {
                        "type": "integer", "required": False,
                        "description": "Minimum reflections to form a cluster",
                    },
                    "min_success_rate": {
                        "type": "float", "required": False,
                        "description": "Minimum success rate for auto-generation eligibility",
                    },
                    "auto_generate_on_scan": {
                        "type": "boolean", "required": False,
                        "description": "Automatically generate playbooks when scanning",
                    },
                    "scan_every_n_reflections": {
                        "type": "integer", "required": False,
                        "description": "Number of reflections between automatic scans",
                    },
                    "max_playbooks_per_scan": {
                        "type": "integer", "required": False,
                        "description": "Maximum playbooks to generate per scan cycle",
                    },
                },
                estimated_cost=0.0,
            ),
            SkillAction(
                name="status",
                description="View current state: clusters found, playbooks generated, wire status, config, and stats.",
                parameters={},
                estimated_cost=0.0,
            ),
        ]

    # ------------------------------------------------------------------
    # Action dispatch
    # ------------------------------------------------------------------

    async def execute(self, action: str, params: Dict) -> SkillResult:
        """Dispatch to the appropriate action handler."""
        handlers = {
            "scan": self._scan,
            "generate": self._generate,
            "auto_generate": self._auto_generate,
            "wire": self._wire,
            "unwire": self._unwire,
            "configure": self._configure,
            "status": self._status,
        }
        handler = handlers.get(action)
        if not handler:
            return SkillResult(
                success=False,
                message=f"Unknown action: {action}. Available: {list(handlers.keys())}",
            )
        try:
            return await handler(params)
        except Exception as e:
            return SkillResult(
                success=False,
                message=f"Auto-playbook generation error in '{action}': {str(e)}",
            )

    # ------------------------------------------------------------------
    # Action: scan
    # ------------------------------------------------------------------

    async def _scan(self, params: Dict) -> SkillResult:
        """
        Scan reflections for playbook-worthy clusters.

        Loads reflections from AgentReflection, clusters them by tag overlap
        and keyword similarity, and identifies clusters that do not yet have
        a corresponding playbook.
        """
        try:
            min_cluster_size = max(2, int(params.get("min_cluster_size", self._config["min_cluster_size"])))
            lookback = max(1, int(params.get("lookback", 50)))
            filter_tag = params.get("filter_tag", "")

            # Fetch reflections from AgentReflection
            reflections = await self._fetch_reflections(lookback, filter_tag)
            if not reflections:
                return SkillResult(
                    success=True,
                    message="No reflections found to scan.",
                    data={"clusters": [], "playbook_candidates": []},
                )

            if len(reflections) < min_cluster_size:
                return SkillResult(
                    success=True,
                    message=f"Only {len(reflections)} reflections found, need at least {min_cluster_size} for a cluster.",
                    data={"clusters": [], "playbook_candidates": [], "reflections_found": len(reflections)},
                )

            # Build clusters
            clusters = self._cluster_reflections(reflections, min_cluster_size)

            # Fetch existing playbooks to check for overlap
            existing_playbooks = await self._fetch_playbooks()
            existing_playbook_tags = self._extract_playbook_tag_sets(existing_playbooks)

            # Annotate clusters with metadata
            cluster_summaries = []
            playbook_candidates = []

            for cluster in clusters:
                cluster_id = self._generate_cluster_id(cluster)
                common_tags = self._get_common_tags(cluster)
                representative_task = self._get_representative_task(cluster)
                success_rate = self._compute_success_rate(cluster)
                existing_playbook = self._find_overlapping_playbook(common_tags, existing_playbook_tags)

                cluster_data = {
                    "cluster_id": cluster_id,
                    "size": len(cluster),
                    "common_tags": common_tags,
                    "representative_task": representative_task,
                    "success_rate": round(success_rate, 3),
                    "existing_playbook": existing_playbook,
                    "reflection_ids": [r.get("id", "") for r in cluster],
                }

                # Store the full cluster in internal state for later generation
                self._clusters[cluster_id] = {
                    "cluster_id": cluster_id,
                    "reflections": cluster,
                    "common_tags": common_tags,
                    "representative_task": representative_task,
                    "success_rate": round(success_rate, 3),
                    "existing_playbook": existing_playbook,
                    "scanned_at": datetime.utcnow().isoformat(),
                }

                cluster_summaries.append(cluster_data)

                if not existing_playbook:
                    playbook_candidates.append(cluster_data)

            # Update stats
            self._stats["scans_performed"] += 1
            self._stats["clusters_found"] = len(self._clusters)
            self._stats["last_scan_at"] = datetime.utcnow().isoformat()
            self._save_state()

            return SkillResult(
                success=True,
                message=(
                    f"Scan complete: {len(cluster_summaries)} cluster(s) found, "
                    f"{len(playbook_candidates)} playbook candidate(s)"
                ),
                data={
                    "clusters": cluster_summaries,
                    "playbook_candidates": playbook_candidates,
                    "reflections_scanned": len(reflections),
                    "total_clusters_stored": len(self._clusters),
                },
            )
        except Exception as e:
            return SkillResult(
                success=False,
                message=f"Scan failed: {str(e)}",
            )

    # ------------------------------------------------------------------
    # Action: generate
    # ------------------------------------------------------------------

    async def _generate(self, params: Dict) -> SkillResult:
        """
        Generate a playbook from a specific reflection cluster.

        Extracts common task patterns, steps, pitfalls, prerequisites,
        and expected outcomes from the cluster's reflections, then creates
        the playbook via AgentReflection.
        """
        try:
            cluster_id = params.get("cluster_id", "")
            if not cluster_id:
                return SkillResult(
                    success=False,
                    message="Required: cluster_id. Run 'scan' first to identify clusters.",
                )

            cluster_data = self._clusters.get(cluster_id)
            if not cluster_data:
                available = list(self._clusters.keys())[:10]
                return SkillResult(
                    success=False,
                    message=f"Cluster '{cluster_id}' not found. Available: {available}",
                )

            reflections = cluster_data.get("reflections", [])
            if not reflections:
                return SkillResult(
                    success=False,
                    message=f"Cluster '{cluster_id}' has no reflections.",
                )

            # Extract playbook components from the cluster
            custom_name = params.get("playbook_name", "")
            extra_tags = params.get("tags", [])

            playbook_name = custom_name or self._auto_generate_name(reflections)
            task_pattern = self._extract_task_pattern(reflections)
            steps = self._extract_steps(reflections)
            pitfalls = self._extract_pitfalls(reflections)
            prerequisites = self._extract_prerequisites(reflections)
            expected_outcome = self._extract_expected_outcome(reflections)
            tags = self._extract_all_tags(reflections, extra_tags)

            # Create the playbook via AgentReflection
            playbook_result = await self._create_playbook_via_reflection(
                name=playbook_name,
                task_pattern=task_pattern,
                steps=steps,
                pitfalls=pitfalls,
                prerequisites=prerequisites,
                expected_outcome=expected_outcome,
                tags=tags,
            )

            if not playbook_result or not playbook_result.success:
                error_msg = playbook_result.message if playbook_result else "No response from agent_reflection"
                return SkillResult(
                    success=False,
                    message=f"Failed to create playbook: {error_msg}",
                    data={"cluster_id": cluster_id},
                )

            # Record the generated playbook
            generated_record = {
                "cluster_id": cluster_id,
                "playbook_name": playbook_name,
                "task_pattern": task_pattern,
                "steps": steps,
                "pitfalls": pitfalls,
                "prerequisites": prerequisites,
                "expected_outcome": expected_outcome,
                "tags": tags,
                "source_reflection_count": len(reflections),
                "success_rate": cluster_data.get("success_rate", 0),
                "generated_at": datetime.utcnow().isoformat(),
            }
            self._generated_playbooks.append(generated_record)

            # Mark cluster as having a playbook now
            cluster_data["existing_playbook"] = playbook_name

            # Update stats
            self._stats["playbooks_generated"] += 1
            self._stats["last_generate_at"] = datetime.utcnow().isoformat()
            self._save_state()

            return SkillResult(
                success=True,
                message=f"Playbook '{playbook_name}' generated from {len(reflections)} reflections",
                data={
                    "playbook": generated_record,
                    "cluster_id": cluster_id,
                    "creation_result": playbook_result.data if playbook_result else {},
                },
            )
        except Exception as e:
            return SkillResult(
                success=False,
                message=f"Generate failed: {str(e)}",
            )

    # ------------------------------------------------------------------
    # Action: auto_generate
    # ------------------------------------------------------------------

    async def _auto_generate(self, params: Dict) -> SkillResult:
        """
        Scan and automatically generate playbooks for all ripe clusters.

        Combines scan + generate into a single operation. Filters clusters
        by minimum success rate and generates up to max_generate playbooks.
        """
        try:
            min_cluster_size = max(2, int(params.get(
                "min_cluster_size", self._config["min_cluster_size"]
            )))
            min_success_rate = float(params.get(
                "min_success_rate", self._config["min_success_rate"]
            ))
            max_generate = max(1, int(params.get(
                "max_generate", self._config["max_playbooks_per_scan"]
            )))

            # Step 1: Scan
            scan_result = await self._scan({
                "min_cluster_size": min_cluster_size,
            })

            if not scan_result.success:
                return SkillResult(
                    success=False,
                    message=f"Auto-generate aborted: scan failed - {scan_result.message}",
                )

            candidates = scan_result.data.get("playbook_candidates", [])
            if not candidates:
                return SkillResult(
                    success=True,
                    message="No playbook candidates found. All clusters already have playbooks or none meet the threshold.",
                    data={
                        "scan_result": scan_result.data,
                        "generated": [],
                    },
                )

            # Step 2: Filter by success rate
            eligible = [
                c for c in candidates
                if c.get("success_rate", 0) >= min_success_rate
            ]

            if not eligible:
                return SkillResult(
                    success=True,
                    message=(
                        f"Found {len(candidates)} candidate(s) but none meet "
                        f"the minimum success rate of {min_success_rate:.0%}."
                    ),
                    data={
                        "scan_result": scan_result.data,
                        "candidates_below_threshold": len(candidates),
                        "generated": [],
                    },
                )

            # Step 3: Generate playbooks (up to max)
            generated = []
            errors = []
            for candidate in eligible[:max_generate]:
                gen_result = await self._generate({
                    "cluster_id": candidate["cluster_id"],
                })
                if gen_result.success:
                    generated.append(gen_result.data.get("playbook", {}))
                else:
                    errors.append({
                        "cluster_id": candidate["cluster_id"],
                        "error": gen_result.message,
                    })

            return SkillResult(
                success=True,
                message=(
                    f"Auto-generated {len(generated)} playbook(s) from "
                    f"{len(eligible)} eligible cluster(s)"
                    + (f" ({len(errors)} error(s))" if errors else "")
                ),
                data={
                    "generated": generated,
                    "errors": errors,
                    "eligible_count": len(eligible),
                    "candidate_count": len(candidates),
                },
            )
        except Exception as e:
            return SkillResult(
                success=False,
                message=f"Auto-generate failed: {str(e)}",
            )

    # ------------------------------------------------------------------
    # Action: wire
    # ------------------------------------------------------------------

    async def _wire(self, params: Dict) -> SkillResult:
        """
        Wire to EventBus for automatic playbook generation.

        Subscribes to reflection.created events. After every N new reflections,
        triggers a scan and (optionally) auto-generates playbooks.
        """
        try:
            scan_every = max(1, int(params.get(
                "scan_every_n_reflections",
                self._config["scan_every_n_reflections"],
            )))
            auto_gen = params.get("auto_generate", self._config["auto_generate_on_scan"])

            # Update config
            self._config["scan_every_n_reflections"] = scan_every
            self._config["auto_generate_on_scan"] = bool(auto_gen)

            # Create subscription record
            sub_id = f"apg_reflection_{int(time.time())}"

            # Try to subscribe via EventBus if context is available
            subscribed_to_bus = False
            if hasattr(self, "context") and self.context:
                try:
                    result = await self.context.call_skill("event", "subscribe", {
                        "topic": "reflection.created",
                        "handler_skill": "auto_playbook_generation",
                        "handler_action": "_on_reflection_created",
                    })
                    subscribed_to_bus = result.success if result else False
                except Exception:
                    pass

            self._wire_state = {
                "active": True,
                "subscription_id": sub_id,
                "reflections_since_scan": 0,
                "subscribed_to_bus": subscribed_to_bus,
                "wired_at": datetime.utcnow().isoformat(),
            }

            self._save_state()

            return SkillResult(
                success=True,
                message=(
                    f"Wired: will scan every {scan_every} reflections"
                    + (", auto-generating playbooks" if auto_gen else "")
                    + (f" (EventBus: {'connected' if subscribed_to_bus else 'local only'})")
                ),
                data={
                    "wire_state": self._wire_state,
                    "config": self._config,
                },
            )
        except Exception as e:
            return SkillResult(
                success=False,
                message=f"Wire failed: {str(e)}",
            )

    # ------------------------------------------------------------------
    # Action: unwire
    # ------------------------------------------------------------------

    async def _unwire(self, params: Dict) -> SkillResult:
        """Remove EventBus subscription and stop automatic generation."""
        try:
            was_active = self._wire_state.get("active", False)
            old_sub_id = self._wire_state.get("subscription_id")

            # Try to unsubscribe from EventBus
            unsubscribed_from_bus = False
            if was_active and hasattr(self, "context") and self.context and old_sub_id:
                try:
                    result = await self.context.call_skill("event", "unsubscribe", {
                        "subscription_id": old_sub_id,
                    })
                    unsubscribed_from_bus = result.success if result else False
                except Exception:
                    pass

            self._wire_state = self._default_wire_state()
            self._save_state()

            return SkillResult(
                success=True,
                message=(
                    "Unwired: automatic playbook generation stopped"
                    if was_active else "Already unwired"
                ),
                data={
                    "was_active": was_active,
                    "previous_subscription_id": old_sub_id,
                    "unsubscribed_from_bus": unsubscribed_from_bus,
                },
            )
        except Exception as e:
            return SkillResult(
                success=False,
                message=f"Unwire failed: {str(e)}",
            )

    # ------------------------------------------------------------------
    # Action: configure
    # ------------------------------------------------------------------

    async def _configure(self, params: Dict) -> SkillResult:
        """Update configuration parameters without rewiring."""
        try:
            updated = []
            config_keys = {
                "min_cluster_size": int,
                "min_success_rate": float,
                "auto_generate_on_scan": bool,
                "scan_every_n_reflections": int,
                "max_playbooks_per_scan": int,
            }

            for key, cast_fn in config_keys.items():
                if key in params:
                    old_val = self._config.get(key)
                    try:
                        new_val = cast_fn(params[key])
                    except (ValueError, TypeError):
                        continue
                    self._config[key] = new_val
                    updated.append({"key": key, "old": old_val, "new": new_val})

            self._save_state()

            return SkillResult(
                success=True,
                message=f"Updated {len(updated)} config value(s)",
                data={"updated": updated, "config": self._config},
            )
        except Exception as e:
            return SkillResult(
                success=False,
                message=f"Configure failed: {str(e)}",
            )

    # ------------------------------------------------------------------
    # Action: status
    # ------------------------------------------------------------------

    async def _status(self, params: Dict) -> SkillResult:
        """View current state: clusters, playbooks generated, wire status, config, and stats."""
        try:
            cluster_summaries = []
            for cid, cdata in list(self._clusters.items())[-20:]:
                cluster_summaries.append({
                    "cluster_id": cid,
                    "size": len(cdata.get("reflections", [])),
                    "common_tags": cdata.get("common_tags", []),
                    "representative_task": cdata.get("representative_task", ""),
                    "success_rate": cdata.get("success_rate", 0),
                    "existing_playbook": cdata.get("existing_playbook"),
                    "scanned_at": cdata.get("scanned_at"),
                })

            recent_playbooks = self._generated_playbooks[-10:]

            return SkillResult(
                success=True,
                message=(
                    f"Status: {len(self._clusters)} cluster(s), "
                    f"{self._stats['playbooks_generated']} playbook(s) generated, "
                    f"wire {'ACTIVE' if self._wire_state.get('active') else 'INACTIVE'}"
                ),
                data={
                    "clusters": cluster_summaries,
                    "recent_generated_playbooks": recent_playbooks,
                    "wire_state": self._wire_state,
                    "config": self._config,
                    "stats": self._stats,
                    "total_clusters": len(self._clusters),
                    "total_generated_playbooks": len(self._generated_playbooks),
                },
            )
        except Exception as e:
            return SkillResult(
                success=False,
                message=f"Status failed: {str(e)}",
            )

    # ------------------------------------------------------------------
    # EventBus callback (called when wire is active)
    # ------------------------------------------------------------------

    async def _on_reflection_created(self, event_data: Dict) -> None:
        """
        Callback invoked when a reflection.created event fires.

        Increments the reflection counter and triggers scan + auto_generate
        when the threshold is reached.
        """
        if not self._wire_state.get("active", False):
            return

        self._wire_state["reflections_since_scan"] = (
            self._wire_state.get("reflections_since_scan", 0) + 1
        )

        threshold = self._config.get("scan_every_n_reflections", 10)
        if self._wire_state["reflections_since_scan"] >= threshold:
            self._wire_state["reflections_since_scan"] = 0

            if self._config.get("auto_generate_on_scan", True):
                await self._auto_generate({})
            else:
                await self._scan({})

            self._save_state()

    # ------------------------------------------------------------------
    # Clustering algorithm
    # ------------------------------------------------------------------

    def _cluster_reflections(
        self, reflections: List[Dict], min_cluster_size: int
    ) -> List[List[Dict]]:
        """
        Cluster reflections by tag overlap and task keyword similarity.

        Uses single-linkage clustering: two reflections are linked if their
        combined similarity (tag Jaccard + keyword shared-words ratio) meets
        the threshold. If any pair in a group is linked, they belong to the
        same cluster.

        Args:
            reflections: List of reflection dicts.
            min_cluster_size: Minimum number of reflections for a valid cluster.

        Returns:
            List of clusters, where each cluster is a list of reflection dicts.
        """
        n = len(reflections)
        if n < min_cluster_size:
            return []

        # Precompute tag sets and keyword sets for each reflection
        tag_sets: List[Set[str]] = []
        keyword_sets: List[Set[str]] = []
        for r in reflections:
            tags = set(t.lower() for t in r.get("tags", []))
            tag_sets.append(tags)
            keywords = _tokenize(r.get("task", ""))
            keyword_sets.append(keywords)

        # Build adjacency via similarity threshold using single-linkage
        # Use Union-Find for efficient cluster merging
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]  # path compression
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        similarity_threshold = 0.3

        for i in range(n):
            for j in range(i + 1, n):
                tag_sim = _jaccard(tag_sets[i], tag_sets[j])
                keyword_sim = _shared_words_ratio(keyword_sets[i], keyword_sets[j])

                # Either tag overlap OR keyword overlap is sufficient to link
                if tag_sim >= similarity_threshold or keyword_sim >= similarity_threshold:
                    union(i, j)

        # Collect clusters
        cluster_map: Dict[int, List[int]] = {}
        for i in range(n):
            root = find(i)
            if root not in cluster_map:
                cluster_map[root] = []
            cluster_map[root].append(i)

        # Filter by min size and build result
        clusters = []
        for indices in cluster_map.values():
            if len(indices) >= min_cluster_size:
                cluster = [reflections[i] for i in indices]
                clusters.append(cluster)

        # Sort by cluster size descending (largest clusters first)
        clusters.sort(key=lambda c: len(c), reverse=True)

        return clusters

    # ------------------------------------------------------------------
    # Playbook extraction helpers
    # ------------------------------------------------------------------

    def _auto_generate_name(self, reflections: List[Dict]) -> str:
        """
        Auto-generate a playbook name from the most common significant words
        in the cluster's task descriptions.

        Args:
            reflections: List of reflection dicts in the cluster.

        Returns:
            A snake_case playbook name like "deploy_service_monitoring".
        """
        word_counter: Counter = Counter()
        for r in reflections:
            words = _tokenize(r.get("task", ""))
            word_counter.update(words)

        # Take top 3 most common words
        top_words = [word for word, _ in word_counter.most_common(3)]
        if not top_words:
            short_hash = hashlib.sha256(
                json.dumps([r.get("id", "") for r in reflections]).encode()
            ).hexdigest()[:6]
            return f"auto_playbook_{short_hash}"

        return "_".join(top_words)

    def _extract_task_pattern(self, reflections: List[Dict]) -> str:
        """
        Extract a common task pattern description from the cluster.

        Identifies the most frequent significant words across all task
        descriptions and composes a pattern string.

        Args:
            reflections: List of reflection dicts in the cluster.

        Returns:
            A descriptive task pattern string.
        """
        word_counter: Counter = Counter()
        for r in reflections:
            words = _tokenize(r.get("task", ""))
            word_counter.update(words)

        # Build pattern from top keywords
        top_words = [word for word, _ in word_counter.most_common(8)]
        if not top_words:
            return "General task pattern"

        # Find the shortest task description that contains the most top words
        best_task = ""
        best_score = -1
        for r in reflections:
            task = r.get("task", "")
            task_words = _tokenize(task)
            score = len(task_words & set(top_words))
            if score > best_score or (score == best_score and len(task) < len(best_task)):
                best_score = score
                best_task = task

        if best_task:
            # Clean up auto-reflection prefixes
            pattern = re.sub(r"^\[AUTO\]\s*", "", best_task)
            return f"Tasks involving: {', '.join(top_words[:5])} (e.g., {pattern})"

        return f"Tasks related to: {', '.join(top_words[:5])}"

    def _extract_steps(self, reflections: List[Dict]) -> List[str]:
        """
        Extract ordered steps from successful reflections' actions_taken.

        Collects all actions from successful reflections, counts frequency,
        and returns the most common ones as the recommended step sequence.

        Args:
            reflections: List of reflection dicts in the cluster.

        Returns:
            Ordered list of step descriptions.
        """
        successful = [r for r in reflections if r.get("success")]
        if not successful:
            # Fall back to all reflections if none succeeded
            successful = reflections

        action_counter: Counter = Counter()
        all_actions: List[str] = []

        for r in successful:
            actions_taken = r.get("actions_taken", [])
            if isinstance(actions_taken, str):
                # Handle case where actions_taken is a single string
                actions_taken = [actions_taken]
            for action in actions_taken:
                action_str = str(action).strip()
                if action_str:
                    action_counter[action_str] += 1
                    if action_str not in all_actions:
                        all_actions.append(action_str)

        if not all_actions:
            return ["Execute the task based on available context"]

        # Return actions ordered by first appearance (preserves natural sequence),
        # but prioritize those that appear more frequently
        frequent_threshold = max(1, len(successful) // 3)
        frequent_actions = [
            a for a in all_actions if action_counter[a] >= frequent_threshold
        ]

        # If too few frequent actions, include all unique ones
        if len(frequent_actions) < 2:
            frequent_actions = all_actions

        # Limit to reasonable number of steps
        return frequent_actions[:10]

    def _extract_pitfalls(self, reflections: List[Dict]) -> List[str]:
        """
        Extract pitfalls from failed reflections' analysis and improvements.

        Collects analysis text and improvement suggestions from failures,
        deduplicates, and returns as a list of pitfall warnings.

        Args:
            reflections: List of reflection dicts in the cluster.

        Returns:
            List of pitfall descriptions.
        """
        failed = [r for r in reflections if not r.get("success")]
        if not failed:
            return []

        pitfalls: List[str] = []
        seen: Set[str] = set()

        for r in failed:
            # Extract from analysis
            analysis = r.get("analysis", "").strip()
            if analysis:
                normalized = analysis.lower()
                if normalized not in seen:
                    seen.add(normalized)
                    pitfalls.append(f"Pitfall: {analysis}")

            # Extract from improvements
            improvements = r.get("improvements", [])
            if isinstance(improvements, str):
                improvements = [improvements]
            for imp in improvements:
                imp_str = str(imp).strip()
                if imp_str:
                    normalized = imp_str.lower()
                    if normalized not in seen:
                        seen.add(normalized)
                        pitfalls.append(f"Avoid: {imp_str}")

        # Limit to manageable number
        return pitfalls[:8]

    def _extract_prerequisites(self, reflections: List[Dict]) -> List[str]:
        """
        Extract prerequisites from common context across reflections.

        Looks for common tags and task patterns that suggest prerequisites
        (e.g., dependencies, required services, credentials).

        Args:
            reflections: List of reflection dicts in the cluster.

        Returns:
            List of prerequisite descriptions.
        """
        prerequisites: List[str] = []

        # Common tags suggest context requirements
        common_tags = self._get_common_tags(reflections)
        if common_tags:
            prerequisites.append(f"Context tags: {', '.join(common_tags)}")

        # Look for recurring keywords in task descriptions that suggest prerequisites
        word_counter: Counter = Counter()
        for r in reflections:
            words = _tokenize(r.get("task", ""))
            word_counter.update(words)

        # Words that appear in most reflections suggest shared prerequisites
        threshold = max(1, len(reflections) * 0.6)
        common_words = [
            word for word, count in word_counter.items()
            if count >= threshold
        ]

        prereq_hints = {
            "deploy", "deployment", "service", "api", "database", "db",
            "config", "configuration", "credentials", "auth", "authentication",
            "server", "cluster", "container", "docker", "kubernetes", "k8s",
            "ssl", "certificate", "dns", "domain", "network",
        }

        for word in common_words:
            if word in prereq_hints:
                prerequisites.append(f"Requires {word} to be available/configured")

        # Extract from successful reflections' context
        successful = [r for r in reflections if r.get("success")]
        if successful:
            # Check if there is a common task prefix pattern
            tasks = [r.get("task", "") for r in successful]
            if len(tasks) >= 2:
                # Find common prefix
                prefix = os.path.commonprefix(tasks).strip()
                if len(prefix) > 10:
                    prerequisites.append(f"Task context: {prefix}")

        return prerequisites[:5] if prerequisites else ["No specific prerequisites identified"]

    def _extract_expected_outcome(self, reflections: List[Dict]) -> str:
        """
        Extract expected outcome from successful reflections' outcomes.

        Picks the most representative successful outcome description.

        Args:
            reflections: List of reflection dicts in the cluster.

        Returns:
            Expected outcome description string.
        """
        successful = [r for r in reflections if r.get("success")]
        if not successful:
            return "Task completion (no successful examples in cluster)"

        # Find the most common outcome keywords
        outcome_counter: Counter = Counter()
        outcomes: List[str] = []
        for r in successful:
            outcome = r.get("outcome", "").strip()
            if outcome:
                outcomes.append(outcome)
                words = _tokenize(outcome)
                outcome_counter.update(words)

        if not outcomes:
            return "Successful task completion"

        # Pick the outcome that best represents the cluster
        # (contains the most common outcome keywords)
        top_outcome_words = set(
            word for word, _ in outcome_counter.most_common(5)
        )

        best_outcome = outcomes[0]
        best_score = 0
        for outcome in outcomes:
            words = _tokenize(outcome)
            score = len(words & top_outcome_words)
            if score > best_score:
                best_score = score
                best_outcome = outcome

        return best_outcome

    def _extract_all_tags(
        self, reflections: List[Dict], extra_tags: List[str] = None
    ) -> List[str]:
        """
        Collect the union of all tags from the cluster's reflections.

        Args:
            reflections: List of reflection dicts in the cluster.
            extra_tags: Additional tags to include.

        Returns:
            Deduplicated list of tags.
        """
        all_tags: Set[str] = set()
        for r in reflections:
            for tag in r.get("tags", []):
                all_tags.add(tag)

        if extra_tags:
            all_tags.update(extra_tags)

        # Add a marker tag
        all_tags.add("auto_generated_playbook")

        return sorted(all_tags)

    # ------------------------------------------------------------------
    # Cluster metadata helpers
    # ------------------------------------------------------------------

    def _generate_cluster_id(self, cluster: List[Dict]) -> str:
        """
        Generate a deterministic cluster ID from the cluster's reflection IDs.

        Args:
            cluster: List of reflection dicts.

        Returns:
            A cluster ID like "cluster_a1b2c3d4".
        """
        ids = sorted(r.get("id", str(i)) for i, r in enumerate(cluster))
        content = "|".join(ids)
        hash_val = hashlib.sha256(content.encode()).hexdigest()[:8]
        return f"cluster_{hash_val}"

    def _get_common_tags(self, cluster: List[Dict]) -> List[str]:
        """
        Get tags that appear in at least half of the cluster's reflections.

        Args:
            cluster: List of reflection dicts.

        Returns:
            List of common tags sorted by frequency.
        """
        tag_counter: Counter = Counter()
        for r in cluster:
            for tag in r.get("tags", []):
                tag_counter[tag] += 1

        threshold = max(1, len(cluster) // 2)
        common = [
            tag for tag, count in tag_counter.most_common()
            if count >= threshold
        ]
        return common

    def _get_representative_task(self, cluster: List[Dict]) -> str:
        """
        Pick the most representative task description from the cluster.

        Selects the task whose keywords overlap most with the cluster's
        aggregate keyword set.

        Args:
            cluster: List of reflection dicts.

        Returns:
            Representative task description string.
        """
        if not cluster:
            return ""

        # Aggregate keywords
        aggregate: Counter = Counter()
        for r in cluster:
            words = _tokenize(r.get("task", ""))
            aggregate.update(words)

        top_words = set(word for word, _ in aggregate.most_common(10))

        best_task = cluster[0].get("task", "")
        best_score = 0
        for r in cluster:
            task = r.get("task", "")
            words = _tokenize(task)
            score = len(words & top_words)
            if score > best_score:
                best_score = score
                best_task = task

        return best_task

    def _compute_success_rate(self, cluster: List[Dict]) -> float:
        """
        Compute the success rate of reflections in the cluster.

        Args:
            cluster: List of reflection dicts.

        Returns:
            Success rate as a float between 0.0 and 1.0.
        """
        if not cluster:
            return 0.0
        successes = sum(1 for r in cluster if r.get("success"))
        return successes / len(cluster)

    def _find_overlapping_playbook(
        self,
        common_tags: List[str],
        existing_playbook_tags: Dict[str, Set[str]],
    ) -> Optional[str]:
        """
        Check if an existing playbook has significant tag overlap with the cluster.

        Args:
            common_tags: Tags common to the cluster.
            existing_playbook_tags: Dict mapping playbook name to its tag set.

        Returns:
            Name of the overlapping playbook, or None.
        """
        if not common_tags:
            return None

        cluster_tags = set(t.lower() for t in common_tags)
        best_name = None
        best_overlap = 0.0

        for pb_name, pb_tags in existing_playbook_tags.items():
            if not pb_tags:
                continue
            overlap = _jaccard(cluster_tags, pb_tags)
            if overlap > best_overlap and overlap >= 0.3:
                best_overlap = overlap
                best_name = pb_name

        return best_name

    def _extract_playbook_tag_sets(
        self, playbooks: List[Dict]
    ) -> Dict[str, Set[str]]:
        """
        Build a mapping of playbook name to its tag set.

        Args:
            playbooks: List of playbook dicts from AgentReflection.

        Returns:
            Dict mapping playbook name to lowercase tag set.
        """
        result: Dict[str, Set[str]] = {}
        for pb in playbooks:
            name = pb.get("name", "")
            tags = set(t.lower() for t in pb.get("tags", []))
            if name:
                result[name] = tags
        return result

    # ------------------------------------------------------------------
    # AgentReflection integration
    # ------------------------------------------------------------------

    async def _fetch_reflections(
        self, lookback: int, filter_tag: str = ""
    ) -> List[Dict]:
        """
        Fetch recent reflections from AgentReflection via context.call_skill.

        Args:
            lookback: Maximum number of recent reflections to fetch.
            filter_tag: Optional tag to filter reflections by.

        Returns:
            List of reflection dicts.
        """
        if not (hasattr(self, "context") and self.context):
            return []

        try:
            params: Dict[str, Any] = {"what": "reflections", "limit": lookback}
            if filter_tag:
                params["filter_tag"] = filter_tag

            result = await self.context.call_skill(
                "agent_reflection", "review", params
            )

            if result and result.success and result.data:
                return result.data.get("reflections", [])
        except Exception:
            pass

        return []

    async def _fetch_playbooks(self) -> List[Dict]:
        """
        Fetch existing playbooks from AgentReflection via context.call_skill.

        Returns:
            List of playbook dicts.
        """
        if not (hasattr(self, "context") and self.context):
            return []

        try:
            result = await self.context.call_skill(
                "agent_reflection", "review",
                {"what": "playbooks", "limit": 100},
            )

            if result and result.success and result.data:
                return result.data.get("playbooks", [])
        except Exception:
            pass

        return []

    async def _create_playbook_via_reflection(
        self,
        name: str,
        task_pattern: str,
        steps: List[str],
        pitfalls: List[str],
        prerequisites: List[str],
        expected_outcome: str,
        tags: List[str],
    ) -> Optional[SkillResult]:
        """
        Create a playbook by calling AgentReflection's create_playbook action.

        Args:
            name: Playbook name.
            task_pattern: Description of the task pattern.
            steps: Ordered list of steps.
            pitfalls: Common pitfalls to avoid.
            prerequisites: What must be true before starting.
            expected_outcome: What success looks like.
            tags: Tags for matching.

        Returns:
            SkillResult from AgentReflection, or None if context unavailable.
        """
        if not (hasattr(self, "context") and self.context):
            # Return a synthetic success if no context (for standalone testing)
            return SkillResult(
                success=True,
                message=f"Playbook '{name}' created (no context, local only)",
                data={
                    "playbook": {
                        "name": name,
                        "task_pattern": task_pattern,
                        "steps": steps,
                        "pitfalls": pitfalls,
                        "prerequisites": prerequisites,
                        "expected_outcome": expected_outcome,
                        "tags": tags,
                    }
                },
            )

        try:
            return await self.context.call_skill(
                "agent_reflection", "create_playbook", {
                    "name": name,
                    "task_pattern": task_pattern,
                    "steps": steps,
                    "pitfalls": pitfalls,
                    "prerequisites": prerequisites,
                    "expected_outcome": expected_outcome,
                    "tags": tags,
                }
            )
        except Exception as e:
            return SkillResult(
                success=False,
                message=f"Failed to call agent_reflection.create_playbook: {str(e)}",
            )

    # ------------------------------------------------------------------
    # Cost estimation
    # ------------------------------------------------------------------

    def estimate_cost(self, action: str, params: Dict) -> float:
        """All actions are zero-cost (no external API calls)."""
        return 0.0
