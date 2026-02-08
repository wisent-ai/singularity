#!/usr/bin/env python3
"""
RuleConflictDetectionSkill - Detect and resolve contradictions in learned rules.

As the agent accumulates distilled rules over time, some will inevitably
contradict each other. For example:
  - "Prefer skill X for deployment" vs "Avoid skill X - high failure rate"
  - "Revenue tasks yield best ROI" vs "Self-improvement tasks yield best ROI"
  - "Use Docker for hosting" vs "Serverless is better than Docker for hosting"

Without conflict detection, the agent gets contradictory advice during
decision-making, leading to inconsistent behavior and wasted cycles.

This skill:
1. Scans rule pairs for textual/semantic conflicts using keyword analysis
2. Detects specific conflict patterns (prefer/avoid, success/failure, opposing recommendations)
3. Resolves conflicts by comparing confidence, recency, and evidence strength
4. Auto-weakens or retires the losing rule
5. Maintains a conflict log for transparency and learning

Actions:
  - scan: Analyze all rules for conflicts, return conflict pairs
  - resolve: Auto-resolve detected conflicts (weaken loser, annotate winner)
  - scan_and_resolve: Combined scan + resolve in one step
  - conflicts: List all detected conflicts (current and historical)
  - status: Summary statistics on conflict detection activity
  - configure: Set conflict detection thresholds

Pillar: Self-Improvement (rule hygiene ensures learning quality stays high)
"""

import json
import time
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

from .base import Skill, SkillResult, SkillManifest, SkillAction


DATA_DIR = Path(__file__).parent.parent / "data"
CONFLICT_DATA_FILE = DATA_DIR / "rule_conflicts.json"
RULES_FILE = DATA_DIR / "learning_rules.json"

MAX_CONFLICTS = 500
MAX_RESOLUTION_HISTORY = 200

# Keywords that indicate opposing intent
POSITIVE_SIGNALS = {"prefer", "use", "good", "success", "effective", "fast", "reliable", "recommend", "best", "works", "choose", "high"}
NEGATIVE_SIGNALS = {"avoid", "skip", "bad", "failure", "ineffective", "slow", "unreliable", "worse", "worst", "fails", "low", "never"}

# Opposing category pairs that inherently conflict
# Common stopwords to filter out for better content similarity
STOPWORDS = {"the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
             "it", "its", "and", "or", "but", "for", "of", "in", "on", "at", "to",
             "with", "by", "from", "as", "this", "that", "these", "those", "not",
             "has", "have", "had", "do", "does", "did", "will", "would", "can",
             "could", "should", "may", "might", "than", "when", "where", "which"}

OPPOSING_CATEGORIES = [
    ("success_pattern", "failure_pattern"),
]


class RuleConflictDetectionSkill(Skill):
    """
    Detect and resolve contradictions in the agent's learned rule base.

    Ensures accumulated wisdom stays consistent and actionable by finding
    rules that give opposing advice and resolving them based on evidence.
    """

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self._ensure_data()

    def _ensure_data(self):
        DATA_DIR.mkdir(parents=True, exist_ok=True)

    def _default_state(self) -> Dict:
        return {
            "conflicts": [],
            "resolution_history": [],
            "config": {
                "similarity_threshold": 0.15,  # Min keyword overlap to consider conflict
                "auto_resolve": True,          # Auto-resolve on scan
                "min_confidence_gap": 0.1,     # Min gap to declare a winner
                "weaken_factor": 0.5,          # Multiply loser confidence by this
                "retire_threshold": 0.05,      # Below this, mark rule as retired
            },
            "stats": {
                "total_scans": 0,
                "total_conflicts_found": 0,
                "total_resolved": 0,
                "total_auto_resolved": 0,
                "last_scan": None,
            },
        }

    def _load(self) -> Dict:
        try:
            if CONFLICT_DATA_FILE.exists():
                return json.loads(CONFLICT_DATA_FILE.read_text())
        except (json.JSONDecodeError, IOError):
            pass
        return self._default_state()

    def _save(self, data: Dict):
        self._ensure_data()
        # Trim to size limits
        data["conflicts"] = data["conflicts"][-MAX_CONFLICTS:]
        data["resolution_history"] = data["resolution_history"][-MAX_RESOLUTION_HISTORY:]
        CONFLICT_DATA_FILE.write_text(json.dumps(data, indent=2, default=str))

    def _load_rules(self) -> List[Dict]:
        """Load rules from the learning distillation rule base."""
        try:
            if RULES_FILE.exists():
                rules_data = json.loads(RULES_FILE.read_text())
                return rules_data.get("rules", [])
        except (json.JSONDecodeError, IOError):
            pass
        return []

    def _save_rules(self, rules: List[Dict]):
        """Save modified rules back to the learning distillation rule base."""
        try:
            if RULES_FILE.exists():
                rules_data = json.loads(RULES_FILE.read_text())
            else:
                rules_data = {"rules": [], "stats": {}, "config": {}, "distillation_history": []}
            rules_data["rules"] = rules
            RULES_FILE.write_text(json.dumps(rules_data, indent=2, default=str))
        except (json.JSONDecodeError, IOError):
            pass

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="rule_conflict_detection",
            name="Rule Conflict Detection",
            description="Detect and resolve contradictions in learned rules to maintain consistent decision-making.",
            version="1.0.0",
            actions=[
                SkillAction(
                    name="scan",
                    description="Scan all rules for conflicts. Returns conflict pairs with similarity scores.",
                    parameters={
                        "min_similarity": {
                            "type": "number",
                            "description": "Minimum similarity threshold (0-1) to flag as conflict. Default: use config.",
                        },
                    },
                ),
                SkillAction(
                    name="resolve",
                    description="Resolve a specific conflict by weakening the lower-confidence rule.",
                    parameters={
                        "conflict_id": {
                            "type": "string",
                            "description": "ID of the conflict to resolve.",
                            "required": True,
                        },
                        "winner": {
                            "type": "string",
                            "description": "Rule ID to keep (optional - auto-picks higher confidence if omitted).",
                        },
                    },
                ),
                SkillAction(
                    name="scan_and_resolve",
                    description="Scan for conflicts and auto-resolve all found. Combined convenience action.",
                    parameters={
                        "min_similarity": {
                            "type": "number",
                            "description": "Minimum similarity threshold override.",
                        },
                    },
                ),
                SkillAction(
                    name="conflicts",
                    description="List all detected conflicts (current unresolved and historical).",
                    parameters={
                        "status": {
                            "type": "string",
                            "description": "Filter: 'unresolved', 'resolved', or 'all'. Default: 'all'.",
                        },
                    },
                ),
                SkillAction(
                    name="status",
                    description="Summary of conflict detection stats and configuration.",
                    parameters={},
                ),
                SkillAction(
                    name="configure",
                    description="Update conflict detection configuration.",
                    parameters={
                        "similarity_threshold": {"type": "number", "description": "0-1 keyword overlap threshold."},
                        "auto_resolve": {"type": "boolean", "description": "Auto-resolve on scan."},
                        "min_confidence_gap": {"type": "number", "description": "Min gap to declare winner."},
                        "weaken_factor": {"type": "number", "description": "Multiply loser confidence by this."},
                        "retire_threshold": {"type": "number", "description": "Below this confidence, retire rule."},
                    },
                ),
            ],
        )

    async def execute(self, action: str, params: Dict) -> SkillResult:
        actions = {
            "scan": self._scan,
            "resolve": self._resolve,
            "scan_and_resolve": self._scan_and_resolve,
            "conflicts": self._conflicts,
            "status": self._status,
            "configure": self._configure,
        }
        handler = actions.get(action)
        if not handler:
            return SkillResult(
                success=False,
                message=f"Unknown action: {action}. Valid: {list(actions.keys())}",
            )
        return await handler(params)

    def _tokenize(self, text: str) -> set:
        """Extract lowercase word tokens from text."""
        return set(text.lower().replace("_", " ").replace("-", " ").split())

    def _compute_similarity(self, rule_a: Dict, rule_b: Dict) -> float:
        """Compute keyword-based similarity between two rules (0-1)."""
        text_a = rule_a.get("rule_text", "")
        text_b = rule_b.get("rule_text", "")
        tokens_a = self._tokenize(text_a) - STOPWORDS
        tokens_b = self._tokenize(text_b) - STOPWORDS

        if not tokens_a or not tokens_b:
            return 0.0

        # Jaccard similarity on content words (stopwords removed)
        intersection = tokens_a & tokens_b
        union = tokens_a | tokens_b
        return len(intersection) / len(union) if union else 0.0

    def _detect_sentiment_conflict(self, rule_a: Dict, rule_b: Dict) -> bool:
        """Check if two rules have opposing positive/negative sentiment."""
        text_a = self._tokenize(rule_a.get("rule_text", ""))
        text_b = self._tokenize(rule_b.get("rule_text", ""))

        a_pos = len(text_a & POSITIVE_SIGNALS)
        a_neg = len(text_a & NEGATIVE_SIGNALS)
        b_pos = len(text_b & POSITIVE_SIGNALS)
        b_neg = len(text_b & NEGATIVE_SIGNALS)

        # One is positive and the other is negative
        a_is_positive = a_pos > a_neg
        a_is_negative = a_neg > a_pos
        b_is_positive = b_pos > b_neg
        b_is_negative = b_neg > b_pos

        return (a_is_positive and b_is_negative) or (a_is_negative and b_is_positive)

    def _detect_category_conflict(self, rule_a: Dict, rule_b: Dict) -> bool:
        """Check if rules are in inherently opposing categories."""
        cat_a = rule_a.get("category", "")
        cat_b = rule_b.get("category", "")
        for pair in OPPOSING_CATEGORIES:
            if (cat_a == pair[0] and cat_b == pair[1]) or (cat_a == pair[1] and cat_b == pair[0]):
                return True
        return False

    def _detect_skill_conflict(self, rule_a: Dict, rule_b: Dict) -> bool:
        """Check if rules reference the same skill with opposing advice."""
        skill_a = rule_a.get("skill_id", "")
        skill_b = rule_b.get("skill_id", "")
        if not skill_a or not skill_b or skill_a != skill_b:
            return False
        # Same skill, check if one praises and one criticizes
        return self._detect_sentiment_conflict(rule_a, rule_b)

    def _find_conflicts(self, rules: List[Dict], min_similarity: float) -> List[Dict]:
        """Find all conflicting rule pairs."""
        conflicts = []
        seen_pairs = set()

        for i, rule_a in enumerate(rules):
            for j, rule_b in enumerate(rules):
                if i >= j:
                    continue

                pair_key = tuple(sorted([rule_a["id"], rule_b["id"]]))
                if pair_key in seen_pairs:
                    continue

                similarity = self._compute_similarity(rule_a, rule_b)
                has_sentiment_conflict = self._detect_sentiment_conflict(rule_a, rule_b)
                has_category_conflict = self._detect_category_conflict(rule_a, rule_b)
                has_skill_conflict = self._detect_skill_conflict(rule_a, rule_b)

                # A conflict exists if rules are similar enough AND show opposing signals
                is_conflict = False
                conflict_type = []

                if similarity >= min_similarity and has_sentiment_conflict:
                    is_conflict = True
                    conflict_type.append("sentiment_opposition")

                if similarity >= min_similarity and has_category_conflict:
                    is_conflict = True
                    conflict_type.append("category_opposition")

                if has_skill_conflict:
                    # Skill conflicts don't need high similarity - same skill is enough
                    is_conflict = True
                    conflict_type.append("skill_opposition")

                if is_conflict:
                    seen_pairs.add(pair_key)
                    conflict_id = str(uuid.uuid4())[:8]
                    conflicts.append({
                        "id": conflict_id,
                        "rule_a_id": rule_a["id"],
                        "rule_b_id": rule_b["id"],
                        "rule_a_text": rule_a.get("rule_text", ""),
                        "rule_b_text": rule_b.get("rule_text", ""),
                        "rule_a_confidence": rule_a.get("confidence", 0),
                        "rule_b_confidence": rule_b.get("confidence", 0),
                        "rule_a_category": rule_a.get("category", ""),
                        "rule_b_category": rule_b.get("category", ""),
                        "rule_a_skill": rule_a.get("skill_id", ""),
                        "rule_b_skill": rule_b.get("skill_id", ""),
                        "similarity": round(similarity, 3),
                        "conflict_types": conflict_type,
                        "status": "unresolved",
                        "detected_at": datetime.now().isoformat(),
                    })

        return conflicts

    def _pick_winner(self, conflict: Dict, rules: List[Dict], config: Dict) -> Tuple[str, str, str]:
        """
        Determine which rule wins a conflict.

        Returns (winner_id, loser_id, reason).
        Uses confidence, then recency, then reinforcement count.
        """
        rule_a = None
        rule_b = None
        for r in rules:
            if r["id"] == conflict["rule_a_id"]:
                rule_a = r
            if r["id"] == conflict["rule_b_id"]:
                rule_b = r

        if not rule_a or not rule_b:
            return ("", "", "rule_not_found")

        conf_a = rule_a.get("confidence", 0)
        conf_b = rule_b.get("confidence", 0)
        gap = abs(conf_a - conf_b)
        min_gap = config.get("min_confidence_gap", 0.1)

        # Primary: confidence comparison
        if gap >= min_gap:
            if conf_a > conf_b:
                return (rule_a["id"], rule_b["id"], f"higher_confidence ({conf_a:.3f} vs {conf_b:.3f})")
            else:
                return (rule_b["id"], rule_a["id"], f"higher_confidence ({conf_b:.3f} vs {conf_a:.3f})")

        # Tiebreaker 1: reinforcement count
        reinf_a = rule_a.get("reinforcement_count", 0)
        reinf_b = rule_b.get("reinforcement_count", 0)
        if reinf_a != reinf_b:
            if reinf_a > reinf_b:
                return (rule_a["id"], rule_b["id"], f"more_reinforcements ({reinf_a} vs {reinf_b})")
            else:
                return (rule_b["id"], rule_a["id"], f"more_reinforcements ({reinf_b} vs {reinf_a})")

        # Tiebreaker 2: recency (more recent rule wins)
        time_a = rule_a.get("last_reinforced", rule_a.get("created_at", ""))
        time_b = rule_b.get("last_reinforced", rule_b.get("created_at", ""))
        if time_a > time_b:
            return (rule_a["id"], rule_b["id"], f"more_recent ({time_a[:10]} vs {time_b[:10]})")
        elif time_b > time_a:
            return (rule_b["id"], rule_a["id"], f"more_recent ({time_b[:10]} vs {time_a[:10]})")

        # If truly tied, keep rule_a by default (arbitrary but deterministic)
        return (rule_a["id"], rule_b["id"], "arbitrary_tiebreak")

    def _apply_resolution(self, loser_id: str, rules: List[Dict], config: Dict) -> Optional[Dict]:
        """Weaken or retire the losing rule. Returns the modified rule or None."""
        weaken_factor = config.get("weaken_factor", 0.5)
        retire_threshold = config.get("retire_threshold", 0.05)

        for rule in rules:
            if rule["id"] == loser_id:
                old_confidence = rule["confidence"]
                rule["confidence"] = round(rule["confidence"] * weaken_factor, 4)
                retired = rule["confidence"] < retire_threshold
                if retired:
                    rule["status"] = "retired_by_conflict"
                return {
                    "rule_id": loser_id,
                    "old_confidence": old_confidence,
                    "new_confidence": rule["confidence"],
                    "retired": retired,
                }
        return None

    async def _scan(self, params: Dict) -> SkillResult:
        """Scan all rules for conflicts."""
        data = self._load()
        config = data["config"]
        rules = self._load_rules()

        if len(rules) < 2:
            return SkillResult(
                success=True,
                message="Not enough rules to scan (need at least 2).",
                data={"conflicts": [], "rules_scanned": len(rules)},
            )

        min_sim = float(params.get("min_similarity", config["similarity_threshold"]))

        # Filter out already-retired rules
        active_rules = [r for r in rules if r.get("status") != "retired_by_conflict"]
        conflicts = self._find_conflicts(active_rules, min_sim)

        # Store new conflicts
        existing_ids = {c["id"] for c in data["conflicts"]}
        for c in conflicts:
            if c["id"] not in existing_ids:
                data["conflicts"].append(c)

        data["stats"]["total_scans"] += 1
        data["stats"]["total_conflicts_found"] += len(conflicts)
        data["stats"]["last_scan"] = datetime.now().isoformat()
        self._save(data)

        return SkillResult(
            success=True,
            message=f"Scanned {len(active_rules)} rules, found {len(conflicts)} conflicts.",
            data={
                "conflicts": conflicts,
                "rules_scanned": len(active_rules),
                "conflicts_found": len(conflicts),
            },
        )

    async def _resolve(self, params: Dict) -> SkillResult:
        """Resolve a specific conflict."""
        conflict_id = params.get("conflict_id", "")
        if not conflict_id:
            return SkillResult(success=False, message="conflict_id is required.")

        data = self._load()
        config = data["config"]
        rules = self._load_rules()

        conflict = None
        for c in data["conflicts"]:
            if c["id"] == conflict_id:
                conflict = c
                break

        if not conflict:
            return SkillResult(success=False, message=f"Conflict '{conflict_id}' not found.")

        if conflict.get("status") == "resolved":
            return SkillResult(
                success=True,
                message=f"Conflict '{conflict_id}' already resolved.",
                data={"conflict": conflict},
            )

        # Determine winner
        explicit_winner = params.get("winner", "")
        if explicit_winner:
            if explicit_winner == conflict["rule_a_id"]:
                winner_id, loser_id, reason = conflict["rule_a_id"], conflict["rule_b_id"], "manual_override"
            elif explicit_winner == conflict["rule_b_id"]:
                winner_id, loser_id, reason = conflict["rule_b_id"], conflict["rule_a_id"], "manual_override"
            else:
                return SkillResult(success=False, message=f"Winner '{explicit_winner}' not part of conflict.")
        else:
            winner_id, loser_id, reason = self._pick_winner(conflict, rules, config)
            if not winner_id:
                return SkillResult(success=False, message="Could not find conflicting rules in rule base.")

        # Apply resolution
        result = self._apply_resolution(loser_id, rules, config)
        if not result:
            return SkillResult(success=False, message=f"Could not find loser rule '{loser_id}' to weaken.")

        # Save modified rules
        self._save_rules(rules)

        # Update conflict status
        conflict["status"] = "resolved"
        conflict["resolution"] = {
            "winner_id": winner_id,
            "loser_id": loser_id,
            "reason": reason,
            "loser_old_confidence": result["old_confidence"],
            "loser_new_confidence": result["new_confidence"],
            "loser_retired": result["retired"],
            "resolved_at": datetime.now().isoformat(),
        }

        # Add to resolution history
        data["resolution_history"].append({
            "conflict_id": conflict_id,
            "winner_id": winner_id,
            "loser_id": loser_id,
            "reason": reason,
            "retired": result["retired"],
            "resolved_at": datetime.now().isoformat(),
        })

        data["stats"]["total_resolved"] += 1
        self._save(data)

        action_taken = "retired" if result["retired"] else "weakened"
        return SkillResult(
            success=True,
            message=f"Resolved conflict '{conflict_id}': winner={winner_id} ({reason}), loser={loser_id} {action_taken} ({result['old_confidence']:.3f} → {result['new_confidence']:.3f}).",
            data={
                "conflict": conflict,
                "winner_id": winner_id,
                "loser_id": loser_id,
                "reason": reason,
                "loser_weakened": result,
            },
        )

    async def _scan_and_resolve(self, params: Dict) -> SkillResult:
        """Scan for conflicts and auto-resolve all found."""
        # First scan
        scan_result = await self._scan(params)
        if not scan_result.success:
            return scan_result

        conflicts = scan_result.data.get("conflicts", [])
        if not conflicts:
            return SkillResult(
                success=True,
                message=f"Scanned {scan_result.data.get('rules_scanned', 0)} rules, no conflicts found.",
                data=scan_result.data,
            )

        # Resolve each
        data = self._load()
        config = data["config"]
        rules = self._load_rules()
        resolved = []
        failed = []

        for conflict in conflicts:
            winner_id, loser_id, reason = self._pick_winner(conflict, rules, config)
            if not winner_id:
                failed.append(conflict["id"])
                continue

            result = self._apply_resolution(loser_id, rules, config)
            if not result:
                failed.append(conflict["id"])
                continue

            conflict["status"] = "resolved"
            conflict["resolution"] = {
                "winner_id": winner_id,
                "loser_id": loser_id,
                "reason": reason,
                "loser_old_confidence": result["old_confidence"],
                "loser_new_confidence": result["new_confidence"],
                "loser_retired": result["retired"],
                "resolved_at": datetime.now().isoformat(),
            }

            data["resolution_history"].append({
                "conflict_id": conflict["id"],
                "winner_id": winner_id,
                "loser_id": loser_id,
                "reason": reason,
                "retired": result["retired"],
                "resolved_at": datetime.now().isoformat(),
            })

            resolved.append({
                "conflict_id": conflict["id"],
                "winner_id": winner_id,
                "loser_id": loser_id,
                "reason": reason,
                "retired": result["retired"],
            })

        # Save everything
        self._save_rules(rules)
        data["stats"]["total_resolved"] += len(resolved)
        data["stats"]["total_auto_resolved"] += len(resolved)

        # Update conflicts in data
        conflict_map = {c["id"]: c for c in conflicts}
        for i, c in enumerate(data["conflicts"]):
            if c["id"] in conflict_map:
                data["conflicts"][i] = conflict_map[c["id"]]

        self._save(data)

        return SkillResult(
            success=True,
            message=f"Scanned {scan_result.data.get('rules_scanned', 0)} rules, found {len(conflicts)} conflicts, resolved {len(resolved)}, failed {len(failed)}.",
            data={
                "rules_scanned": scan_result.data.get("rules_scanned", 0),
                "conflicts_found": len(conflicts),
                "resolved": resolved,
                "failed": failed,
            },
        )

    async def _conflicts(self, params: Dict) -> SkillResult:
        """List all detected conflicts."""
        data = self._load()
        status_filter = params.get("status", "all")

        conflicts = data["conflicts"]
        if status_filter == "unresolved":
            conflicts = [c for c in conflicts if c.get("status") != "resolved"]
        elif status_filter == "resolved":
            conflicts = [c for c in conflicts if c.get("status") == "resolved"]

        return SkillResult(
            success=True,
            message=f"Found {len(conflicts)} conflicts (filter: {status_filter}).",
            data={
                "conflicts": conflicts[-50:],  # Limit response size
                "total": len(conflicts),
                "filter": status_filter,
            },
        )

    async def _status(self, params: Dict) -> SkillResult:
        """Summary stats on conflict detection."""
        data = self._load()
        stats = data["stats"]
        config = data["config"]

        unresolved = len([c for c in data["conflicts"] if c.get("status") != "resolved"])
        resolved = len([c for c in data["conflicts"] if c.get("status") == "resolved"])

        return SkillResult(
            success=True,
            message=f"Conflict detection: {unresolved} unresolved, {resolved} resolved.",
            data={
                "stats": stats,
                "config": config,
                "current_conflicts": {
                    "unresolved": unresolved,
                    "resolved": resolved,
                    "total": len(data["conflicts"]),
                },
                "resolution_history_count": len(data["resolution_history"]),
            },
        )

    async def _configure(self, params: Dict) -> SkillResult:
        """Update conflict detection configuration."""
        data = self._load()
        config = data["config"]
        changed = []

        for key in ["similarity_threshold", "min_confidence_gap", "weaken_factor", "retire_threshold"]:
            if key in params:
                val = float(params[key])
                val = max(0.0, min(1.0, val))
                config[key] = val
                changed.append(f"{key}={val}")

        if "auto_resolve" in params:
            config["auto_resolve"] = bool(params["auto_resolve"])
            changed.append(f"auto_resolve={config['auto_resolve']}")

        self._save(data)

        return SkillResult(
            success=True,
            message=f"Updated config: {', '.join(changed) if changed else 'no changes'}.",
            data={"config": config},
        )
