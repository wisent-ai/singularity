#!/usr/bin/env python3
"""
CheckpointComparisonAnalyticsSkill - Track progress across checkpoints with diff analysis.

The agent creates checkpoints over time via AgentCheckpointSkill. This skill
analyzes series of checkpoints to provide:

1. Growth trends - how data files grow/shrink over time
2. Progress scoring - goal completion, skill maturity, experiment outcomes between checkpoints
3. Timeline view - agent evolution across checkpoint history
4. Regression detection - flag when agent state regresses
5. Pillar health tracking - per-pillar progress from checkpoint data

This is the analytical layer on top of raw checkpoints, turning snapshots into
actionable progress intelligence.

Pillars:
- Goal Setting: Quantitative progress tracking across sessions
- Self-Improvement: Detect regressions and learning plateaus
- Revenue: Track revenue-related data growth (customers, services, earnings)
- Replication: Compare replica checkpoints to measure divergence

Actions:
- compare: Deep comparison of two checkpoints with semantic analysis
- timeline: Show agent evolution across all checkpoints
- trends: Detect growth/shrinkage patterns across checkpoint series
- progress_score: Compute progress score between two checkpoints
- regressions: Detect state regressions across checkpoints
- pillar_health: Per-pillar health from checkpoint data evolution
- report: Full progress report combining all analytics
- status: Show analytics health and last analysis state
"""

import json
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .base import Skill, SkillAction, SkillManifest, SkillResult

DATA_DIR = Path(__file__).parent.parent / "data"
CHECKPOINT_DIR = DATA_DIR / "checkpoints"
CHECKPOINT_INDEX = CHECKPOINT_DIR / "index.json"
ANALYTICS_FILE = DATA_DIR / "checkpoint_comparison_analytics.json"

# File classification for pillar attribution
PILLAR_FILE_PATTERNS = {
    "self_improvement": [
        "feedback_loop", "performance", "self_modify", "self_test",
        "self_tuning", "self_assess", "experiment", "tuning",
        "skill_composer", "steering",
    ],
    "revenue": [
        "revenue", "marketplace", "usage_tracking", "service_catalog",
        "service_monitor", "api_gateway", "payment", "funding",
        "billing", "customer", "catalog", "serverless",
    ],
    "replication": [
        "agent_spawner", "agent_network", "fleet", "replica",
        "knowledge_sharing", "consensus", "delegation",
    ],
    "goal_setting": [
        "goal", "strategy", "planner", "outcome", "decision_log",
        "dashboard", "observability", "workflow_analytics",
        "autonomous_loop", "session_bootstrap",
    ],
}


def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _load_json(path: Path) -> Optional[Dict]:
    try:
        if path.exists():
            return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        pass
    return None


def _save_json(path: Path, data: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str))


def _load_checkpoint_index() -> Dict:
    return _load_json(CHECKPOINT_INDEX) or {"checkpoints": []}


def _load_checkpoint_metadata(checkpoint_id: str) -> Optional[Dict]:
    meta_path = CHECKPOINT_DIR / checkpoint_id / "metadata.json"
    return _load_json(meta_path)


def _classify_file(filename: str) -> str:
    """Classify a file into a pillar based on naming patterns."""
    name_lower = filename.lower()
    for pillar, patterns in PILLAR_FILE_PATTERNS.items():
        if any(p in name_lower for p in patterns):
            return pillar
    return "general"


def _compute_file_diff(manifest_a: Dict, manifest_b: Dict) -> Dict:
    """Compute detailed diff between two file manifests."""
    all_files = set(list(manifest_a.keys()) + list(manifest_b.keys()))
    added, removed, modified, unchanged = [], [], [], []
    size_delta = 0

    for f in sorted(all_files):
        in_a = f in manifest_a
        in_b = f in manifest_b
        if in_a and not in_b:
            removed.append({"file": f, "size": manifest_a[f].get("size", 0)})
            size_delta -= manifest_a[f].get("size", 0)
        elif not in_a and in_b:
            added.append({"file": f, "size": manifest_b[f].get("size", 0)})
            size_delta += manifest_b[f].get("size", 0)
        elif manifest_a[f].get("hash") != manifest_b[f].get("hash"):
            delta = manifest_b[f].get("size", 0) - manifest_a[f].get("size", 0)
            modified.append({
                "file": f,
                "size_before": manifest_a[f].get("size", 0),
                "size_after": manifest_b[f].get("size", 0),
                "size_delta": delta,
                "pillar": _classify_file(f),
            })
            size_delta += delta
        else:
            unchanged.append(f)

    return {
        "added": added,
        "removed": removed,
        "modified": modified,
        "unchanged_count": len(unchanged),
        "size_delta": size_delta,
        "files_delta": len(added) - len(removed),
    }


def _compute_progress_score(diff: Dict, meta_a: Dict, meta_b: Dict) -> Dict:
    """Compute a progress score (0-100) from checkpoint diff."""
    scores = {}

    # Data growth score (0-25) - more data = more learned
    total_a = meta_a.get("total_size_bytes", 0) or 1
    total_b = meta_b.get("total_size_bytes", 0) or 0
    growth_ratio = (total_b - total_a) / total_a if total_a > 0 else 0
    # Cap growth contribution at 25 points
    growth_score = min(25, max(0, int(growth_ratio * 100)))
    scores["data_growth"] = growth_score

    # File diversity score (0-25) - more files = more capabilities
    files_a = meta_a.get("files_count", 0) or 1
    files_b = meta_b.get("files_count", 0) or 0
    file_growth = (files_b - files_a) / files_a if files_a > 0 else 0
    diversity_score = min(25, max(0, int(file_growth * 50)))
    scores["capability_diversity"] = diversity_score

    # Modification activity (0-25) - active modification = active learning
    mod_count = len(diff.get("modified", []))
    activity_score = min(25, mod_count * 5)
    scores["modification_activity"] = activity_score

    # No regressions bonus (0-25) - not losing data is good
    removed_count = len(diff.get("removed", []))
    regression_penalty = min(25, removed_count * 5)
    stability_score = 25 - regression_penalty
    scores["stability"] = stability_score

    total = sum(scores.values())
    scores["total"] = total
    scores["grade"] = (
        "A" if total >= 80 else
        "B" if total >= 60 else
        "C" if total >= 40 else
        "D" if total >= 20 else
        "F"
    )

    return scores


def _detect_regressions(diff: Dict) -> List[Dict]:
    """Detect state regressions from a checkpoint diff."""
    regressions = []

    # Removed files = lost data
    for item in diff.get("removed", []):
        regressions.append({
            "type": "file_removed",
            "file": item["file"],
            "severity": "high" if item.get("size", 0) > 1000 else "medium",
            "description": f"Data file '{item['file']}' was removed ({item.get('size', 0)} bytes lost)",
        })

    # Modified files that shrunk significantly
    for item in diff.get("modified", []):
        if item.get("size_delta", 0) < 0:
            shrink_pct = abs(item["size_delta"]) / max(item.get("size_before", 1), 1) * 100
            if shrink_pct > 20:
                regressions.append({
                    "type": "data_shrinkage",
                    "file": item["file"],
                    "severity": "high" if shrink_pct > 50 else "medium",
                    "shrink_percent": round(shrink_pct, 1),
                    "description": f"'{item['file']}' shrunk by {round(shrink_pct, 1)}% ({item['size_delta']} bytes)",
                })

    return regressions


class CheckpointComparisonAnalyticsSkill(Skill):
    """
    Track progress across checkpoints with diff analysis, trend detection,
    and progress scoring for quantitative self-awareness.
    """

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        DATA_DIR.mkdir(parents=True, exist_ok=True)

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="checkpoint_comparison",
            name="Checkpoint Comparison Analytics",
            version="1.0.0",
            category="analytics",
            description=(
                "Analyze progress across checkpoints with diff analysis, "
                "trend detection, progress scoring, and regression detection. "
                "Turns raw checkpoint snapshots into actionable progress intelligence."
            ),
            actions=[
                SkillAction(
                    name="compare",
                    description="Deep comparison of two checkpoints with semantic analysis",
                    parameters={
                        "checkpoint_a": {"type": "string", "required": True, "description": "First checkpoint ID"},
                        "checkpoint_b": {"type": "string", "required": True, "description": "Second checkpoint ID"},
                    },
                ),
                SkillAction(
                    name="timeline",
                    description="Show agent evolution across all checkpoints",
                    parameters={
                        "limit": {"type": "int", "required": False, "description": "Max checkpoints to include"},
                    },
                ),
                SkillAction(
                    name="trends",
                    description="Detect growth/shrinkage patterns across checkpoint series",
                    parameters={
                        "metric": {"type": "string", "required": False, "description": "Metric to analyze: size, files, pillar (default: all)"},
                        "last_n": {"type": "int", "required": False, "description": "Analyze last N checkpoints (default: 10)"},
                    },
                ),
                SkillAction(
                    name="progress_score",
                    description="Compute progress score between two checkpoints",
                    parameters={
                        "checkpoint_a": {"type": "string", "required": True, "description": "Earlier checkpoint ID"},
                        "checkpoint_b": {"type": "string", "required": True, "description": "Later checkpoint ID"},
                    },
                ),
                SkillAction(
                    name="regressions",
                    description="Detect state regressions across recent checkpoints",
                    parameters={
                        "last_n": {"type": "int", "required": False, "description": "Check last N checkpoints (default: 5)"},
                    },
                ),
                SkillAction(
                    name="pillar_health",
                    description="Per-pillar health from checkpoint data evolution",
                    parameters={
                        "checkpoint_a": {"type": "string", "required": False, "description": "Baseline checkpoint (default: earliest)"},
                        "checkpoint_b": {"type": "string", "required": False, "description": "Current checkpoint (default: latest)"},
                    },
                ),
                SkillAction(
                    name="report",
                    description="Full progress report combining all analytics",
                    parameters={
                        "last_n": {"type": "int", "required": False, "description": "Include last N checkpoints (default: 10)"},
                    },
                ),
                SkillAction(
                    name="status",
                    description="Show analytics health and last analysis state",
                    parameters={},
                ),
            ],
            required_credentials=[],
        )

    async def execute(self, action: str, params: Dict[str, Any] = None) -> SkillResult:
        params = params or {}
        handlers = {
            "compare": self._compare,
            "timeline": self._timeline,
            "trends": self._trends,
            "progress_score": self._progress_score,
            "regressions": self._regressions,
            "pillar_health": self._pillar_health,
            "report": self._report,
            "status": self._status,
        }
        handler = handlers.get(action)
        if not handler:
            return SkillResult(success=False, message=f"Unknown action: {action}")
        try:
            return await handler(params)
        except Exception as e:
            return SkillResult(success=False, message=f"Checkpoint comparison {action} failed: {e}")

    def _get_sorted_checkpoints(self, limit: int = 50) -> List[Dict]:
        """Get checkpoints sorted by creation time (oldest first)."""
        index = _load_checkpoint_index()
        checkpoints = index.get("checkpoints", [])
        checkpoints = sorted(checkpoints, key=lambda x: x.get("created_at", ""))
        return checkpoints[-limit:] if limit else checkpoints

    def _get_manifest(self, checkpoint_id: str) -> Optional[Dict]:
        """Get file manifest for a checkpoint."""
        meta = _load_checkpoint_metadata(checkpoint_id)
        if meta:
            return meta.get("file_manifest", {})
        return None

    async def _compare(self, params: Dict) -> SkillResult:
        """Deep comparison of two checkpoints with semantic analysis."""
        cp_a = params.get("checkpoint_a", "")
        cp_b = params.get("checkpoint_b", "")

        if not cp_a or not cp_b:
            return SkillResult(success=False, message="Both checkpoint_a and checkpoint_b required")

        meta_a = _load_checkpoint_metadata(cp_a)
        meta_b = _load_checkpoint_metadata(cp_b)

        if not meta_a:
            return SkillResult(success=False, message=f"Checkpoint '{cp_a}' not found")
        if not meta_b:
            return SkillResult(success=False, message=f"Checkpoint '{cp_b}' not found")

        manifest_a = meta_a.get("file_manifest", {})
        manifest_b = meta_b.get("file_manifest", {})

        diff = _compute_file_diff(manifest_a, manifest_b)
        score = _compute_progress_score(diff, meta_a, meta_b)
        regressions = _detect_regressions(diff)

        # Classify changes by pillar
        pillar_changes = defaultdict(lambda: {"added": 0, "removed": 0, "modified": 0, "size_delta": 0})
        for item in diff.get("added", []):
            pillar = _classify_file(item["file"])
            pillar_changes[pillar]["added"] += 1
            pillar_changes[pillar]["size_delta"] += item.get("size", 0)
        for item in diff.get("removed", []):
            pillar = _classify_file(item["file"])
            pillar_changes[pillar]["removed"] += 1
            pillar_changes[pillar]["size_delta"] -= item.get("size", 0)
        for item in diff.get("modified", []):
            pillar = item.get("pillar", "general")
            pillar_changes[pillar]["modified"] += 1
            pillar_changes[pillar]["size_delta"] += item.get("size_delta", 0)

        # Record in analytics history
        analytics = _load_json(ANALYTICS_FILE) or {"comparisons": [], "last_report": None}
        analytics["comparisons"].append({
            "checkpoint_a": cp_a,
            "checkpoint_b": cp_b,
            "timestamp": _now_iso(),
            "score": score["total"],
            "grade": score["grade"],
            "regressions_count": len(regressions),
        })
        analytics["comparisons"] = analytics["comparisons"][-100:]  # Keep last 100
        _save_json(ANALYTICS_FILE, analytics)

        time_a = meta_a.get("created_at", "unknown")
        time_b = meta_b.get("created_at", "unknown")

        return SkillResult(
            success=True,
            message=(
                f"Comparison {cp_a} → {cp_b}: "
                f"{len(diff['added'])} added, {len(diff['removed'])} removed, "
                f"{len(diff['modified'])} modified | "
                f"Score: {score['total']}/100 ({score['grade']}) | "
                f"{len(regressions)} regressions"
            ),
            data={
                "checkpoint_a": {"id": cp_a, "created_at": time_a, "files": meta_a.get("files_count", 0), "size": meta_a.get("total_size_bytes", 0)},
                "checkpoint_b": {"id": cp_b, "created_at": time_b, "files": meta_b.get("files_count", 0), "size": meta_b.get("total_size_bytes", 0)},
                "diff": diff,
                "progress_score": score,
                "regressions": regressions,
                "pillar_changes": dict(pillar_changes),
            },
        )

    async def _timeline(self, params: Dict) -> SkillResult:
        """Show agent evolution across all checkpoints."""
        limit = params.get("limit", 20)
        checkpoints = self._get_sorted_checkpoints(limit)

        if not checkpoints:
            return SkillResult(success=True, message="No checkpoints found", data={"timeline": []})

        timeline = []
        prev_meta = None
        for cp in checkpoints:
            meta = _load_checkpoint_metadata(cp["id"])
            entry = {
                "id": cp["id"],
                "label": cp.get("label", ""),
                "created_at": cp.get("created_at", ""),
                "files_count": cp.get("files_count", 0),
                "total_size_bytes": cp.get("total_size_bytes", 0),
                "reason": cp.get("reason", ""),
            }

            # Compare with previous checkpoint
            if prev_meta and meta:
                manifest_a = prev_meta.get("file_manifest", {})
                manifest_b = meta.get("file_manifest", {})
                diff = _compute_file_diff(manifest_a, manifest_b)
                entry["delta"] = {
                    "files_added": len(diff["added"]),
                    "files_removed": len(diff["removed"]),
                    "files_modified": len(diff["modified"]),
                    "size_delta": diff["size_delta"],
                }
            else:
                entry["delta"] = None

            timeline.append(entry)
            if meta:
                prev_meta = meta

        # Compute overall evolution
        first = checkpoints[0]
        last = checkpoints[-1]
        total_growth = (last.get("total_size_bytes", 0) or 0) - (first.get("total_size_bytes", 0) or 0)
        file_growth = (last.get("files_count", 0) or 0) - (first.get("files_count", 0) or 0)

        return SkillResult(
            success=True,
            message=f"Timeline: {len(timeline)} checkpoints | Size growth: {total_growth} bytes | File growth: {file_growth}",
            data={
                "timeline": timeline,
                "summary": {
                    "checkpoint_count": len(timeline),
                    "time_span_from": first.get("created_at", ""),
                    "time_span_to": last.get("created_at", ""),
                    "total_size_growth": total_growth,
                    "total_file_growth": file_growth,
                },
            },
        )

    async def _trends(self, params: Dict) -> SkillResult:
        """Detect growth/shrinkage patterns across checkpoint series."""
        last_n = params.get("last_n", 10)
        metric = params.get("metric", "all")

        checkpoints = self._get_sorted_checkpoints(last_n)
        if len(checkpoints) < 2:
            return SkillResult(success=True, message="Need at least 2 checkpoints for trend analysis", data={"trends": {}})

        # Collect time-series data
        sizes = []
        file_counts = []
        pillar_sizes = defaultdict(list)

        for cp in checkpoints:
            sizes.append(cp.get("total_size_bytes", 0) or 0)
            file_counts.append(cp.get("files_count", 0) or 0)

            meta = _load_checkpoint_metadata(cp["id"])
            if meta:
                manifest = meta.get("file_manifest", {})
                pillar_totals = defaultdict(int)
                for fname, finfo in manifest.items():
                    pillar = _classify_file(fname)
                    pillar_totals[pillar] += finfo.get("size", 0)
                for pillar, total in pillar_totals.items():
                    pillar_sizes[pillar].append(total)

        def _trend_direction(series: List[int]) -> str:
            if len(series) < 2:
                return "unknown"
            diffs = [series[i+1] - series[i] for i in range(len(series)-1)]
            positive = sum(1 for d in diffs if d > 0)
            negative = sum(1 for d in diffs if d < 0)
            if positive > negative * 2:
                return "growing"
            elif negative > positive * 2:
                return "shrinking"
            elif positive > negative:
                return "slightly_growing"
            elif negative > positive:
                return "slightly_shrinking"
            return "stable"

        def _avg_change(series: List[int]) -> float:
            if len(series) < 2:
                return 0.0
            diffs = [series[i+1] - series[i] for i in range(len(series)-1)]
            return round(sum(diffs) / len(diffs), 1)

        trends = {}

        if metric in ("all", "size"):
            trends["size"] = {
                "direction": _trend_direction(sizes),
                "avg_change_per_checkpoint": _avg_change(sizes),
                "min": min(sizes) if sizes else 0,
                "max": max(sizes) if sizes else 0,
                "latest": sizes[-1] if sizes else 0,
                "series": sizes,
            }

        if metric in ("all", "files"):
            trends["files"] = {
                "direction": _trend_direction(file_counts),
                "avg_change_per_checkpoint": _avg_change(file_counts),
                "min": min(file_counts) if file_counts else 0,
                "max": max(file_counts) if file_counts else 0,
                "latest": file_counts[-1] if file_counts else 0,
                "series": file_counts,
            }

        if metric in ("all", "pillar"):
            for pillar, series in pillar_sizes.items():
                trends[f"pillar_{pillar}"] = {
                    "direction": _trend_direction(series),
                    "avg_change_per_checkpoint": _avg_change(series),
                    "latest": series[-1] if series else 0,
                    "series": series,
                }

        # Summarize
        growing = [k for k, v in trends.items() if "growing" in v.get("direction", "")]
        shrinking = [k for k, v in trends.items() if "shrinking" in v.get("direction", "")]

        return SkillResult(
            success=True,
            message=f"Trends across {len(checkpoints)} checkpoints: {len(growing)} growing, {len(shrinking)} shrinking",
            data={
                "trends": trends,
                "growing_metrics": growing,
                "shrinking_metrics": shrinking,
                "checkpoints_analyzed": len(checkpoints),
            },
        )

    async def _progress_score(self, params: Dict) -> SkillResult:
        """Compute progress score between two checkpoints."""
        cp_a = params.get("checkpoint_a", "")
        cp_b = params.get("checkpoint_b", "")

        if not cp_a or not cp_b:
            return SkillResult(success=False, message="Both checkpoint_a and checkpoint_b required")

        meta_a = _load_checkpoint_metadata(cp_a)
        meta_b = _load_checkpoint_metadata(cp_b)

        if not meta_a:
            return SkillResult(success=False, message=f"Checkpoint '{cp_a}' not found")
        if not meta_b:
            return SkillResult(success=False, message=f"Checkpoint '{cp_b}' not found")

        manifest_a = meta_a.get("file_manifest", {})
        manifest_b = meta_b.get("file_manifest", {})

        diff = _compute_file_diff(manifest_a, manifest_b)
        score = _compute_progress_score(diff, meta_a, meta_b)

        return SkillResult(
            success=True,
            message=f"Progress {cp_a} → {cp_b}: {score['total']}/100 ({score['grade']})",
            data={
                "score": score,
                "checkpoint_a": cp_a,
                "checkpoint_b": cp_b,
                "size_a": meta_a.get("total_size_bytes", 0),
                "size_b": meta_b.get("total_size_bytes", 0),
                "files_a": meta_a.get("files_count", 0),
                "files_b": meta_b.get("files_count", 0),
            },
        )

    async def _regressions(self, params: Dict) -> SkillResult:
        """Detect state regressions across recent checkpoints."""
        last_n = params.get("last_n", 5)
        checkpoints = self._get_sorted_checkpoints(last_n)

        if len(checkpoints) < 2:
            return SkillResult(success=True, message="Need at least 2 checkpoints", data={"regressions": []})

        all_regressions = []
        for i in range(len(checkpoints) - 1):
            cp_a = checkpoints[i]
            cp_b = checkpoints[i + 1]

            meta_a = _load_checkpoint_metadata(cp_a["id"])
            meta_b = _load_checkpoint_metadata(cp_b["id"])

            if not meta_a or not meta_b:
                continue

            manifest_a = meta_a.get("file_manifest", {})
            manifest_b = meta_b.get("file_manifest", {})

            diff = _compute_file_diff(manifest_a, manifest_b)
            regs = _detect_regressions(diff)

            for r in regs:
                r["between"] = f"{cp_a['id']} → {cp_b['id']}"
                r["timestamp"] = cp_b.get("created_at", "")

            all_regressions.extend(regs)

        # Sort by severity
        severity_order = {"high": 0, "medium": 1, "low": 2}
        all_regressions.sort(key=lambda r: severity_order.get(r.get("severity", "low"), 2))

        high_count = sum(1 for r in all_regressions if r.get("severity") == "high")
        medium_count = sum(1 for r in all_regressions if r.get("severity") == "medium")

        return SkillResult(
            success=True,
            message=f"Found {len(all_regressions)} regressions across {len(checkpoints)} checkpoints ({high_count} high, {medium_count} medium)",
            data={
                "regressions": all_regressions,
                "high_severity": high_count,
                "medium_severity": medium_count,
                "checkpoints_analyzed": len(checkpoints),
            },
        )

    async def _pillar_health(self, params: Dict) -> SkillResult:
        """Per-pillar health from checkpoint data evolution."""
        checkpoints = self._get_sorted_checkpoints(50)

        if not checkpoints:
            return SkillResult(success=True, message="No checkpoints found", data={"pillars": {}})

        cp_a_id = params.get("checkpoint_a", checkpoints[0]["id"])
        cp_b_id = params.get("checkpoint_b", checkpoints[-1]["id"])

        meta_a = _load_checkpoint_metadata(cp_a_id)
        meta_b = _load_checkpoint_metadata(cp_b_id)

        if not meta_a or not meta_b:
            return SkillResult(success=False, message="Could not load checkpoint metadata")

        manifest_a = meta_a.get("file_manifest", {})
        manifest_b = meta_b.get("file_manifest", {})

        # Compute per-pillar stats
        pillars = {}
        for pillar_name in ["self_improvement", "revenue", "replication", "goal_setting"]:
            # Count files in each pillar
            files_a = {f: info for f, info in manifest_a.items() if _classify_file(f) == pillar_name}
            files_b = {f: info for f, info in manifest_b.items() if _classify_file(f) == pillar_name}

            size_a = sum(info.get("size", 0) for info in files_a.values())
            size_b = sum(info.get("size", 0) for info in files_b.values())

            # Health score: combination of presence, growth, and breadth
            presence_score = min(30, len(files_b) * 10)  # Up to 30 for having files
            growth_score = 0
            if size_a > 0:
                growth_ratio = (size_b - size_a) / size_a
                growth_score = min(40, max(0, int(growth_ratio * 100)))  # Up to 40 for growth
            elif size_b > 0:
                growth_score = 40  # New data = maximum growth
            stability = 30 if size_b >= size_a else max(0, 30 - int((size_a - size_b) / max(size_a, 1) * 30))

            health = presence_score + growth_score + stability

            pillars[pillar_name] = {
                "health_score": min(100, health),
                "files_before": len(files_a),
                "files_after": len(files_b),
                "size_before": size_a,
                "size_after": size_b,
                "size_growth": size_b - size_a,
                "file_growth": len(files_b) - len(files_a),
            }

        # Overall health
        avg_health = round(sum(p["health_score"] for p in pillars.values()) / max(len(pillars), 1), 1)
        weakest = min(pillars, key=lambda k: pillars[k]["health_score"])
        strongest = max(pillars, key=lambda k: pillars[k]["health_score"])

        return SkillResult(
            success=True,
            message=f"Pillar health: avg={avg_health}/100 | strongest={strongest} | weakest={weakest}",
            data={
                "pillars": pillars,
                "average_health": avg_health,
                "weakest_pillar": weakest,
                "strongest_pillar": strongest,
                "baseline": cp_a_id,
                "current": cp_b_id,
            },
        )

    async def _report(self, params: Dict) -> SkillResult:
        """Full progress report combining all analytics."""
        last_n = params.get("last_n", 10)

        # Get timeline
        timeline_result = await self._timeline({"limit": last_n})
        # Get trends
        trends_result = await self._trends({"last_n": last_n})
        # Get regressions
        regressions_result = await self._regressions({"last_n": last_n})
        # Get pillar health
        pillar_result = await self._pillar_health(params)

        # Compute overall score from latest two checkpoints
        checkpoints = self._get_sorted_checkpoints(last_n)
        overall_score = None
        if len(checkpoints) >= 2:
            first = checkpoints[0]
            last = checkpoints[-1]
            meta_first = _load_checkpoint_metadata(first["id"])
            meta_last = _load_checkpoint_metadata(last["id"])
            if meta_first and meta_last:
                diff = _compute_file_diff(
                    meta_first.get("file_manifest", {}),
                    meta_last.get("file_manifest", {}),
                )
                overall_score = _compute_progress_score(diff, meta_first, meta_last)

        # Build summary
        growing = trends_result.data.get("growing_metrics", [])
        shrinking = trends_result.data.get("shrinking_metrics", [])
        reg_count = regressions_result.data.get("high_severity", 0) + regressions_result.data.get("medium_severity", 0)
        avg_health = pillar_result.data.get("average_health", 0) if pillar_result.success else 0

        # Save report
        analytics = _load_json(ANALYTICS_FILE) or {"comparisons": [], "last_report": None}
        analytics["last_report"] = {
            "timestamp": _now_iso(),
            "checkpoints_analyzed": len(checkpoints),
            "overall_score": overall_score["total"] if overall_score else None,
            "regressions": reg_count,
            "avg_pillar_health": avg_health,
        }
        _save_json(ANALYTICS_FILE, analytics)

        grade = overall_score["grade"] if overall_score else "N/A"
        score_val = overall_score["total"] if overall_score else 0

        return SkillResult(
            success=True,
            message=(
                f"Progress Report: {len(checkpoints)} checkpoints | "
                f"Score: {score_val}/100 ({grade}) | "
                f"Pillar health: {avg_health}/100 | "
                f"{len(growing)} growing, {len(shrinking)} shrinking | "
                f"{reg_count} regressions"
            ),
            data={
                "overall_score": overall_score,
                "timeline_summary": timeline_result.data.get("summary", {}),
                "trends": trends_result.data.get("trends", {}),
                "regressions": regressions_result.data.get("regressions", []),
                "pillar_health": pillar_result.data.get("pillars", {}),
                "weakest_pillar": pillar_result.data.get("weakest_pillar", ""),
                "strongest_pillar": pillar_result.data.get("strongest_pillar", ""),
                "growing_metrics": growing,
                "shrinking_metrics": shrinking,
            },
        )

    async def _status(self, params: Dict) -> SkillResult:
        """Show analytics health and last analysis state."""
        analytics = _load_json(ANALYTICS_FILE) or {"comparisons": [], "last_report": None}
        index = _load_checkpoint_index()
        checkpoint_count = len(index.get("checkpoints", []))

        return SkillResult(
            success=True,
            message=f"Checkpoint analytics: {checkpoint_count} checkpoints, {len(analytics.get('comparisons', []))} comparisons recorded",
            data={
                "checkpoint_count": checkpoint_count,
                "comparisons_recorded": len(analytics.get("comparisons", [])),
                "last_report": analytics.get("last_report"),
                "analytics_file": str(ANALYTICS_FILE),
            },
        )
