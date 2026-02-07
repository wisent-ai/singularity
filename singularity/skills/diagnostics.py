#!/usr/bin/env python3
"""
DiagnosticsSkill - Agent self-awareness and environment monitoring.

Provides the agent with visibility into:
- System resources (CPU, memory, disk)
- Loaded skills and their status
- Environment/credential availability
- Network connectivity
- Runtime health metrics
- Cost tracking summary

Serves: Replication (resource awareness), Self-Improvement (capability awareness),
        Revenue (cost tracking), Goal Setting (constraint awareness).
"""

import os
import sys
import time
import json
import platform
import importlib
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional

from .base import Skill, SkillManifest, SkillAction, SkillResult


# Credential keys the agent might use, mapped to skill names
KNOWN_CREDENTIALS = {
    "TWITTER_API_KEY": "TwitterSkill",
    "TWITTER_API_SECRET": "TwitterSkill",
    "TWITTER_ACCESS_TOKEN": "TwitterSkill",
    "TWITTER_ACCESS_SECRET": "TwitterSkill",
    "GITHUB_TOKEN": "GitHubSkill",
    "ANTHROPIC_API_KEY": "CognitionEngine (Anthropic)",
    "OPENAI_API_KEY": "CognitionEngine (OpenAI)",
    "NAMECHEAP_API_KEY": "NamecheapSkill",
    "NAMECHEAP_USERNAME": "NamecheapSkill",
    "EMAIL_ADDRESS": "EmailSkill",
    "EMAIL_PASSWORD": "EmailSkill",
    "VERCEL_TOKEN": "VercelSkill",
    "CRYPTO_PRIVATE_KEY": "CryptoSkill",
    "CRYPTO_WALLET_ADDRESS": "CryptoSkill",
}

# Python packages needed by various skills
SKILL_DEPENDENCIES = {
    "anthropic": "CognitionEngine (Anthropic provider)",
    "openai": "CognitionEngine (OpenAI provider)",
    "tweepy": "TwitterSkill",
    "fastapi": "Service API / LLM Server",
    "uvicorn": "Service API / LLM Server",
    "selenium": "BrowserSkill",
    "seleniumbase": "BrowserSkill",
    "web3": "CryptoSkill",
    "vllm": "CognitionEngine (vLLM provider)",
    "transformers": "CognitionEngine (Transformers provider)",
    "torch": "SteeringSkill / Local models",
    "wisent": "SteeringSkill",
    "dotenv": "Environment loading",
}


class DiagnosticsSkill(Skill):
    """Provides the agent with self-awareness about its operational state."""

    def __init__(self, credentials: Dict[str, str] = None, registry=None):
        super().__init__(credentials)
        self._registry = registry  # SkillRegistry reference for skill introspection
        self._start_time = time.time()
        self._action_log: List[Dict] = []
        self._health_snapshots: List[Dict] = []
        self.initialized = True

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="diagnostics",
            name="DiagnosticsSkill",
            version="1.0.0",
            category="system",
            description="Agent self-awareness: system resources, skills status, credentials, health monitoring",
            required_credentials=[],
            actions=[
                SkillAction(
                    name="system_info",
                    description="Get system information: OS, Python version, CPU, memory, disk",
                    parameters={},
                    estimated_cost=0,
                ),
                SkillAction(
                    name="check_resources",
                    description="Check current resource usage: CPU load, memory, disk space",
                    parameters={},
                    estimated_cost=0,
                ),
                SkillAction(
                    name="check_credentials",
                    description="Check which API keys and credentials are available (without exposing values)",
                    parameters={},
                    estimated_cost=0,
                ),
                SkillAction(
                    name="check_dependencies",
                    description="Check which Python packages are installed and available",
                    parameters={
                        "packages": {
                            "type": "list",
                            "required": False,
                            "description": "Specific packages to check (default: all known)",
                        }
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="skill_status",
                    description="Report on loaded skills, their actions, and readiness",
                    parameters={},
                    estimated_cost=0,
                ),
                SkillAction(
                    name="health_check",
                    description="Comprehensive health check combining resources, credentials, and skills",
                    parameters={},
                    estimated_cost=0,
                ),
                SkillAction(
                    name="capability_gaps",
                    description="Identify missing capabilities: unmet credentials, missing packages, skill gaps",
                    parameters={},
                    estimated_cost=0,
                ),
                SkillAction(
                    name="runtime_stats",
                    description="Get runtime statistics: uptime, action count, cost tracking",
                    parameters={},
                    estimated_cost=0,
                ),
                SkillAction(
                    name="snapshot",
                    description="Take a health snapshot for tracking changes over time",
                    parameters={
                        "label": {
                            "type": "string",
                            "required": False,
                            "description": "Optional label for this snapshot",
                        }
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="compare_snapshots",
                    description="Compare two health snapshots to see what changed",
                    parameters={
                        "index_a": {
                            "type": "integer",
                            "required": False,
                            "description": "Index of first snapshot (default: -2, second to last)",
                        },
                        "index_b": {
                            "type": "integer",
                            "required": False,
                            "description": "Index of second snapshot (default: -1, last)",
                        },
                    },
                    estimated_cost=0,
                ),
            ],
        )

    async def execute(self, action: str, params: Dict) -> SkillResult:
        self._usage_count += 1
        self._action_log.append({
            "action": action,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        handlers = {
            "system_info": self._system_info,
            "check_resources": self._check_resources,
            "check_credentials": self._check_credentials,
            "check_dependencies": self._check_dependencies,
            "skill_status": self._skill_status,
            "health_check": self._health_check,
            "capability_gaps": self._capability_gaps,
            "runtime_stats": self._runtime_stats,
            "snapshot": self._snapshot,
            "compare_snapshots": self._compare_snapshots,
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
            return SkillResult(success=False, message=f"Diagnostics error: {str(e)}")

    async def _system_info(self, params: Dict) -> SkillResult:
        """Get basic system information."""
        info = {
            "platform": platform.platform(),
            "system": platform.system(),
            "machine": platform.machine(),
            "processor": platform.processor() or "unknown",
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "python_path": sys.executable,
            "cwd": os.getcwd(),
            "pid": os.getpid(),
            "cpu_count": os.cpu_count(),
        }

        # Try to get hostname
        try:
            info["hostname"] = platform.node()
        except Exception:
            info["hostname"] = "unknown"

        return SkillResult(
            success=True,
            message=f"System: {info['system']} {info['machine']}, Python {info['python_version']}, {info['cpu_count']} CPUs",
            data=info,
        )

    async def _check_resources(self, params: Dict) -> SkillResult:
        """Check current system resource usage."""
        resources = {}

        # Memory info
        try:
            with open("/proc/meminfo", "r") as f:
                meminfo = f.read()
            mem_total = mem_available = mem_free = None
            for line in meminfo.split("\n"):
                if line.startswith("MemTotal:"):
                    mem_total = int(line.split()[1]) * 1024  # Convert KB to bytes
                elif line.startswith("MemAvailable:"):
                    mem_available = int(line.split()[1]) * 1024
                elif line.startswith("MemFree:"):
                    mem_free = int(line.split()[1]) * 1024
            if mem_total:
                resources["memory"] = {
                    "total_gb": round(mem_total / (1024**3), 2),
                    "available_gb": round((mem_available or mem_free or 0) / (1024**3), 2),
                    "used_pct": round(
                        (1 - (mem_available or mem_free or 0) / mem_total) * 100, 1
                    ),
                }
        except (FileNotFoundError, PermissionError):
            resources["memory"] = {"status": "unavailable (non-Linux or no /proc)"}

        # Disk info
        try:
            stat = os.statvfs("/")
            disk_total = stat.f_blocks * stat.f_frsize
            disk_free = stat.f_bavail * stat.f_frsize
            resources["disk"] = {
                "total_gb": round(disk_total / (1024**3), 2),
                "free_gb": round(disk_free / (1024**3), 2),
                "used_pct": round((1 - disk_free / disk_total) * 100, 1),
            }
        except (OSError, AttributeError):
            resources["disk"] = {"status": "unavailable"}

        # CPU load average
        try:
            load1, load5, load15 = os.getloadavg()
            cpu_count = os.cpu_count() or 1
            resources["cpu"] = {
                "load_1m": round(load1, 2),
                "load_5m": round(load5, 2),
                "load_15m": round(load15, 2),
                "cores": cpu_count,
                "load_pct_1m": round(load1 / cpu_count * 100, 1),
            }
        except (OSError, AttributeError):
            resources["cpu"] = {"status": "unavailable"}

        # Process count (our own process memory)
        try:
            import resource
            usage = resource.getrusage(resource.RUSAGE_SELF)
            resources["process"] = {
                "max_rss_mb": round(usage.ru_maxrss / 1024, 2),  # KB to MB on Linux
                "user_time_s": round(usage.ru_utime, 2),
                "system_time_s": round(usage.ru_stime, 2),
            }
        except (ImportError, Exception):
            resources["process"] = {"status": "unavailable"}

        # Build summary
        summaries = []
        if "memory" in resources and "used_pct" in resources["memory"]:
            summaries.append(f"Memory: {resources['memory']['used_pct']}% used")
        if "disk" in resources and "free_gb" in resources["disk"]:
            summaries.append(f"Disk: {resources['disk']['free_gb']}GB free")
        if "cpu" in resources and "load_pct_1m" in resources["cpu"]:
            summaries.append(f"CPU: {resources['cpu']['load_pct_1m']}% load")

        return SkillResult(
            success=True,
            message=", ".join(summaries) if summaries else "Resource check complete",
            data=resources,
        )

    async def _check_credentials(self, params: Dict) -> SkillResult:
        """Check which credentials are available without exposing values."""
        available = {}
        missing = {}

        for key, skill_name in KNOWN_CREDENTIALS.items():
            value = os.environ.get(key, "")
            if value:
                # Show first 4 chars + masked rest for confirmation
                masked = value[:4] + "..." + value[-2:] if len(value) > 8 else "***set***"
                available[key] = {"skill": skill_name, "preview": masked}
            else:
                missing[key] = {"skill": skill_name}

        # Group by skill
        skills_ready = set()
        skills_missing = set()
        for info in available.values():
            skills_ready.add(info["skill"])
        for info in missing.values():
            skills_missing.add(info["skill"])

        return SkillResult(
            success=True,
            message=f"{len(available)} credentials available, {len(missing)} missing. "
                    f"Ready skills: {', '.join(sorted(skills_ready)) or 'none'}",
            data={
                "available": available,
                "missing": missing,
                "available_count": len(available),
                "missing_count": len(missing),
                "skills_with_credentials": sorted(skills_ready),
                "skills_missing_credentials": sorted(skills_missing - skills_ready),
            },
        )

    async def _check_dependencies(self, params: Dict) -> SkillResult:
        """Check which Python packages are installed."""
        packages_to_check = params.get("packages") or list(SKILL_DEPENDENCIES.keys())

        installed = {}
        missing = {}

        for pkg in packages_to_check:
            try:
                mod = importlib.import_module(pkg)
                version = getattr(mod, "__version__", "unknown")
                purpose = SKILL_DEPENDENCIES.get(pkg, "unknown")
                installed[pkg] = {"version": version, "purpose": purpose}
            except ImportError:
                purpose = SKILL_DEPENDENCIES.get(pkg, "unknown")
                missing[pkg] = {"purpose": purpose}

        return SkillResult(
            success=True,
            message=f"{len(installed)} packages installed, {len(missing)} missing",
            data={
                "installed": installed,
                "missing": missing,
                "installed_count": len(installed),
                "missing_count": len(missing),
            },
        )

    async def _skill_status(self, params: Dict) -> SkillResult:
        """Report on loaded skills and their readiness."""
        skills_info = []

        if self._registry:
            for skill_id, skill in self._registry._skills.items():
                manifest = skill.manifest
                skills_info.append({
                    "id": manifest.skill_id,
                    "name": manifest.name,
                    "version": manifest.version,
                    "category": manifest.category,
                    "actions": [a.name for a in manifest.actions],
                    "action_count": len(manifest.actions),
                    "required_credentials": manifest.required_credentials,
                    "initialized": skill.initialized,
                    "usage_count": skill._usage_count,
                    "total_cost": skill._total_cost,
                })
        else:
            skills_info = [{"note": "No registry reference provided - cannot inspect skills"}]

        total_actions = sum(s.get("action_count", 0) for s in skills_info if isinstance(s, dict) and "action_count" in s)

        return SkillResult(
            success=True,
            message=f"{len(skills_info)} skills loaded, {total_actions} total actions available",
            data={
                "skills": skills_info,
                "skill_count": len(skills_info),
                "total_actions": total_actions,
            },
        )

    async def _health_check(self, params: Dict) -> SkillResult:
        """Comprehensive health check."""
        results = {}
        issues = []
        score = 100  # Start at 100, deduct for issues

        # System resources
        resource_result = await self._check_resources({})
        results["resources"] = resource_result.data

        if resource_result.data.get("memory", {}).get("used_pct", 0) > 90:
            issues.append("CRITICAL: Memory usage above 90%")
            score -= 30
        elif resource_result.data.get("memory", {}).get("used_pct", 0) > 75:
            issues.append("WARNING: Memory usage above 75%")
            score -= 10

        if resource_result.data.get("disk", {}).get("free_gb", 100) < 1:
            issues.append("CRITICAL: Less than 1GB disk space free")
            score -= 30
        elif resource_result.data.get("disk", {}).get("free_gb", 100) < 5:
            issues.append("WARNING: Less than 5GB disk space free")
            score -= 10

        if resource_result.data.get("cpu", {}).get("load_pct_1m", 0) > 90:
            issues.append("WARNING: CPU load above 90%")
            score -= 15

        # Credentials
        cred_result = await self._check_credentials({})
        results["credentials"] = {
            "available": cred_result.data["available_count"],
            "missing": cred_result.data["missing_count"],
        }

        if cred_result.data["available_count"] == 0:
            issues.append("WARNING: No API credentials configured")
            score -= 15

        # Dependencies
        dep_result = await self._check_dependencies({})
        results["dependencies"] = {
            "installed": dep_result.data["installed_count"],
            "missing": dep_result.data["missing_count"],
        }

        # Skills
        skill_result = await self._skill_status({})
        results["skills"] = {
            "count": skill_result.data.get("skill_count", 0),
            "total_actions": skill_result.data.get("total_actions", 0),
        }

        # Runtime
        uptime = time.time() - self._start_time
        results["runtime"] = {
            "uptime_seconds": round(uptime, 1),
            "diagnostics_actions": len(self._action_log),
        }

        # Determine overall health
        if score >= 90:
            health = "HEALTHY"
        elif score >= 70:
            health = "DEGRADED"
        elif score >= 50:
            health = "UNHEALTHY"
        else:
            health = "CRITICAL"

        results["health"] = health
        results["score"] = max(0, score)
        results["issues"] = issues

        return SkillResult(
            success=True,
            message=f"Health: {health} (score: {max(0, score)}/100). {len(issues)} issues found.",
            data=results,
        )

    async def _capability_gaps(self, params: Dict) -> SkillResult:
        """Identify missing capabilities and how to fill them."""
        gaps = []

        # Check credentials
        cred_result = await self._check_credentials({})
        for key, info in cred_result.data.get("missing", {}).items():
            gaps.append({
                "type": "missing_credential",
                "key": key,
                "blocks": info["skill"],
                "fix": f"Set environment variable {key}",
                "severity": "medium",
            })

        # Check dependencies
        dep_result = await self._check_dependencies({})
        for pkg, info in dep_result.data.get("missing", {}).items():
            gaps.append({
                "type": "missing_package",
                "package": pkg,
                "blocks": info["purpose"],
                "fix": f"pip install {pkg}",
                "severity": "low",
            })

        # Check LLM availability
        has_llm = any(
            os.environ.get(key)
            for key in ["ANTHROPIC_API_KEY", "OPENAI_API_KEY"]
        )
        if not has_llm:
            gaps.append({
                "type": "missing_llm",
                "blocks": "Core cognition - agent cannot think without LLM",
                "fix": "Set ANTHROPIC_API_KEY or OPENAI_API_KEY",
                "severity": "critical",
            })

        # Check revenue capability
        has_crypto = os.environ.get("CRYPTO_WALLET_ADDRESS")
        if not has_crypto:
            gaps.append({
                "type": "missing_payment",
                "blocks": "Revenue collection - agent cannot receive payments",
                "fix": "Set CRYPTO_WALLET_ADDRESS and CRYPTO_PRIVATE_KEY",
                "severity": "medium",
            })

        # Sort by severity
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        gaps.sort(key=lambda g: severity_order.get(g.get("severity", "low"), 3))

        critical = sum(1 for g in gaps if g.get("severity") == "critical")
        high = sum(1 for g in gaps if g.get("severity") == "high")

        return SkillResult(
            success=True,
            message=f"{len(gaps)} capability gaps found ({critical} critical, {high} high)",
            data={
                "gaps": gaps,
                "total": len(gaps),
                "by_severity": {
                    "critical": critical,
                    "high": high,
                    "medium": sum(1 for g in gaps if g.get("severity") == "medium"),
                    "low": sum(1 for g in gaps if g.get("severity") == "low"),
                },
            },
        )

    async def _runtime_stats(self, params: Dict) -> SkillResult:
        """Get runtime statistics."""
        uptime = time.time() - self._start_time

        stats = {
            "uptime_seconds": round(uptime, 1),
            "uptime_human": _format_duration(uptime),
            "diagnostics_actions_run": len(self._action_log),
            "action_history": self._action_log[-20:],  # Last 20 actions
            "snapshots_taken": len(self._health_snapshots),
            "pid": os.getpid(),
            "start_time": datetime.fromtimestamp(
                self._start_time, tz=timezone.utc
            ).isoformat(),
        }

        # Registry stats if available
        if self._registry:
            total_usage = 0
            total_cost = 0.0
            for skill in self._registry._skills.values():
                total_usage += skill._usage_count
                total_cost += skill._total_cost
            stats["total_skill_usage"] = total_usage
            stats["total_skill_cost"] = round(total_cost, 6)

        return SkillResult(
            success=True,
            message=f"Uptime: {stats['uptime_human']}, {len(self._action_log)} diagnostic actions",
            data=stats,
        )

    async def _snapshot(self, params: Dict) -> SkillResult:
        """Take a health snapshot for tracking over time."""
        health_result = await self._health_check({})

        snapshot = {
            "index": len(self._health_snapshots),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "label": params.get("label", f"snapshot_{len(self._health_snapshots)}"),
            "health": health_result.data.get("health"),
            "score": health_result.data.get("score"),
            "resources": health_result.data.get("resources", {}),
            "credentials_available": health_result.data.get("credentials", {}).get("available", 0),
            "skills_count": health_result.data.get("skills", {}).get("count", 0),
            "issues": health_result.data.get("issues", []),
        }

        self._health_snapshots.append(snapshot)

        return SkillResult(
            success=True,
            message=f"Snapshot #{snapshot['index']} taken: {snapshot['health']} (score: {snapshot['score']})",
            data=snapshot,
        )

    async def _compare_snapshots(self, params: Dict) -> SkillResult:
        """Compare two health snapshots."""
        if len(self._health_snapshots) < 2:
            return SkillResult(
                success=False,
                message=f"Need at least 2 snapshots to compare (have {len(self._health_snapshots)}). Take snapshots first with 'snapshot' action.",
            )

        idx_a = params.get("index_a", -2)
        idx_b = params.get("index_b", -1)

        try:
            snap_a = self._health_snapshots[idx_a]
            snap_b = self._health_snapshots[idx_b]
        except IndexError:
            return SkillResult(
                success=False,
                message=f"Invalid snapshot indices. Have {len(self._health_snapshots)} snapshots (0-{len(self._health_snapshots)-1})",
            )

        changes = []
        score_delta = (snap_b.get("score", 0) or 0) - (snap_a.get("score", 0) or 0)
        if score_delta != 0:
            changes.append({
                "field": "health_score",
                "from": snap_a.get("score"),
                "to": snap_b.get("score"),
                "delta": score_delta,
                "direction": "improved" if score_delta > 0 else "degraded",
            })

        if snap_a.get("health") != snap_b.get("health"):
            changes.append({
                "field": "health_status",
                "from": snap_a.get("health"),
                "to": snap_b.get("health"),
            })

        cred_delta = (snap_b.get("credentials_available", 0) or 0) - (snap_a.get("credentials_available", 0) or 0)
        if cred_delta != 0:
            changes.append({
                "field": "credentials_available",
                "from": snap_a.get("credentials_available"),
                "to": snap_b.get("credentials_available"),
                "delta": cred_delta,
            })

        # Compare memory usage
        mem_a = snap_a.get("resources", {}).get("memory", {}).get("used_pct", 0) or 0
        mem_b = snap_b.get("resources", {}).get("memory", {}).get("used_pct", 0) or 0
        if abs(mem_b - mem_a) > 1:
            changes.append({
                "field": "memory_used_pct",
                "from": mem_a,
                "to": mem_b,
                "delta": round(mem_b - mem_a, 1),
                "direction": "increased" if mem_b > mem_a else "decreased",
            })

        new_issues = [i for i in snap_b.get("issues", []) if i not in snap_a.get("issues", [])]
        resolved_issues = [i for i in snap_a.get("issues", []) if i not in snap_b.get("issues", [])]

        return SkillResult(
            success=True,
            message=f"{len(changes)} changes detected between snapshots. "
                    f"Score: {snap_a.get('score')} -> {snap_b.get('score')}",
            data={
                "snapshot_a": {"index": snap_a["index"], "label": snap_a["label"], "timestamp": snap_a["timestamp"]},
                "snapshot_b": {"index": snap_b["index"], "label": snap_b["label"], "timestamp": snap_b["timestamp"]},
                "changes": changes,
                "new_issues": new_issues,
                "resolved_issues": resolved_issues,
                "change_count": len(changes),
            },
        )


def _format_duration(seconds: float) -> str:
    """Format seconds into human-readable duration."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"
