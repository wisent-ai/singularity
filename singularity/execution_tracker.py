"""
ExecutionTracker - Tracks tool execution outcomes for agent self-awareness.

Provides:
- Per-tool success/failure/error counts
- Fuzzy tool name matching when LLM makes typos
- Session-level execution summaries
- Tool health ratings for LLM context
"""

import difflib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple


@dataclass
class ToolStats:
    """Execution statistics for a single tool."""
    success: int = 0
    failed: int = 0
    errors: int = 0
    last_error: str = ""
    last_used: str = ""

    @property
    def total(self) -> int:
        return self.success + self.failed + self.errors

    @property
    def success_rate(self) -> float:
        if self.total == 0:
            return 1.0
        return self.success / self.total

    def record(self, status: str, error_msg: str = ""):
        self.last_used = datetime.now().isoformat()
        if status == "success":
            self.success += 1
        elif status == "error":
            self.errors += 1
            self.last_error = error_msg
        else:
            self.failed += 1
            self.last_error = error_msg


class ExecutionTracker:
    """Tracks execution outcomes and provides intelligent tool resolution."""

    def __init__(self):
        self.tool_stats: Dict[str, ToolStats] = {}
        self.session_start = datetime.now().isoformat()
        self.total_executions = 0
        self.corrections_made = 0

    def record(self, tool: str, status: str, error_msg: str = ""):
        """Record an execution outcome."""
        if tool not in self.tool_stats:
            self.tool_stats[tool] = ToolStats()
        self.tool_stats[tool].record(status, error_msg)
        self.total_executions += 1

    def find_closest_tool(
        self, tool_name: str, available_tools: List[str], cutoff: float = 0.6
    ) -> Optional[str]:
        """Find the closest matching tool name using fuzzy matching.

        Returns the best match if confidence is above cutoff, else None.
        """
        if tool_name in available_tools:
            return tool_name

        # Try direct fuzzy match
        matches = difflib.get_close_matches(tool_name, available_tools, n=1, cutoff=cutoff)
        if matches:
            return matches[0]

        # Try component matching: split skill:action and match each part
        if ":" in tool_name:
            parts = tool_name.split(":", 1)
            skill_id, action_name = parts[0], parts[1]

            # Find matching skills
            skill_ids = set()
            action_map: Dict[str, List[str]] = {}
            for t in available_tools:
                if ":" in t:
                    s, a = t.split(":", 1)
                    skill_ids.add(s)
                    if s not in action_map:
                        action_map[s] = []
                    action_map[s].append(a)

            # Match skill first
            skill_matches = difflib.get_close_matches(
                skill_id, list(skill_ids), n=1, cutoff=0.5
            )
            if skill_matches:
                matched_skill = skill_matches[0]
                # Now match action within that skill
                if matched_skill in action_map:
                    action_matches = difflib.get_close_matches(
                        action_name, action_map[matched_skill], n=1, cutoff=0.5
                    )
                    if action_matches:
                        result = f"{matched_skill}:{action_matches[0]}"
                        if result in available_tools:
                            return result

        return None

    def suggest_tools(self, tool_name: str, available_tools: List[str]) -> List[str]:
        """Suggest similar tool names for error messages."""
        return difflib.get_close_matches(tool_name, available_tools, n=3, cutoff=0.4)

    def get_summary(self) -> Dict:
        """Get execution summary for LLM context."""
        summary = {
            "total_executions": self.total_executions,
            "corrections_made": self.corrections_made,
            "tools_used": {},
        }
        for tool, stats in self.tool_stats.items():
            if stats.total > 0:
                entry = {
                    "calls": stats.total,
                    "success_rate": round(stats.success_rate, 2),
                }
                if stats.last_error:
                    entry["last_error"] = stats.last_error[:100]
                summary["tools_used"][tool] = entry
        return summary

    def get_prompt_context(self) -> str:
        """Generate a brief text summary for inclusion in LLM prompt."""
        if not self.tool_stats:
            return ""

        lines = []
        failing_tools = []
        for tool, stats in self.tool_stats.items():
            if stats.total > 0 and stats.success_rate < 0.5:
                failing_tools.append(
                    f"  - {tool}: {stats.success}/{stats.total} succeeded"
                    + (f" (last error: {stats.last_error[:80]})" if stats.last_error else "")
                )

        if failing_tools:
            lines.append("Tools with low success rate:")
            lines.extend(failing_tools)

        if self.corrections_made > 0:
            lines.append(
                f"Note: {self.corrections_made} tool name(s) were auto-corrected this session."
            )

        return "\n".join(lines)
