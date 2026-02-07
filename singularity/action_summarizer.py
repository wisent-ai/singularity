"""
ActionSummarizer - Enriches the LLM prompt with intelligent action history analysis.

Instead of showing bare "tool: status" lines, this module produces:
1. Detailed action summaries with error messages and relevant params
2. Failure streak detection (warns when consecutive failures occur)
3. Loop detection (warns when the same action is repeated)
4. Per-tool success rates from recent history
5. Contextual warnings and suggestions

This directly improves the agent's decision-making by giving the LLM
better information about what has happened and what's going wrong.

Pillar: Self-Improvement (act → measure → adapt feedback loop)
"""

from collections import Counter
from typing import Dict, List, Optional


def summarize_actions(recent_actions: List[Dict], max_display: int = 8) -> str:
    """
    Produce a rich summary of recent actions for the LLM prompt.

    Args:
        recent_actions: List of action dicts from the agent loop.
            Each has: tool, params, result (dict with status/message/data),
            api_cost_usd, tokens, cycle
        max_display: Maximum number of recent actions to show in detail.

    Returns:
        Formatted string to inject into the LLM prompt.
    """
    if not recent_actions:
        return ""

    sections = []

    # Section 1: Recent action details
    display_actions = recent_actions[-max_display:]
    action_lines = []
    for a in display_actions:
        line = _format_action(a)
        action_lines.append(line)

    sections.append("Recent actions:\n" + "\n".join(action_lines))

    # Section 2: Warnings (loops, streaks, failing tools)
    warnings = _detect_warnings(recent_actions)
    if warnings:
        sections.append("⚠ WARNINGS:\n" + "\n".join(f"- {w}" for w in warnings))

    # Section 3: Tool success rates (if enough history)
    if len(recent_actions) >= 3:
        stats = _tool_stats(recent_actions)
        if stats:
            sections.append("Tool stats:\n" + "\n".join(f"- {s}" for s in stats))

    return "\n\n".join(sections)


def _format_action(action: Dict) -> str:
    """Format a single action with useful detail."""
    tool = action.get("tool", "unknown")
    result = action.get("result", {})
    status = result.get("status", "unknown")
    cycle = action.get("cycle", "?")

    # Build the main line
    parts = [f"  [{cycle}] {tool} → {status}"]

    # Add error message for failures
    if status in ("error", "failed"):
        msg = result.get("message", "")
        if msg:
            # Truncate long error messages
            msg = msg[:120] + "..." if len(msg) > 120 else msg
            parts.append(f"       Error: {msg}")

    # Add key params (exclude very long values)
    params = action.get("params", {})
    if params:
        param_summary = _summarize_params(params)
        if param_summary:
            parts.append(f"       Params: {param_summary}")

    # Add success data summary for successful actions
    if status == "success":
        data = result.get("data", {})
        msg = result.get("message", "")
        data_summary = _summarize_result_data(data, msg)
        if data_summary:
            parts.append(f"       Result: {data_summary}")

    return "\n".join(parts)


def _summarize_params(params: Dict, max_len: int = 100) -> str:
    """Summarize action parameters, hiding long values."""
    if not params:
        return ""

    summary_parts = []
    for key, value in params.items():
        if isinstance(value, str) and len(value) > 60:
            summary_parts.append(f"{key}=<{len(value)} chars>")
        elif isinstance(value, (list, dict)):
            summary_parts.append(f"{key}=<{type(value).__name__}>")
        else:
            summary_parts.append(f"{key}={value}")

    result = ", ".join(summary_parts)
    if len(result) > max_len:
        result = result[:max_len] + "..."
    return result


def _summarize_result_data(data: Dict, message: str) -> str:
    """Summarize successful result data."""
    if not data and not message:
        return ""

    # Prefer message if it's concise
    if message and len(message) <= 100:
        return message

    if not data:
        return message[:100] + "..." if message else ""

    # Show key fields from data
    summary_parts = []
    for key in list(data.keys())[:3]:
        val = data[key]
        if isinstance(val, str) and len(val) > 50:
            summary_parts.append(f"{key}=<{len(val)} chars>")
        elif isinstance(val, (list, dict)):
            summary_parts.append(f"{key}=<{type(val).__name__}>")
        else:
            summary_parts.append(f"{key}={val}")

    return ", ".join(summary_parts) if summary_parts else ""


def _detect_warnings(recent_actions: List[Dict]) -> List[str]:
    """Detect problematic patterns in recent action history."""
    warnings = []

    if not recent_actions:
        return warnings

    # 1. Failure streak detection
    streak = _failure_streak(recent_actions)
    if streak >= 3:
        warnings.append(
            f"FAILURE STREAK: Last {streak} actions failed in a row. "
            "Consider a different approach or tool."
        )

    # 2. Loop detection (same tool+params repeated)
    loop_info = _detect_loop(recent_actions)
    if loop_info:
        tool, count = loop_info
        warnings.append(
            f"LOOP DETECTED: '{tool}' called {count} times recently "
            "with similar parameters. Try a different strategy."
        )

    # 3. Same error repeated
    repeated_error = _detect_repeated_errors(recent_actions)
    if repeated_error:
        tool, error, count = repeated_error
        warnings.append(
            f"REPEATED ERROR: '{tool}' has failed {count} times with: {error[:80]}. "
            "This tool may not work in current state."
        )

    # 4. High cost warning
    total_cost = sum(a.get("api_cost_usd", 0) for a in recent_actions[-10:])
    if total_cost > 0.50:
        warnings.append(
            f"HIGH COST: Last {min(10, len(recent_actions))} actions cost "
            f"${total_cost:.4f}. Consider more efficient approaches."
        )

    return warnings


def _failure_streak(recent_actions: List[Dict]) -> int:
    """Count consecutive failures from the end of the action list."""
    streak = 0
    for action in reversed(recent_actions):
        status = action.get("result", {}).get("status", "unknown")
        if status in ("error", "failed"):
            streak += 1
        else:
            break
    return streak


def _detect_loop(recent_actions: List[Dict], window: int = 6) -> Optional[tuple]:
    """
    Detect if the same tool is being called repeatedly.

    Returns (tool_name, count) if a loop is found, None otherwise.
    """
    if len(recent_actions) < 3:
        return None

    window_actions = recent_actions[-window:]
    tool_counts = Counter(a.get("tool", "") for a in window_actions)

    for tool, count in tool_counts.most_common(1):
        if count >= 3 and tool != "wait":
            return (tool, count)

    return None


def _detect_repeated_errors(
    recent_actions: List[Dict], window: int = 8
) -> Optional[tuple]:
    """
    Detect if the same tool is failing with the same error message.

    Returns (tool_name, error_message, count) if found, None otherwise.
    """
    if len(recent_actions) < 2:
        return None

    window_actions = recent_actions[-window:]
    error_tracker: Dict[str, Dict[str, int]] = {}

    for action in window_actions:
        result = action.get("result", {})
        status = result.get("status", "")
        if status not in ("error", "failed"):
            continue

        tool = action.get("tool", "")
        msg = result.get("message", "unknown")
        # Normalize error messages (strip variable parts)
        msg_key = msg[:80]

        if tool not in error_tracker:
            error_tracker[tool] = {}
        error_tracker[tool][msg_key] = error_tracker[tool].get(msg_key, 0) + 1

    # Find the worst offender
    for tool, errors in error_tracker.items():
        for msg, count in errors.items():
            if count >= 2:
                return (tool, msg, count)

    return None


def _tool_stats(recent_actions: List[Dict]) -> List[str]:
    """
    Calculate per-tool success rates from recent history.

    Only shows tools that have been used more than once.
    """
    tool_results: Dict[str, Dict[str, int]] = {}

    for action in recent_actions:
        tool = action.get("tool", "unknown")
        status = action.get("result", {}).get("status", "unknown")
        if tool not in tool_results:
            tool_results[tool] = {"success": 0, "failed": 0, "error": 0, "other": 0}

        if status == "success":
            tool_results[tool]["success"] += 1
        elif status == "failed":
            tool_results[tool]["failed"] += 1
        elif status == "error":
            tool_results[tool]["error"] += 1
        else:
            tool_results[tool]["other"] += 1

    stats = []
    for tool, counts in sorted(tool_results.items()):
        total = sum(counts.values())
        if total < 2:
            continue
        success_rate = counts["success"] / total * 100
        stats.append(f"{tool}: {counts['success']}/{total} succeeded ({success_rate:.0f}%)")

    return stats
