#!/usr/bin/env python3
"""
Result Feedback - Rich action result formatting for LLM context.

Currently the agent only sees "tool: status" for previous actions.
This module formats full action results so the LLM can reason about
what actually happened - seeing file contents, command outputs,
error messages, API responses, etc.

This is critical for multi-step reasoning: the agent needs to see
the result of step 1 to decide what to do in step 2.
"""

import json
from typing import Any, Dict, List, Optional


def _truncate(text: str, max_len: int = 500) -> str:
    """Truncate text with ellipsis indicator."""
    if len(text) <= max_len:
        return text
    half = (max_len - 20) // 2
    return text[:half] + f"\n... [{len(text) - max_len} chars truncated] ...\n" + text[-half:]


def _format_data(data: Any, max_len: int = 500) -> str:
    """Format result data for display, handling various types."""
    if data is None:
        return ""
    if isinstance(data, str):
        return _truncate(data, max_len)
    if isinstance(data, dict):
        # For dicts, show key-value pairs, prioritizing important keys
        priority_keys = ["content", "output", "stdout", "result", "text", "message",
                         "path", "url", "status", "error", "count", "items"]
        lines = []
        seen = set()

        # Show priority keys first
        for key in priority_keys:
            if key in data:
                val = data[key]
                if isinstance(val, str) and len(val) > 200:
                    val = _truncate(val, 200)
                elif isinstance(val, (dict, list)):
                    val = _truncate(json.dumps(val, default=str), 200)
                lines.append(f"    {key}: {val}")
                seen.add(key)

        # Then show remaining keys (truncated)
        for key, val in data.items():
            if key in seen:
                continue
            if isinstance(val, str) and len(val) > 100:
                val = _truncate(val, 100)
            elif isinstance(val, (dict, list)):
                val = _truncate(json.dumps(val, default=str), 100)
            lines.append(f"    {key}: {val}")

        result = "\n".join(lines)
        return _truncate(result, max_len)

    if isinstance(data, list):
        if len(data) == 0:
            return "    (empty list)"
        items = []
        for i, item in enumerate(data[:5]):
            items.append(f"    [{i}]: {_truncate(str(item), 100)}")
        if len(data) > 5:
            items.append(f"    ... and {len(data) - 5} more items")
        return "\n".join(items)

    return _truncate(str(data), max_len)


def format_action_result(action: Dict, max_result_len: int = 500) -> str:
    """
    Format a single action record for LLM context.

    Args:
        action: Dict with keys: tool, params, result, cycle, api_cost_usd, tokens
        max_result_len: Max chars for result data

    Returns:
        Formatted string showing tool, status, and key result data
    """
    tool = action.get("tool", "unknown")
    result = action.get("result", {})
    status = result.get("status", "unknown")
    cycle = action.get("cycle", "?")

    # Status indicator
    icon = "✓" if status == "success" else "✗" if status in ("error", "failed") else "○"

    lines = [f"  {icon} [{cycle}] {tool} → {status}"]

    # Show error/message if present
    message = result.get("message", "")
    if message and status in ("error", "failed"):
        lines.append(f"    error: {_truncate(message, 200)}")
    elif message and status == "success":
        lines.append(f"    message: {_truncate(message, 150)}")

    # Show result data for successful actions
    data = result.get("data")
    if data is not None:
        formatted = _format_data(data, max_result_len)
        if formatted.strip():
            lines.append(f"    data:\n{formatted}")

    return "\n".join(lines)


def format_recent_actions(
    actions: List[Dict],
    max_actions: int = 5,
    max_total_len: int = 2000,
    max_per_result: int = 400,
) -> str:
    """
    Format recent actions for inclusion in the LLM prompt.

    Shows full results for the most recent actions so the LLM can
    reason about what happened. Older actions get progressively
    less detail.

    Args:
        actions: List of action records (most recent last)
        max_actions: Max number of actions to show
        max_total_len: Max total length of formatted output
        max_per_result: Max chars per individual result

    Returns:
        Formatted string for prompt inclusion
    """
    if not actions:
        return ""

    recent = actions[-max_actions:]
    formatted_parts = []

    for i, action in enumerate(recent):
        # Most recent actions get more detail
        is_latest = (i == len(recent) - 1)
        is_recent = (i >= len(recent) - 2)

        if is_latest:
            result_len = max_per_result
        elif is_recent:
            result_len = max_per_result // 2
        else:
            result_len = 100

        formatted = format_action_result(action, max_result_len=result_len)
        formatted_parts.append(formatted)

    result = "\nRecent actions (oldest to newest):\n" + "\n".join(formatted_parts)

    # Final truncation if still too long
    if len(result) > max_total_len:
        result = result[:max_total_len - 50] + "\n  ... (earlier actions truncated)"

    return result
