"""
Output Manager - Smart result truncation and action history formatting.

Prevents context window overflow and token waste by:
1. Truncating large skill outputs to a configurable max size
2. Formatting action history with relevant detail levels
3. Detecting and summarizing repetitive outputs
"""

from typing import Dict, List, Any, Optional


# Default max characters for a single result
DEFAULT_MAX_RESULT_CHARS = 2000
# Default max characters for the entire action history block
DEFAULT_MAX_HISTORY_CHARS = 4000


def truncate_result(result: Dict, max_chars: int = DEFAULT_MAX_RESULT_CHARS) -> Dict:
    """
    Truncate a skill execution result to fit within max_chars.

    Preserves status and message, truncates data fields that are too long.
    Adds a truncation notice so the agent knows data was cut.

    Args:
        result: Skill execution result dict
        max_chars: Maximum total characters for the result

    Returns:
        Truncated result dict
    """
    result_str = str(result)
    if len(result_str) <= max_chars:
        return result

    truncated = {
        "status": result.get("status", "unknown"),
        "message": result.get("message", ""),
    }

    data = result.get("data")
    if data is not None:
        data_str = str(data)
        # Calculate how much space we have for data
        overhead = len(str(truncated)) + 50  # room for truncation notice
        available = max(100, max_chars - overhead)

        if len(data_str) > available:
            if isinstance(data, str):
                truncated["data"] = data[:available] + f"\n... [truncated, {len(data)} chars total]"
            elif isinstance(data, dict):
                truncated["data"] = _truncate_dict(data, available)
            elif isinstance(data, list):
                truncated["data"] = _truncate_list(data, available)
            else:
                truncated["data"] = data_str[:available] + f"\n... [truncated]"
        else:
            truncated["data"] = data

    truncated["_truncated"] = True
    truncated["_original_size"] = len(result_str)

    return truncated


def _truncate_dict(d: dict, max_chars: int) -> dict:
    """Truncate a dict, keeping keys but shortening long values."""
    result = {}
    used = 2  # for {}
    for key, value in d.items():
        val_str = str(value)
        remaining = max_chars - used - len(str(key)) - 5  # overhead for key: val,

        if remaining <= 0:
            result["_more_keys"] = f"... {len(d) - len(result)} more keys"
            break

        if len(val_str) > remaining:
            if isinstance(value, str):
                result[key] = value[:remaining] + f"... [{len(value)} chars]"
            elif isinstance(value, (list, dict)):
                result[key] = f"[{type(value).__name__}, {len(val_str)} chars]"
            else:
                result[key] = val_str[:remaining] + "..."
        else:
            result[key] = value

        used += len(str(key)) + len(str(result[key])) + 4

    return result


def _truncate_list(lst: list, max_chars: int) -> list:
    """Truncate a list, keeping first items and noting how many were cut."""
    result = []
    used = 2  # for []
    for i, item in enumerate(lst):
        item_str = str(item)
        remaining = max_chars - used - 4  # overhead for comma, space

        if remaining <= 0 or used > max_chars:
            result.append(f"... [{len(lst) - i} more items]")
            break

        if len(item_str) > remaining:
            if isinstance(item, str):
                result.append(item[:remaining] + "...")
            else:
                result.append(f"[item {i}, {len(item_str)} chars]")
            break
        else:
            result.append(item)
            used += len(item_str) + 2

    return result


def format_action_history(
    recent_actions: List[Dict],
    max_actions: int = 10,
    max_total_chars: int = DEFAULT_MAX_HISTORY_CHARS,
    detail_level: str = "normal",
) -> str:
    """
    Format recent actions into a concise, informative string for LLM context.

    Args:
        recent_actions: List of action dicts from agent
        max_actions: Maximum number of actions to include
        max_total_chars: Maximum total characters for output
        detail_level: "minimal", "normal", or "detailed"

    Returns:
        Formatted string of action history
    """
    if not recent_actions:
        return ""

    actions = recent_actions[-max_actions:]
    lines = []

    for action in actions:
        cycle = action.get("cycle", "?")
        tool = action.get("tool", "unknown")
        result = action.get("result", {})
        status = result.get("status", "unknown")
        params = action.get("params", {})
        cost = action.get("api_cost_usd", 0)

        # Status indicator
        if status == "success":
            indicator = "\u2713"
        elif status == "error" or status == "failed":
            indicator = "\u2717"
        else:
            indicator = "~"

        if detail_level == "minimal":
            lines.append(f"  {indicator} #{cycle} {tool} -> {status}")
        elif detail_level == "normal":
            # Include brief result info
            msg = result.get("message", "")
            if msg:
                msg = msg[:80]
            param_hint = _summarize_params(params)
            line = f"  {indicator} #{cycle} {tool}"
            if param_hint:
                line += f" ({param_hint})"
            line += f" -> {status}"
            if msg:
                line += f": {msg}"
            lines.append(line)
        else:  # detailed
            msg = result.get("message", "")[:120]
            data_summary = _summarize_data(result.get("data"))
            param_hint = _summarize_params(params, max_len=100)
            line = f"  {indicator} #{cycle} {tool}"
            if param_hint:
                line += f"\n    params: {param_hint}"
            line += f"\n    result: {status}"
            if msg:
                line += f" - {msg}"
            if data_summary:
                line += f"\n    data: {data_summary}"
            if cost > 0:
                line += f" (${cost:.6f})"
            lines.append(line)

    # Detect patterns
    pattern_note = _detect_patterns(actions)

    header = f"\nRecent actions ({len(actions)} of {len(recent_actions)} total):"
    body = "\n".join(lines)
    result = header + "\n" + body

    if pattern_note:
        result += f"\n\n\u26a0 Pattern detected: {pattern_note}"

    # Final truncation if needed
    if len(result) > max_total_chars:
        result = result[:max_total_chars - 50] + f"\n... [history truncated, {len(recent_actions)} total actions]"

    return result


def _summarize_params(params: Dict, max_len: int = 60) -> str:
    """Create a brief summary of action parameters."""
    if not params:
        return ""

    parts = []
    for key, value in params.items():
        val_str = str(value)
        if len(val_str) > 30:
            val_str = val_str[:27] + "..."
        parts.append(f"{key}={val_str}")

    summary = ", ".join(parts)
    if len(summary) > max_len:
        summary = summary[:max_len - 3] + "..."
    return summary


def _summarize_data(data: Any, max_len: int = 80) -> str:
    """Create a brief summary of result data."""
    if data is None:
        return ""

    if isinstance(data, str):
        if len(data) > max_len:
            return f'"{data[:max_len - 10]}..." ({len(data)} chars)'
        return f'"{data}"'
    elif isinstance(data, dict):
        keys = list(data.keys())[:5]
        if len(keys) < len(data):
            return f"{{{', '.join(keys)}, ... +{len(data)-len(keys)} more}}"
        return f"{{{', '.join(keys)}}}"
    elif isinstance(data, list):
        return f"[{len(data)} items]"
    else:
        s = str(data)
        if len(s) > max_len:
            return s[:max_len - 3] + "..."
        return s


def _detect_patterns(actions: List[Dict]) -> str:
    """Detect repetitive patterns in action history."""
    if len(actions) < 3:
        return ""

    # Check for same tool repeated
    last_tools = [a.get("tool", "") for a in actions[-5:]]
    if len(set(last_tools)) == 1 and len(last_tools) >= 3:
        return f"{last_tools[0]} has been called {len(last_tools)} times in a row. Consider a different approach."

    # Check for repeated failures
    last_statuses = [a.get("result", {}).get("status", "") for a in actions[-5:]]
    fail_count = sum(1 for s in last_statuses if s in ("error", "failed"))
    if fail_count >= 3:
        return f"{fail_count} of last {len(last_statuses)} actions failed. Review your approach."

    # Check for tool-status alternation (retry loops)
    if len(actions) >= 4:
        last_4 = [(a.get("tool"), a.get("result", {}).get("status")) for a in actions[-4:]]
        if last_4[0] == last_4[2] and last_4[1] == last_4[3]:
            return "Alternating pattern detected - you may be stuck in a retry loop."

    return ""
