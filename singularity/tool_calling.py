#!/usr/bin/env python3
"""
Native Tool Calling Support for Singularity Agent

Converts skill actions into structured tool definitions for LLM providers
that support native tool/function calling (Anthropic, OpenAI).

Benefits over text-based JSON parsing:
- Eliminates regex parsing errors
- LLM sees exact parameter schemas
- Structured responses instead of free-text JSON
- Better parameter validation
"""

import re
from typing import Any, Dict, List, Optional

from .cognition import Action


def _param_description_to_schema(param_name: str, description: Any) -> dict:
    """Convert a parameter description to a JSON Schema property.

    Handles both formats:
    - Simple string: "glob pattern"
    - Dict: {"type": "string", "required": True, "description": "..."}
    """
    if isinstance(description, dict):
        schema = {"type": description.get("type", "string")}
        if "description" in description:
            schema["description"] = description["description"]
        return schema, description.get("required", True)

    # Simple string description
    desc_str = str(description)
    is_optional = "optional" in desc_str.lower()

    # Infer type from description hints
    param_type = "string"
    lower_desc = desc_str.lower()
    if any(kw in lower_desc for kw in ["number", "count", "max", "limit", "offset", "amount", "seconds"]):
        param_type = "number"
    elif any(kw in lower_desc for kw in ["true", "false", "boolean", "flag", "enable", "disable"]):
        param_type = "boolean"

    # Param name hints
    lower_name = param_name.lower()
    if lower_name in ("limit", "offset", "count", "max_results", "timeout", "depth", "max_lines", "retries"):
        param_type = "number"
    elif lower_name in ("recursive", "force", "dry_run", "verbose", "overwrite", "append"):
        param_type = "boolean"

    return {"type": param_type, "description": desc_str}, not is_optional


def tools_to_anthropic_format(tools: List[Dict]) -> List[Dict]:
    """Convert agent tools list to Anthropic tool definitions.

    Args:
        tools: List of tool dicts with 'name', 'description', 'parameters'

    Returns:
        List of Anthropic tool definition dicts with JSON Schema input_schema
    """
    tool_defs = []
    for tool in tools:
        name = tool.get("name", "")
        if name == "wait":
            continue

        # Anthropic tool names must match [a-zA-Z0-9_-]
        safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", name)

        properties = {}
        required = []

        params = tool.get("parameters", {})
        if isinstance(params, dict):
            for param_name, param_desc in params.items():
                prop_schema, is_required = _param_description_to_schema(param_name, param_desc)
                properties[param_name] = prop_schema
                if is_required:
                    required.append(param_name)

        tool_def = {
            "name": safe_name,
            "description": tool.get("description", ""),
            "input_schema": {
                "type": "object",
                "properties": properties,
            }
        }
        if required:
            tool_def["input_schema"]["required"] = required

        tool_defs.append(tool_def)

    return tool_defs


def tools_to_openai_format(tools: List[Dict]) -> List[Dict]:
    """Convert agent tools list to OpenAI function calling format.

    Args:
        tools: List of tool dicts with 'name', 'description', 'parameters'

    Returns:
        List of OpenAI tool definition dicts
    """
    tool_defs = []
    for tool in tools:
        name = tool.get("name", "")
        if name == "wait":
            continue

        safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", name)

        properties = {}
        required = []

        params = tool.get("parameters", {})
        if isinstance(params, dict):
            for param_name, param_desc in params.items():
                prop_schema, is_required = _param_description_to_schema(param_name, param_desc)
                properties[param_name] = prop_schema
                if is_required:
                    required.append(param_name)

        function_def = {
            "name": safe_name,
            "description": tool.get("description", ""),
            "parameters": {
                "type": "object",
                "properties": properties,
            }
        }
        if required:
            function_def["parameters"]["required"] = required

        tool_defs.append({
            "type": "function",
            "function": function_def,
        })

    return tool_defs


# Mapping from safe names back to original tool names
def _build_name_map(tools: List[Dict]) -> Dict[str, str]:
    """Build mapping from sanitized names back to original skill:action names."""
    name_map = {}
    for tool in tools:
        name = tool.get("name", "")
        safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
        name_map[safe_name] = name
    return name_map


def parse_anthropic_tool_use(response, tools: List[Dict]) -> Optional[Action]:
    """Parse an Anthropic response that may contain tool_use blocks.

    Args:
        response: Anthropic API response object
        tools: Original tools list (for name mapping)

    Returns:
        Action if tool_use found, None otherwise
    """
    name_map = _build_name_map(tools)

    reasoning = ""
    tool_name = None
    tool_input = {}

    for block in response.content:
        if block.type == "text":
            reasoning = block.text
        elif block.type == "tool_use":
            safe_name = block.name
            tool_name = name_map.get(safe_name, safe_name.replace("_", ":", 1))
            tool_input = block.input or {}

    if tool_name:
        return Action(
            tool=tool_name,
            params=tool_input,
            reasoning=reasoning,
        )

    return None


def parse_openai_tool_call(response, tools: List[Dict]) -> Optional[Action]:
    """Parse an OpenAI response that may contain tool calls.

    Args:
        response: OpenAI API response object
        tools: Original tools list (for name mapping)

    Returns:
        Action if tool call found, None otherwise
    """
    import json

    name_map = _build_name_map(tools)

    message = response.choices[0].message
    if message.tool_calls:
        call = message.tool_calls[0]
        safe_name = call.function.name
        tool_name = name_map.get(safe_name, safe_name.replace("_", ":", 1))

        try:
            params = json.loads(call.function.arguments)
        except (json.JSONDecodeError, TypeError):
            params = {}

        reasoning = message.content or ""
        return Action(tool=tool_name, params=params, reasoning=reasoning)

    return None


def format_tools_for_text_prompt(tools: List[Dict]) -> str:
    """Format tools with parameter details for text-based prompts.

    Used as fallback when native tool calling is not available.
    Includes parameter names, types, and descriptions.

    Args:
        tools: List of tool dicts with 'name', 'description', 'parameters'

    Returns:
        Formatted string for inclusion in LLM prompt
    """
    lines = []
    for tool in tools:
        name = tool.get("name", "")
        desc = tool.get("description", "")
        params = tool.get("parameters", {})

        if not isinstance(params, dict) or not params:
            lines.append(f"- {name}: {desc}")
            continue

        param_parts = []
        for pname, pdesc in params.items():
            if isinstance(pdesc, dict):
                ptype = pdesc.get("type", "string")
                pdescription = pdesc.get("description", "")
                is_opt = not pdesc.get("required", True)
            else:
                pdescription = str(pdesc)
                is_opt = "optional" in pdescription.lower()
                ptype = "string"

            opt_marker = "?" if is_opt else ""
            param_parts.append(f"{pname}{opt_marker}: {pdescription}")

        params_str = ", ".join(param_parts)
        lines.append(f"- {name}({params_str}): {desc}")

    return "\n".join(lines)
