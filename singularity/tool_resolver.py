#!/usr/bin/env python3
"""
ToolResolver - Fuzzy tool name matching and parameter validation.

When the LLM produces a tool name that doesn't exactly match an installed tool,
ToolResolver finds the closest match and auto-corrects it. It also validates
that required parameters are present before execution.

This is a core self-improvement capability: the agent becomes more resilient
to LLM output errors without requiring additional API calls.
"""

import difflib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class ToolMatch:
    """Result of resolving a tool name."""
    original: str
    resolved: str
    was_corrected: bool
    confidence: float  # 0.0 to 1.0
    suggestions: List[str] = field(default_factory=list)
    error: str = ""


@dataclass
class ParamValidation:
    """Result of validating parameters against a tool's schema."""
    valid: bool
    missing_required: List[str] = field(default_factory=list)
    unknown_params: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class ToolResolver:
    """
    Resolves tool names with fuzzy matching and validates parameters.

    Usage:
        resolver = ToolResolver(tools)
        match = resolver.resolve("filesytem:red_file")  # typo
        # match.resolved == "filesystem:read_file"
        # match.was_corrected == True
    """

    # Minimum similarity ratio to auto-correct (0.0 to 1.0)
    AUTO_CORRECT_THRESHOLD = 0.80

    # Minimum similarity ratio to suggest as alternative
    SUGGEST_THRESHOLD = 0.50

    # Maximum number of suggestions to return
    MAX_SUGGESTIONS = 3

    def __init__(self, tools: List[Dict]):
        """
        Initialize with available tools.

        Args:
            tools: List of tool dicts with 'name', 'description', 'parameters' keys
        """
        self._tools: Dict[str, Dict] = {}
        self._tool_names: List[str] = []
        self._skill_ids: List[str] = []
        self._action_names_by_skill: Dict[str, List[str]] = {}

        for tool in tools:
            name = tool.get("name", "")
            self._tools[name] = tool
            self._tool_names.append(name)

            if ":" in name:
                skill_id, action_name = name.split(":", 1)
                if skill_id not in self._skill_ids:
                    self._skill_ids.append(skill_id)
                if skill_id not in self._action_names_by_skill:
                    self._action_names_by_skill[skill_id] = []
                self._action_names_by_skill[skill_id].append(action_name)

        # Track corrections for learning
        self._corrections: List[Dict] = []

    def resolve(self, tool_name: str) -> ToolMatch:
        """
        Resolve a tool name, applying fuzzy matching if needed.

        Args:
            tool_name: The tool name from LLM output

        Returns:
            ToolMatch with resolved name and metadata
        """
        # Exact match - fast path
        if tool_name in self._tools:
            return ToolMatch(
                original=tool_name,
                resolved=tool_name,
                was_corrected=False,
                confidence=1.0,
            )

        # Handle "wait" special case
        if tool_name == "wait":
            return ToolMatch(
                original=tool_name,
                resolved=tool_name,
                was_corrected=False,
                confidence=1.0,
            )

        # Try fuzzy matching on the full tool name
        match = self._fuzzy_match_full(tool_name)
        if match:
            return match

        # Try component-level matching: fix skill_id and action separately
        if ":" in tool_name:
            match = self._fuzzy_match_components(tool_name)
            if match:
                return match

        # No match found - return error with suggestions
        suggestions = self._get_suggestions(tool_name)
        suggestion_text = ""
        if suggestions:
            suggestion_text = f"Did you mean: {', '.join(suggestions)}?"

        return ToolMatch(
            original=tool_name,
            resolved=tool_name,
            was_corrected=False,
            confidence=0.0,
            suggestions=suggestions,
            error=f"Unknown tool: {tool_name}. {suggestion_text}".strip(),
        )

    def _fuzzy_match_full(self, tool_name: str) -> Optional[ToolMatch]:
        """Try fuzzy matching against all full tool names."""
        matches = difflib.get_close_matches(
            tool_name,
            self._tool_names,
            n=self.MAX_SUGGESTIONS,
            cutoff=self.SUGGEST_THRESHOLD,
        )

        if not matches:
            return None

        best = matches[0]
        confidence = difflib.SequenceMatcher(None, tool_name, best).ratio()

        if confidence >= self.AUTO_CORRECT_THRESHOLD:
            self._record_correction(tool_name, best, confidence)
            return ToolMatch(
                original=tool_name,
                resolved=best,
                was_corrected=True,
                confidence=confidence,
                suggestions=matches[1:] if len(matches) > 1 else [],
            )

        # Not confident enough to auto-correct, but have suggestions
        return ToolMatch(
            original=tool_name,
            resolved=tool_name,
            was_corrected=False,
            confidence=confidence,
            suggestions=matches,
            error=f"Unknown tool: {tool_name}. Did you mean: {', '.join(matches)}?",
        )

    def _fuzzy_match_components(self, tool_name: str) -> Optional[ToolMatch]:
        """Try matching skill_id and action_name separately for better precision."""
        skill_id, action_name = tool_name.split(":", 1)

        # Find closest skill_id
        skill_matches = difflib.get_close_matches(
            skill_id,
            self._skill_ids,
            n=1,
            cutoff=self.SUGGEST_THRESHOLD,
        )

        if not skill_matches:
            return None

        resolved_skill = skill_matches[0]
        skill_confidence = difflib.SequenceMatcher(None, skill_id, resolved_skill).ratio()

        # Find closest action within that skill
        available_actions = self._action_names_by_skill.get(resolved_skill, [])
        if not available_actions:
            return None

        action_matches = difflib.get_close_matches(
            action_name,
            available_actions,
            n=1,
            cutoff=self.SUGGEST_THRESHOLD,
        )

        if not action_matches:
            # Skill matched but action didn't - suggest valid actions
            suggestions = [f"{resolved_skill}:{a}" for a in available_actions[:self.MAX_SUGGESTIONS]]
            return ToolMatch(
                original=tool_name,
                resolved=tool_name,
                was_corrected=False,
                confidence=skill_confidence * 0.5,
                suggestions=suggestions,
                error=f"Unknown action '{action_name}' for skill '{resolved_skill}'. "
                      f"Available: {', '.join(available_actions)}",
            )

        resolved_action = action_matches[0]
        action_confidence = difflib.SequenceMatcher(None, action_name, resolved_action).ratio()

        # Combined confidence
        combined_confidence = (skill_confidence + action_confidence) / 2
        resolved_name = f"{resolved_skill}:{resolved_action}"

        if combined_confidence >= self.AUTO_CORRECT_THRESHOLD:
            self._record_correction(tool_name, resolved_name, combined_confidence)
            return ToolMatch(
                original=tool_name,
                resolved=resolved_name,
                was_corrected=True,
                confidence=combined_confidence,
            )

        return ToolMatch(
            original=tool_name,
            resolved=tool_name,
            was_corrected=False,
            confidence=combined_confidence,
            suggestions=[resolved_name],
            error=f"Unknown tool: {tool_name}. Did you mean: {resolved_name}?",
        )

    def _get_suggestions(self, tool_name: str) -> List[str]:
        """Get general suggestions when no good fuzzy match exists."""
        matches = difflib.get_close_matches(
            tool_name,
            self._tool_names,
            n=self.MAX_SUGGESTIONS,
            cutoff=0.3,  # Lower cutoff for general suggestions
        )
        return matches

    def _record_correction(self, original: str, corrected: str, confidence: float):
        """Record a correction for potential learning."""
        self._corrections.append({
            "original": original,
            "corrected": corrected,
            "confidence": confidence,
        })

    def validate_params(self, tool_name: str, params: Dict) -> ParamValidation:
        """
        Validate parameters against a tool's parameter schema.

        Args:
            tool_name: The resolved tool name
            params: Parameters to validate

        Returns:
            ParamValidation with results
        """
        tool = self._tools.get(tool_name)
        if not tool:
            return ParamValidation(valid=True)  # Can't validate unknown tools

        schema = tool.get("parameters", {})
        if not schema:
            return ParamValidation(valid=True)

        missing = []
        unknown = []
        warnings = []

        # Check required params
        for param_name, param_def in schema.items():
            if isinstance(param_def, dict) and param_def.get("required", False):
                if param_name not in params:
                    missing.append(param_name)

        # Check for unknown params
        known_params = set(schema.keys())
        for param_name in params:
            if param_name not in known_params and known_params:
                # Try fuzzy match on param names
                close = difflib.get_close_matches(param_name, list(known_params), n=1, cutoff=0.6)
                if close:
                    warnings.append(
                        f"Unknown param '{param_name}' - did you mean '{close[0]}'?"
                    )
                unknown.append(param_name)

        return ParamValidation(
            valid=len(missing) == 0,
            missing_required=missing,
            unknown_params=unknown,
            warnings=warnings,
        )

    @property
    def corrections(self) -> List[Dict]:
        """Get list of all auto-corrections made."""
        return list(self._corrections)

    @property
    def correction_count(self) -> int:
        """Number of auto-corrections made."""
        return len(self._corrections)
