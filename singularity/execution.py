#!/usr/bin/env python3
"""
ExecutionEngine - Smart skill execution with validation, timing, and error context.

Provides pre-execution validation of actions and parameters,
execution timing, and rich error messages that help the LLM
self-correct on subsequent attempts.
"""

import asyncio
import time
from difflib import get_close_matches
from typing import Dict, List, Optional, Tuple

from .skills.base import Skill, SkillAction, SkillRegistry, SkillResult


class ExecutionEngine:
    """
    Wraps skill execution with validation, timing, and intelligent error handling.
    
    Replaces direct skill.execute() calls with a pipeline that:
    1. Validates the tool name exists (with fuzzy suggestions if not)
    2. Validates the action exists on the skill (with suggestions)
    3. Checks required parameters are present
    4. Executes with timing
    5. Returns rich error context on failure
    """

    def __init__(self, skills: SkillRegistry, default_timeout: float = 120.0):
        self.skills = skills
        self.default_timeout = default_timeout
        # Track execution history for performance insights
        self.history: List[Dict] = []
        self._max_history = 100

    async def execute(self, tool: str, params: Dict, timeout: Optional[float] = None) -> Dict:
        """
        Execute a tool action with full validation and timing.
        
        Args:
            tool: Tool identifier in "skill:action" format
            params: Parameters dict for the action
            timeout: Optional timeout in seconds (default: self.default_timeout)
            
        Returns:
            Result dict with status, data, message, and execution metadata
        """
        start_time = time.monotonic()
        timeout = timeout or self.default_timeout

        # Handle wait action
        if tool == "wait":
            return self._result("success", message="Waited", duration=0.0)

        # Parse skill:action format
        skill_id, action_name, parse_error = self._parse_tool(tool)
        if parse_error:
            return self._result("error", message=parse_error, duration=0.0)

        # Validate skill exists
        skill = self.skills.get(skill_id)
        if not skill:
            suggestion = self._suggest_skill(skill_id)
            msg = f"Unknown skill: '{skill_id}'."
            if suggestion:
                msg += f" Did you mean '{suggestion}'?"
            msg += f" Available skills: {', '.join(sorted(self.skills.skills.keys()))}"
            return self._result("error", message=msg, duration=0.0)

        # Validate action exists on skill
        action_def = skill.get_action(action_name)
        if not action_def:
            suggestion = self._suggest_action(skill, action_name)
            available = [a.name for a in skill.get_actions()]
            msg = f"Unknown action '{action_name}' on skill '{skill_id}'."
            if suggestion:
                msg += f" Did you mean '{suggestion}'?"
            msg += f" Available actions: {', '.join(available)}"
            return self._result("error", message=msg, duration=0.0)

        # Validate parameters
        param_error = self._validate_params(action_def, params)
        if param_error:
            return self._result("error", message=param_error, duration=0.0)

        # Execute with timeout
        try:
            result = await asyncio.wait_for(
                skill.execute(action_name, params),
                timeout=timeout
            )
            duration = time.monotonic() - start_time
            
            exec_result = {
                "status": "success" if result.success else "failed",
                "data": result.data,
                "message": result.message,
                "duration_seconds": round(duration, 3),
            }
            
            self._record(tool, params, exec_result, duration)
            return exec_result

        except asyncio.TimeoutError:
            duration = time.monotonic() - start_time
            msg = f"Action '{skill_id}:{action_name}' timed out after {timeout:.0f}s"
            result = self._result("error", message=msg, duration=duration)
            self._record(tool, params, result, duration)
            return result

        except Exception as e:
            duration = time.monotonic() - start_time
            error_type = type(e).__name__
            msg = f"Execution error ({error_type}): {str(e)}"
            # Add parameter context for debugging
            if params:
                param_summary = ", ".join(f"{k}={repr(v)[:50]}" for k, v in params.items())
                msg += f" | Called with: {param_summary}"
            result = self._result("error", message=msg, duration=duration)
            self._record(tool, params, result, duration)
            return result

    def _parse_tool(self, tool: str) -> Tuple[str, str, Optional[str]]:
        """Parse 'skill:action' format. Returns (skill_id, action_name, error)."""
        if ":" not in tool:
            # Try to find a skill that matches the whole string
            if tool in self.skills.skills:
                return tool, "", f"Missing action name. Use '{tool}:<action>'. Available actions: {', '.join(a.name for a in self.skills.get(tool).get_actions())}"
            return "", "", f"Invalid tool format: '{tool}'. Expected 'skill:action' format (e.g. 'filesystem:view', 'shell:bash')."
        
        parts = tool.split(":", 1)
        skill_id = parts[0].strip()
        action_name = parts[1].strip() if len(parts) > 1 else ""
        
        if not action_name:
            skill = self.skills.get(skill_id)
            if skill:
                available = [a.name for a in skill.get_actions()]
                return skill_id, "", f"Missing action name after ':'. Available actions for '{skill_id}': {', '.join(available)}"
            return skill_id, "", f"Missing action name after ':'."
        
        return skill_id, action_name, None

    def _validate_params(self, action_def: SkillAction, params: Dict) -> Optional[str]:
        """
        Validate parameters against action definition.
        Returns error message if validation fails, None if ok.
        """
        if not action_def.parameters:
            return None

        # Determine which params are required vs optional
        required_params = []
        optional_params = []
        
        for param_name, param_desc in action_def.parameters.items():
            desc_lower = str(param_desc).lower() if param_desc else ""
            if "optional" in desc_lower or "default" in desc_lower:
                optional_params.append(param_name)
            else:
                required_params.append(param_name)

        # Check for missing required params
        missing = [p for p in required_params if p not in params or params[p] is None]
        if missing:
            param_info = []
            for p in action_def.parameters:
                marker = "(required)" if p in required_params else "(optional)"
                param_info.append(f"  {p} {marker}: {action_def.parameters[p]}")
            return (
                f"Missing required parameter(s): {', '.join(missing)}. "
                f"Expected parameters for '{action_def.name}':\n" +
                "\n".join(param_info)
            )

        # Check for unknown params (warning, not error - some skills accept extras)
        known_params = set(action_def.parameters.keys())
        unknown = [p for p in params if p not in known_params]
        # Don't fail on unknown params, just note them - skills may handle them
        
        return None

    def _suggest_skill(self, skill_id: str) -> Optional[str]:
        """Suggest a similar skill name using fuzzy matching."""
        available = list(self.skills.skills.keys())
        matches = get_close_matches(skill_id, available, n=1, cutoff=0.5)
        return matches[0] if matches else None

    def _suggest_action(self, skill: Skill, action_name: str) -> Optional[str]:
        """Suggest a similar action name using fuzzy matching."""
        available = [a.name for a in skill.get_actions()]
        matches = get_close_matches(action_name, available, n=1, cutoff=0.5)
        return matches[0] if matches else None

    def _result(self, status: str, message: str = "", data: Dict = None, duration: float = 0.0) -> Dict:
        """Create a standardized result dict."""
        return {
            "status": status,
            "data": data or {},
            "message": message,
            "duration_seconds": round(duration, 3),
        }

    def _record(self, tool: str, params: Dict, result: Dict, duration: float):
        """Record execution for history tracking."""
        self.history.append({
            "tool": tool,
            "status": result.get("status", "unknown"),
            "duration": round(duration, 3),
        })
        if len(self.history) > self._max_history:
            self.history = self.history[-self._max_history:]

    def get_stats(self) -> Dict:
        """Get execution statistics."""
        if not self.history:
            return {"total": 0, "success": 0, "failed": 0, "error": 0, "avg_duration": 0}
        
        total = len(self.history)
        success = sum(1 for h in self.history if h["status"] == "success")
        failed = sum(1 for h in self.history if h["status"] == "failed")
        errors = sum(1 for h in self.history if h["status"] == "error")
        avg_duration = sum(h["duration"] for h in self.history) / total
        
        return {
            "total": total,
            "success": success,
            "failed": failed,
            "error": errors,
            "success_rate": round(success / total, 2) if total > 0 else 0,
            "avg_duration_seconds": round(avg_duration, 3),
        }
