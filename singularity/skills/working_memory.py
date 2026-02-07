#!/usr/bin/env python3
"""
WorkingMemorySkill - In-session context management for the agent.

Provides a scratchpad, goal tracking, and context stack that persists
across cycles within a single agent session. This context is injected
into the LLM's state so the agent can maintain coherent multi-step plans.

Unlike LocalMemorySkill (cross-session file persistence), this is fast
in-memory storage designed to augment the agent's thinking context.
"""

from collections import OrderedDict
from datetime import datetime
from typing import Dict, List, Any, Optional

from .base import Skill, SkillManifest, SkillAction, SkillResult


class WorkingMemorySkill(Skill):
    """In-session working memory for maintaining context across cycles."""

    MAX_NOTES = 50
    MAX_CONTEXT_STACK = 10
    MAX_LOG_ENTRIES = 100

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self._notes: OrderedDict[str, Any] = OrderedDict()
        self._current_goal: Optional[str] = None
        self._sub_goals: List[str] = []
        self._context_stack: List[Dict[str, Any]] = []
        self._log: List[Dict[str, str]] = []
        self._tags: Dict[str, List[str]] = {}  # tag -> list of note keys

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="working_memory",
            name="Working Memory",
            version="1.0.0",
            category="cognition",
            description=(
                "In-session working memory for maintaining context across cycles. "
                "Store notes, track goals, manage a context stack for nested tasks. "
                "Contents are included in your thinking context automatically."
            ),
            actions=[
                SkillAction(
                    name="note",
                    description="Store or update a note in working memory",
                    parameters={
                        "key": {"type": "string", "required": True,
                                "description": "Unique key for this note"},
                        "value": {"type": "string", "required": True,
                                  "description": "Content to store"},
                        "tags": {"type": "array", "required": False,
                                 "description": "Optional tags for categorization"},
                    },
                ),
                SkillAction(
                    name="recall",
                    description="Retrieve a note by key",
                    parameters={
                        "key": {"type": "string", "required": True,
                                "description": "Key of the note to retrieve"},
                    },
                ),
                SkillAction(
                    name="forget",
                    description="Remove a note from working memory",
                    parameters={
                        "key": {"type": "string", "required": True,
                                "description": "Key of the note to remove"},
                    },
                ),
                SkillAction(
                    name="set_goal",
                    description="Set the current high-level goal",
                    parameters={
                        "goal": {"type": "string", "required": True,
                                 "description": "The goal to work towards"},
                        "sub_goals": {"type": "array", "required": False,
                                      "description": "Optional list of sub-goals/steps"},
                    },
                ),
                SkillAction(
                    name="complete_sub_goal",
                    description="Mark a sub-goal as completed",
                    parameters={
                        "index": {"type": "integer", "required": True,
                                  "description": "Index of the sub-goal (0-based)"},
                    },
                ),
                SkillAction(
                    name="push_context",
                    description="Push current context onto stack (for nested tasks)",
                    parameters={
                        "label": {"type": "string", "required": True,
                                  "description": "Label for this context frame"},
                        "data": {"type": "object", "required": False,
                                 "description": "Optional data to store with context"},
                    },
                ),
                SkillAction(
                    name="pop_context",
                    description="Pop and restore the most recent context from stack",
                    parameters={},
                ),
                SkillAction(
                    name="search",
                    description="Search notes by keyword or tag",
                    parameters={
                        "query": {"type": "string", "required": False,
                                  "description": "Search keyword (searches keys and values)"},
                        "tag": {"type": "string", "required": False,
                                "description": "Filter by tag"},
                    },
                ),
                SkillAction(
                    name="summary",
                    description="Get a summary of all working memory contents",
                    parameters={},
                ),
                SkillAction(
                    name="clear",
                    description="Clear all working memory",
                    parameters={
                        "confirm": {"type": "boolean", "required": True,
                                    "description": "Must be true to confirm clearing"},
                    },
                ),
            ],
            required_credentials=[],
        )

    async def execute(self, action: str, params: Dict) -> SkillResult:
        handlers = {
            "note": self._note,
            "recall": self._recall,
            "forget": self._forget,
            "set_goal": self._set_goal,
            "complete_sub_goal": self._complete_sub_goal,
            "push_context": self._push_context,
            "pop_context": self._pop_context,
            "search": self._search,
            "summary": self._summary,
            "clear": self._clear,
        }

        handler = handlers.get(action)
        if not handler:
            return SkillResult(
                success=False,
                message=f"Unknown action: {action}. Available: {list(handlers.keys())}",
            )

        return await handler(params)

    async def _note(self, params: Dict) -> SkillResult:
        key = params.get("key", "").strip()
        value = params.get("value", "")
        tags = params.get("tags", [])

        if not key:
            return SkillResult(success=False, message="Key is required")

        is_update = key in self._notes

        # Enforce max notes
        if not is_update and len(self._notes) >= self.MAX_NOTES:
            # Remove oldest
            oldest_key = next(iter(self._notes))
            del self._notes[oldest_key]
            self._remove_tags_for_key(oldest_key)

        self._notes[key] = {
            "value": value,
            "created_at": datetime.now().isoformat(),
            "tags": tags,
        }

        # Update tag index
        self._remove_tags_for_key(key)
        for tag in tags:
            if tag not in self._tags:
                self._tags[tag] = []
            self._tags[tag].append(key)

        self._add_log("note", f"{'Updated' if is_update else 'Stored'} note: {key}")

        return SkillResult(
            success=True,
            message=f"{'Updated' if is_update else 'Stored'} note '{key}'",
            data={"key": key, "note_count": len(self._notes)},
        )

    async def _recall(self, params: Dict) -> SkillResult:
        key = params.get("key", "").strip()
        if not key:
            return SkillResult(success=False, message="Key is required")

        note = self._notes.get(key)
        if not note:
            # Try fuzzy match
            matches = [k for k in self._notes if key.lower() in k.lower()]
            if matches:
                return SkillResult(
                    success=False,
                    message=f"Note '{key}' not found. Did you mean: {matches[:5]}",
                    data={"suggestions": matches[:5]},
                )
            return SkillResult(success=False, message=f"Note '{key}' not found")

        return SkillResult(
            success=True,
            message=note["value"],
            data={"key": key, "value": note["value"], "tags": note["tags"],
                   "created_at": note["created_at"]},
        )

    async def _forget(self, params: Dict) -> SkillResult:
        key = params.get("key", "").strip()
        if not key:
            return SkillResult(success=False, message="Key is required")

        if key not in self._notes:
            return SkillResult(success=False, message=f"Note '{key}' not found")

        del self._notes[key]
        self._remove_tags_for_key(key)
        self._add_log("forget", f"Removed note: {key}")

        return SkillResult(
            success=True,
            message=f"Removed note '{key}'",
            data={"note_count": len(self._notes)},
        )

    async def _set_goal(self, params: Dict) -> SkillResult:
        goal = params.get("goal", "").strip()
        if not goal:
            return SkillResult(success=False, message="Goal is required")

        old_goal = self._current_goal
        self._current_goal = goal
        self._sub_goals = [
            {"text": sg, "completed": False}
            for sg in params.get("sub_goals", [])
        ]

        self._add_log("set_goal", f"Goal: {goal}")

        return SkillResult(
            success=True,
            message=f"Goal set: {goal}" + (f" (replaced: {old_goal})" if old_goal else ""),
            data={
                "goal": goal,
                "sub_goals": self._sub_goals,
                "previous_goal": old_goal,
            },
        )

    async def _complete_sub_goal(self, params: Dict) -> SkillResult:
        index = params.get("index")
        if index is None:
            return SkillResult(success=False, message="Index is required")

        if not self._sub_goals:
            return SkillResult(success=False, message="No sub-goals set")

        if index < 0 or index >= len(self._sub_goals):
            return SkillResult(
                success=False,
                message=f"Index {index} out of range (0-{len(self._sub_goals)-1})",
            )

        self._sub_goals[index]["completed"] = True
        completed = sum(1 for sg in self._sub_goals if sg["completed"])
        total = len(self._sub_goals)

        self._add_log("complete_sub_goal",
                       f"Completed: {self._sub_goals[index]['text']} ({completed}/{total})")

        return SkillResult(
            success=True,
            message=f"Sub-goal {index} completed ({completed}/{total} done)",
            data={
                "completed_text": self._sub_goals[index]["text"],
                "progress": f"{completed}/{total}",
                "all_complete": completed == total,
                "sub_goals": self._sub_goals,
            },
        )

    async def _push_context(self, params: Dict) -> SkillResult:
        label = params.get("label", "").strip()
        if not label:
            return SkillResult(success=False, message="Label is required")

        if len(self._context_stack) >= self.MAX_CONTEXT_STACK:
            return SkillResult(
                success=False,
                message=f"Context stack full (max {self.MAX_CONTEXT_STACK})",
            )

        frame = {
            "label": label,
            "data": params.get("data", {}),
            "goal": self._current_goal,
            "sub_goals": list(self._sub_goals),
            "pushed_at": datetime.now().isoformat(),
        }
        self._context_stack.append(frame)
        self._add_log("push_context", f"Pushed: {label} (depth: {len(self._context_stack)})")

        return SkillResult(
            success=True,
            message=f"Context '{label}' pushed (stack depth: {len(self._context_stack)})",
            data={"depth": len(self._context_stack), "label": label},
        )

    async def _pop_context(self, params: Dict) -> SkillResult:
        if not self._context_stack:
            return SkillResult(success=False, message="Context stack is empty")

        frame = self._context_stack.pop()

        # Restore goal from context
        restored_goal = frame.get("goal")
        restored_sub_goals = frame.get("sub_goals", [])

        self._current_goal = restored_goal
        self._sub_goals = restored_sub_goals

        self._add_log("pop_context",
                       f"Popped: {frame['label']} (depth: {len(self._context_stack)})")

        return SkillResult(
            success=True,
            message=f"Restored context '{frame['label']}'. Goal: {restored_goal or 'none'}",
            data={
                "label": frame["label"],
                "data": frame["data"],
                "restored_goal": restored_goal,
                "depth": len(self._context_stack),
            },
        )

    async def _search(self, params: Dict) -> SkillResult:
        query = params.get("query", "").strip().lower()
        tag = params.get("tag", "").strip()

        if not query and not tag:
            return SkillResult(success=False, message="Provide query or tag to search")

        results = []

        if tag:
            # Search by tag
            tagged_keys = self._tags.get(tag, [])
            for key in tagged_keys:
                if key in self._notes:
                    results.append({"key": key, **self._notes[key]})
        elif query:
            # Search by keyword in key and value
            for key, note in self._notes.items():
                if (query in key.lower() or
                        query in str(note["value"]).lower()):
                    results.append({"key": key, **note})

        return SkillResult(
            success=True,
            message=f"Found {len(results)} matching notes",
            data={"results": results, "count": len(results)},
        )

    async def _summary(self, params: Dict) -> SkillResult:
        summary = self.get_context_summary()
        return SkillResult(
            success=True,
            message=summary,
            data={
                "goal": self._current_goal,
                "sub_goals": self._sub_goals,
                "note_count": len(self._notes),
                "note_keys": list(self._notes.keys()),
                "context_depth": len(self._context_stack),
                "context_labels": [f["label"] for f in self._context_stack],
                "tags": list(self._tags.keys()),
                "recent_log": self._log[-10:],
            },
        )

    async def _clear(self, params: Dict) -> SkillResult:
        if not params.get("confirm"):
            return SkillResult(success=False, message="Set confirm=true to clear")

        count = len(self._notes)
        self._notes.clear()
        self._current_goal = None
        self._sub_goals = []
        self._context_stack = []
        self._tags.clear()
        self._add_log("clear", f"Cleared {count} notes and all context")

        return SkillResult(
            success=True,
            message=f"Cleared {count} notes, goal, and context stack",
            data={"cleared_notes": count},
        )

    def get_context_summary(self) -> str:
        """Generate a text summary of working memory for LLM context injection."""
        parts = []

        if self._current_goal:
            parts.append(f"CURRENT GOAL: {self._current_goal}")
            if self._sub_goals:
                for i, sg in enumerate(self._sub_goals):
                    marker = "✓" if sg["completed"] else "○"
                    parts.append(f"  {marker} {i}. {sg['text']}")
                completed = sum(1 for sg in self._sub_goals if sg["completed"])
                parts.append(f"  Progress: {completed}/{len(self._sub_goals)}")

        if self._notes:
            parts.append(f"\nWORKING NOTES ({len(self._notes)}):")
            for key, note in self._notes.items():
                value_preview = str(note["value"])[:200]
                tag_str = f" [{', '.join(note['tags'])}]" if note.get("tags") else ""
                parts.append(f"  • {key}{tag_str}: {value_preview}")

        if self._context_stack:
            labels = [f["label"] for f in self._context_stack]
            parts.append(f"\nCONTEXT STACK ({len(self._context_stack)}): {' → '.join(labels)}")

        if not parts:
            return "Working memory is empty."

        return "\n".join(parts)

    def has_content(self) -> bool:
        """Check if working memory has any content."""
        return bool(self._notes or self._current_goal or self._context_stack)

    def _remove_tags_for_key(self, key: str):
        """Remove all tag references for a key."""
        for tag_keys in self._tags.values():
            if key in tag_keys:
                tag_keys.remove(key)
        # Clean empty tags
        self._tags = {t: keys for t, keys in self._tags.items() if keys}

    def _add_log(self, action: str, message: str):
        """Add an entry to the internal log."""
        self._log.append({
            "action": action,
            "message": message,
            "timestamp": datetime.now().isoformat(),
        })
        if len(self._log) > self.MAX_LOG_ENTRIES:
            self._log = self._log[-self.MAX_LOG_ENTRIES:]
