"""
AgentJournal - Persistent session memory for agent continuity.

Provides cross-session awareness by recording what the agent did,
what worked, what failed, and what it planned to do next. On startup,
recent journal entries are summarized and injected into the LLM prompt
via project_context, giving the agent memory of its past.

Architecture:
    - Append-only JSONL file for durability (one JSON object per line)
    - Session start/end markers with summaries
    - Per-action outcome tracking (successes, failures, key results)
    - Configurable context window (how many past sessions to remember)
    - Auto-generates a concise context summary for the LLM prompt
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


DEFAULT_JOURNAL_PATH = Path(__file__).parent / "data" / "journal.jsonl"
MAX_CONTEXT_SESSIONS = 5  # How many past sessions to include in context
MAX_CONTEXT_CHARS = 2000  # Max chars for context summary


class AgentJournal:
    """Persistent journal that gives the agent cross-session memory.

    Usage:
        journal = AgentJournal()
        journal.start_session("MyAgent", "AGENT")

        # During run loop, record outcomes:
        journal.record_action("github:create_repo", {"name": "test"}, "success", "Created repo")
        journal.record_insight("Writing code is my strongest skill")
        journal.record_goal("Build a REST API for code review service")

        # At session end:
        journal.end_session(cycles=50, total_cost=0.15)

        # On next startup, get context for LLM:
        context = journal.get_context_summary()
    """

    def __init__(self, path: Optional[Path] = None):
        self.path = path or DEFAULT_JOURNAL_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._session_id: Optional[str] = None
        self._session_actions: List[Dict] = []
        self._session_insights: List[str] = []
        self._session_goals: List[str] = []
        self._session_start: Optional[float] = None

    def start_session(self, agent_name: str, ticker: str, agent_type: str = "general") -> str:
        """Mark the start of a new agent session. Returns session_id."""
        self._session_id = f"{ticker}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        self._session_actions = []
        self._session_insights = []
        self._session_goals = []
        self._session_start = time.time()

        self._append({
            "type": "session_start",
            "session_id": self._session_id,
            "agent_name": agent_name,
            "ticker": ticker,
            "agent_type": agent_type,
            "timestamp": datetime.now().isoformat(),
        })
        return self._session_id

    def record_action(self, tool: str, params: Dict, status: str, message: str = ""):
        """Record an action outcome during the current session."""
        entry = {
            "type": "action",
            "session_id": self._session_id,
            "tool": tool,
            "status": status,
            "message": message[:200],
            "timestamp": datetime.now().isoformat(),
        }
        self._append(entry)
        self._session_actions.append(entry)

    def record_insight(self, insight: str):
        """Record a lesson learned or insight during this session."""
        self._append({
            "type": "insight",
            "session_id": self._session_id,
            "insight": insight[:500],
            "timestamp": datetime.now().isoformat(),
        })
        self._session_insights.append(insight[:500])

    def record_goal(self, goal: str, priority: str = "medium"):
        """Record a goal or intention for current/future sessions."""
        self._append({
            "type": "goal",
            "session_id": self._session_id,
            "goal": goal[:500],
            "priority": priority,
            "timestamp": datetime.now().isoformat(),
        })
        self._session_goals.append(goal[:500])

    def end_session(self, cycles: int = 0, total_cost: float = 0.0,
                    balance_remaining: float = 0.0):
        """Mark the end of the current session with a summary."""
        duration = time.time() - self._session_start if self._session_start else 0

        successes = sum(1 for a in self._session_actions if a["status"] == "success")
        failures = sum(1 for a in self._session_actions if a["status"] != "success")

        # Identify most-used tools
        tool_counts: Dict[str, int] = {}
        for a in self._session_actions:
            tool = a["tool"]
            tool_counts[tool] = tool_counts.get(tool, 0) + 1
        top_tools = sorted(tool_counts.items(), key=lambda x: -x[1])[:5]

        self._append({
            "type": "session_end",
            "session_id": self._session_id,
            "cycles": cycles,
            "total_cost_usd": round(total_cost, 6),
            "balance_remaining": round(balance_remaining, 4),
            "duration_seconds": round(duration, 1),
            "actions_total": len(self._session_actions),
            "actions_succeeded": successes,
            "actions_failed": failures,
            "top_tools": top_tools,
            "insights": self._session_insights[:10],
            "goals": self._session_goals[:10],
            "timestamp": datetime.now().isoformat(),
        })

        self._session_id = None
        self._session_actions = []
        self._session_insights = []
        self._session_goals = []
        self._session_start = None

    def get_context_summary(self, max_sessions: int = MAX_CONTEXT_SESSIONS,
                            max_chars: int = MAX_CONTEXT_CHARS) -> str:
        """Generate a context summary from recent journal entries for the LLM prompt.

        Returns a concise text block summarizing:
        - Recent session outcomes
        - Key insights learned
        - Outstanding goals
        """
        entries = self._load_entries()
        if not entries:
            return ""

        # Collect session summaries (from session_end entries)
        sessions = [e for e in entries if e.get("type") == "session_end"]
        recent_sessions = sessions[-max_sessions:]

        # Collect all insights and goals
        all_insights = [e["insight"] for e in entries if e.get("type") == "insight"]
        all_goals = [e["goal"] for e in entries if e.get("type") == "goal"]

        # Deduplicate insights (keep last occurrence)
        seen_insights = set()
        unique_insights = []
        for insight in reversed(all_insights):
            normalized = insight.strip().lower()
            if normalized not in seen_insights:
                seen_insights.add(normalized)
                unique_insights.append(insight)
        unique_insights = list(reversed(unique_insights[-10:]))

        # Build context
        parts = []
        parts.append("=== AGENT JOURNAL (Past Sessions) ===")

        if recent_sessions:
            parts.append(f"\nRecent Sessions ({len(recent_sessions)} of {len(sessions)} total):")
            for s in recent_sessions:
                sid = s.get("session_id", "unknown")
                ts = s.get("timestamp", "")[:10]
                cycles = s.get("cycles", 0)
                cost = s.get("total_cost_usd", 0)
                ok = s.get("actions_succeeded", 0)
                fail = s.get("actions_failed", 0)
                tools = ", ".join(t[0] for t in s.get("top_tools", [])[:3])
                parts.append(f"  [{ts}] {sid}: {cycles} cycles, ${cost:.4f} cost, "
                             f"{ok} ok/{fail} fail. Tools: {tools}")

                # Include session-level insights/goals
                for insight in s.get("insights", [])[:2]:
                    parts.append(f"    Insight: {insight}")
                for goal in s.get("goals", [])[:2]:
                    parts.append(f"    Goal: {goal}")

        if unique_insights:
            parts.append(f"\nKey Insights ({len(unique_insights)}):")
            for insight in unique_insights[-5:]:
                parts.append(f"  - {insight}")

        if all_goals:
            parts.append(f"\nOutstanding Goals ({len(all_goals)}):")
            for goal in all_goals[-5:]:
                parts.append(f"  - {goal}")

        context = "\n".join(parts)

        # Truncate if too long
        if len(context) > max_chars:
            context = context[:max_chars - 20] + "\n... (truncated)"

        return context

    def get_session_count(self) -> int:
        """Return the total number of completed sessions."""
        entries = self._load_entries()
        return sum(1 for e in entries if e.get("type") == "session_end")

    def get_recent_goals(self, n: int = 5) -> List[str]:
        """Return the most recent n goals."""
        entries = self._load_entries()
        goals = [e["goal"] for e in entries if e.get("type") == "goal"]
        return goals[-n:]

    def get_recent_insights(self, n: int = 5) -> List[str]:
        """Return the most recent n insights."""
        entries = self._load_entries()
        insights = [e["insight"] for e in entries if e.get("type") == "insight"]
        return insights[-n:]

    def clear(self):
        """Clear all journal entries. Use with caution."""
        if self.path.exists():
            self.path.unlink()

    def _append(self, entry: Dict):
        """Append a single JSON entry to the journal file."""
        with open(self.path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def _load_entries(self) -> List[Dict]:
        """Load all journal entries from the JSONL file."""
        if not self.path.exists():
            return []
        entries = []
        with open(self.path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return entries
