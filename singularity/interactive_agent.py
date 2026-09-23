#!/usr/bin/env python3
"""InteractiveAgent - Conversational AI agent with tool execution."""
import asyncio
import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from .cognition import CognitionEngine, AgentState, Action, TokenUsage, calculate_api_cost
from .skills.base import SkillRegistry


@dataclass
class ChatMessage:
    role: str
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChatResponse:
    message: str
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    total_cost_usd: float = 0.0
    total_tokens: int = 0
    thinking_steps: int = 0


class InteractiveAgent:
    """Conversational AI agent that processes user messages and responds using tools."""

    def __init__(self, name="Assistant", llm_provider="anthropic",
                 llm_model="claude-sonnet-4-20250514", llm_base_url="http://localhost:8000/v1",
                 anthropic_api_key="", openai_api_key="", system_prompt=None,
                 system_prompt_file=None, max_tool_calls_per_message=10, skills=None):
        self.name = name
        self.max_tool_calls = max_tool_calls_per_message
        self.total_cost = 0.0
        self.total_tokens = 0
        self.message_count = 0
        self.history: List[ChatMessage] = []
        interactive_prompt = system_prompt or self._default_system_prompt()
        self.cognition = CognitionEngine(
            llm_provider=llm_provider,
            anthropic_api_key=anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY", ""),
            openai_api_key=openai_api_key or os.environ.get("OPENAI_API_KEY", ""),
            openai_base_url=llm_base_url, llm_model=llm_model,
            agent_name=name, agent_ticker="CHAT", agent_type="interactive",
            agent_specialty="conversational assistant", system_prompt=interactive_prompt,
            system_prompt_file=system_prompt_file)
        self.skills = SkillRegistry()
        self._init_skills(filter_skills=skills)

    def _default_system_prompt(self):
        return (
            "You are " + self.name + ", an interactive AI assistant with tools.\n"
            "To use a tool: {\"tool\": \"skill:action\", \"params\": {}, \"reasoning\": \"why\"}\n"
            "To respond: {\"tool\": \"respond\", \"params\": {\"message\": \"your response\"}, \"reasoning\": \"why\"}\n"
            "Always end with respond when you have enough info."
        )

    def _init_skills(self, filter_skills=None):
        creds = {k: os.environ.get(k, "") for k in ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GITHUB_TOKEN"]}
        self.skills.set_credentials(creds)
        from .skills.filesystem import FilesystemSkill
        from .skills.shell import ShellSkill
        from .skills.request import RequestSkill
        for sc in [FilesystemSkill, ShellSkill, RequestSkill]:
            try:
                self.skills.install(sc)
                s = self.skills.get(sc(creds).manifest.skill_id)
                if s and s.check_credentials():
                    if filter_skills and s.manifest.skill_id not in filter_skills:
                        self.skills.uninstall(s.manifest.skill_id)
                else:
                    self.skills.uninstall(sc(creds).manifest.skill_id)
            except Exception:
                pass

    def _get_tools(self):
        tools = []
        for skill in self.skills.skills.values():
            for action in skill.manifest.actions:
                tools.append({"name": skill.manifest.skill_id + ":" + action.name,
                    "description": action.description, "parameters": action.parameters})
        tools.append({"name": "respond", "description": "Send response to user",
            "parameters": {"message": "Response text"}})
        return tools

    async def chat(self, message, context=""):
        self.message_count += 1
        self.history.append(ChatMessage(role="user", content=message))
        tools = self._get_tools()
        tool_calls = []
        total_cost = 0.0
        total_tokens = 0
        steps = 0
        history_text = self._format_history(limit=20)
        for _ in range(self.max_tool_calls):
            steps += 1
            state = AgentState(balance=999.0, burn_rate=0.0, runway_hours=999.0, tools=tools,
                recent_actions=[], cycle=steps,
                project_context=self._build_context(message, history_text, tool_calls, context))
            decision = await self.cognition.think(state)
            total_cost += decision.api_cost_usd
            total_tokens += decision.token_usage.total_tokens()
            action = decision.action
            if action.tool == "respond":
                resp = action.params.get("message", action.reasoning or "I am not sure.")
                self.history.append(ChatMessage(role="assistant", content=resp,
                    metadata={"tool_calls": len(tool_calls), "cost": total_cost}))
                self.total_cost += total_cost
                self.total_tokens += total_tokens
                return ChatResponse(message=resp, tool_calls=tool_calls, total_cost_usd=total_cost,
                    total_tokens=total_tokens, thinking_steps=steps)
            result = await self._execute(action)
            tool_calls.append({"tool": action.tool, "params": action.params,
                "result": result, "reasoning": action.reasoning})
        resp = self._synthesize_response(message, tool_calls)
        self.history.append(ChatMessage(role="assistant", content=resp,
            metadata={"tool_calls": len(tool_calls), "cost": total_cost, "forced": True}))
        self.total_cost += total_cost
        self.total_tokens += total_tokens
        return ChatResponse(message=resp, tool_calls=tool_calls, total_cost_usd=total_cost,
            total_tokens=total_tokens, thinking_steps=steps)

    def _build_context(self, msg, history, tool_calls, extra=""):
        parts = []
        if history:
            parts.append("CONVERSATION HISTORY:\n" + history)
        parts.append("CURRENT USER MESSAGE:\n" + msg)
        if tool_calls:
            tc = "\n".join(["  - " + t["tool"] + ": " + json.dumps(t["result"])[:300] for t in tool_calls])
            parts.append("TOOL RESULTS:\n" + tc + "\nUse respond tool to answer.")
        if extra:
            parts.append("CONTEXT:\n" + extra)
        return "\n\n".join(parts)

    def _format_history(self, limit=20):
        relevant = self.history[:-1][-limit:] if self.history else []
        if not relevant:
            return ""
        return "\n".join(["[" + m.role.upper() + "]: " + m.content[:500] for m in relevant])

    def _synthesize_response(self, msg, tool_calls):
        if not tool_calls:
            return "I was not able to process your request. Could you try rephrasing?"
        results = []
        for t in tool_calls:
            if t["result"].get("status") == "success":
                val = str(t["result"].get("message", t["result"].get("data", "")))[:200]
                results.append("- " + t["tool"] + ": " + val)
        if results:
            return "Here is what I found:\n" + "\n".join(results)
        return "I tried but could not get a clear result. Could you provide more details?"

    async def _execute(self, action):
        if action.tool in ("wait", "respond"):
            return {"status": "skipped"}
        if ":" in action.tool:
            sid, aname = action.tool.split(":", 1)
            skill = self.skills.get(sid)
            if skill:
                try:
                    r = await skill.execute(aname, action.params)
                    return {"status": "success" if r.success else "failed", "data": r.data, "message": r.message}
                except Exception as e:
                    return {"status": "error", "message": str(e)}
        return {"status": "error", "message": "Unknown tool: " + action.tool}

    def clear_history(self):
        self.history.clear()

    def get_stats(self):
        return {"messages": self.message_count, "total_cost_usd": round(self.total_cost, 6),
                "total_tokens": self.total_tokens, "history_length": len(self.history),
                "skills_loaded": len(self.skills.skills)}

    def get_history(self):
        return [{"role": m.role, "content": m.content, "timestamp": m.timestamp,
                 "metadata": m.metadata} for m in self.history]
