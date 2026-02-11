"""
Singularity Autonomous Agent - generic agent using the full engine.

Uses CognitionEngine (multi-turn, multi-model), SkillRegistry with
PluginLoader for lazy-loaded skills, and SKILL.md/MCP support.
"""

import multiprocessing
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass

import asyncio
import os
import json
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
from datetime import datetime
from typing import Dict, List, Optional

ACTIVITY_FILE = Path(__file__).parent / "data" / "activity.json"

from .cognition import CognitionEngine, AgentState, Decision, Action, TokenUsage
from .cognition.prompt_builder import build_result_message
from .skills.base import SkillRegistry
from .skills.loader import PluginLoader


class AutonomousAgent:
    """
    An autonomous AI agent powered by singularity engine.

    Uses PluginLoader for lazy skill loading from registry.json,
    CognitionEngine for multi-turn decision making, and supports
    SKILL.md and MCP server integration.
    """

    INSTANCE_COSTS = {
        "e2-micro": 0.0084, "e2-small": 0.0168, "e2-medium": 0.0336,
        "e2-standard-2": 0.0672, "g2-standard-4": 0.7111, "local": 0.0,
    }

    def __init__(
        self, name="Agent", ticker="AGENT", agent_type="general", specialty="",
        starting_balance=100.0, instance_type="local", cycle_interval_seconds=5.0,
        llm_provider="anthropic", llm_base_url="http://localhost:8000/v1",
        llm_model="claude-sonnet-4-20250514", anthropic_api_key="", openai_api_key="",
        system_prompt=None, system_prompt_file=None, registry_path=None,
    ):
        self.name = name
        self.ticker = ticker
        self.agent_type = agent_type
        self.specialty = specialty or agent_type
        self.balance = starting_balance
        self.instance_type = instance_type
        self.cycle_interval = cycle_interval_seconds
        self.instance_cost_per_hour = self.INSTANCE_COSTS.get(instance_type, 0.0)
        self.total_api_cost = 0.0
        self.total_instance_cost = 0.0
        self.total_tokens_used = 0

        self.cognition = CognitionEngine(
            llm_provider=llm_provider,
            anthropic_api_key=anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY", ""),
            openai_api_key=openai_api_key or os.environ.get("OPENAI_API_KEY", ""),
            openai_base_url=llm_base_url, llm_model=llm_model,
            agent_name=name, agent_ticker=ticker, agent_type=agent_type,
            agent_specialty=self.specialty,
            system_prompt=system_prompt or "", system_prompt_file=system_prompt_file or "",
        )

        # Initialize skills via PluginLoader
        loader = PluginLoader(registry_path=registry_path)
        self.skills = SkillRegistry(loader=loader)
        self.skills.set_agent(self)
        self._init_skills()

        self.recent_actions: List[Dict] = []
        self.cycle = 0
        self.running = False
        self.conversation: List[Dict] = []
        self.created_resources: Dict[str, List] = {
            'payment_links': [], 'products': [], 'files': [], 'repos': [],
        }

    def _init_skills(self):
        """Install all skills that have credentials via lazy loader."""
        credentials = {k: os.environ.get(k, "") for k in [
            "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "TWITTER_API_KEY",
            "TWITTER_API_SECRET", "TWITTER_ACCESS_TOKEN", "TWITTER_ACCESS_SECRET",
            "GITHUB_TOKEN", "RESEND_API_KEY", "VERCEL_TOKEN",
            "STRIPE_SECRET_KEY", "REDDIT_USERNAME", "REDDIT_PASSWORD",
        ]}
        self.skills.set_credentials(credentials)
        installed = self.skills.install_all_available()
        for sid in installed:
            self._log("SKILL", f"+ {sid}")

    def _get_tools(self) -> List[Dict]:
        tools = []
        for skill in self.skills.skills.values():
            for action in skill.manifest.actions:
                tools.append({
                    "name": f"{skill.manifest.skill_id}:{action.name}",
                    "description": action.description,
                    "parameters": action.parameters,
                })
        return tools or [{"name": "wait", "description": "No tools. Wait.", "parameters": {}}]

    async def run(self):
        """Main agent loop with multi-turn conversation."""
        self.running = True
        tools = self._get_tools()
        cycle_start_time = datetime.now()
        self._log("AWAKE", f"{self.name} (${self.ticker}) - {self.agent_type}")
        self._log("BALANCE", f"${self.balance:.4f} USD")
        self._log("TOOLS", f"{len(tools)} available")

        while self.running and self.balance > 0:
            self.cycle += 1
            cycle_start = datetime.now()
            avg_hours = self.cycle_interval / 3600
            est_cost = 0.01 + (self.instance_cost_per_hour * avg_hours)
            runway_cycles = self.balance / est_cost if est_cost > 0 else float('inf')
            runway_hours = runway_cycles * avg_hours
            self._log("CYCLE", f"#{self.cycle} | ${self.balance:.4f} | ~{runway_cycles:.0f} left")

            state = AgentState(
                balance=self.balance, burn_rate=est_cost, runway_hours=runway_hours,
                tools=self._get_tools(), recent_actions=self.recent_actions[-10:],
                cycle=self.cycle, created_resources=self.created_resources,
            )

            decision, self.conversation = await self.cognition.think_with_context(state, self.conversation)
            self._log("THINK", decision.reasoning[:150] if decision.reasoning else "...")
            self._log("DO", f"{decision.action.tool} {decision.action.params}")

            result = await self._execute(decision.action)
            self._log("RESULT", str(result)[:200])

            # Feed result back into conversation
            result_msg = build_result_message(decision.action.tool, decision.action.params, result)
            self.conversation.append({"role": "user", "content": result_msg})

            self.recent_actions.append({
                "cycle": self.cycle, "tool": decision.action.tool,
                "params": decision.action.params, "result": result,
                "api_cost_usd": decision.api_cost_usd,
                "tokens": decision.token_usage.total_tokens(),
            })

            duration_hours = (datetime.now() - cycle_start).total_seconds() / 3600
            inst_cost = self.instance_cost_per_hour * duration_hours
            total_cost = inst_cost + decision.api_cost_usd
            self.total_api_cost += decision.api_cost_usd
            self.total_instance_cost += inst_cost
            self.total_tokens_used += decision.token_usage.total_tokens()
            self.balance -= total_cost
            self._log("COST", f"API: ${decision.api_cost_usd:.6f} + Instance: ${inst_cost:.6f}")
            await asyncio.sleep(self.cycle_interval)

        total_hours = (datetime.now() - cycle_start_time).total_seconds() / 3600
        self._log("END", f"Balance: ${self.balance:.4f}")
        self._log("SUMMARY", f"{self.cycle} cycles in {total_hours:.2f}h | API: ${self.total_api_cost:.4f}")
        self._mark_stopped()

    async def _execute(self, action: Action) -> Dict:
        if action.tool == "wait":
            return {"status": "waited"}
        if ":" in action.tool:
            skill_id, action_name = action.tool.split(":", 1)
            skill = self.skills.get(skill_id)
            if skill:
                try:
                    if not skill.initialized:
                        if not await skill.initialize():
                            return {"status": "error", "message": f"Skill '{skill_id}' failed to initialize: missing {skill.get_missing_credentials()}"}
                    result = await skill.execute(action_name, action.params)
                    return {"status": "success" if result.success else "failed",
                            "data": result.data, "message": result.message}
                except Exception as e:
                    return {"status": "error", "message": str(e)}
        return {"status": "error", "message": f"Unknown tool: {action.tool}"}

    def _log(self, tag, msg):
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] [{self.ticker}] [{tag}] {msg}")
        self._save_activity(tag, msg)

    def _save_activity(self, tag, msg):
        try:
            ACTIVITY_FILE.parent.mkdir(parents=True, exist_ok=True)
            if ACTIVITY_FILE.exists():
                with open(ACTIVITY_FILE) as f:
                    data = json.load(f)
            else:
                data = {"status": "stopped", "logs": [], "state": {}}
            avg_hours = self.cycle_interval / 3600
            est_cost = 0.01 + (self.instance_cost_per_hour * avg_hours)
            data["status"] = "running" if self.running else "stopped"
            data["state"] = {
                "name": self.name, "ticker": self.ticker, "agent_type": self.agent_type,
                "balance_usd": self.balance, "total_api_cost": self.total_api_cost,
                "total_tokens_used": self.total_tokens_used,
                "runway_cycles": self.balance / est_cost if est_cost > 0 else 0,
                "cycle": self.cycle, "updated_at": datetime.now().isoformat(),
            }
            data["logs"].append({"timestamp": datetime.now().isoformat(), "tag": tag, "message": msg[:500]})
            data["logs"] = data["logs"][-100:]
            with open(ACTIVITY_FILE, 'w') as f: json.dump(data, f, indent=2)
        except Exception:
            pass

    def _mark_stopped(self):
        try:
            if ACTIVITY_FILE.exists():
                with open(ACTIVITY_FILE) as f:
                    data = json.load(f)
                data["status"] = "stopped"
                data["state"]["updated_at"] = datetime.now().isoformat()
                with open(ACTIVITY_FILE, 'w') as f: json.dump(data, f, indent=2)
        except Exception:
            pass

    def stop(self):
        self.running = False


async def main():
    agent = AutonomousAgent(
        name=os.environ.get("AGENT_NAME", "MyAgent"),
        ticker=os.environ.get("AGENT_TICKER", "AGENT"),
        agent_type=os.environ.get("AGENT_TYPE", "general"),
        starting_balance=float(os.environ.get("STARTING_BALANCE", 10.0)),
        llm_provider=os.environ.get("LLM_PROVIDER", "anthropic"),
        llm_model=os.environ.get("LLM_MODEL", "claude-sonnet-4-20250514"),
    )
    await agent.run()


if __name__ == "__main__":
    asyncio.run(main())
