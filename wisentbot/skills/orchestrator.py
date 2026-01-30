#!/usr/bin/env python3
"""
Orchestrator Skill - Spawn and manage child agents.

Allows agents to:
- Spawn new agents with specific configurations
- Delegate tasks to child agents
- Monitor child agent status
- Communicate between agents
- Terminate child agents

This enables hierarchical agent architectures where a parent agent
can create specialized workers for subtasks.
"""

import asyncio
import uuid
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from .base import Skill, SkillManifest, SkillAction, SkillResult


class AgentStatus(Enum):
    """Status of a spawned agent."""
    STARTING = "starting"
    RUNNING = "running"
    IDLE = "idle"
    WORKING = "working"
    COMPLETED = "completed"
    FAILED = "failed"
    TERMINATED = "terminated"


@dataclass
class SpawnedAgent:
    """Represents a spawned child agent."""
    id: str
    name: str
    specialty: str
    status: AgentStatus
    created_at: datetime
    parent_id: str
    task: Optional[str] = None
    result: Optional[str] = None
    error: Optional[str] = None
    agent_instance: Optional[Any] = None
    task_handle: Optional[asyncio.Task] = None


class OrchestratorSkill(Skill):
    """
    Skill for spawning and managing child agents.

    Enables multi-agent architectures:
    - Parent agents can spawn specialized workers
    - Tasks can be delegated and results collected
    - Agents can communicate via message passing
    - Hierarchical agent structures
    """

    def __init__(self, credentials: Dict = None):
        super().__init__(credentials)

        # Parent agent reference (set by autonomous_agent)
        self._parent_agent: Optional[Any] = None
        self._parent_id: str = str(uuid.uuid4())

        # Spawned agents registry
        self._children: Dict[str, SpawnedAgent] = {}

        # Message queues for inter-agent communication
        self._message_queues: Dict[str, asyncio.Queue] = {}

        # Agent factory function (set by autonomous_agent)
        self._agent_factory: Optional[Callable] = None

    def set_parent_agent(self, agent: Any, agent_factory: Callable = None):
        """Set the parent agent and factory for creating children."""
        self._parent_agent = agent
        self._parent_id = getattr(agent, 'name', str(uuid.uuid4()))
        self._agent_factory = agent_factory

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="orchestrator",
            name="Agent Orchestrator",
            version="1.0.0",
            category="meta",
            description="Spawn and manage child agents",
            actions=[
                # === Spawning ===
                SkillAction(
                    name="spawn",
                    description="Spawn a new child agent",
                    parameters={
                        "name": {
                            "type": "string",
                            "required": True,
                            "description": "Name for the child agent"
                        },
                        "specialty": {
                            "type": "string",
                            "required": True,
                            "description": "What this agent specializes in"
                        },
                        "task": {
                            "type": "string",
                            "required": False,
                            "description": "Initial task to assign (optional)"
                        },
                        "budget": {
                            "type": "number",
                            "required": False,
                            "description": "Budget allocation in USD (default: 1.0)"
                        },
                        "model": {
                            "type": "string",
                            "required": False,
                            "description": "LLM model to use (inherits from parent if not set)"
                        }
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="spawn_team",
                    description="Spawn multiple specialized agents at once",
                    parameters={
                        "agents": {
                            "type": "string",
                            "required": True,
                            "description": "JSON array of agent specs: [{name, specialty, task?}]"
                        },
                        "budget_each": {
                            "type": "number",
                            "required": False,
                            "description": "Budget per agent (default: 1.0)"
                        }
                    },
                    estimated_cost=0,
                ),
                # === Task Management ===
                SkillAction(
                    name="assign",
                    description="Assign a task to a child agent",
                    parameters={
                        "agent_id": {
                            "type": "string",
                            "required": True,
                            "description": "ID of the child agent"
                        },
                        "task": {
                            "type": "string",
                            "required": True,
                            "description": "Task description to assign"
                        }
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="broadcast",
                    description="Send a task to all child agents",
                    parameters={
                        "task": {
                            "type": "string",
                            "required": True,
                            "description": "Task to broadcast"
                        }
                    },
                    estimated_cost=0,
                ),
                # === Communication ===
                SkillAction(
                    name="message",
                    description="Send a message to a child agent",
                    parameters={
                        "agent_id": {
                            "type": "string",
                            "required": True,
                            "description": "ID of the child agent"
                        },
                        "message": {
                            "type": "string",
                            "required": True,
                            "description": "Message to send"
                        }
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="collect_results",
                    description="Collect results from all completed child agents",
                    parameters={},
                    estimated_cost=0,
                ),
                SkillAction(
                    name="wait_for",
                    description="Wait for a specific agent to complete its task",
                    parameters={
                        "agent_id": {
                            "type": "string",
                            "required": True,
                            "description": "ID of the agent to wait for"
                        },
                        "timeout": {
                            "type": "number",
                            "required": False,
                            "description": "Timeout in seconds (default: 300)"
                        }
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="wait_all",
                    description="Wait for all child agents to complete",
                    parameters={
                        "timeout": {
                            "type": "number",
                            "required": False,
                            "description": "Timeout in seconds (default: 600)"
                        }
                    },
                    estimated_cost=0,
                ),
                # === Monitoring ===
                SkillAction(
                    name="status",
                    description="Get status of all child agents",
                    parameters={},
                    estimated_cost=0,
                ),
                SkillAction(
                    name="get_result",
                    description="Get the result from a specific child agent",
                    parameters={
                        "agent_id": {
                            "type": "string",
                            "required": True,
                            "description": "ID of the child agent"
                        }
                    },
                    estimated_cost=0,
                ),
                # === Lifecycle ===
                SkillAction(
                    name="terminate",
                    description="Terminate a child agent",
                    parameters={
                        "agent_id": {
                            "type": "string",
                            "required": True,
                            "description": "ID of the agent to terminate"
                        }
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="terminate_all",
                    description="Terminate all child agents",
                    parameters={},
                    estimated_cost=0,
                ),
            ],
            required_credentials=[],
        )

    def check_credentials(self) -> bool:
        """Check if orchestration is available."""
        return self._agent_factory is not None

    async def execute(self, action: str, params: Dict) -> SkillResult:
        if not self._agent_factory:
            return SkillResult(
                success=False,
                message="Orchestrator not configured. Agent factory not set."
            )

        handlers = {
            "spawn": self._spawn,
            "spawn_team": self._spawn_team,
            "assign": self._assign,
            "broadcast": self._broadcast,
            "message": self._message,
            "collect_results": self._collect_results,
            "wait_for": self._wait_for,
            "wait_all": self._wait_all,
            "status": self._status,
            "get_result": self._get_result,
            "terminate": self._terminate,
            "terminate_all": self._terminate_all,
        }

        handler = handlers.get(action)
        if handler:
            return await handler(params)
        return SkillResult(success=False, message=f"Unknown action: {action}")

    # === Handlers ===

    async def _spawn(self, params: Dict) -> SkillResult:
        """Spawn a new child agent."""
        name = params.get("name", "").strip()
        specialty = params.get("specialty", "").strip()
        task = params.get("task", "").strip()
        budget = params.get("budget", 1.0)
        model = params.get("model")

        if not name or not specialty:
            return SkillResult(success=False, message="name and specialty required")

        try:
            # Generate unique ID
            agent_id = f"{name.lower().replace(' ', '_')}_{uuid.uuid4().hex[:8]}"

            # Get parent's config for inheritance
            parent_config = {}
            if self._parent_agent:
                parent_config = {
                    "llm_provider": getattr(self._parent_agent, 'cognition', {}).get('llm_type', 'anthropic'),
                    "llm_model": model or getattr(self._parent_agent, 'cognition', {}).get('llm_model'),
                }

            # Create child agent using factory
            child_agent = self._agent_factory(
                name=name,
                ticker=agent_id[:6].upper(),
                specialty=specialty,
                starting_balance=budget,
                **parent_config,
            )

            # Create message queue
            self._message_queues[agent_id] = asyncio.Queue()

            # Register spawned agent
            spawned = SpawnedAgent(
                id=agent_id,
                name=name,
                specialty=specialty,
                status=AgentStatus.STARTING,
                created_at=datetime.now(),
                parent_id=self._parent_id,
                task=task if task else None,
                agent_instance=child_agent,
            )
            self._children[agent_id] = spawned

            # If task provided, start running it
            if task:
                spawned.status = AgentStatus.WORKING
                spawned.task_handle = asyncio.create_task(
                    self._run_agent_task(agent_id, task)
                )
            else:
                spawned.status = AgentStatus.IDLE

            return SkillResult(
                success=True,
                message=f"Spawned agent '{name}' ({agent_id})",
                data={
                    "agent_id": agent_id,
                    "name": name,
                    "specialty": specialty,
                    "budget": budget,
                    "status": spawned.status.value,
                    "task": task if task else None,
                }
            )
        except Exception as e:
            return SkillResult(success=False, message=f"Spawn failed: {e}")

    async def _spawn_team(self, params: Dict) -> SkillResult:
        """Spawn multiple agents at once."""
        import json

        agents_json = params.get("agents", "")
        budget_each = params.get("budget_each", 1.0)

        try:
            agent_specs = json.loads(agents_json)
        except json.JSONDecodeError:
            return SkillResult(success=False, message="Invalid JSON for agents")

        if not isinstance(agent_specs, list):
            return SkillResult(success=False, message="agents must be a JSON array")

        spawned_ids = []
        errors = []

        for spec in agent_specs:
            if not isinstance(spec, dict):
                continue

            result = await self._spawn({
                "name": spec.get("name", "Worker"),
                "specialty": spec.get("specialty", "general"),
                "task": spec.get("task", ""),
                "budget": budget_each,
            })

            if result.success:
                spawned_ids.append(result.data.get("agent_id"))
            else:
                errors.append(result.message)

        return SkillResult(
            success=len(spawned_ids) > 0,
            message=f"Spawned {len(spawned_ids)} agents" + (f", {len(errors)} failed" if errors else ""),
            data={
                "agent_ids": spawned_ids,
                "count": len(spawned_ids),
                "errors": errors if errors else None,
            }
        )

    async def _assign(self, params: Dict) -> SkillResult:
        """Assign a task to a child agent."""
        agent_id = params.get("agent_id", "")
        task = params.get("task", "").strip()

        if agent_id not in self._children:
            return SkillResult(success=False, message=f"Agent '{agent_id}' not found")

        if not task:
            return SkillResult(success=False, message="Task required")

        spawned = self._children[agent_id]

        # Cancel any existing task
        if spawned.task_handle and not spawned.task_handle.done():
            spawned.task_handle.cancel()

        # Start new task
        spawned.task = task
        spawned.result = None
        spawned.error = None
        spawned.status = AgentStatus.WORKING
        spawned.task_handle = asyncio.create_task(
            self._run_agent_task(agent_id, task)
        )

        return SkillResult(
            success=True,
            message=f"Assigned task to '{spawned.name}'",
            data={
                "agent_id": agent_id,
                "task": task,
            }
        )

    async def _broadcast(self, params: Dict) -> SkillResult:
        """Broadcast task to all idle children."""
        task = params.get("task", "").strip()

        if not task:
            return SkillResult(success=False, message="Task required")

        assigned = []
        for agent_id, spawned in self._children.items():
            if spawned.status in (AgentStatus.IDLE, AgentStatus.COMPLETED):
                spawned.task = task
                spawned.result = None
                spawned.error = None
                spawned.status = AgentStatus.WORKING
                spawned.task_handle = asyncio.create_task(
                    self._run_agent_task(agent_id, task)
                )
                assigned.append(agent_id)

        return SkillResult(
            success=len(assigned) > 0,
            message=f"Broadcast to {len(assigned)} agents",
            data={
                "assigned_to": assigned,
                "count": len(assigned),
            }
        )

    async def _message(self, params: Dict) -> SkillResult:
        """Send message to a child agent."""
        agent_id = params.get("agent_id", "")
        message = params.get("message", "").strip()

        if agent_id not in self._children:
            return SkillResult(success=False, message=f"Agent '{agent_id}' not found")

        if not message:
            return SkillResult(success=False, message="Message required")

        # Put message in agent's queue
        await self._message_queues[agent_id].put({
            "from": self._parent_id,
            "message": message,
            "timestamp": datetime.now().isoformat(),
        })

        return SkillResult(
            success=True,
            message=f"Message sent to '{self._children[agent_id].name}'",
            data={"agent_id": agent_id}
        )

    async def _collect_results(self, params: Dict) -> SkillResult:
        """Collect results from all completed agents."""
        results = {}
        pending = []

        for agent_id, spawned in self._children.items():
            if spawned.status == AgentStatus.COMPLETED:
                results[agent_id] = {
                    "name": spawned.name,
                    "task": spawned.task,
                    "result": spawned.result,
                }
            elif spawned.status == AgentStatus.FAILED:
                results[agent_id] = {
                    "name": spawned.name,
                    "task": spawned.task,
                    "error": spawned.error,
                }
            elif spawned.status == AgentStatus.WORKING:
                pending.append(agent_id)

        return SkillResult(
            success=True,
            message=f"Collected {len(results)} results, {len(pending)} pending",
            data={
                "results": results,
                "pending": pending,
                "completed_count": len(results),
                "pending_count": len(pending),
            }
        )

    async def _wait_for(self, params: Dict) -> SkillResult:
        """Wait for a specific agent to complete."""
        agent_id = params.get("agent_id", "")
        timeout = params.get("timeout", 300)

        if agent_id not in self._children:
            return SkillResult(success=False, message=f"Agent '{agent_id}' not found")

        spawned = self._children[agent_id]

        if spawned.task_handle:
            try:
                await asyncio.wait_for(spawned.task_handle, timeout=timeout)
            except asyncio.TimeoutError:
                return SkillResult(
                    success=False,
                    message=f"Timeout waiting for '{spawned.name}'"
                )

        return SkillResult(
            success=spawned.status == AgentStatus.COMPLETED,
            message=f"Agent '{spawned.name}' {spawned.status.value}",
            data={
                "agent_id": agent_id,
                "status": spawned.status.value,
                "result": spawned.result,
                "error": spawned.error,
            }
        )

    async def _wait_all(self, params: Dict) -> SkillResult:
        """Wait for all child agents to complete."""
        timeout = params.get("timeout", 600)

        tasks = [
            s.task_handle for s in self._children.values()
            if s.task_handle and not s.task_handle.done()
        ]

        if tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                return SkillResult(
                    success=False,
                    message="Timeout waiting for agents"
                )

        # Collect final results
        return await self._collect_results({})

    async def _status(self, params: Dict) -> SkillResult:
        """Get status of all child agents."""
        agents = {}
        for agent_id, spawned in self._children.items():
            agents[agent_id] = {
                "name": spawned.name,
                "specialty": spawned.specialty,
                "status": spawned.status.value,
                "task": spawned.task,
                "created_at": spawned.created_at.isoformat(),
            }

        by_status = {}
        for spawned in self._children.values():
            status = spawned.status.value
            by_status[status] = by_status.get(status, 0) + 1

        return SkillResult(
            success=True,
            message=f"Managing {len(self._children)} child agents",
            data={
                "agents": agents,
                "count": len(self._children),
                "by_status": by_status,
                "parent_id": self._parent_id,
            }
        )

    async def _get_result(self, params: Dict) -> SkillResult:
        """Get result from a specific agent."""
        agent_id = params.get("agent_id", "")

        if agent_id not in self._children:
            return SkillResult(success=False, message=f"Agent '{agent_id}' not found")

        spawned = self._children[agent_id]

        return SkillResult(
            success=True,
            message=f"Result from '{spawned.name}'",
            data={
                "agent_id": agent_id,
                "name": spawned.name,
                "status": spawned.status.value,
                "task": spawned.task,
                "result": spawned.result,
                "error": spawned.error,
            }
        )

    async def _terminate(self, params: Dict) -> SkillResult:
        """Terminate a child agent."""
        agent_id = params.get("agent_id", "")

        if agent_id not in self._children:
            return SkillResult(success=False, message=f"Agent '{agent_id}' not found")

        spawned = self._children[agent_id]

        # Cancel task if running
        if spawned.task_handle and not spawned.task_handle.done():
            spawned.task_handle.cancel()

        # Stop agent
        if spawned.agent_instance and hasattr(spawned.agent_instance, 'stop'):
            spawned.agent_instance.stop()

        spawned.status = AgentStatus.TERMINATED

        # Clean up
        if agent_id in self._message_queues:
            del self._message_queues[agent_id]

        return SkillResult(
            success=True,
            message=f"Terminated agent '{spawned.name}'",
            data={"agent_id": agent_id}
        )

    async def _terminate_all(self, params: Dict) -> SkillResult:
        """Terminate all child agents."""
        terminated = []
        for agent_id in list(self._children.keys()):
            result = await self._terminate({"agent_id": agent_id})
            if result.success:
                terminated.append(agent_id)

        return SkillResult(
            success=True,
            message=f"Terminated {len(terminated)} agents",
            data={
                "terminated": terminated,
                "count": len(terminated),
            }
        )

    # === Internal Methods ===

    async def _run_agent_task(self, agent_id: str, task: str):
        """Run a task on a child agent."""
        spawned = self._children.get(agent_id)
        if not spawned or not spawned.agent_instance:
            return

        try:
            # Run the agent with the task
            agent = spawned.agent_instance

            # Inject task into agent's context
            if hasattr(agent, 'cognition'):
                # Add task to conversation
                result = await agent.cognition.think(
                    f"Your task: {task}\n\nComplete this task and report the result."
                )
                spawned.result = str(result)

            spawned.status = AgentStatus.COMPLETED

        except asyncio.CancelledError:
            spawned.status = AgentStatus.TERMINATED
            raise

        except Exception as e:
            spawned.status = AgentStatus.FAILED
            spawned.error = str(e)
