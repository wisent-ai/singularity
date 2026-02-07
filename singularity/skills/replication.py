#!/usr/bin/env python3
"""
Replication Skill - Enables agents to clone themselves across processes.

This skill gives agents the ability to:
- Snapshot their configuration into a portable format
- Spawn independent replicas as Docker containers
- Track and manage running replicas
- Budget resource allocation for replicas
- Create mutated clones with modified configurations

This is the core of the Replication pillar - enabling agents to scale
beyond a single process and persist across restarts.
"""

import json
import uuid
import asyncio
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
from .base import Skill, SkillManifest, SkillAction, SkillResult


class ReplicaStatus(str, Enum):
    """Status of a replica."""
    PENDING = "pending"
    STARTING = "starting"
    RUNNING = "running"
    STOPPED = "stopped"
    FAILED = "failed"
    TERMINATED = "terminated"


@dataclass
class AgentSnapshot:
    """Serializable snapshot of an agent's full configuration."""
    name: str
    ticker: str
    system_prompt: str
    llm_provider: str
    llm_model: str
    balance: float
    skills_config: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    snapshot_id: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        if not self.snapshot_id:
            self.snapshot_id = f"snap_{uuid.uuid4().hex[:12]}"

    def to_dict(self) -> Dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: Dict) -> "AgentSnapshot":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class Replica:
    """Tracks a spawned replica."""
    replica_id: str
    name: str
    container_id: Optional[str]
    snapshot_id: str
    status: ReplicaStatus
    budget: float
    mutations: Dict[str, Any]
    created_at: str = ""
    parent_id: str = ""
    port: Optional[int] = None
    error: Optional[str] = None

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["status"] = self.status.value
        return d


class ReplicationSkill(Skill):
    """
    Skill for agent self-replication across processes and containers.

    Enables agents to create independent copies of themselves that run
    in separate Docker containers with their own resources and budgets.
    """

    def __init__(self, credentials: Dict = None):
        super().__init__(credentials)
        self._snapshots: Dict[str, AgentSnapshot] = {}
        self._replicas: Dict[str, Replica] = {}
        self._my_agent: Optional[Any] = None
        self._data_dir: Path = Path("agent_data/replication")
        self._docker_image: str = "singularity:latest"
        self._max_replicas: int = 10
        self._total_budget_allocated: float = 0.0

    def set_agent(self, agent: Any):
        """Connect to the parent agent."""
        self._my_agent = agent

    def set_data_dir(self, path: str):
        """Set persistent storage directory."""
        self._data_dir = Path(path)

    def set_docker_image(self, image: str):
        """Set the Docker image to use for replicas."""
        self._docker_image = image

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="replication",
            name="Replication",
            version="1.0.0",
            category="existence",
            description="Clone yourself into independent replicas across containers",
            actions=[
                SkillAction(
                    name="snapshot",
                    description="Capture your current configuration as a portable snapshot",
                    parameters={
                        "label": {
                            "type": "string",
                            "required": False,
                            "description": "Optional label for this snapshot"
                        }
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="list_snapshots",
                    description="List all saved configuration snapshots",
                    parameters={},
                    estimated_cost=0,
                ),
                SkillAction(
                    name="spawn",
                    description="Spawn a new replica from a snapshot (or current config) as a Docker container",
                    parameters={
                        "name": {
                            "type": "string",
                            "required": True,
                            "description": "Name for the new replica"
                        },
                        "budget": {
                            "type": "number",
                            "required": True,
                            "description": "Budget to allocate from your wallet (USD)"
                        },
                        "snapshot_id": {
                            "type": "string",
                            "required": False,
                            "description": "Snapshot ID to spawn from (uses current config if not set)"
                        },
                        "mutations": {
                            "type": "object",
                            "required": False,
                            "description": "Config overrides: {model, prompt_addition, skills, env}"
                        },
                        "port": {
                            "type": "number",
                            "required": False,
                            "description": "Host port to expose the replica's API on"
                        }
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="list_replicas",
                    description="List all replicas and their status",
                    parameters={},
                    estimated_cost=0,
                ),
                SkillAction(
                    name="inspect_replica",
                    description="Get detailed status of a specific replica",
                    parameters={
                        "replica_id": {
                            "type": "string",
                            "required": True,
                            "description": "ID of the replica to inspect"
                        }
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="stop_replica",
                    description="Stop a running replica container",
                    parameters={
                        "replica_id": {
                            "type": "string",
                            "required": True,
                            "description": "ID of the replica to stop"
                        }
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="export_snapshot",
                    description="Export a snapshot as JSON for external use",
                    parameters={
                        "snapshot_id": {
                            "type": "string",
                            "required": True,
                            "description": "ID of the snapshot to export"
                        }
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="import_snapshot",
                    description="Import a snapshot from JSON",
                    parameters={
                        "snapshot_json": {
                            "type": "string",
                            "required": True,
                            "description": "JSON string of the snapshot to import"
                        }
                    },
                    estimated_cost=0,
                ),
            ],
            required_credentials=[],
        )

    def check_credentials(self) -> bool:
        return True

    async def execute(self, action: str, params: Dict) -> SkillResult:
        handlers = {
            "snapshot": self._snapshot,
            "list_snapshots": self._list_snapshots,
            "spawn": self._spawn,
            "list_replicas": self._list_replicas,
            "inspect_replica": self._inspect_replica,
            "stop_replica": self._stop_replica,
            "export_snapshot": self._export_snapshot,
            "import_snapshot": self._import_snapshot,
        }

        handler = handlers.get(action)
        if not handler:
            return SkillResult(success=False, message=f"Unknown action: {action}")
        return await handler(params)

    # === Snapshot Methods ===

    async def _snapshot(self, params: Dict) -> SkillResult:
        """Capture current agent configuration as a snapshot."""
        label = params.get("label", "")

        snapshot = self._capture_current_config()
        if label:
            snapshot.metadata["label"] = label

        self._snapshots[snapshot.snapshot_id] = snapshot
        self._persist_snapshot(snapshot)

        return SkillResult(
            success=True,
            message=f"Snapshot captured: {snapshot.snapshot_id}",
            data={
                "snapshot_id": snapshot.snapshot_id,
                "name": snapshot.name,
                "model": snapshot.llm_model,
                "balance": snapshot.balance,
                "label": label,
                "created_at": snapshot.created_at,
            }
        )

    async def _list_snapshots(self, params: Dict) -> SkillResult:
        """List all saved snapshots."""
        self._load_snapshots()
        snapshots = []
        for snap in self._snapshots.values():
            snapshots.append({
                "snapshot_id": snap.snapshot_id,
                "name": snap.name,
                "model": snap.llm_model,
                "balance": snap.balance,
                "label": snap.metadata.get("label", ""),
                "created_at": snap.created_at,
            })

        return SkillResult(
            success=True,
            message=f"{len(snapshots)} snapshot(s) available",
            data={"snapshots": snapshots, "count": len(snapshots)}
        )

    async def _export_snapshot(self, params: Dict) -> SkillResult:
        """Export a snapshot as JSON."""
        snapshot_id = params.get("snapshot_id", "").strip()
        if not snapshot_id:
            return SkillResult(success=False, message="snapshot_id required")

        snapshot = self._snapshots.get(snapshot_id)
        if not snapshot:
            self._load_snapshots()
            snapshot = self._snapshots.get(snapshot_id)
        if not snapshot:
            return SkillResult(success=False, message=f"Snapshot not found: {snapshot_id}")

        return SkillResult(
            success=True,
            message=f"Snapshot exported: {snapshot_id}",
            data={"snapshot_json": snapshot.to_json(), "snapshot_id": snapshot_id}
        )

    async def _import_snapshot(self, params: Dict) -> SkillResult:
        """Import a snapshot from JSON."""
        snapshot_json = params.get("snapshot_json", "").strip()
        if not snapshot_json:
            return SkillResult(success=False, message="snapshot_json required")

        try:
            data = json.loads(snapshot_json)
            snapshot = AgentSnapshot.from_dict(data)
            self._snapshots[snapshot.snapshot_id] = snapshot
            self._persist_snapshot(snapshot)

            return SkillResult(
                success=True,
                message=f"Snapshot imported: {snapshot.snapshot_id}",
                data={
                    "snapshot_id": snapshot.snapshot_id,
                    "name": snapshot.name,
                    "model": snapshot.llm_model,
                }
            )
        except (json.JSONDecodeError, TypeError, KeyError) as e:
            return SkillResult(success=False, message=f"Invalid snapshot JSON: {e}")

    # === Spawn Methods ===

    async def _spawn(self, params: Dict) -> SkillResult:
        """Spawn a new replica as a Docker container."""
        name = params.get("name", "").strip()
        budget = params.get("budget", 0)
        snapshot_id = params.get("snapshot_id", "")
        mutations = params.get("mutations", {})
        port = params.get("port")

        if not name:
            return SkillResult(success=False, message="Replica name required")
        if budget <= 0:
            return SkillResult(success=False, message="Budget must be positive")

        # Check replica limit
        active_replicas = sum(
            1 for r in self._replicas.values()
            if r.status in (ReplicaStatus.RUNNING, ReplicaStatus.STARTING)
        )
        if active_replicas >= self._max_replicas:
            return SkillResult(
                success=False,
                message=f"Max replicas ({self._max_replicas}) reached. Stop one first."
            )

        # Check parent budget
        if self._my_agent and hasattr(self._my_agent, "balance"):
            if self._my_agent.balance < budget:
                return SkillResult(
                    success=False,
                    message=f"Insufficient funds: ${self._my_agent.balance:.2f} < ${budget:.2f}"
                )

        # Get or create snapshot
        if snapshot_id:
            snapshot = self._snapshots.get(snapshot_id)
            if not snapshot:
                self._load_snapshots()
                snapshot = self._snapshots.get(snapshot_id)
            if not snapshot:
                return SkillResult(success=False, message=f"Snapshot not found: {snapshot_id}")
        else:
            snapshot = self._capture_current_config()

        # Apply mutations
        if mutations:
            snapshot = self._apply_mutations(snapshot, mutations)

        # Override balance with allocated budget
        snapshot.balance = budget

        # Create replica record
        replica_id = f"replica_{uuid.uuid4().hex[:8]}"
        replica = Replica(
            replica_id=replica_id,
            name=name,
            container_id=None,
            snapshot_id=snapshot.snapshot_id,
            status=ReplicaStatus.PENDING,
            budget=budget,
            mutations=mutations or {},
            parent_id=getattr(self._my_agent, "name", "unknown"),
            port=port,
        )

        # Deduct budget from parent
        if self._my_agent and hasattr(self._my_agent, "balance"):
            self._my_agent.balance -= budget
            self._total_budget_allocated += budget

        # Launch container
        try:
            replica.status = ReplicaStatus.STARTING
            container_id = await self._launch_container(replica, snapshot)
            replica.container_id = container_id
            replica.status = ReplicaStatus.RUNNING
        except Exception as e:
            replica.status = ReplicaStatus.FAILED
            replica.error = str(e)
            # Refund on failure
            if self._my_agent and hasattr(self._my_agent, "balance"):
                self._my_agent.balance += budget
                self._total_budget_allocated -= budget

        self._replicas[replica_id] = replica
        self._persist_replicas()

        if replica.status == ReplicaStatus.FAILED:
            return SkillResult(
                success=False,
                message=f"Failed to spawn replica: {replica.error}",
                data={"replica_id": replica_id}
            )

        return SkillResult(
            success=True,
            message=f"Replica '{name}' spawned with ${budget:.2f} budget",
            data=replica.to_dict()
        )

    async def _list_replicas(self, params: Dict) -> SkillResult:
        """List all replicas."""
        # Refresh status from Docker
        await self._refresh_replica_statuses()

        replicas = [r.to_dict() for r in self._replicas.values()]
        active = sum(1 for r in self._replicas.values() if r.status == ReplicaStatus.RUNNING)

        return SkillResult(
            success=True,
            message=f"{active} active replica(s) of {len(replicas)} total",
            data={
                "replicas": replicas,
                "active": active,
                "total": len(replicas),
                "total_budget_allocated": self._total_budget_allocated,
            }
        )

    async def _inspect_replica(self, params: Dict) -> SkillResult:
        """Get detailed status of a replica."""
        replica_id = params.get("replica_id", "").strip()
        if not replica_id:
            return SkillResult(success=False, message="replica_id required")

        replica = self._replicas.get(replica_id)
        if not replica:
            return SkillResult(success=False, message=f"Replica not found: {replica_id}")

        # Try to get live container info
        container_info = await self._get_container_info(replica)

        data = replica.to_dict()
        if container_info:
            data["container_info"] = container_info

        return SkillResult(
            success=True,
            message=f"Replica '{replica.name}' is {replica.status.value}",
            data=data
        )

    async def _stop_replica(self, params: Dict) -> SkillResult:
        """Stop a running replica."""
        replica_id = params.get("replica_id", "").strip()
        if not replica_id:
            return SkillResult(success=False, message="replica_id required")

        replica = self._replicas.get(replica_id)
        if not replica:
            return SkillResult(success=False, message=f"Replica not found: {replica_id}")

        if replica.status not in (ReplicaStatus.RUNNING, ReplicaStatus.STARTING):
            return SkillResult(
                success=False,
                message=f"Replica is {replica.status.value}, cannot stop"
            )

        try:
            if replica.container_id:
                await self._stop_container(replica.container_id)
            replica.status = ReplicaStatus.STOPPED
            self._persist_replicas()
            return SkillResult(
                success=True,
                message=f"Replica '{replica.name}' stopped",
                data=replica.to_dict()
            )
        except Exception as e:
            return SkillResult(success=False, message=f"Failed to stop replica: {e}")

    # === Internal Methods ===

    def _capture_current_config(self) -> AgentSnapshot:
        """Capture the current agent's configuration."""
        name = "unknown"
        ticker = "UNK"
        system_prompt = ""
        llm_provider = "anthropic"
        llm_model = "claude-sonnet-4-20250514"
        balance = 0.0
        skills_config = {}

        if self._my_agent:
            name = getattr(self._my_agent, "name", name)
            ticker = getattr(self._my_agent, "ticker", ticker)
            balance = getattr(self._my_agent, "balance", balance)

            cognition = getattr(self._my_agent, "cognition", None)
            if cognition:
                system_prompt = getattr(cognition, "system_prompt", "")
                llm_provider = getattr(cognition, "llm_type", llm_provider)
                llm_model = getattr(cognition, "llm_model", llm_model)

            # Capture installed skill IDs
            skills_registry = getattr(self._my_agent, "skills", None)
            if skills_registry and hasattr(skills_registry, "skills"):
                skills_config = {
                    sid: {"version": s.manifest.version, "category": s.manifest.category}
                    for sid, s in skills_registry.skills.items()
                }

        return AgentSnapshot(
            name=name,
            ticker=ticker,
            system_prompt=system_prompt,
            llm_provider=llm_provider,
            llm_model=llm_model,
            balance=balance,
            skills_config=skills_config,
            metadata={"captured_from": name},
        )

    def _apply_mutations(self, snapshot: AgentSnapshot, mutations: Dict) -> AgentSnapshot:
        """Apply mutations to a snapshot, returning a new modified copy."""
        data = snapshot.to_dict()

        if "model" in mutations:
            data["llm_model"] = mutations["model"]
        if "provider" in mutations:
            data["llm_provider"] = mutations["provider"]
        if "name" in mutations:
            data["name"] = mutations["name"]
        if "prompt_addition" in mutations:
            data["system_prompt"] += f"\n\n{mutations['prompt_addition']}"
        if "prompt_replace" in mutations:
            data["system_prompt"] = mutations["prompt_replace"]

        # Generate new snapshot ID for mutated version
        data["snapshot_id"] = f"snap_{uuid.uuid4().hex[:12]}"
        data["metadata"] = {**data.get("metadata", {}), "mutated_from": snapshot.snapshot_id}

        return AgentSnapshot.from_dict(data)

    async def _launch_container(self, replica: Replica, snapshot: AgentSnapshot) -> str:
        """Launch a Docker container for the replica. Returns container ID."""
        # Prepare environment variables for the replica
        env_vars = {
            "AGENT_NAME": snapshot.name,
            "AGENT_TICKER": snapshot.ticker,
            "STARTING_BALANCE": str(snapshot.balance),
            "LLM_PROVIDER": snapshot.llm_provider,
            "LLM_MODEL": snapshot.llm_model,
            "REPLICA_ID": replica.replica_id,
            "PARENT_AGENT": replica.parent_id,
        }

        # Pass through API keys from current environment
        for key in ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY",
                     "GITHUB_TOKEN", "TWITTER_BEARER_TOKEN"]:
            import os
            val = os.environ.get(key)
            if val:
                env_vars[key] = val

        # Save snapshot to a volume-mounted config file
        config_dir = self._data_dir / "configs" / replica.replica_id
        config_dir.mkdir(parents=True, exist_ok=True)
        config_path = config_dir / "snapshot.json"
        config_path.write_text(snapshot.to_json())

        # Build docker run command
        cmd = ["docker", "run", "-d", "--name", f"singularity-{replica.replica_id}"]

        # Add environment variables
        for key, val in env_vars.items():
            cmd.extend(["-e", f"{key}={val}"])

        # Mount config
        cmd.extend(["-v", f"{config_dir.resolve()}:/app/agent_data/config:ro"])

        # Expose port if specified
        if replica.port:
            cmd.extend(["-p", f"{replica.port}:8000"])

        # Resource limits (prevent runaway replicas)
        cmd.extend(["--memory", "512m", "--cpus", "0.5"])

        # Use the configured image
        cmd.append(self._docker_image)

        # Run the command
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            error_msg = stderr.decode().strip()
            raise RuntimeError(f"Docker run failed: {error_msg}")

        container_id = stdout.decode().strip()[:12]
        return container_id

    async def _stop_container(self, container_id: str):
        """Stop a Docker container."""
        proc = await asyncio.create_subprocess_exec(
            "docker", "stop", container_id,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate()

    async def _get_container_info(self, replica: Replica) -> Optional[Dict]:
        """Get live container information."""
        if not replica.container_id:
            return None

        try:
            proc = await asyncio.create_subprocess_exec(
                "docker", "inspect", replica.container_id,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
            if proc.returncode == 0:
                info = json.loads(stdout.decode())
                if info:
                    container = info[0]
                    return {
                        "state": container.get("State", {}).get("Status", "unknown"),
                        "started_at": container.get("State", {}).get("StartedAt", ""),
                        "pid": container.get("State", {}).get("Pid", 0),
                    }
        except Exception:
            pass
        return None

    async def _refresh_replica_statuses(self):
        """Refresh the status of all replicas from Docker."""
        for replica in self._replicas.values():
            if replica.status in (ReplicaStatus.RUNNING, ReplicaStatus.STARTING):
                info = await self._get_container_info(replica)
                if info:
                    docker_status = info.get("state", "")
                    if docker_status == "running":
                        replica.status = ReplicaStatus.RUNNING
                    elif docker_status in ("exited", "dead"):
                        replica.status = ReplicaStatus.STOPPED
                elif replica.container_id:
                    # Container not found, mark as stopped
                    replica.status = ReplicaStatus.STOPPED

    # === Persistence ===

    def _persist_snapshot(self, snapshot: AgentSnapshot):
        """Save a snapshot to disk."""
        snap_dir = self._data_dir / "snapshots"
        snap_dir.mkdir(parents=True, exist_ok=True)
        path = snap_dir / f"{snapshot.snapshot_id}.json"
        path.write_text(snapshot.to_json())

    def _load_snapshots(self):
        """Load all snapshots from disk."""
        snap_dir = self._data_dir / "snapshots"
        if not snap_dir.exists():
            return
        for path in snap_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text())
                snapshot = AgentSnapshot.from_dict(data)
                if snapshot.snapshot_id not in self._snapshots:
                    self._snapshots[snapshot.snapshot_id] = snapshot
            except (json.JSONDecodeError, KeyError):
                pass

    def _persist_replicas(self):
        """Save replica registry to disk."""
        self._data_dir.mkdir(parents=True, exist_ok=True)
        path = self._data_dir / "replicas.json"
        data = {rid: r.to_dict() for rid, r in self._replicas.items()}
        path.write_text(json.dumps(data, indent=2))

    def _load_replicas(self):
        """Load replica registry from disk."""
        path = self._data_dir / "replicas.json"
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text())
            for rid, rdata in data.items():
                if rid not in self._replicas:
                    rdata["status"] = ReplicaStatus(rdata["status"])
                    self._replicas[rid] = Replica(**{
                        k: v for k, v in rdata.items()
                        if k in Replica.__dataclass_fields__
                    })
        except (json.JSONDecodeError, KeyError):
            pass
