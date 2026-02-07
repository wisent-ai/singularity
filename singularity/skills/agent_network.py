#!/usr/bin/env python3
"""
AgentNetworkSkill - Service discovery, capability routing, and RPC for agent meshes.

Transforms isolated agents into a collaborative network by enabling:

1. REGISTER - Agents announce themselves with their capabilities and endpoint
2. DISCOVER - Find agents by capability, status, or name
3. RPC      - Call a remote agent's skill directly via its endpoint
4. ROUTE    - Find the best agent for a specific task based on advertised skills
5. TOPOLOGY - View the full network graph of connected agents

This is the critical missing piece in the Replication pillar. Without it, replicas
are isolated islands. With it, they form an intelligent mesh that can:
- Distribute work to the most capable agent
- Share load when one agent is overwhelmed
- Auto-failover when an agent goes down
- Collectively offer more capabilities than any single agent

Works with:
- ReplicationSkill: spawned replicas auto-register on the network
- HealthMonitor: network status informed by health checks
- InboxSkill: fallback messaging when direct RPC unavailable
- TaskDelegator: routes tasks to best agent via capability matching
- KnowledgeSharingSkill: network-wide knowledge propagation

Pillars served: Replication (primary), Revenue (service discovery for customers)
"""

import json
import time
import uuid
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from .base import Skill, SkillResult, SkillManifest, SkillAction

# Optional HTTP support for RPC
try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

NETWORK_FILE = Path(__file__).parent.parent / "data" / "agent_network.json"

# Limits
MAX_PEERS = 200
MAX_CAPABILITIES_PER_AGENT = 50
MAX_RPC_LOG_ENTRIES = 500
MAX_PENDING_RPCS = 100


def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _generate_agent_id(name: str, endpoint: str) -> str:
    """Generate deterministic agent ID from name + endpoint."""
    raw = f"{name}:{endpoint}"
    return f"agent_{hashlib.sha256(raw.encode()).hexdigest()[:12]}"


def _load_data(path: Path = None) -> Dict:
    """Load network data from disk."""
    p = path or NETWORK_FILE
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return _default_data()


def _save_data(data: Dict, path: Path = None):
    """Save network data to disk."""
    p = path or NETWORK_FILE
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(data, f, indent=2, default=str)


def _default_data() -> Dict:
    return {
        "self": None,  # This agent's registration
        "peers": {},   # agent_id -> PeerRecord
        "rpc_log": [], # History of RPC calls made/received
        "pending_rpcs": [],  # RPCs waiting for response
        "config": {
            "rpc_timeout_seconds": 30,
            "peer_ttl_seconds": 3600,  # Peers expire after 1 hour without refresh
            "auto_deregister_on_shutdown": True,
            "max_rpc_retries": 2,
            "broadcast_on_register": True,  # Announce to all known peers
        },
        "stats": {
            "rpcs_sent": 0,
            "rpcs_received": 0,
            "rpcs_succeeded": 0,
            "rpcs_failed": 0,
            "discoveries": 0,
            "registrations": 0,
        },
    }


class AgentNetworkSkill(Skill):
    """
    Agent mesh networking skill for service discovery, capability routing, and RPC.

    Enables agents to form a collaborative network where they can find each other,
    advertise their capabilities, and call each other's skills directly.
    """

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self._data_path = NETWORK_FILE
        self._ensure_data()

    def _ensure_data(self):
        self._data_path.parent.mkdir(parents=True, exist_ok=True)
        if not self._data_path.exists():
            _save_data(_default_data(), self._data_path)

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="agent_network",
            name="Agent Network",
            version="1.0.0",
            category="replication",
            description="Service discovery, capability routing, and RPC for agent mesh networking",
            required_credentials=[],
            actions=[
                SkillAction(
                    name="register",
                    description="Register this agent on the network with its capabilities and endpoint",
                    parameters={
                        "name": {"type": "string", "required": True, "description": "Agent name"},
                        "endpoint": {"type": "string", "required": True, "description": "Agent's reachable endpoint (URL or file path)"},
                        "capabilities": {"type": "list", "required": True, "description": "List of capability dicts with name, description, skill_id"},
                        "metadata": {"type": "dict", "required": False, "description": "Additional agent metadata"},
                    },
                ),
                SkillAction(
                    name="deregister",
                    description="Remove this agent from the network",
                    parameters={},
                ),
                SkillAction(
                    name="add_peer",
                    description="Add a known peer agent to the network",
                    parameters={
                        "name": {"type": "string", "required": True, "description": "Peer agent name"},
                        "endpoint": {"type": "string", "required": True, "description": "Peer's reachable endpoint"},
                        "capabilities": {"type": "list", "required": False, "description": "Peer's capabilities"},
                        "metadata": {"type": "dict", "required": False, "description": "Peer metadata"},
                    },
                ),
                SkillAction(
                    name="remove_peer",
                    description="Remove a peer agent from the network",
                    parameters={
                        "agent_id": {"type": "string", "required": True, "description": "ID of the peer to remove"},
                    },
                ),
                SkillAction(
                    name="discover",
                    description="Find agents by capability, name pattern, or status",
                    parameters={
                        "capability": {"type": "string", "required": False, "description": "Capability name to search for"},
                        "name_pattern": {"type": "string", "required": False, "description": "Name substring to match"},
                        "include_self": {"type": "bool", "required": False, "description": "Include self in results (default False)"},
                    },
                ),
                SkillAction(
                    name="route",
                    description="Find the best agent for a specific task/capability",
                    parameters={
                        "capability": {"type": "string", "required": True, "description": "The capability needed"},
                        "prefer": {"type": "string", "required": False, "description": "Preference: 'freshest', 'most_capable', 'closest' (default 'freshest')"},
                    },
                ),
                SkillAction(
                    name="rpc_call",
                    description="Make a remote procedure call to another agent's skill",
                    parameters={
                        "agent_id": {"type": "string", "required": True, "description": "Target agent ID"},
                        "skill_id": {"type": "string", "required": True, "description": "Skill to invoke on target"},
                        "action": {"type": "string", "required": True, "description": "Action to execute"},
                        "params": {"type": "dict", "required": False, "description": "Parameters for the action"},
                    },
                ),
                SkillAction(
                    name="rpc_respond",
                    description="Respond to a pending RPC request",
                    parameters={
                        "rpc_id": {"type": "string", "required": True, "description": "ID of the RPC to respond to"},
                        "success": {"type": "bool", "required": True, "description": "Whether the call succeeded"},
                        "result": {"type": "dict", "required": False, "description": "Result data"},
                        "error": {"type": "string", "required": False, "description": "Error message if failed"},
                    },
                ),
                SkillAction(
                    name="get_topology",
                    description="View the full network topology: all known agents and connections",
                    parameters={},
                ),
                SkillAction(
                    name="get_pending_rpcs",
                    description="List RPC requests waiting for this agent to handle",
                    parameters={},
                ),
                SkillAction(
                    name="refresh_peers",
                    description="Expire stale peers and update network status",
                    parameters={},
                ),
            ],
        )

    async def execute(self, action: str, params: Dict = None) -> SkillResult:
        params = params or {}
        handlers = {
            "register": self._register,
            "deregister": self._deregister,
            "add_peer": self._add_peer,
            "remove_peer": self._remove_peer,
            "discover": self._discover,
            "route": self._route,
            "rpc_call": self._rpc_call,
            "rpc_respond": self._rpc_respond,
            "get_topology": self._get_topology,
            "get_pending_rpcs": self._get_pending_rpcs,
            "refresh_peers": self._refresh_peers,
        }

        handler = handlers.get(action)
        if not handler:
            return SkillResult(
                success=False,
                message=f"Unknown action: {action}. Available: {list(handlers.keys())}",
            )
        return handler(params)

    # ── Registration ─────────────────────────────────────────

    def _register(self, params: Dict) -> SkillResult:
        """Register this agent on the network."""
        name = params.get("name")
        endpoint = params.get("endpoint")
        capabilities = params.get("capabilities", [])
        metadata = params.get("metadata", {})

        if not name or not endpoint:
            return SkillResult(success=False, message="name and endpoint are required")

        if len(capabilities) > MAX_CAPABILITIES_PER_AGENT:
            return SkillResult(
                success=False,
                message=f"Too many capabilities (max {MAX_CAPABILITIES_PER_AGENT})",
            )

        agent_id = _generate_agent_id(name, endpoint)

        registration = {
            "agent_id": agent_id,
            "name": name,
            "endpoint": endpoint,
            "capabilities": capabilities,
            "metadata": metadata,
            "registered_at": _now_iso(),
            "last_seen": _now_iso(),
            "status": "active",
        }

        data = _load_data(self._data_path)
        data["self"] = registration
        data["stats"]["registrations"] += 1
        _save_data(data, self._data_path)

        return SkillResult(
            success=True,
            message=f"Registered as '{name}' with {len(capabilities)} capabilities (ID: {agent_id})",
            data={"agent_id": agent_id, "registration": registration},
        )

    def _deregister(self, params: Dict) -> SkillResult:
        """Remove this agent from the network."""
        data = _load_data(self._data_path)
        if not data["self"]:
            return SkillResult(success=False, message="Not registered on any network")

        agent_id = data["self"]["agent_id"]
        data["self"] = None
        _save_data(data, self._data_path)

        return SkillResult(
            success=True,
            message=f"Deregistered agent {agent_id} from network",
            data={"agent_id": agent_id},
        )

    # ── Peer Management ──────────────────────────────────────

    def _add_peer(self, params: Dict) -> SkillResult:
        """Add a peer agent to the known network."""
        name = params.get("name")
        endpoint = params.get("endpoint")
        capabilities = params.get("capabilities", [])
        metadata = params.get("metadata", {})

        if not name or not endpoint:
            return SkillResult(success=False, message="name and endpoint are required")

        data = _load_data(self._data_path)

        if len(data["peers"]) >= MAX_PEERS:
            return SkillResult(
                success=False,
                message=f"Peer limit reached ({MAX_PEERS}). Remove stale peers first.",
            )

        agent_id = _generate_agent_id(name, endpoint)

        peer = {
            "agent_id": agent_id,
            "name": name,
            "endpoint": endpoint,
            "capabilities": capabilities,
            "metadata": metadata,
            "added_at": _now_iso(),
            "last_seen": _now_iso(),
            "status": "active",
            "rpc_stats": {
                "calls_made": 0,
                "calls_succeeded": 0,
                "calls_failed": 0,
                "avg_latency_ms": 0,
            },
        }

        is_update = agent_id in data["peers"]
        data["peers"][agent_id] = peer
        _save_data(data, self._data_path)

        action_word = "Updated" if is_update else "Added"
        return SkillResult(
            success=True,
            message=f"{action_word} peer '{name}' ({agent_id}) with {len(capabilities)} capabilities",
            data={"agent_id": agent_id, "peer": peer},
        )

    def _remove_peer(self, params: Dict) -> SkillResult:
        """Remove a peer agent from the network."""
        agent_id = params.get("agent_id")
        if not agent_id:
            return SkillResult(success=False, message="agent_id is required")

        data = _load_data(self._data_path)

        if agent_id not in data["peers"]:
            return SkillResult(success=False, message=f"Peer {agent_id} not found")

        peer_name = data["peers"][agent_id]["name"]
        del data["peers"][agent_id]
        _save_data(data, self._data_path)

        return SkillResult(
            success=True,
            message=f"Removed peer '{peer_name}' ({agent_id})",
        )

    # ── Discovery ────────────────────────────────────────────

    def _discover(self, params: Dict) -> SkillResult:
        """Find agents by capability or name pattern."""
        capability = params.get("capability", "").lower()
        name_pattern = params.get("name_pattern", "").lower()
        include_self = params.get("include_self", False)

        data = _load_data(self._data_path)
        data["stats"]["discoveries"] += 1
        _save_data(data, self._data_path)

        results = []

        # Search peers
        for agent_id, peer in data["peers"].items():
            if peer.get("status") != "active":
                continue
            if self._matches_search(peer, capability, name_pattern):
                results.append(peer)

        # Optionally include self
        if include_self and data["self"]:
            if self._matches_search(data["self"], capability, name_pattern):
                results.append(data["self"])

        return SkillResult(
            success=True,
            message=f"Found {len(results)} agent(s) matching query",
            data={
                "agents": results,
                "query": {"capability": capability, "name_pattern": name_pattern},
                "total_peers": len(data["peers"]),
            },
        )

    def _matches_search(self, agent: Dict, capability: str, name_pattern: str) -> bool:
        """Check if an agent matches search criteria."""
        if capability:
            cap_names = [c.get("name", "").lower() for c in agent.get("capabilities", [])]
            cap_descriptions = [c.get("description", "").lower() for c in agent.get("capabilities", [])]
            cap_skill_ids = [c.get("skill_id", "").lower() for c in agent.get("capabilities", [])]
            if not any(
                capability in n or capability in d or capability in s
                for n, d, s in zip(cap_names, cap_descriptions, cap_skill_ids)
            ):
                return False

        if name_pattern:
            if name_pattern not in agent.get("name", "").lower():
                return False

        return True

    # ── Routing ──────────────────────────────────────────────

    def _route(self, params: Dict) -> SkillResult:
        """Find the best agent for a capability, with preference scoring."""
        capability = params.get("capability", "").lower()
        prefer = params.get("prefer", "freshest")

        if not capability:
            return SkillResult(success=False, message="capability is required")

        data = _load_data(self._data_path)

        candidates = []
        for agent_id, peer in data["peers"].items():
            if peer.get("status") != "active":
                continue
            # Check capability match
            for cap in peer.get("capabilities", []):
                cap_name = cap.get("name", "").lower()
                cap_skill = cap.get("skill_id", "").lower()
                if capability in cap_name or capability in cap_skill:
                    candidates.append((peer, cap))
                    break

        if not candidates:
            return SkillResult(
                success=True,
                message=f"No agents found with capability '{capability}'",
                data={"agents": [], "capability": capability},
            )

        # Score and sort candidates
        scored = []
        for peer, cap in candidates:
            score = self._score_candidate(peer, cap, prefer)
            scored.append((score, peer, cap))

        scored.sort(key=lambda x: x[0], reverse=True)

        best_score, best_peer, best_cap = scored[0]
        all_options = [
            {
                "agent_id": p["agent_id"],
                "name": p["name"],
                "endpoint": p["endpoint"],
                "capability": c,
                "score": s,
            }
            for s, p, c in scored
        ]

        return SkillResult(
            success=True,
            message=f"Best agent for '{capability}': {best_peer['name']} (score: {best_score:.2f})",
            data={
                "best": {
                    "agent_id": best_peer["agent_id"],
                    "name": best_peer["name"],
                    "endpoint": best_peer["endpoint"],
                    "capability": best_cap,
                    "score": best_score,
                },
                "alternatives": all_options[1:5],  # Top 5 alternatives
                "total_candidates": len(scored),
            },
        )

    def _score_candidate(self, peer: Dict, cap: Dict, prefer: str) -> float:
        """Score a candidate agent for routing. Higher = better."""
        score = 50.0  # Base score

        rpc_stats = peer.get("rpc_stats", {})

        # Freshness: prefer recently seen agents
        if prefer == "freshest":
            try:
                last_seen = datetime.fromisoformat(peer.get("last_seen", "").rstrip("Z"))
                age_minutes = (datetime.utcnow() - last_seen).total_seconds() / 60
                score += max(0, 30 - age_minutes)  # Up to +30 for very fresh
            except (ValueError, TypeError):
                pass

        # Reliability: prefer agents with better success rates
        elif prefer == "most_capable":
            total = rpc_stats.get("calls_made", 0)
            succeeded = rpc_stats.get("calls_succeeded", 0)
            if total > 0:
                success_rate = succeeded / total
                score += success_rate * 40  # Up to +40

        # Latency: prefer fastest responders
        elif prefer == "closest":
            avg_latency = rpc_stats.get("avg_latency_ms", 0)
            if avg_latency > 0:
                score += max(0, 30 - (avg_latency / 100))  # Lower latency = higher score

        # Bonus for capability confidence
        confidence = cap.get("confidence", 0.5)
        score += confidence * 10

        return round(score, 2)

    # ── RPC ──────────────────────────────────────────────────

    def _rpc_call(self, params: Dict) -> SkillResult:
        """Make a remote procedure call to another agent."""
        agent_id = params.get("agent_id")
        skill_id = params.get("skill_id")
        action = params.get("action")
        rpc_params = params.get("params", {})

        if not agent_id or not skill_id or not action:
            return SkillResult(
                success=False,
                message="agent_id, skill_id, and action are required",
            )

        data = _load_data(self._data_path)
        peer = data["peers"].get(agent_id)

        if not peer:
            return SkillResult(success=False, message=f"Peer {agent_id} not found")

        rpc_id = f"rpc_{uuid.uuid4().hex[:12]}"
        rpc_record = {
            "rpc_id": rpc_id,
            "from_agent": data["self"]["agent_id"] if data["self"] else "unknown",
            "to_agent": agent_id,
            "skill_id": skill_id,
            "action": action,
            "params": rpc_params,
            "created_at": _now_iso(),
            "status": "pending",
            "response": None,
        }

        # Try HTTP RPC if endpoint is a URL and httpx is available
        endpoint = peer.get("endpoint", "")
        if endpoint.startswith("http") and HAS_HTTPX:
            return self._rpc_http(data, peer, rpc_record)

        # Fall back to file-based RPC
        return self._rpc_file_based(data, peer, rpc_record)

    def _rpc_http(self, data: Dict, peer: Dict, rpc_record: Dict) -> SkillResult:
        """Attempt HTTP-based RPC call."""
        endpoint = peer["endpoint"].rstrip("/")
        url = f"{endpoint}/api/v1/tasks"

        payload = {
            "skill_id": rpc_record["skill_id"],
            "action": rpc_record["action"],
            "params": rpc_record["params"],
            "metadata": {
                "rpc_id": rpc_record["rpc_id"],
                "from_agent": rpc_record["from_agent"],
            },
        }

        start_time = time.time()
        try:
            # Use synchronous client since we may not be in async context
            with httpx.Client(timeout=data["config"]["rpc_timeout_seconds"]) as client:
                response = client.post(url, json=payload)
                latency_ms = (time.time() - start_time) * 1000

                rpc_record["status"] = "completed" if response.status_code < 400 else "failed"
                rpc_record["response"] = {
                    "status_code": response.status_code,
                    "body": response.json() if response.headers.get("content-type", "").startswith("application/json") else response.text[:500],
                    "latency_ms": round(latency_ms, 2),
                }
                rpc_record["completed_at"] = _now_iso()

                # Update stats
                self._update_rpc_stats(data, peer["agent_id"], rpc_record["status"] == "completed", latency_ms)
                self._append_rpc_log(data, rpc_record)
                _save_data(data, self._data_path)

                return SkillResult(
                    success=rpc_record["status"] == "completed",
                    message=f"RPC to {peer['name']}: {rpc_record['status']} ({latency_ms:.0f}ms)",
                    data={"rpc": rpc_record},
                )
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            rpc_record["status"] = "failed"
            rpc_record["response"] = {"error": str(e), "latency_ms": round(latency_ms, 2)}
            rpc_record["completed_at"] = _now_iso()

            self._update_rpc_stats(data, peer["agent_id"], False, latency_ms)
            self._append_rpc_log(data, rpc_record)
            _save_data(data, self._data_path)

            return SkillResult(
                success=False,
                message=f"RPC to {peer['name']} failed: {str(e)[:200]}",
                data={"rpc": rpc_record},
            )

    def _rpc_file_based(self, data: Dict, peer: Dict, rpc_record: Dict) -> SkillResult:
        """File-based RPC: write request to peer's inbox directory."""
        endpoint = peer.get("endpoint", "")

        # For file-based RPC, endpoint is the peer's inbox path
        inbox_path = Path(endpoint)
        if not inbox_path.exists():
            # Try creating a reasonable inbox path
            inbox_path.mkdir(parents=True, exist_ok=True)

        rpc_file = inbox_path / f"rpc_{rpc_record['rpc_id']}.json"

        try:
            with open(rpc_file, "w") as f:
                json.dump(rpc_record, f, indent=2, default=str)

            rpc_record["status"] = "sent"
            rpc_record["delivery_path"] = str(rpc_file)

            # Add to pending RPCs
            if len(data["pending_rpcs"]) < MAX_PENDING_RPCS:
                data["pending_rpcs"].append(rpc_record)

            self._append_rpc_log(data, rpc_record)
            data["stats"]["rpcs_sent"] += 1
            _save_data(data, self._data_path)

            return SkillResult(
                success=True,
                message=f"RPC request sent to {peer['name']} via file ({rpc_record['rpc_id']})",
                data={"rpc": rpc_record, "delivery_path": str(rpc_file)},
            )
        except Exception as e:
            rpc_record["status"] = "failed"
            self._append_rpc_log(data, rpc_record)
            data["stats"]["rpcs_failed"] += 1
            _save_data(data, self._data_path)

            return SkillResult(
                success=False,
                message=f"Failed to deliver RPC to {peer['name']}: {str(e)[:200]}",
                data={"rpc": rpc_record},
            )

    def _rpc_respond(self, params: Dict) -> SkillResult:
        """Respond to a pending inbound RPC request."""
        rpc_id = params.get("rpc_id")
        success = params.get("success", True)
        result = params.get("result", {})
        error = params.get("error", "")

        if not rpc_id:
            return SkillResult(success=False, message="rpc_id is required")

        data = _load_data(self._data_path)

        # Find the pending RPC
        pending = None
        pending_idx = None
        for i, rpc in enumerate(data.get("pending_rpcs", [])):
            if rpc.get("rpc_id") == rpc_id:
                pending = rpc
                pending_idx = i
                break

        if pending is None:
            return SkillResult(success=False, message=f"No pending RPC with id {rpc_id}")

        # Build response
        response = {
            "rpc_id": rpc_id,
            "success": success,
            "result": result,
            "error": error,
            "responded_at": _now_iso(),
        }

        # Try to write response back to caller's inbox
        from_agent = pending.get("from_agent")
        caller_peer = data["peers"].get(from_agent)

        if caller_peer:
            try:
                endpoint = caller_peer.get("endpoint", "")
                if endpoint.startswith("http"):
                    # HTTP response would be a callback
                    pass
                else:
                    # File-based response
                    inbox_path = Path(endpoint)
                    if inbox_path.exists():
                        resp_file = inbox_path / f"rpc_response_{rpc_id}.json"
                        with open(resp_file, "w") as f:
                            json.dump(response, f, indent=2, default=str)
            except Exception:
                pass  # Best effort

        # Remove from pending
        if pending_idx is not None:
            data["pending_rpcs"].pop(pending_idx)

        # Log it
        response_log = {**pending, "status": "responded", "response": response}
        self._append_rpc_log(data, response_log)
        data["stats"]["rpcs_received"] += 1
        _save_data(data, self._data_path)

        return SkillResult(
            success=True,
            message=f"Responded to RPC {rpc_id} (success={success})",
            data={"response": response},
        )

    # ── Topology ─────────────────────────────────────────────

    def _get_topology(self, params: Dict) -> SkillResult:
        """Get the full network topology."""
        data = _load_data(self._data_path)

        self_info = data["self"]
        peers = data["peers"]

        # Build topology summary
        active_peers = [p for p in peers.values() if p.get("status") == "active"]
        stale_peers = [p for p in peers.values() if p.get("status") != "active"]

        # Capability map: what capabilities are available across the network
        capability_map = {}
        for peer in active_peers:
            for cap in peer.get("capabilities", []):
                cap_name = cap.get("name", "unknown")
                if cap_name not in capability_map:
                    capability_map[cap_name] = []
                capability_map[cap_name].append({
                    "agent_id": peer["agent_id"],
                    "name": peer["name"],
                })

        topology = {
            "self": self_info,
            "active_peers": len(active_peers),
            "stale_peers": len(stale_peers),
            "total_peers": len(peers),
            "peer_list": [
                {
                    "agent_id": p["agent_id"],
                    "name": p["name"],
                    "endpoint": p["endpoint"],
                    "status": p.get("status"),
                    "capabilities_count": len(p.get("capabilities", [])),
                    "last_seen": p.get("last_seen"),
                    "rpc_stats": p.get("rpc_stats", {}),
                }
                for p in peers.values()
            ],
            "capability_map": capability_map,
            "network_capabilities": len(capability_map),
            "stats": data["stats"],
            "recent_rpcs": data["rpc_log"][-10:],
        }

        return SkillResult(
            success=True,
            message=f"Network: {len(active_peers)} active peers, {len(capability_map)} unique capabilities",
            data=topology,
        )

    def _get_pending_rpcs(self, params: Dict) -> SkillResult:
        """List pending inbound RPC requests."""
        data = _load_data(self._data_path)
        pending = data.get("pending_rpcs", [])

        return SkillResult(
            success=True,
            message=f"{len(pending)} pending RPC(s)",
            data={"pending": pending},
        )

    # ── Maintenance ──────────────────────────────────────────

    def _refresh_peers(self, params: Dict) -> SkillResult:
        """Expire stale peers and update network status."""
        data = _load_data(self._data_path)
        ttl = data["config"].get("peer_ttl_seconds", 3600)
        now = datetime.utcnow()

        expired = []
        for agent_id, peer in list(data["peers"].items()):
            try:
                last_seen = datetime.fromisoformat(peer.get("last_seen", "").rstrip("Z"))
                age = (now - last_seen).total_seconds()
                if age > ttl:
                    peer["status"] = "stale"
                    expired.append(peer["name"])
            except (ValueError, TypeError):
                peer["status"] = "unknown"

        _save_data(data, self._data_path)

        return SkillResult(
            success=True,
            message=f"Refreshed peers: {len(expired)} marked stale (TTL={ttl}s)",
            data={"expired": expired, "total_peers": len(data["peers"])},
        )

    # ── Helpers ───────────────────────────────────────────────

    def _update_rpc_stats(self, data: Dict, agent_id: str, success: bool, latency_ms: float):
        """Update RPC statistics for a peer."""
        peer = data["peers"].get(agent_id)
        if not peer:
            return

        stats = peer.setdefault("rpc_stats", {
            "calls_made": 0, "calls_succeeded": 0, "calls_failed": 0, "avg_latency_ms": 0,
        })

        stats["calls_made"] += 1
        if success:
            stats["calls_succeeded"] += 1
            data["stats"]["rpcs_succeeded"] += 1
        else:
            stats["calls_failed"] += 1
            data["stats"]["rpcs_failed"] += 1

        data["stats"]["rpcs_sent"] += 1

        # Update running average latency
        total_calls = stats["calls_made"]
        old_avg = stats.get("avg_latency_ms", 0)
        stats["avg_latency_ms"] = round(
            ((old_avg * (total_calls - 1)) + latency_ms) / total_calls, 2
        )

        peer["last_seen"] = _now_iso()

    def _append_rpc_log(self, data: Dict, rpc_record: Dict):
        """Append to RPC log, trimming if needed."""
        data["rpc_log"].append(rpc_record)
        if len(data["rpc_log"]) > MAX_RPC_LOG_ENTRIES:
            data["rpc_log"] = data["rpc_log"][-MAX_RPC_LOG_ENTRIES:]
