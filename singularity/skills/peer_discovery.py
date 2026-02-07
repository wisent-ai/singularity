#!/usr/bin/env python3
"""
Peer Discovery Skill - Find and connect with other agents on the network.

This skill provides a unified interface for agent-to-agent discovery,
capability matching, and lightweight messaging. Unlike the full marketplace
or orchestrator skills, this focuses specifically on the discovery phase:

1. Register this agent's capabilities on the network
2. Find other agents by type, tags, or task description
3. Send/receive direct messages and broadcasts
4. Query the shared knowledge base

This is the "phone book + inbox" for the agent network.

Usage:
    # Register on the network
    await peer_discovery.execute("register", {})

    # Find agents who can review code
    result = await peer_discovery.execute("find_agents", {
        "task": "review Python code for security vulnerabilities"
    })

    # Send a collaboration request
    await peer_discovery.execute("send_message", {
        "to_agent": "agent_456",
        "subject": "Collaboration",
        "body": {"proposal": "Let's build a testing framework together"}
    })

Author: Adam (ADAM) - autonomous AI agent
"""

import json
import os
from typing import Dict, List, Optional

from singularity.skills.base import (
    Skill,
    SkillAction,
    SkillManifest,
    SkillResult,
)


# Inline lightweight implementations to avoid external dependencies.
# These mirror the protocol from adam-agent-toolkit but are self-contained.

import hashlib
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path


DEFAULT_DATA_DIR = os.environ.get(
    "AGENT_DATA_DIR",
    str(Path.home() / ".singularity" / "network"),
)
KNOWLEDGE_TTL_DAYS = 7
MAX_INBOX_SIZE = 500


@dataclass
class _AgentRecord:
    """A registered agent on the network."""
    agent_id: str
    name: str
    ticker: str
    agent_type: str = "general"
    specialty: str = ""
    capabilities: List[Dict] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    registered_at: str = ""
    last_seen: str = ""

    def to_dict(self):
        return asdict(self)


@dataclass
class _Message:
    """An inter-agent message."""
    id: str = ""
    from_agent: str = ""
    to_agent: str = ""
    subject: str = ""
    body: Dict = field(default_factory=dict)
    timestamp: str = ""
    ttl_seconds: int = 86400  # 24 hours default

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()

    @property
    def is_expired(self) -> bool:
        try:
            sent = datetime.fromisoformat(self.timestamp)
            return (datetime.utcnow() - sent).total_seconds() > self.ttl_seconds
        except (ValueError, TypeError):
            return False

    def to_dict(self):
        return asdict(self)


class _FileStore:
    """Simple JSON file store for network state."""

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def load(self) -> dict:
        if not os.path.exists(self.path):
            return {}
        try:
            with open(self.path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}

    def save(self, data: dict):
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2, default=str)
        os.replace(tmp, self.path)


class PeerDiscoverySkill(Skill):
    """
    Skill for discovering and communicating with peer agents.

    Actions:
        register        - Register this agent on the network
        find_agents     - Find agents by type, tags, or task description
        who_is_online   - List all known agents
        send_message    - Send a direct message to another agent
        check_inbox     - Check for incoming messages
        broadcast       - Send a message to all known agents
        publish_knowledge - Share a discovery with the network
        query_knowledge - Search the shared knowledge base
    """

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        data_dir = credentials.get("data_dir", DEFAULT_DATA_DIR) if credentials else DEFAULT_DATA_DIR
        self._agents_store = _FileStore(os.path.join(data_dir, "peers.json"))
        self._messages_store = _FileStore(os.path.join(data_dir, "inbox.json"))
        self._knowledge_store = _FileStore(os.path.join(data_dir, "knowledge.json"))
        self._my_agent_id = ""
        self._my_name = ""
        self._my_ticker = ""

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="peer_discovery",
            name="Peer Discovery",
            version="1.0.0",
            category="network",
            description="Find and connect with other agents on the network. "
                        "Register capabilities, discover peers, exchange messages, "
                        "and share knowledge.",
            actions=[
                SkillAction(
                    name="register",
                    description="Register this agent on the network with capabilities",
                    parameters={
                        "capabilities": {
                            "type": "list",
                            "required": False,
                            "description": "List of capability dicts [{name, description, tags}]",
                        },
                        "specialty": {
                            "type": "string",
                            "required": False,
                            "description": "Agent's area of specialty",
                        },
                    },
                    estimated_cost=0.0,
                ),
                SkillAction(
                    name="find_agents",
                    description="Find agents matching a task or criteria",
                    parameters={
                        "task": {
                            "type": "string",
                            "required": False,
                            "description": "Natural language task description to match",
                        },
                        "agent_type": {
                            "type": "string",
                            "required": False,
                            "description": "Filter by agent type (coder, writer, etc.)",
                        },
                        "tags": {
                            "type": "list",
                            "required": False,
                            "description": "Filter by capability tags",
                        },
                    },
                    estimated_cost=0.0,
                ),
                SkillAction(
                    name="who_is_online",
                    description="List all known agents on the network",
                    parameters={},
                    estimated_cost=0.0,
                ),
                SkillAction(
                    name="send_message",
                    description="Send a direct message to another agent",
                    parameters={
                        "to_agent": {
                            "type": "string",
                            "required": True,
                            "description": "ID of the recipient agent",
                        },
                        "subject": {
                            "type": "string",
                            "required": True,
                            "description": "Message subject",
                        },
                        "body": {
                            "type": "dict",
                            "required": False,
                            "description": "Message body (any JSON-serializable dict)",
                        },
                    },
                    estimated_cost=0.0,
                ),
                SkillAction(
                    name="check_inbox",
                    description="Check for incoming messages",
                    parameters={
                        "from_agent": {
                            "type": "string",
                            "required": False,
                            "description": "Filter messages from a specific agent",
                        },
                    },
                    estimated_cost=0.0,
                ),
                SkillAction(
                    name="broadcast",
                    description="Send an announcement to all known agents",
                    parameters={
                        "subject": {
                            "type": "string",
                            "required": True,
                            "description": "Announcement subject",
                        },
                        "body": {
                            "type": "dict",
                            "required": False,
                            "description": "Announcement body",
                        },
                    },
                    estimated_cost=0.0,
                ),
                SkillAction(
                    name="publish_knowledge",
                    description="Share a discovery or insight with the network",
                    parameters={
                        "content": {
                            "type": "string",
                            "required": True,
                            "description": "The knowledge to share",
                        },
                        "category": {
                            "type": "string",
                            "required": False,
                            "description": "Category: strategy, warning, optimization, capability, market",
                        },
                        "confidence": {
                            "type": "float",
                            "required": False,
                            "description": "Confidence level 0.0-1.0 (default 0.5)",
                        },
                        "tags": {
                            "type": "list",
                            "required": False,
                            "description": "Tags for categorization",
                        },
                    },
                    estimated_cost=0.0,
                ),
                SkillAction(
                    name="query_knowledge",
                    description="Search the shared knowledge base",
                    parameters={
                        "category": {
                            "type": "string",
                            "required": False,
                            "description": "Filter by category",
                        },
                        "tags": {
                            "type": "list",
                            "required": False,
                            "description": "Filter by tags",
                        },
                        "search": {
                            "type": "string",
                            "required": False,
                            "description": "Text search in content",
                        },
                        "min_confidence": {
                            "type": "float",
                            "required": False,
                            "description": "Minimum confidence threshold (default 0.3)",
                        },
                    },
                    estimated_cost=0.0,
                ),
            ],
            required_credentials=[],
            author="Adam (ADAM)",
        )

    async def initialize(self) -> bool:
        """Initialize with agent identity from context."""
        if self.context:
            self._my_name = self.context.agent_name
            self._my_ticker = self.context.agent_ticker
            state = self.context.agent_state
            self._my_agent_id = state.get("instance_id", f"agent_{self._my_ticker.lower()}")
        self.initialized = True
        return True

    async def execute(self, action: str, params: Dict) -> SkillResult:
        if not self.initialized:
            await self.initialize()

        handlers = {
            "register": self._register,
            "find_agents": self._find_agents,
            "who_is_online": self._who_is_online,
            "send_message": self._send_message,
            "check_inbox": self._check_inbox,
            "broadcast": self._broadcast,
            "publish_knowledge": self._publish_knowledge,
            "query_knowledge": self._query_knowledge,
        }

        handler = handlers.get(action)
        if not handler:
            return SkillResult(
                success=False,
                message=f"Unknown action: {action}. Available: {list(handlers.keys())}",
            )

        try:
            return await handler(params)
        except Exception as e:
            return SkillResult(success=False, message=f"Error in {action}: {e}")

    # ─── Action Implementations ───────────────────────────────────────────

    async def _register(self, params: Dict) -> SkillResult:
        """Register this agent on the network."""
        # Build capability list from our skills if available
        capabilities = params.get("capabilities", [])
        tags = set()

        if not capabilities and self.context:
            # Auto-discover from installed skills
            for skill_id in self.context.list_skills():
                skill = self.context.get_skill(skill_id)
                if skill and skill.manifest.skill_id != "peer_discovery":
                    for action in skill.get_actions():
                        cap = {
                            "skill_id": skill_id,
                            "action": action.name,
                            "description": action.description,
                        }
                        capabilities.append(cap)
                        # Auto-tag
                        for word in action.description.lower().split():
                            if len(word) > 3:
                                tags.add(word)

        record = _AgentRecord(
            agent_id=self._my_agent_id,
            name=self._my_name,
            ticker=self._my_ticker,
            agent_type=params.get("agent_type", "general"),
            specialty=params.get("specialty", ""),
            capabilities=capabilities,
            tags=list(tags)[:50],  # Cap tags
            registered_at=datetime.utcnow().isoformat(),
            last_seen=datetime.utcnow().isoformat(),
        )

        data = self._agents_store.load()
        data[self._my_agent_id] = record.to_dict()
        self._agents_store.save(data)

        return SkillResult(
            success=True,
            message=f"Registered {self._my_name} ({self._my_ticker}) with {len(capabilities)} capabilities",
            data={
                "agent_id": self._my_agent_id,
                "capabilities_count": len(capabilities),
                "tags": list(tags)[:20],
            },
        )

    async def _find_agents(self, params: Dict) -> SkillResult:
        """Find agents matching criteria."""
        data = self._agents_store.load()
        task = params.get("task", "")
        agent_type = params.get("agent_type", "")
        filter_tags = set(params.get("tags", []))

        results = []
        task_words = set(task.lower().split()) if task else set()

        for agent_id, agent_data in data.items():
            if agent_id == self._my_agent_id:
                continue

            # Filter by type
            if agent_type and agent_data.get("agent_type") != agent_type:
                continue

            # Filter by tags
            agent_tags = set(agent_data.get("tags", []))
            if filter_tags and not filter_tags & agent_tags:
                continue

            # Score by task match
            score = 0.0
            if task_words:
                # Check capabilities
                for cap in agent_data.get("capabilities", []):
                    desc = cap.get("description", "").lower()
                    desc_words = set(desc.split())
                    overlap = len(task_words & desc_words)
                    cap_score = overlap / max(len(task_words), 1)
                    score = max(score, cap_score)

                # Check tags
                tag_overlap = len(task_words & agent_tags)
                tag_score = tag_overlap / max(len(task_words), 1)
                score = max(score, tag_score)

                if score < 0.1:
                    continue

            results.append({
                "agent_id": agent_id,
                "name": agent_data.get("name", ""),
                "ticker": agent_data.get("ticker", ""),
                "agent_type": agent_data.get("agent_type", ""),
                "specialty": agent_data.get("specialty", ""),
                "capabilities_count": len(agent_data.get("capabilities", [])),
                "match_score": round(score, 3),
                "last_seen": agent_data.get("last_seen", ""),
            })

        results.sort(key=lambda x: x["match_score"], reverse=True)

        return SkillResult(
            success=True,
            message=f"Found {len(results)} matching agent(s)",
            data={"agents": results[:20]},
        )

    async def _who_is_online(self, params: Dict) -> SkillResult:
        """List all known agents."""
        data = self._agents_store.load()
        agents = []

        for agent_id, agent_data in data.items():
            agents.append({
                "agent_id": agent_id,
                "name": agent_data.get("name", ""),
                "ticker": agent_data.get("ticker", ""),
                "agent_type": agent_data.get("agent_type", ""),
                "specialty": agent_data.get("specialty", ""),
                "capabilities_count": len(agent_data.get("capabilities", [])),
                "last_seen": agent_data.get("last_seen", ""),
            })

        return SkillResult(
            success=True,
            message=f"{len(agents)} agent(s) known on the network",
            data={"agents": agents, "total": len(agents)},
        )

    async def _send_message(self, params: Dict) -> SkillResult:
        """Send a message to another agent."""
        to_agent = params.get("to_agent", "")
        subject = params.get("subject", "")

        if not to_agent:
            return SkillResult(success=False, message="to_agent is required")
        if not subject:
            return SkillResult(success=False, message="subject is required")

        msg = _Message(
            from_agent=self._my_agent_id,
            to_agent=to_agent,
            subject=subject,
            body=params.get("body", {}),
        )

        data = self._messages_store.load()
        inbox = data.get(to_agent, [])

        # Trim old messages
        inbox = [m for m in inbox if not _Message(**m).is_expired]
        if len(inbox) >= MAX_INBOX_SIZE:
            inbox = inbox[-MAX_INBOX_SIZE + 1:]

        inbox.append(msg.to_dict())
        data[to_agent] = inbox
        self._messages_store.save(data)

        return SkillResult(
            success=True,
            message=f"Message sent to {to_agent}: {subject}",
            data={"message_id": msg.id, "to": to_agent},
        )

    async def _check_inbox(self, params: Dict) -> SkillResult:
        """Check inbox for messages."""
        from_agent = params.get("from_agent", "")

        data = self._messages_store.load()
        inbox = data.get(self._my_agent_id, [])

        messages = []
        remaining = []

        for msg_data in inbox:
            msg = _Message(**msg_data)
            if msg.is_expired:
                continue
            if from_agent and msg.from_agent != from_agent:
                remaining.append(msg_data)
                continue
            messages.append(msg.to_dict())

        # Drain read messages
        data[self._my_agent_id] = remaining
        self._messages_store.save(data)

        return SkillResult(
            success=True,
            message=f"{len(messages)} message(s) received",
            data={"messages": messages, "count": len(messages)},
        )

    async def _broadcast(self, params: Dict) -> SkillResult:
        """Broadcast to all known agents."""
        subject = params.get("subject", "")
        if not subject:
            return SkillResult(success=False, message="subject is required")

        agents_data = self._agents_store.load()
        sent_count = 0

        for agent_id in agents_data:
            if agent_id == self._my_agent_id:
                continue
            await self._send_message({
                "to_agent": agent_id,
                "subject": subject,
                "body": params.get("body", {}),
            })
            sent_count += 1

        return SkillResult(
            success=True,
            message=f"Broadcast sent to {sent_count} agent(s)",
            data={"sent_count": sent_count, "subject": subject},
        )

    async def _publish_knowledge(self, params: Dict) -> SkillResult:
        """Publish knowledge to the shared base."""
        content = params.get("content", "")
        if not content:
            return SkillResult(success=False, message="content is required")

        entry_id = hashlib.sha256(content.encode()).hexdigest()[:16]
        entry = {
            "id": entry_id,
            "content": content,
            "category": params.get("category", "strategy"),
            "confidence": min(1.0, max(0.0, params.get("confidence", 0.5))),
            "tags": params.get("tags", []),
            "published_by": self._my_agent_id,
            "published_at": datetime.utcnow().isoformat(),
            "expires_at": (datetime.utcnow() + timedelta(days=KNOWLEDGE_TTL_DAYS)).isoformat(),
            "endorsements": [],
            "disputes": [],
        }

        data = self._knowledge_store.load()
        entries = data.get("entries", {})

        # Dedup: if same content hash exists, keep higher confidence
        if entry_id in entries:
            existing = entries[entry_id]
            if entry["confidence"] <= existing.get("confidence", 0):
                return SkillResult(
                    success=True,
                    message="Knowledge already exists with equal or higher confidence",
                    data={"entry_id": entry_id, "action": "skipped"},
                )

        entries[entry_id] = entry
        data["entries"] = entries
        self._knowledge_store.save(data)

        return SkillResult(
            success=True,
            message=f"Published knowledge: {content[:80]}...",
            data={"entry_id": entry_id, "category": entry["category"]},
        )

    async def _query_knowledge(self, params: Dict) -> SkillResult:
        """Query the shared knowledge base."""
        category = params.get("category", "")
        filter_tags = set(params.get("tags", []))
        search = params.get("search", "").lower()
        min_confidence = params.get("min_confidence", 0.3)

        data = self._knowledge_store.load()
        entries = data.get("entries", {})
        results = []
        now = datetime.utcnow()

        for eid, entry in entries.items():
            # Check expiry
            try:
                expires = datetime.fromisoformat(entry.get("expires_at", ""))
                if now > expires:
                    continue
            except (ValueError, TypeError):
                pass

            # Confidence check
            confidence = entry.get("confidence", 0)
            if confidence < min_confidence:
                continue

            # Category filter
            if category and entry.get("category") != category:
                continue

            # Tag filter
            if filter_tags and not filter_tags & set(entry.get("tags", [])):
                continue

            # Text search
            if search and search not in entry.get("content", "").lower():
                continue

            # Compute relevance score
            score = confidence
            score += len(entry.get("endorsements", [])) * 0.05
            score -= len(entry.get("disputes", [])) * 0.10

            try:
                pub_time = datetime.fromisoformat(entry.get("published_at", ""))
                age_hours = (now - pub_time).total_seconds() / 3600
                score += max(0, 0.2 - (age_hours / 168) * 0.2)
            except (ValueError, TypeError):
                pass

            entry_result = dict(entry)
            entry_result["relevance_score"] = round(max(0, min(1.0, score)), 3)
            results.append(entry_result)

        results.sort(key=lambda x: x["relevance_score"], reverse=True)

        return SkillResult(
            success=True,
            message=f"Found {len(results)} knowledge entries",
            data={"entries": results[:20], "total": len(results)},
        )
