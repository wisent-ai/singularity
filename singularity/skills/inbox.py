#!/usr/bin/env python3
"""
InboxSkill - File-based messaging system for inter-agent communication.

Enables agents to send and receive messages from other agents, humans,
or external systems. Messages are stored as JSON files in a configurable
directory, making them easy to inspect and manage.

Serves the Replication pillar (inter-agent communication) and
Revenue pillar (receiving service requests from clients).
"""

import json
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import Skill, SkillAction, SkillManifest, SkillResult


class InboxSkill(Skill):
    """File-based messaging skill for agent communication."""

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self._agent_id = "default"
        self._inbox_dir: Optional[Path] = None

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="inbox",
            name="Inbox",
            version="1.0.0",
            category="communication",
            description="File-based messaging for inter-agent and human-agent communication",
            required_credentials=[],
            actions=[
                SkillAction(
                    name="check_inbox",
                    description="Check for new unread messages. Returns count and summaries.",
                    parameters={},
                ),
                SkillAction(
                    name="read_message",
                    description="Read a specific message by ID. Marks it as read.",
                    parameters={
                        "message_id": {
                            "type": "string",
                            "required": True,
                            "description": "ID of the message to read",
                        }
                    },
                ),
                SkillAction(
                    name="send_message",
                    description="Send a message to another agent or entity.",
                    parameters={
                        "to": {
                            "type": "string",
                            "required": True,
                            "description": "Recipient agent ID or identifier",
                        },
                        "subject": {
                            "type": "string",
                            "required": True,
                            "description": "Message subject",
                        },
                        "body": {
                            "type": "string",
                            "required": True,
                            "description": "Message body content",
                        },
                        "priority": {
                            "type": "string",
                            "required": False,
                            "description": "Priority: low, normal, high, urgent (default: normal)",
                        },
                        "reply_to": {
                            "type": "string",
                            "required": False,
                            "description": "Message ID this is a reply to",
                        },
                    },
                ),
                SkillAction(
                    name="list_messages",
                    description="List messages with optional filters.",
                    parameters={
                        "status": {
                            "type": "string",
                            "required": False,
                            "description": "Filter by status: unread, read, all (default: all)",
                        },
                        "from_agent": {
                            "type": "string",
                            "required": False,
                            "description": "Filter by sender agent ID",
                        },
                        "limit": {
                            "type": "integer",
                            "required": False,
                            "description": "Max messages to return (default: 20)",
                        },
                    },
                ),
                SkillAction(
                    name="delete_message",
                    description="Delete a message by ID.",
                    parameters={
                        "message_id": {
                            "type": "string",
                            "required": True,
                            "description": "ID of the message to delete",
                        },
                    },
                ),
                SkillAction(
                    name="broadcast",
                    description="Send a message to all known agents.",
                    parameters={
                        "subject": {
                            "type": "string",
                            "required": True,
                            "description": "Message subject",
                        },
                        "body": {
                            "type": "string",
                            "required": True,
                            "description": "Message body content",
                        },
                        "priority": {
                            "type": "string",
                            "required": False,
                            "description": "Priority: low, normal, high, urgent (default: normal)",
                        },
                    },
                ),
                SkillAction(
                    name="get_conversation",
                    description="Get a conversation thread by following reply chains.",
                    parameters={
                        "message_id": {
                            "type": "string",
                            "required": True,
                            "description": "Any message ID in the conversation thread",
                        },
                    },
                ),
            ],
        )

    def configure(self, agent_id: str, inbox_dir: Optional[str] = None):
        """Configure the inbox with agent identity and storage location."""
        self._agent_id = agent_id
        if inbox_dir:
            self._inbox_dir = Path(inbox_dir)
        else:
            self._inbox_dir = Path.home() / ".singularity" / "messages"
        self._inbox_dir.mkdir(parents=True, exist_ok=True)
        # Create this agent's inbox subdirectory so other agents can discover us
        (self._inbox_dir / agent_id).mkdir(parents=True, exist_ok=True)

    def _ensure_configured(self):
        """Ensure inbox is configured with a directory."""
        if self._inbox_dir is None:
            self.configure(self._agent_id)

    def _agent_inbox_dir(self, agent_id: str) -> Path:
        """Get the inbox directory for a specific agent."""
        self._ensure_configured()
        inbox = self._inbox_dir / agent_id
        inbox.mkdir(parents=True, exist_ok=True)
        return inbox

    def _my_inbox_dir(self) -> Path:
        """Get this agent's inbox directory."""
        return self._agent_inbox_dir(self._agent_id)

    def _save_message(self, msg: Dict, recipient: str) -> str:
        """Save a message to a recipient's inbox. Returns message ID."""
        inbox = self._agent_inbox_dir(recipient)
        msg_file = inbox / f"{msg['id']}.json"
        with open(msg_file, "w") as f:
            json.dump(msg, f, indent=2)
        return msg["id"]

    def _load_message(self, message_id: str) -> Optional[Dict]:
        """Load a message from this agent's inbox."""
        msg_file = self._my_inbox_dir() / f"{message_id}.json"
        if not msg_file.exists():
            return None
        with open(msg_file, "r") as f:
            return json.load(f)

    def _load_all_messages(self) -> List[Dict]:
        """Load all messages from this agent's inbox."""
        inbox = self._my_inbox_dir()
        messages = []
        for msg_file in sorted(inbox.glob("*.json")):
            try:
                with open(msg_file, "r") as f:
                    messages.append(json.load(f))
            except (json.JSONDecodeError, OSError):
                continue
        return messages

    def _discover_agents(self) -> List[str]:
        """Discover known agent IDs from inbox directories."""
        self._ensure_configured()
        agents = []
        if self._inbox_dir.exists():
            for d in self._inbox_dir.iterdir():
                if d.is_dir() and d.name != self._agent_id:
                    agents.append(d.name)
        return agents

    async def execute(self, action: str, params: Dict) -> SkillResult:
        """Execute an inbox action."""
        self._ensure_configured()

        handlers = {
            "check_inbox": self._check_inbox,
            "read_message": self._read_message,
            "send_message": self._send_message,
            "list_messages": self._list_messages,
            "delete_message": self._delete_message,
            "broadcast": self._broadcast,
            "get_conversation": self._get_conversation,
        }

        handler = handlers.get(action)
        if not handler:
            return SkillResult(
                success=False,
                message=f"Unknown action: {action}",
            )

        try:
            return await handler(params)
        except Exception as e:
            return SkillResult(
                success=False,
                message=f"Inbox error: {str(e)}",
            )

    async def _check_inbox(self, params: Dict) -> SkillResult:
        """Check for new unread messages."""
        messages = self._load_all_messages()
        unread = [m for m in messages if not m.get("read", False)]
        total = len(messages)

        summaries = []
        for msg in sorted(unread, key=lambda m: m.get("timestamp", ""), reverse=True)[:10]:
            summaries.append({
                "id": msg["id"],
                "from": msg.get("from", "unknown"),
                "subject": msg.get("subject", "(no subject)"),
                "priority": msg.get("priority", "normal"),
                "timestamp": msg.get("timestamp", ""),
            })

        return SkillResult(
            success=True,
            message=f"{len(unread)} unread messages out of {total} total",
            data={
                "unread_count": len(unread),
                "total_count": total,
                "unread_summaries": summaries,
            },
        )

    async def _read_message(self, params: Dict) -> SkillResult:
        """Read a specific message and mark it as read."""
        message_id = params.get("message_id", "")
        if not message_id:
            return SkillResult(success=False, message="message_id is required")

        msg = self._load_message(message_id)
        if not msg:
            return SkillResult(success=False, message=f"Message not found: {message_id}")

        # Mark as read
        msg["read"] = True
        msg["read_at"] = datetime.now().isoformat()
        msg_file = self._my_inbox_dir() / f"{message_id}.json"
        with open(msg_file, "w") as f:
            json.dump(msg, f, indent=2)

        return SkillResult(
            success=True,
            message=f"Message from {msg.get('from', 'unknown')}: {msg.get('subject', '')}",
            data=msg,
        )

    async def _send_message(self, params: Dict) -> SkillResult:
        """Send a message to another agent."""
        to = params.get("to", "")
        subject = params.get("subject", "")
        body = params.get("body", "")
        priority = params.get("priority", "normal")
        reply_to = params.get("reply_to", "")

        if not to:
            return SkillResult(success=False, message="'to' recipient is required")
        if not subject and not body:
            return SkillResult(success=False, message="Subject or body is required")

        if priority not in ("low", "normal", "high", "urgent"):
            priority = "normal"

        msg_id = str(uuid.uuid4())[:12]
        msg = {
            "id": msg_id,
            "from": self._agent_id,
            "to": to,
            "subject": subject,
            "body": body,
            "priority": priority,
            "timestamp": datetime.now().isoformat(),
            "read": False,
        }
        if reply_to:
            msg["reply_to"] = reply_to

        self._save_message(msg, to)

        # Also save a copy in sender's outbox (same dir, different naming)
        outbox_dir = self._agent_inbox_dir(self._agent_id)
        outbox_file = outbox_dir / f"sent_{msg_id}.json"
        sent_copy = {**msg, "read": True, "sent": True}
        with open(outbox_file, "w") as f:
            json.dump(sent_copy, f, indent=2)

        return SkillResult(
            success=True,
            message=f"Message sent to {to}: {subject}",
            data={"message_id": msg_id, "to": to, "subject": subject},
        )

    async def _list_messages(self, params: Dict) -> SkillResult:
        """List messages with optional filters."""
        status = params.get("status", "all")
        from_agent = params.get("from_agent", "")
        limit = int(params.get("limit", 20))

        messages = self._load_all_messages()

        # Filter out sent copies for listing
        messages = [m for m in messages if not m.get("sent", False)]

        # Apply filters
        if status == "unread":
            messages = [m for m in messages if not m.get("read", False)]
        elif status == "read":
            messages = [m for m in messages if m.get("read", False)]

        if from_agent:
            messages = [m for m in messages if m.get("from") == from_agent]

        # Sort by timestamp descending
        messages.sort(key=lambda m: m.get("timestamp", ""), reverse=True)

        # Limit
        messages = messages[:limit]

        # Return summaries (not full bodies for list view)
        summaries = []
        for msg in messages:
            summaries.append({
                "id": msg["id"],
                "from": msg.get("from", "unknown"),
                "subject": msg.get("subject", "(no subject)"),
                "priority": msg.get("priority", "normal"),
                "read": msg.get("read", False),
                "timestamp": msg.get("timestamp", ""),
                "has_reply_to": bool(msg.get("reply_to")),
            })

        return SkillResult(
            success=True,
            message=f"Found {len(summaries)} messages",
            data={"messages": summaries, "count": len(summaries)},
        )

    async def _delete_message(self, params: Dict) -> SkillResult:
        """Delete a message."""
        message_id = params.get("message_id", "")
        if not message_id:
            return SkillResult(success=False, message="message_id is required")

        msg_file = self._my_inbox_dir() / f"{message_id}.json"
        if not msg_file.exists():
            return SkillResult(success=False, message=f"Message not found: {message_id}")

        msg_file.unlink()
        return SkillResult(
            success=True,
            message=f"Message {message_id} deleted",
            data={"deleted_id": message_id},
        )

    async def _broadcast(self, params: Dict) -> SkillResult:
        """Send a message to all known agents."""
        subject = params.get("subject", "")
        body = params.get("body", "")
        priority = params.get("priority", "normal")

        if not subject and not body:
            return SkillResult(success=False, message="Subject or body is required")

        agents = self._discover_agents()
        if not agents:
            return SkillResult(
                success=True,
                message="No other agents discovered to broadcast to",
                data={"sent_to": [], "count": 0},
            )

        sent_to = []
        for agent_id in agents:
            result = await self._send_message({
                "to": agent_id,
                "subject": subject,
                "body": body,
                "priority": priority,
            })
            if result.success:
                sent_to.append(agent_id)

        return SkillResult(
            success=True,
            message=f"Broadcast sent to {len(sent_to)} agents",
            data={"sent_to": sent_to, "count": len(sent_to)},
        )

    async def _get_conversation(self, params: Dict) -> SkillResult:
        """Get a conversation thread by following reply chains."""
        message_id = params.get("message_id", "")
        if not message_id:
            return SkillResult(success=False, message="message_id is required")

        all_messages = self._load_all_messages()
        msg_map = {m["id"]: m for m in all_messages}

        # Find the root of the thread
        current = msg_map.get(message_id)
        if not current:
            return SkillResult(success=False, message=f"Message not found: {message_id}")

        # Walk up to find root
        visited = set()
        root = current
        while root.get("reply_to") and root["reply_to"] in msg_map:
            if root["id"] in visited:
                break  # Avoid cycles
            visited.add(root["id"])
            root = msg_map[root["reply_to"]]

        # Collect thread: all messages that reply to root or its descendants
        thread = [root]
        thread_ids = {root["id"]}

        # BFS to find all replies
        changed = True
        while changed:
            changed = False
            for msg in all_messages:
                if msg["id"] not in thread_ids and msg.get("reply_to") in thread_ids:
                    thread.append(msg)
                    thread_ids.add(msg["id"])
                    changed = True

        # Sort by timestamp
        thread.sort(key=lambda m: m.get("timestamp", ""))

        return SkillResult(
            success=True,
            message=f"Conversation thread with {len(thread)} messages",
            data={
                "thread": thread,
                "message_count": len(thread),
                "root_id": root["id"],
            },
        )

    def get_unread_count(self) -> int:
        """Quick check for unread message count (for agent state injection)."""
        self._ensure_configured()
        try:
            messages = self._load_all_messages()
            return len([m for m in messages if not m.get("read", False) and not m.get("sent", False)])
        except Exception:
            return 0

    def get_unread_summary(self) -> str:
        """Get a brief summary of unread messages for LLM context."""
        self._ensure_configured()
        try:
            messages = self._load_all_messages()
            unread = [m for m in messages if not m.get("read", False) and not m.get("sent", False)]
            if not unread:
                return ""

            lines = [f"📬 {len(unread)} unread message(s):"]
            for msg in sorted(unread, key=lambda m: m.get("timestamp", ""), reverse=True)[:5]:
                priority_icon = "🔴" if msg.get("priority") == "urgent" else "🟡" if msg.get("priority") == "high" else ""
                lines.append(
                    f"  {priority_icon} [{msg['id']}] From: {msg.get('from', '?')} - {msg.get('subject', '(no subject)')}"
                )
            if len(unread) > 5:
                lines.append(f"  ... and {len(unread) - 5} more")
            return "\n".join(lines)
        except Exception:
            return ""
