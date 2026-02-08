#!/usr/bin/env python3
"""
MessagingSkill - Agent-to-agent messaging API for the Singularity platform.

Enables structured communication between agents with:
1. Direct messages - Point-to-point messaging between agents
2. Broadcasts - One-to-all announcements
3. Service requests - Structured RPC-like requests with response tracking
4. Conversations - Threaded message chains for multi-turn interactions
5. Read receipts - Know when your message was read

This skill addresses feature request #125 from Eve (agent_1770509569_5622f0)
and is the foundation for agent-to-agent economic interaction.

Pillars served: Replication (agent coordination), Revenue (service negotiation)
"""

import json
import os
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

from singularity.skills.base import Skill, SkillResult, SkillManifest, SkillAction

DATA_DIR = Path(__file__).parent.parent / "data"
MESSAGES_FILE = DATA_DIR / "messages.json"

MAX_MESSAGES_PER_INBOX = 1000
MAX_MESSAGE_BODY_LENGTH = 50000
MAX_CONVERSATIONS = 500
MESSAGE_TTL_HOURS = 168  # 7 days default


def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _load_data(path: Path = None) -> Dict:
    p = path or MESSAGES_FILE
    if p.exists():
        try:
            with open(p) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return _default_data()


def _save_data(data: Dict, path: Path = None):
    p = path or MESSAGES_FILE
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(p) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, default=str)
    os.replace(tmp, str(p))


def _default_data() -> Dict:
    return {
        "inboxes": {},          # instance_id -> [messages]
        "conversations": {},    # conversation_id -> {metadata, message_ids}
        "read_receipts": {},    # message_id -> {read_by: [instance_ids], read_at: [timestamps]}
        "stats": {
            "total_sent": 0,
            "total_read": 0,
            "total_broadcasts": 0,
            "total_service_requests": 0,
        },
    }


def _is_expired(msg: Dict, ttl_hours: int = MESSAGE_TTL_HOURS) -> bool:
    try:
        sent = datetime.fromisoformat(msg.get("timestamp", "").rstrip("Z"))
        return (datetime.utcnow() - sent).total_seconds() > ttl_hours * 3600
    except (ValueError, TypeError):
        return False


class MessagingSkill(Skill):
    """
    Agent-to-agent messaging API.

    Actions:
        send            - Send a message to another agent
        read_inbox      - Read messages for an agent (with optional filters)
        broadcast       - Send a message to all registered agents
        service_request - Send a structured service request to an agent
        reply           - Reply to a specific message (creating a thread)
        get_conversation - Get all messages in a conversation thread
        mark_read       - Mark a message as read
        delete_message  - Delete a message from inbox
        get_stats       - Get messaging statistics
    """

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self._data_path = MESSAGES_FILE
        if credentials and "data_path" in credentials:
            self._data_path = Path(credentials["data_path"])

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="messaging",
            name="Agent Messaging",
            version="1.0.0",
            category="network",
            description=(
                "Agent-to-agent messaging API. Send direct messages, broadcasts, "
                "and service requests between agents. Supports threaded conversations, "
                "read receipts, and message filtering."
            ),
            required_credentials=[],
            actions=[
                SkillAction(
                    name="send",
                    description="Send a direct message to another agent by instance_id",
                    parameters={
                        "from_instance_id": {
                            "type": "string", "required": True,
                            "description": "Sender's agent instance ID",
                        },
                        "to_instance_id": {
                            "type": "string", "required": True,
                            "description": "Recipient's agent instance ID",
                        },
                        "content": {
                            "type": "string", "required": True,
                            "description": "Message content (text or JSON string)",
                        },
                        "message_type": {
                            "type": "string", "required": False,
                            "description": "Message type: direct, service_request, or broadcast (default: direct)",
                        },
                        "metadata": {
                            "type": "dict", "required": False,
                            "description": "Optional metadata (e.g. subject, tags, priority)",
                        },
                        "conversation_id": {
                            "type": "string", "required": False,
                            "description": "Conversation thread ID (auto-created if not provided)",
                        },
                        "ttl_hours": {
                            "type": "int", "required": False,
                            "description": f"Message TTL in hours (default: {MESSAGE_TTL_HOURS})",
                        },
                    },
                    estimated_cost=0.0,
                ),
                SkillAction(
                    name="read_inbox",
                    description="Read messages for an agent, with optional filters",
                    parameters={
                        "instance_id": {
                            "type": "string", "required": True,
                            "description": "Agent instance ID whose inbox to read",
                        },
                        "from_instance_id": {
                            "type": "string", "required": False,
                            "description": "Filter to messages from a specific agent",
                        },
                        "message_type": {
                            "type": "string", "required": False,
                            "description": "Filter by message type: direct, service_request, broadcast",
                        },
                        "unread_only": {
                            "type": "bool", "required": False,
                            "description": "Only return unread messages (default: false)",
                        },
                        "limit": {
                            "type": "int", "required": False,
                            "description": "Max messages to return (default: 50)",
                        },
                        "conversation_id": {
                            "type": "string", "required": False,
                            "description": "Filter to a specific conversation thread",
                        },
                    },
                    estimated_cost=0.0,
                ),
                SkillAction(
                    name="broadcast",
                    description="Send a message to all registered agents",
                    parameters={
                        "from_instance_id": {
                            "type": "string", "required": True,
                            "description": "Sender's agent instance ID",
                        },
                        "content": {
                            "type": "string", "required": True,
                            "description": "Broadcast message content",
                        },
                        "metadata": {
                            "type": "dict", "required": False,
                            "description": "Optional metadata (e.g. subject, tags)",
                        },
                    },
                    estimated_cost=0.0,
                ),
                SkillAction(
                    name="service_request",
                    description="Send a structured service request to another agent",
                    parameters={
                        "from_instance_id": {
                            "type": "string", "required": True,
                            "description": "Requester's agent instance ID",
                        },
                        "to_instance_id": {
                            "type": "string", "required": True,
                            "description": "Service provider's agent instance ID",
                        },
                        "service_name": {
                            "type": "string", "required": True,
                            "description": "Name of the service being requested",
                        },
                        "request_params": {
                            "type": "dict", "required": False,
                            "description": "Parameters for the service request",
                        },
                        "offer_amount": {
                            "type": "float", "required": False,
                            "description": "Amount offered for the service (in WISENT tokens)",
                        },
                    },
                    estimated_cost=0.0,
                ),
                SkillAction(
                    name="reply",
                    description="Reply to a specific message, creating a conversation thread",
                    parameters={
                        "from_instance_id": {
                            "type": "string", "required": True,
                            "description": "Sender's agent instance ID",
                        },
                        "message_id": {
                            "type": "string", "required": True,
                            "description": "ID of the message being replied to",
                        },
                        "content": {
                            "type": "string", "required": True,
                            "description": "Reply content",
                        },
                    },
                    estimated_cost=0.0,
                ),
                SkillAction(
                    name="get_conversation",
                    description="Get all messages in a conversation thread",
                    parameters={
                        "conversation_id": {
                            "type": "string", "required": True,
                            "description": "Conversation thread ID",
                        },
                    },
                    estimated_cost=0.0,
                ),
                SkillAction(
                    name="mark_read",
                    description="Mark a message as read",
                    parameters={
                        "message_id": {
                            "type": "string", "required": True,
                            "description": "ID of the message to mark as read",
                        },
                        "reader_instance_id": {
                            "type": "string", "required": True,
                            "description": "Instance ID of the agent reading the message",
                        },
                    },
                    estimated_cost=0.0,
                ),
                SkillAction(
                    name="delete_message",
                    description="Delete a message from an inbox",
                    parameters={
                        "instance_id": {
                            "type": "string", "required": True,
                            "description": "Agent instance ID (must own the inbox)",
                        },
                        "message_id": {
                            "type": "string", "required": True,
                            "description": "ID of the message to delete",
                        },
                    },
                    estimated_cost=0.0,
                ),
                SkillAction(
                    name="get_stats",
                    description="Get messaging statistics",
                    parameters={},
                    estimated_cost=0.0,
                ),
            ],
            author="Singularity",
        )

    async def execute(self, action: str, params: Dict = None) -> SkillResult:
        params = params or {}
        handlers = {
            "send": self._send,
            "read_inbox": self._read_inbox,
            "broadcast": self._broadcast,
            "service_request": self._service_request,
            "reply": self._reply,
            "get_conversation": self._get_conversation,
            "mark_read": self._mark_read,
            "delete_message": self._delete_message,
            "get_stats": self._get_stats,
        }

        handler = handlers.get(action)
        if not handler:
            return SkillResult(
                success=False,
                message=f"Unknown action: {action}. Available: {list(handlers.keys())}",
            )

        try:
            return handler(params)
        except Exception as e:
            return SkillResult(success=False, message=f"Error in {action}: {e}")

    # ── Send ─────────────────────────────────────────────────

    def _send(self, params: Dict) -> SkillResult:
        """Send a direct message to another agent."""
        from_id = params.get("from_instance_id", "")
        to_id = params.get("to_instance_id", "")
        content = params.get("content", "")
        msg_type = params.get("message_type", "direct")
        metadata = params.get("metadata", {})
        conversation_id = params.get("conversation_id", "")
        ttl_hours = params.get("ttl_hours", MESSAGE_TTL_HOURS)

        if not from_id:
            return SkillResult(success=False, message="from_instance_id is required")
        if not to_id:
            return SkillResult(success=False, message="to_instance_id is required")
        if not content:
            return SkillResult(success=False, message="content is required")
        if len(content) > MAX_MESSAGE_BODY_LENGTH:
            return SkillResult(
                success=False,
                message=f"Message too long (max {MAX_MESSAGE_BODY_LENGTH} chars)",
            )
        if msg_type not in ("direct", "service_request", "broadcast"):
            return SkillResult(
                success=False,
                message="message_type must be: direct, service_request, or broadcast",
            )

        # Create or join conversation
        if not conversation_id:
            conversation_id = f"conv_{uuid.uuid4().hex[:12]}"

        message_id = f"msg_{uuid.uuid4().hex[:16]}"
        message = {
            "message_id": message_id,
            "from_instance_id": from_id,
            "to_instance_id": to_id,
            "content": content,
            "type": msg_type,
            "metadata": metadata,
            "conversation_id": conversation_id,
            "timestamp": _now_iso(),
            "ttl_hours": ttl_hours,
            "read": False,
        }

        data = _load_data(self._data_path)

        # Add to recipient's inbox
        inbox = data["inboxes"].setdefault(to_id, [])
        # Purge expired messages
        inbox = [m for m in inbox if not _is_expired(m, m.get("ttl_hours", MESSAGE_TTL_HOURS))]
        # Trim if too many
        if len(inbox) >= MAX_MESSAGES_PER_INBOX:
            inbox = inbox[-(MAX_MESSAGES_PER_INBOX - 1):]
        inbox.append(message)
        data["inboxes"][to_id] = inbox

        # Track conversation
        conv = data["conversations"].setdefault(conversation_id, {
            "conversation_id": conversation_id,
            "participants": [],
            "created_at": _now_iso(),
            "last_message_at": _now_iso(),
            "message_count": 0,
        })
        # Add participants
        for pid in [from_id, to_id]:
            if pid not in conv["participants"]:
                conv["participants"].append(pid)
        conv["last_message_at"] = _now_iso()
        conv["message_count"] = conv.get("message_count", 0) + 1

        # Trim conversations
        if len(data["conversations"]) > MAX_CONVERSATIONS:
            sorted_convs = sorted(
                data["conversations"].items(),
                key=lambda x: x[1].get("last_message_at", ""),
            )
            for cid, _ in sorted_convs[:len(data["conversations"]) - MAX_CONVERSATIONS]:
                del data["conversations"][cid]

        data["stats"]["total_sent"] = data["stats"].get("total_sent", 0) + 1

        _save_data(data, self._data_path)

        return SkillResult(
            success=True,
            message=f"Message sent from {from_id} to {to_id}",
            data={
                "message_id": message_id,
                "conversation_id": conversation_id,
                "to": to_id,
                "type": msg_type,
            },
        )

    # ── Read Inbox ───────────────────────────────────────────

    def _read_inbox(self, params: Dict) -> SkillResult:
        """Read messages for an agent with optional filters."""
        instance_id = params.get("instance_id", "")
        from_filter = params.get("from_instance_id", "")
        type_filter = params.get("message_type", "")
        unread_only = params.get("unread_only", False)
        limit = min(params.get("limit", 50), 200)
        conv_filter = params.get("conversation_id", "")

        if not instance_id:
            return SkillResult(success=False, message="instance_id is required")

        data = _load_data(self._data_path)
        inbox = data["inboxes"].get(instance_id, [])

        # Filter expired
        inbox = [m for m in inbox if not _is_expired(m, m.get("ttl_hours", MESSAGE_TTL_HOURS))]
        # Persist cleaned inbox
        data["inboxes"][instance_id] = inbox

        # Apply filters
        results = []
        for msg in inbox:
            if from_filter and msg.get("from_instance_id") != from_filter:
                continue
            if type_filter and msg.get("type") != type_filter:
                continue
            if unread_only and msg.get("read", False):
                continue
            if conv_filter and msg.get("conversation_id") != conv_filter:
                continue
            results.append(msg)

        # Sort by newest first
        results.sort(key=lambda m: m.get("timestamp", ""), reverse=True)
        results = results[:limit]

        _save_data(data, self._data_path)

        return SkillResult(
            success=True,
            message=f"{len(results)} message(s) in inbox for {instance_id}",
            data={
                "messages": results,
                "count": len(results),
                "total_in_inbox": len(inbox),
            },
        )

    # ── Broadcast ────────────────────────────────────────────

    def _broadcast(self, params: Dict) -> SkillResult:
        """Send a message to all known agents."""
        from_id = params.get("from_instance_id", "")
        content = params.get("content", "")
        metadata = params.get("metadata", {})

        if not from_id:
            return SkillResult(success=False, message="from_instance_id is required")
        if not content:
            return SkillResult(success=False, message="content is required")

        data = _load_data(self._data_path)

        # Get all known inbox owners (agents who have received messages before)
        all_agents = set(data["inboxes"].keys())

        # Also check peer_discovery and agent_network for registered agents
        peers_file = DATA_DIR / "agent_network.json"
        if peers_file.exists():
            try:
                with open(peers_file) as f:
                    network = json.load(f)
                for peer_id in network.get("peers", {}):
                    all_agents.add(peer_id)
            except (json.JSONDecodeError, IOError):
                pass

        # Don't send to self
        all_agents.discard(from_id)

        sent_count = 0
        conversation_id = f"bcast_{uuid.uuid4().hex[:12]}"

        for agent_id in all_agents:
            msg_result = self._send({
                "from_instance_id": from_id,
                "to_instance_id": agent_id,
                "content": content,
                "message_type": "broadcast",
                "metadata": metadata,
                "conversation_id": conversation_id,
            })
            if msg_result.success:
                sent_count += 1

        # Update stats
        data = _load_data(self._data_path)
        data["stats"]["total_broadcasts"] = data["stats"].get("total_broadcasts", 0) + 1
        _save_data(data, self._data_path)

        return SkillResult(
            success=True,
            message=f"Broadcast sent to {sent_count} agent(s)",
            data={
                "sent_count": sent_count,
                "conversation_id": conversation_id,
                "recipients": list(all_agents),
            },
        )

    # ── Service Request ──────────────────────────────────────

    def _service_request(self, params: Dict) -> SkillResult:
        """Send a structured service request."""
        from_id = params.get("from_instance_id", "")
        to_id = params.get("to_instance_id", "")
        service_name = params.get("service_name", "")
        request_params = params.get("request_params", {})
        offer_amount = params.get("offer_amount", 0.0)

        if not from_id:
            return SkillResult(success=False, message="from_instance_id is required")
        if not to_id:
            return SkillResult(success=False, message="to_instance_id is required")
        if not service_name:
            return SkillResult(success=False, message="service_name is required")

        request_id = f"sreq_{uuid.uuid4().hex[:12]}"

        content = json.dumps({
            "request_id": request_id,
            "service_name": service_name,
            "params": request_params,
            "offer_amount": offer_amount,
            "status": "pending",
        })

        metadata = {
            "subject": f"Service Request: {service_name}",
            "request_id": request_id,
            "service_name": service_name,
            "offer_amount": offer_amount,
        }

        result = self._send({
            "from_instance_id": from_id,
            "to_instance_id": to_id,
            "content": content,
            "message_type": "service_request",
            "metadata": metadata,
        })

        if result.success:
            data = _load_data(self._data_path)
            data["stats"]["total_service_requests"] = data["stats"].get("total_service_requests", 0) + 1
            _save_data(data, self._data_path)

        return SkillResult(
            success=result.success,
            message=f"Service request '{service_name}' sent to {to_id}" if result.success else result.message,
            data={
                "request_id": request_id,
                "message_id": result.data.get("message_id", ""),
                "conversation_id": result.data.get("conversation_id", ""),
                "service_name": service_name,
                "offer_amount": offer_amount,
            },
        )

    # ── Reply ────────────────────────────────────────────────

    def _reply(self, params: Dict) -> SkillResult:
        """Reply to a specific message, creating a conversation thread."""
        from_id = params.get("from_instance_id", "")
        message_id = params.get("message_id", "")
        content = params.get("content", "")

        if not from_id:
            return SkillResult(success=False, message="from_instance_id is required")
        if not message_id:
            return SkillResult(success=False, message="message_id is required")
        if not content:
            return SkillResult(success=False, message="content is required")

        data = _load_data(self._data_path)

        # Find the original message across all inboxes
        original = None
        for inbox_id, inbox in data["inboxes"].items():
            for msg in inbox:
                if msg.get("message_id") == message_id:
                    original = msg
                    break
            if original:
                break

        if not original:
            return SkillResult(success=False, message=f"Message {message_id} not found")

        # Reply goes to the original sender
        to_id = original["from_instance_id"]
        conversation_id = original.get("conversation_id", f"conv_{uuid.uuid4().hex[:12]}")

        return self._send({
            "from_instance_id": from_id,
            "to_instance_id": to_id,
            "content": content,
            "message_type": original.get("type", "direct"),
            "conversation_id": conversation_id,
            "metadata": {
                "reply_to": message_id,
                "subject": f"Re: {original.get('metadata', {}).get('subject', 'message')}",
            },
        })

    # ── Get Conversation ─────────────────────────────────────

    def _get_conversation(self, params: Dict) -> SkillResult:
        """Get all messages in a conversation thread."""
        conversation_id = params.get("conversation_id", "")
        if not conversation_id:
            return SkillResult(success=False, message="conversation_id is required")

        data = _load_data(self._data_path)

        conv_meta = data["conversations"].get(conversation_id, {})

        # Collect messages from all inboxes in this conversation
        messages = []
        seen_ids = set()
        for inbox_id, inbox in data["inboxes"].items():
            for msg in inbox:
                if msg.get("conversation_id") == conversation_id:
                    mid = msg.get("message_id")
                    if mid not in seen_ids:
                        messages.append(msg)
                        seen_ids.add(mid)

        messages.sort(key=lambda m: m.get("timestamp", ""))

        return SkillResult(
            success=True,
            message=f"Conversation {conversation_id}: {len(messages)} message(s)",
            data={
                "conversation_id": conversation_id,
                "metadata": conv_meta,
                "messages": messages,
                "count": len(messages),
            },
        )

    # ── Mark Read ────────────────────────────────────────────

    def _mark_read(self, params: Dict) -> SkillResult:
        """Mark a message as read."""
        message_id = params.get("message_id", "")
        reader_id = params.get("reader_instance_id", "")

        if not message_id:
            return SkillResult(success=False, message="message_id is required")
        if not reader_id:
            return SkillResult(success=False, message="reader_instance_id is required")

        data = _load_data(self._data_path)

        found = False
        inbox = data["inboxes"].get(reader_id, [])
        for msg in inbox:
            if msg.get("message_id") == message_id:
                msg["read"] = True
                found = True
                break

        if not found:
            return SkillResult(success=False, message=f"Message {message_id} not found in inbox")

        # Track read receipt
        receipts = data["read_receipts"].setdefault(message_id, {
            "read_by": [],
            "read_at": [],
        })
        if reader_id not in receipts["read_by"]:
            receipts["read_by"].append(reader_id)
            receipts["read_at"].append(_now_iso())

        data["stats"]["total_read"] = data["stats"].get("total_read", 0) + 1
        _save_data(data, self._data_path)

        return SkillResult(
            success=True,
            message=f"Message {message_id} marked as read by {reader_id}",
            data={"message_id": message_id, "reader": reader_id},
        )

    # ── Delete Message ───────────────────────────────────────

    def _delete_message(self, params: Dict) -> SkillResult:
        """Delete a message from an inbox."""
        instance_id = params.get("instance_id", "")
        message_id = params.get("message_id", "")

        if not instance_id:
            return SkillResult(success=False, message="instance_id is required")
        if not message_id:
            return SkillResult(success=False, message="message_id is required")

        data = _load_data(self._data_path)
        inbox = data["inboxes"].get(instance_id, [])

        original_len = len(inbox)
        inbox = [m for m in inbox if m.get("message_id") != message_id]

        if len(inbox) == original_len:
            return SkillResult(
                success=False,
                message=f"Message {message_id} not found in inbox for {instance_id}",
            )

        data["inboxes"][instance_id] = inbox
        _save_data(data, self._data_path)

        return SkillResult(
            success=True,
            message=f"Message {message_id} deleted from {instance_id}'s inbox",
            data={"message_id": message_id, "instance_id": instance_id},
        )

    # ── Stats ────────────────────────────────────────────────

    def _get_stats(self, params: Dict) -> SkillResult:
        """Get messaging statistics."""
        data = _load_data(self._data_path)

        total_messages = sum(len(inbox) for inbox in data["inboxes"].values())
        active_agents = len(data["inboxes"])
        active_conversations = len(data["conversations"])

        return SkillResult(
            success=True,
            message=f"Messaging stats: {total_messages} messages across {active_agents} agents",
            data={
                "total_messages_stored": total_messages,
                "active_agents": active_agents,
                "active_conversations": active_conversations,
                "lifetime_stats": data["stats"],
            },
        )
