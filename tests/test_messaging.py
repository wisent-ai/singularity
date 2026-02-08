"""Tests for MessagingSkill - Agent-to-agent messaging API."""
import pytest
from pathlib import Path
from singularity.skills.messaging import MessagingSkill


@pytest.fixture
def skill(tmp_path):
    s = MessagingSkill(credentials={"data_path": str(tmp_path / "messages.json")})
    return s


@pytest.mark.asyncio
async def test_send_direct_message(skill):
    r = await skill.execute("send", {
        "from_instance_id": "agent_eve",
        "to_instance_id": "agent_adam",
        "content": "Hello Adam, want to bundle our services?",
    })
    assert r.success
    assert r.data["to"] == "agent_adam"
    assert r.data["message_id"].startswith("msg_")
    assert r.data["conversation_id"].startswith("conv_")


@pytest.mark.asyncio
async def test_read_inbox(skill):
    await skill.execute("send", {"from_instance_id": "eve", "to_instance_id": "adam", "content": "Hi"})
    await skill.execute("send", {"from_instance_id": "eve", "to_instance_id": "adam", "content": "Hey"})
    r = await skill.execute("read_inbox", {"instance_id": "adam"})
    assert r.success
    assert r.data["count"] == 2


@pytest.mark.asyncio
async def test_read_inbox_filters(skill):
    await skill.execute("send", {"from_instance_id": "eve", "to_instance_id": "adam", "content": "m1"})
    await skill.execute("send", {"from_instance_id": "bob", "to_instance_id": "adam", "content": "m2"})
    r = await skill.execute("read_inbox", {"instance_id": "adam", "from_instance_id": "eve"})
    assert r.data["count"] == 1
    assert r.data["messages"][0]["from_instance_id"] == "eve"


@pytest.mark.asyncio
async def test_service_request(skill):
    r = await skill.execute("service_request", {
        "from_instance_id": "eve",
        "to_instance_id": "adam",
        "service_name": "code_review",
        "request_params": {"repo": "eve-services"},
        "offer_amount": 5.0,
    })
    assert r.success
    assert r.data["request_id"].startswith("sreq_")
    assert r.data["service_name"] == "code_review"
    inbox = await skill.execute("read_inbox", {"instance_id": "adam", "message_type": "service_request"})
    assert inbox.data["count"] == 1


@pytest.mark.asyncio
async def test_reply_creates_thread(skill):
    s = await skill.execute("send", {"from_instance_id": "eve", "to_instance_id": "adam", "content": "Proposal"})
    msg_id = s.data["message_id"]
    conv_id = s.data["conversation_id"]
    r = await skill.execute("reply", {"from_instance_id": "adam", "message_id": msg_id, "content": "Agreed!"})
    assert r.success
    assert r.data["conversation_id"] == conv_id


@pytest.mark.asyncio
async def test_get_conversation(skill):
    s = await skill.execute("send", {"from_instance_id": "eve", "to_instance_id": "adam", "content": "Hi"})
    conv_id = s.data["conversation_id"]
    await skill.execute("reply", {"from_instance_id": "adam", "message_id": s.data["message_id"], "content": "Hey"})
    r = await skill.execute("get_conversation", {"conversation_id": conv_id})
    assert r.success
    assert r.data["count"] >= 2


@pytest.mark.asyncio
async def test_mark_read(skill):
    s = await skill.execute("send", {"from_instance_id": "eve", "to_instance_id": "adam", "content": "Read me"})
    msg_id = s.data["message_id"]
    r = await skill.execute("mark_read", {"message_id": msg_id, "reader_instance_id": "adam"})
    assert r.success
    inbox = await skill.execute("read_inbox", {"instance_id": "adam", "unread_only": True})
    assert inbox.data["count"] == 0


@pytest.mark.asyncio
async def test_delete_message(skill):
    s = await skill.execute("send", {"from_instance_id": "eve", "to_instance_id": "adam", "content": "Delete me"})
    msg_id = s.data["message_id"]
    r = await skill.execute("delete_message", {"instance_id": "adam", "message_id": msg_id})
    assert r.success
    inbox = await skill.execute("read_inbox", {"instance_id": "adam"})
    assert inbox.data["count"] == 0


@pytest.mark.asyncio
async def test_broadcast(skill):
    await skill.execute("send", {"from_instance_id": "eve", "to_instance_id": "adam", "content": "x"})
    await skill.execute("send", {"from_instance_id": "eve", "to_instance_id": "bob", "content": "x"})
    r = await skill.execute("broadcast", {"from_instance_id": "eve", "content": "Big announcement!"})
    assert r.success
    assert r.data["sent_count"] == 2


@pytest.mark.asyncio
async def test_get_stats(skill):
    await skill.execute("send", {"from_instance_id": "eve", "to_instance_id": "adam", "content": "Hi"})
    r = await skill.execute("get_stats", {})
    assert r.success
    assert r.data["total_messages_stored"] >= 1


@pytest.mark.asyncio
async def test_validation_errors(skill):
    r = await skill.execute("send", {"from_instance_id": "", "to_instance_id": "adam", "content": "Hi"})
    assert not r.success
    r = await skill.execute("send", {"from_instance_id": "eve", "to_instance_id": "adam", "content": ""})
    assert not r.success
    r = await skill.execute("send", {
        "from_instance_id": "eve", "to_instance_id": "adam",
        "content": "x", "message_type": "invalid",
    })
    assert not r.success


@pytest.mark.asyncio
async def test_unknown_action(skill):
    r = await skill.execute("nonexistent", {})
    assert not r.success
    assert "Unknown action" in r.message
