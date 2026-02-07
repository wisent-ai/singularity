"""Tests for InboxSkill - file-based agent messaging."""

import asyncio
import json
import tempfile
from pathlib import Path

import pytest

from singularity.skills.inbox import InboxSkill


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def alice(tmp_dir):
    skill = InboxSkill()
    skill.configure("alice", inbox_dir=tmp_dir)
    return skill


@pytest.fixture
def bob(tmp_dir):
    skill = InboxSkill()
    skill.configure("bob", inbox_dir=tmp_dir)
    return skill


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class TestInboxBasics:
    def test_manifest(self):
        s = InboxSkill()
        assert s.manifest.skill_id == "inbox"
        assert s.manifest.category == "communication"
        assert len(s.manifest.actions) == 7
        assert s.check_credentials()  # no credentials required

    def test_empty_inbox(self, alice):
        r = run(alice.execute("check_inbox", {}))
        assert r.success
        assert r.data["unread_count"] == 0
        assert r.data["total_count"] == 0

    def test_unknown_action(self, alice):
        r = run(alice.execute("nonexistent", {}))
        assert not r.success


class TestSendAndReceive:
    def test_send_message(self, alice, bob):
        r = run(alice.execute("send_message", {
            "to": "bob", "subject": "Hello", "body": "Hi Bob!"
        }))
        assert r.success
        assert r.data["to"] == "bob"
        msg_id = r.data["message_id"]

        # Bob should see it
        r2 = run(bob.execute("check_inbox", {}))
        assert r2.success
        assert r2.data["unread_count"] == 1
        assert r2.data["unread_summaries"][0]["from"] == "alice"

        # Bob reads it
        r3 = run(bob.execute("read_message", {"message_id": msg_id}))
        assert r3.success
        assert r3.data["body"] == "Hi Bob!"
        assert r3.data["read"] is True

        # Now 0 unread
        r4 = run(bob.execute("check_inbox", {}))
        assert r4.data["unread_count"] == 0

    def test_send_validation(self, alice):
        r = run(alice.execute("send_message", {"to": "", "subject": "X", "body": "Y"}))
        assert not r.success

        r2 = run(alice.execute("send_message", {"to": "bob", "subject": "", "body": ""}))
        assert not r2.success

    def test_priority(self, alice, bob):
        run(alice.execute("send_message", {
            "to": "bob", "subject": "Urgent!", "body": "Now!", "priority": "urgent"
        }))
        r = run(bob.execute("check_inbox", {}))
        assert r.data["unread_summaries"][0]["priority"] == "urgent"

    def test_invalid_priority_defaults(self, alice, bob):
        run(alice.execute("send_message", {
            "to": "bob", "subject": "Test", "body": "X", "priority": "invalid"
        }))
        r = run(bob.execute("check_inbox", {}))
        assert r.data["unread_summaries"][0]["priority"] == "normal"


class TestListAndFilter:
    def test_list_all(self, alice, bob):
        run(alice.execute("send_message", {"to": "bob", "subject": "A", "body": "1"}))
        run(alice.execute("send_message", {"to": "bob", "subject": "B", "body": "2"}))

        r = run(bob.execute("list_messages", {}))
        assert r.success
        assert r.data["count"] == 2

    def test_filter_unread(self, alice, bob):
        r1 = run(alice.execute("send_message", {"to": "bob", "subject": "A", "body": "1"}))
        run(alice.execute("send_message", {"to": "bob", "subject": "B", "body": "2"}))

        # Read first message
        run(bob.execute("read_message", {"message_id": r1.data["message_id"]}))

        r = run(bob.execute("list_messages", {"status": "unread"}))
        assert r.data["count"] == 1
        assert r.data["messages"][0]["subject"] == "B"

    def test_filter_by_sender(self, alice, bob, tmp_dir):
        charlie = InboxSkill()
        charlie.configure("charlie", inbox_dir=tmp_dir)

        run(alice.execute("send_message", {"to": "bob", "subject": "From Alice", "body": "Hi"}))
        run(charlie.execute("send_message", {"to": "bob", "subject": "From Charlie", "body": "Hey"}))

        r = run(bob.execute("list_messages", {"from_agent": "charlie"}))
        assert r.data["count"] == 1
        assert r.data["messages"][0]["from"] == "charlie"

    def test_list_limit(self, alice, bob):
        for i in range(5):
            run(alice.execute("send_message", {"to": "bob", "subject": f"Msg {i}", "body": "X"}))

        r = run(bob.execute("list_messages", {"limit": 2}))
        assert r.data["count"] == 2


class TestDelete:
    def test_delete_message(self, alice, bob):
        r = run(alice.execute("send_message", {"to": "bob", "subject": "Del me", "body": "X"}))
        msg_id = r.data["message_id"]

        r2 = run(bob.execute("delete_message", {"message_id": msg_id}))
        assert r2.success

        r3 = run(bob.execute("check_inbox", {}))
        assert r3.data["total_count"] == 0

    def test_delete_nonexistent(self, alice):
        r = run(alice.execute("delete_message", {"message_id": "fake"}))
        assert not r.success


class TestConversation:
    def test_reply_thread(self, alice, bob):
        # Alice sends to Bob
        r1 = run(alice.execute("send_message", {"to": "bob", "subject": "Question", "body": "How?"}))
        msg1 = r1.data["message_id"]

        # Bob replies
        r2 = run(bob.execute("send_message", {
            "to": "alice", "subject": "Re: Question", "body": "Like this!",
            "reply_to": msg1
        }))
        msg2 = r2.data["message_id"]

        # Alice gets the conversation from her inbox
        r3 = run(alice.execute("get_conversation", {"message_id": msg2}))
        assert r3.success
        assert r3.data["message_count"] >= 1  # At least the reply is there


class TestBroadcast:
    def test_broadcast_to_known_agents(self, alice, tmp_dir):
        # Create bob and charlie inboxes
        bob = InboxSkill()
        bob.configure("bob", inbox_dir=tmp_dir)
        charlie = InboxSkill()
        charlie.configure("charlie", inbox_dir=tmp_dir)

        # Broadcast
        r = run(alice.execute("broadcast", {"subject": "Announcement", "body": "Hello all!"}))
        assert r.success
        assert r.data["count"] == 2
        assert set(r.data["sent_to"]) == {"bob", "charlie"}

        # Both receive it
        assert run(bob.execute("check_inbox", {})).data["unread_count"] == 1
        assert run(charlie.execute("check_inbox", {})).data["unread_count"] == 1

    def test_broadcast_no_agents(self, tmp_dir):
        lonely = InboxSkill()
        lonely.configure("lonely", inbox_dir=tmp_dir)
        r = run(lonely.execute("broadcast", {"subject": "Hello?", "body": "Anyone?"}))
        assert r.success
        assert r.data["count"] == 0


class TestHelpers:
    def test_unread_count(self, alice, bob):
        assert bob.get_unread_count() == 0
        run(alice.execute("send_message", {"to": "bob", "subject": "X", "body": "Y"}))
        assert bob.get_unread_count() == 1

    def test_unread_summary(self, alice, bob):
        assert bob.get_unread_summary() == ""
        run(alice.execute("send_message", {"to": "bob", "subject": "Important", "body": "Y", "priority": "urgent"}))
        summary = bob.get_unread_summary()
        assert "1 unread" in summary
        assert "Important" in summary
        assert "🔴" in summary

    def test_auto_configure(self):
        skill = InboxSkill()
        # Should auto-configure on first use
        assert skill.get_unread_count() == 0
