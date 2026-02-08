"""Tests for ConversationCompressorSkill."""
import pytest
import json
from pathlib import Path
from singularity.skills.conversation_compressor import ConversationCompressorSkill

@pytest.fixture
def skill(tmp_path, monkeypatch):
    monkeypatch.setattr("singularity.skills.conversation_compressor.DATA_DIR", tmp_path)
    monkeypatch.setattr("singularity.skills.conversation_compressor.COMPRESSOR_FILE", tmp_path / "conversation_compressor.json")
    return ConversationCompressorSkill()

def make_messages(n_pairs):
    """Generate n pairs of user/assistant messages."""
    msgs = []
    for i in range(n_pairs):
        msgs.append({"role": "user", "content": f"Request {i}: Please execute task number {i} for the project"})
        msgs.append({"role": "assistant", "content": f"Result {i}: I will execute task {i}. Action completed successfully with status ok."})
    return msgs

@pytest.mark.asyncio
async def test_analyze(skill):
    msgs = make_messages(10)
    r = await skill.execute("analyze", {"messages": msgs})
    assert r.success
    assert r.data["total_messages"] == 20
    assert r.data["total_tokens"] > 0

@pytest.mark.asyncio
async def test_analyze_over_budget(skill):
    msgs = make_messages(50)  # Many messages should exceed default 8000 token budget
    r = await skill.execute("analyze", {"messages": msgs})
    assert r.success
    assert r.data["compressible_messages"] > 0

@pytest.mark.asyncio
async def test_compress(skill):
    msgs = make_messages(10)
    r = await skill.execute("compress", {"messages": msgs})
    assert r.success
    assert r.data["compressed"] is True
    assert r.data["tokens_saved"] > 0
    assert len(r.data["preserved_messages"]) == 12  # default preserve_recent=6 pairs

@pytest.mark.asyncio
async def test_compress_nothing_to_compress(skill):
    msgs = make_messages(3)  # Only 3 pairs, all within preservation window (6 pairs)
    r = await skill.execute("compress", {"messages": msgs})
    assert r.success
    assert r.data["compressed"] is False

@pytest.mark.asyncio
async def test_extract_facts(skill):
    msgs = [
        {"role": "user", "content": "Deploy the service to production"},
        {"role": "assistant", "content": "I will deploy the service to production. Result: deployment successful."},
    ]
    r = await skill.execute("extract_facts", {"messages": msgs})
    assert r.success
    assert r.data["extracted_facts"]

@pytest.mark.asyncio
async def test_add_and_list_facts(skill):
    r = await skill.execute("add_fact", {"fact": "System uses Python 3.11", "category": "context"})
    assert r.success
    r = await skill.execute("facts")
    assert r.success
    assert r.data["total"] == 1
    assert "Python 3.11" in r.data["facts"][0]["text"]

@pytest.mark.asyncio
async def test_remove_fact(skill):
    await skill.execute("add_fact", {"fact": "Fact 1"})
    await skill.execute("add_fact", {"fact": "Fact 2"})
    r = await skill.execute("remove_fact", {"index": 0})
    assert r.success
    r = await skill.execute("facts")
    assert r.data["total"] == 1

@pytest.mark.asyncio
async def test_inject_empty(skill):
    r = await skill.execute("inject")
    assert r.success
    assert r.data["has_context"] is False

@pytest.mark.asyncio
async def test_inject_with_facts(skill):
    await skill.execute("add_fact", {"fact": "Important context"})
    r = await skill.execute("inject")
    assert r.success
    assert r.data["has_context"] is True
    assert "Important context" in r.data["context_block"]

@pytest.mark.asyncio
async def test_stats(skill):
    r = await skill.execute("stats")
    assert r.success
    assert "stats" in r.data

@pytest.mark.asyncio
async def test_configure(skill):
    r = await skill.execute("configure", {"max_tokens": 16000, "preserve_recent": 10})
    assert r.success
    assert r.data["settings"]["max_tokens"] == 16000
    assert r.data["settings"]["preserve_recent"] == 10

@pytest.mark.asyncio
async def test_reset(skill):
    await skill.execute("add_fact", {"fact": "Test fact"})
    r = await skill.execute("reset")
    assert r.success
    r = await skill.execute("facts")
    assert r.data["total"] == 0

def test_auto_compress_under_budget(skill):
    msgs = make_messages(3)
    result = skill.auto_compress_if_needed(msgs)
    assert result["compressed"] is False
    assert len(result["messages"]) == 6

@pytest.mark.asyncio
async def test_auto_compress_over_budget(skill):
    # Lower budget so our test messages exceed it
    await skill.execute("configure", {"max_tokens": 500, "preserve_recent": 3})
    msgs = make_messages(50)  # 100 messages, should exceed 500 token budget
    result = skill.auto_compress_if_needed(msgs)
    assert result["compressed"] is True
    assert result["tokens_saved"] > 0
    assert len(result["messages"]) < len(msgs)
    assert result["context_preamble"] != ""

def test_estimate_tokens():
    assert ConversationCompressorSkill.estimate_tokens("") == 0
    assert ConversationCompressorSkill.estimate_tokens("hello world") > 0

def test_estimate_message_tokens():
    msgs = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "world"}]
    tokens = ConversationCompressorSkill.estimate_message_tokens(msgs)
    assert tokens > 0
