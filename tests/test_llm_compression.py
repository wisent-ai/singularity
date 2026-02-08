"""Tests for LLM-powered compression in ConversationCompressorSkill."""
import pytest
import json
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path
from singularity.skills.conversation_compressor import (
    ConversationCompressorSkill,
    SUMMARIZE_SYSTEM_PROMPT,
    EXTRACT_FACTS_SYSTEM_PROMPT,
)
from singularity.cognition import TokenUsage

@pytest.fixture
def skill(tmp_path, monkeypatch):
    monkeypatch.setattr("singularity.skills.conversation_compressor.DATA_DIR", tmp_path)
    monkeypatch.setattr("singularity.skills.conversation_compressor.COMPRESSOR_FILE", tmp_path / "compressor.json")
    return ConversationCompressorSkill()

@pytest.fixture
def mock_engine():
    engine = MagicMock()
    engine.llm = MagicMock()
    engine.llm_type = "anthropic"
    engine._call_with_fallback = AsyncMock(return_value=(
        "The agent decided to deploy the API. Task completed successfully.",
        TokenUsage(), "anthropic", "claude-3"
    ))
    return engine

def make_messages(n_pairs, long=False):
    msgs = []
    for i in range(n_pairs):
        extra = " This is a detailed request with comprehensive analysis required." * 3 if long else ""
        msgs.append({"role": "user", "content": f"Request {i}: Please execute task {i} with high priority.{extra}"})
        extra_r = " Execution completed with full diagnostic output and status reporting." * 3 if long else ""
        msgs.append({"role": "assistant", "content": f"Result {i}: Task {i} completed. Balance $0.50. Goal achieved.{extra_r}"})
    return msgs

def test_has_llm_false_without_engine(skill):
    assert skill.has_llm() is False

def test_has_llm_true_with_engine(skill, mock_engine):
    skill.set_cognition_engine(mock_engine)
    assert skill.has_llm() is True

def test_has_llm_false_when_disabled(skill, mock_engine):
    skill.set_cognition_engine(mock_engine)
    skill._llm_compression_enabled = False
    assert skill.has_llm() is False

def test_has_llm_false_when_no_llm_type(skill):
    engine = MagicMock()
    engine.llm = None
    engine.llm_type = "none"
    skill.set_cognition_engine(engine)
    assert skill.has_llm() is False

def test_format_messages_for_llm(skill):
    msgs = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
    ]
    result = skill._format_messages_for_llm(msgs)
    assert "[USER]: Hello" in result
    assert "[ASSISTANT]: Hi there" in result

def test_format_messages_truncates_long(skill):
    msgs = [{"role": "user", "content": "x" * 2000}]
    result = skill._format_messages_for_llm(msgs)
    assert len(result) < 1100

@pytest.mark.asyncio
async def test_llm_compress_messages(skill, mock_engine):
    skill.set_cognition_engine(mock_engine)
    msgs = make_messages(5)
    result = await skill.llm_compress_messages(msgs)
    assert "deploy" in result.lower() or "completed" in result.lower()
    mock_engine._call_with_fallback.assert_called_once()

@pytest.mark.asyncio
async def test_llm_compress_messages_fallback_on_error(skill, mock_engine):
    mock_engine._call_with_fallback = AsyncMock(side_effect=Exception("API error"))
    skill.set_cognition_engine(mock_engine)
    msgs = make_messages(3)
    result = await skill.llm_compress_messages(msgs)
    # Should fall back to regex compression
    assert len(result) > 0

@pytest.mark.asyncio
async def test_llm_compress_without_engine(skill):
    msgs = make_messages(3)
    result = await skill.llm_compress_messages(msgs)
    # Falls back to regex
    assert len(result) > 0

@pytest.mark.asyncio
async def test_llm_extract_facts(skill, mock_engine):
    mock_engine._call_with_fallback = AsyncMock(return_value=(
        "The agent deployed version 2.0\nBalance was $0.50\nGoal: improve self-improvement pillar",
        TokenUsage(), "anthropic", "claude-3"
    ))
    skill.set_cognition_engine(mock_engine)
    msgs = make_messages(3)
    facts = await skill.llm_extract_facts(msgs)
    assert len(facts) > 0
    assert any("deploy" in f.lower() or "balance" in f.lower() for f in facts)

@pytest.mark.asyncio
async def test_llm_extract_facts_fallback(skill, mock_engine):
    mock_engine._call_with_fallback = AsyncMock(side_effect=Exception("err"))
    skill.set_cognition_engine(mock_engine)
    msgs = make_messages(3)
    facts = await skill.llm_extract_facts(msgs)
    assert isinstance(facts, list)

@pytest.mark.asyncio
async def test_async_auto_compress_below_budget(skill, mock_engine):
    skill.set_cognition_engine(mock_engine)
    msgs = make_messages(2)  # small, under budget
    result = await skill.async_auto_compress_if_needed(msgs)
    assert result["compressed"] is False
    assert result["llm_powered"] is False

@pytest.mark.asyncio
async def test_async_auto_compress_uses_llm(skill, mock_engine):
    skill.set_cognition_engine(mock_engine)
    # Create enough messages to exceed the default 8000 token budget
    msgs = make_messages(100, long=True)
    result = await skill.async_auto_compress_if_needed(msgs)
    assert result["compressed"] is True
    assert result["llm_powered"] is True
    assert result["tokens_saved"] > 0
    assert len(result["messages"]) < len(msgs)

@pytest.mark.asyncio
async def test_async_auto_compress_regex_fallback(skill):
    # No engine set - should use regex
    msgs = make_messages(100, long=True)
    result = await skill.async_auto_compress_if_needed(msgs)
    assert result["compressed"] is True
    assert result["llm_powered"] is False

@pytest.mark.asyncio
async def test_llm_compress_action(skill, mock_engine):
    skill.set_cognition_engine(mock_engine)
    msgs = make_messages(100, long=True)
    result = await skill.execute("llm_compress", {"messages": msgs})
    assert result.success is True
    assert "LLM" in result.message or "compress" in result.message.lower()

@pytest.mark.asyncio
async def test_llm_compress_action_no_engine(skill):
    result = await skill.execute("llm_compress", {"messages": make_messages(5)})
    assert result.success is False
    assert "cognition engine" in result.message.lower()

@pytest.mark.asyncio
async def test_stats_show_llm_info(skill, mock_engine):
    skill.set_cognition_engine(mock_engine)
    result = await skill.execute("stats")
    assert result.success is True
    assert "llm_available" in result.data
