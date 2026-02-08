#!/usr/bin/env python3
"""Tests for ErrorRecoverySkill."""

import json
import pytest
from pathlib import Path
from unittest.mock import patch
from singularity.skills.error_recovery import (
    ErrorRecoverySkill,
    ErrorCategory,
    ERROR_FILE,
)


@pytest.fixture
def skill(tmp_path):
    """Create an ErrorRecoverySkill with temp storage."""
    test_file = tmp_path / "error_recovery.json"
    with patch("singularity.skills.error_recovery.ERROR_FILE", test_file):
        s = ErrorRecoverySkill()
        yield s


@pytest.mark.asyncio
async def test_record_error(skill):
    result = await skill.execute("record", {
        "skill_id": "github",
        "action": "create_pr",
        "error_message": "Connection timed out after 30s",
        "error_type": "TimeoutError",
    })
    assert result.success
    assert "transient" in result.message
    assert result.data["classification"]["category"] == ErrorCategory.TRANSIENT
    assert result.data["classification"]["retryable"] is True


@pytest.mark.asyncio
async def test_classify_permission_error(skill):
    result = await skill.execute("classify", {
        "error_message": "401 Unauthorized: invalid API key",
        "error_type": "AuthError",
    })
    assert result.success
    assert result.data["classification"]["category"] == ErrorCategory.PERMISSION


@pytest.mark.asyncio
async def test_classify_dependency_error(skill):
    result = await skill.execute("classify", {
        "error_message": "No module named 'nonexistent_pkg'",
        "error_type": "ModuleNotFoundError",
    })
    assert result.success
    assert result.data["classification"]["category"] == ErrorCategory.DEPENDENCY


@pytest.mark.asyncio
async def test_classify_unknown_error(skill):
    result = await skill.execute("classify", {
        "error_message": "Something very unusual happened xyz",
    })
    assert result.success
    assert result.data["classification"]["category"] == ErrorCategory.UNKNOWN


@pytest.mark.asyncio
async def test_suggest_recovery_default_strategies(skill):
    result = await skill.execute("suggest_recovery", {
        "error_message": "Connection timed out",
    })
    assert result.success
    assert len(result.data["strategies"]) > 0
    assert result.data["has_learned_strategies"] is False
    assert result.data["recommended"] is not None


@pytest.mark.asyncio
async def test_suggest_recovery_with_learned_strategies(skill):
    # Record an error and a successful recovery
    await skill.execute("record", {
        "skill_id": "github",
        "action": "push",
        "error_message": "Connection reset by peer",
    })
    await skill.execute("record_recovery", {
        "error_message": "Connection reset by peer",
        "strategy_used": "retry_with_backoff",
        "success": True,
        "skill_id": "github",
        "notes": "Worked after 2 retries",
    })

    # Now suggest recovery for a similar error
    result = await skill.execute("suggest_recovery", {
        "error_message": "Connection reset by peer",
    })
    assert result.success
    assert result.data["has_learned_strategies"] is True
    learned = [s for s in result.data["strategies"] if s["source"] == "knowledge_base"]
    assert len(learned) >= 1


@pytest.mark.asyncio
async def test_record_recovery_updates_knowledge_base(skill):
    result = await skill.execute("record_recovery", {
        "error_message": "Rate limit exceeded",
        "strategy_used": "wait_and_retry",
        "success": True,
        "notes": "Waited 60 seconds",
    })
    assert result.success
    assert result.data["knowledge_base_updated"] is True

    # Query knowledge base
    kb = await skill.execute("knowledge", {})
    assert kb.success
    assert kb.data["total"] >= 1


@pytest.mark.asyncio
async def test_record_recovery_failed_no_kb_update(skill):
    result = await skill.execute("record_recovery", {
        "error_message": "Some error",
        "strategy_used": "retry_once",
        "success": False,
    })
    assert result.success
    assert result.data["knowledge_base_updated"] is False


@pytest.mark.asyncio
async def test_patterns_analysis(skill):
    # Record several errors
    for msg in ["Timeout error", "Timeout again", "Permission denied", "KeyError: 'foo'"]:
        await skill.execute("record", {
            "skill_id": "test",
            "action": "run",
            "error_message": msg,
        })

    result = await skill.execute("patterns", {"timeframe_hours": 24})
    assert result.success
    assert result.data["total_errors"] == 4
    assert len(result.data["by_category"]) > 0


@pytest.mark.asyncio
async def test_reset_preserves_knowledge(skill):
    # Build some knowledge
    await skill.execute("record", {
        "skill_id": "test", "action": "a", "error_message": "err1",
    })
    await skill.execute("record_recovery", {
        "error_message": "err1",
        "strategy_used": "fix_it",
        "success": True,
    })

    # Reset
    result = await skill.execute("reset", {})
    assert result.success
    assert result.data["knowledge_base_preserved"] >= 1

    # Knowledge should still be there
    kb = await skill.execute("knowledge", {})
    assert kb.data["total"] >= 1

    # But errors should be gone
    patterns = await skill.execute("patterns", {})
    assert patterns.data["total_errors"] == 0


@pytest.mark.asyncio
async def test_knowledge_query_with_filter(skill):
    await skill.execute("record_recovery", {
        "error_message": "API rate limit exceeded",
        "strategy_used": "wait_and_retry",
        "success": True,
    })
    await skill.execute("record_recovery", {
        "error_message": "Disk full",
        "strategy_used": "cleanup",
        "success": True,
    })

    result = await skill.execute("knowledge", {"query": "rate limit"})
    assert result.success
    assert result.data["total"] >= 1


@pytest.mark.asyncio
async def test_unknown_action(skill):
    result = await skill.execute("nonexistent", {})
    assert not result.success
    assert "Unknown action" in result.message


@pytest.mark.asyncio
async def test_missing_required_params(skill):
    result = await skill.execute("record", {"skill_id": "x"})
    assert not result.success

    result = await skill.execute("classify", {})
    assert not result.success
