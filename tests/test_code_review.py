#!/usr/bin/env python3
"""Tests for CodeReviewSkill."""

import pytest
import json
from pathlib import Path
from singularity.skills.code_review import CodeReviewSkill, REVIEW_LOG


@pytest.fixture
def skill(tmp_path, monkeypatch):
    monkeypatch.setattr("singularity.skills.code_review.REVIEW_LOG", tmp_path / "reviews.json")
    s = CodeReviewSkill()
    s._ensure_data()
    return s


@pytest.mark.asyncio
async def test_analyze_clean_code(skill):
    code = '''
def add(a, b):
    """Add two numbers."""
    return a + b
'''
    result = await skill.execute("analyze", {"code": code})
    assert result.success
    assert result.data["total_issues"] == 0


@pytest.mark.asyncio
async def test_analyze_finds_eval(skill):
    code = 'result = eval(user_input)\n'
    result = await skill.execute("analyze", {"code": code})
    assert result.success
    assert result.data["critical_count"] >= 1
    assert any("eval" in i["message"].lower() for i in result.data["issues"])


@pytest.mark.asyncio
async def test_analyze_finds_bare_except(skill):
    code = '''
try:
    x = 1
except:
    pass
'''
    result = await skill.execute("analyze", {"code": code})
    assert result.success
    assert result.data["total_issues"] >= 1


@pytest.mark.asyncio
async def test_analyze_syntax_error(skill):
    code = 'def broken(\n'
    result = await skill.execute("analyze", {"code": code})
    assert result.success
    assert result.data["critical_count"] >= 1
    assert any("syntax" in i["message"].lower() for i in result.data["issues"])


@pytest.mark.asyncio
async def test_analyze_empty_code(skill):
    result = await skill.execute("analyze", {"code": ""})
    assert not result.success


@pytest.mark.asyncio
async def test_analyze_mutable_default(skill):
    code = 'def foo(items=[]):\n    return items\n'
    result = await skill.execute("analyze", {"code": code})
    assert result.success
    assert any("mutable" in i["message"].lower() for i in result.data["issues"])


@pytest.mark.asyncio
async def test_review_diff_clean(skill):
    diff = """+++ b/main.py
+def greet(name):
+    return f"Hello, {name}"
"""
    result = await skill.execute("review_diff", {"diff": diff})
    assert result.success
    assert result.data["lines_added"] == 2


@pytest.mark.asyncio
async def test_review_diff_security_issue(skill):
    diff = """+++ b/main.py
+result = eval(user_input)
"""
    result = await skill.execute("review_diff", {"diff": diff})
    assert result.success
    assert result.data["total_findings"] >= 1


@pytest.mark.asyncio
async def test_review_diff_empty(skill):
    result = await skill.execute("review_diff", {"diff": ""})
    assert not result.success


@pytest.mark.asyncio
async def test_security_scan_clean(skill):
    code = '''
import hashlib
def hash_password(pw):
    return hashlib.sha256(pw.encode()).hexdigest()
'''
    result = await skill.execute("security_scan", {"code": code})
    assert result.success
    assert result.data["critical_count"] == 0


@pytest.mark.asyncio
async def test_security_scan_finds_issues(skill):
    code = '''
import pickle, os
data = pickle.load(open("data.pkl", "rb"))
os.system("rm -rf /")
password = "hunter2"
'''
    result = await skill.execute("security_scan", {"code": code})
    assert result.success
    assert result.data["critical_count"] >= 2
    assert len(result.data["by_category"]) >= 2


@pytest.mark.asyncio
async def test_score_good_code(skill):
    code = '''
def factorial(n):
    """Compute factorial of n."""
    if n <= 1:
        return 1
    return n * factorial(n - 1)
'''
    result = await skill.execute("score", {"code": code})
    assert result.success
    assert result.data["score"] >= 70
    assert "breakdown" in result.data


@pytest.mark.asyncio
async def test_score_bad_code(skill):
    code = 'eval(input())\nexec(input())\npassword = "secret123"\n'
    result = await skill.execute("score", {"code": code})
    assert result.success
    assert result.data["score"] < 60


@pytest.mark.asyncio
async def test_history_empty(skill):
    result = await skill.execute("history", {})
    assert result.success
    assert result.data["reviews"] == []


@pytest.mark.asyncio
async def test_history_after_reviews(skill):
    await skill.execute("analyze", {"code": "x = 1\n"})
    await skill.execute("score", {"code": "x = 1\n"})
    result = await skill.execute("history", {"limit": 5})
    assert result.success
    assert len(result.data["reviews"]) == 2
    assert result.data["trends"]["total_reviews"] == 2


@pytest.mark.asyncio
async def test_unknown_action(skill):
    result = await skill.execute("unknown_action", {})
    assert not result.success


@pytest.mark.asyncio
async def test_metrics_computation(skill):
    code = '''
class Calculator:
    """A calculator class."""
    def add(self, a, b):
        """Add two numbers."""
        return a + b
    def subtract(self, a, b):
        return a - b
'''
    result = await skill.execute("analyze", {"code": code})
    m = result.data["metrics"]
    assert m["function_count"] == 2
    assert m["class_count"] == 1
    assert m["docstring_count"] >= 1


@pytest.mark.asyncio
async def test_manifest(skill):
    m = skill.manifest()
    assert m.skill_id == "code_review"
    assert m.category == "dev"
    assert len(m.actions) == 5
    assert m.required_credentials == []
