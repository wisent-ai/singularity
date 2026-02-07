"""Tests for CodeAnalysisSkill."""

import os
import pytest
import tempfile
from singularity.skills.code_analysis import CodeAnalysisSkill


@pytest.fixture
def skill():
    return CodeAnalysisSkill()


@pytest.fixture
def sample_file(tmp_path):
    code = '''
import os
import json
from pathlib import Path

def greet(name="World"):
    """Say hello."""
    print(f"Hello, {name}!")

class Calculator:
    def add(self, a, b):
        return a + b

    def divide(self, a, b):
        try:
            return a / b
        except:
            pass
'''
    f = tmp_path / "sample.py"
    f.write_text(code)
    return str(f)


@pytest.fixture
def bad_code_file(tmp_path):
    code = '''
import os
from sys import *
password = "supersecret123"

def process(data=[]):
    eval(data[0])
    exec(data[1])

def mega_func(a, b, c, d, e, f, g, h, i):
    # TODO: refactor this
    # FIXME: broken logic
    global x
    if a:
        if b:
            if c:
                for i in range(d):
                    while e:
                        pass
    return None
'''
    f = tmp_path / "bad.py"
    f.write_text(code)
    return str(f)


@pytest.mark.asyncio
async def test_analyze(skill, sample_file):
    result = await skill.execute("analyze", {"path": sample_file})
    assert result.success
    assert "Score:" in result.message
    assert result.data["quality_score"] > 0
    assert result.data["metrics"]["functions"] >= 1
    assert result.data["metrics"]["classes"] >= 1


@pytest.mark.asyncio
async def test_analyze_code_string(skill):
    result = await skill.execute("analyze_code", {"code": "def hello(): return 42"})
    assert result.success
    assert result.data["quality_score"] > 0


@pytest.mark.asyncio
async def test_metrics(skill, sample_file):
    result = await skill.execute("metrics", {"path": sample_file})
    assert result.success
    m = result.data["metrics"]
    assert m["total_lines"] > 0
    assert m["code_lines"] > 0
    assert m["functions"] >= 1


@pytest.mark.asyncio
async def test_issues_bare_except(skill, sample_file):
    result = await skill.execute("issues", {"path": sample_file})
    assert result.success
    codes = [i["code"] for i in result.data["issues"]]
    assert "W001" in codes  # bare except
    assert "W002" in codes  # except pass


@pytest.mark.asyncio
async def test_issues_bad_code(skill, bad_code_file):
    result = await skill.execute("issues", {"path": bad_code_file})
    assert result.success
    codes = [i["code"] for i in result.data["issues"]]
    assert "E001" in codes  # mutable default
    assert "W003" in codes  # star import
    assert "W004" in codes  # too many args
    assert "I001" in codes  # global


@pytest.mark.asyncio
async def test_security_scan(skill, bad_code_file):
    result = await skill.execute("security", {"path": bad_code_file})
    assert result.success
    assert len(result.data["security_issues"]) >= 2  # eval + exec + password


@pytest.mark.asyncio
async def test_dependencies(skill, sample_file):
    result = await skill.execute("dependencies", {"path": sample_file})
    assert result.success
    deps = result.data["dependencies"]
    assert "os" in deps["stdlib"]
    assert "json" in deps["stdlib"]


@pytest.mark.asyncio
async def test_complexity(skill, bad_code_file):
    result = await skill.execute("complexity", {"path": bad_code_file, "threshold": 1})
    assert result.success
    assert len(result.data["functions"]) >= 1


@pytest.mark.asyncio
async def test_scan_directory(skill, tmp_path):
    (tmp_path / "a.py").write_text("def f(): pass")
    (tmp_path / "b.py").write_text("x = 1\ny = 2")
    result = await skill.execute("scan_directory", {"path": str(tmp_path)})
    assert result.success
    assert result.data["file_count"] == 2


@pytest.mark.asyncio
async def test_file_not_found(skill):
    result = await skill.execute("analyze", {"path": "/nonexistent.py"})
    assert not result.success


@pytest.mark.asyncio
async def test_not_python_file(skill, tmp_path):
    f = tmp_path / "readme.txt"
    f.write_text("hello")
    result = await skill.execute("analyze", {"path": str(f)})
    assert not result.success


@pytest.mark.asyncio
async def test_syntax_error_file(skill, tmp_path):
    f = tmp_path / "broken.py"
    f.write_text("def f(\n")
    result = await skill.execute("analyze", {"path": str(f)})
    assert not result.success
    assert "Syntax error" in result.message


@pytest.mark.asyncio
async def test_manifest(skill):
    m = skill.manifest
    assert m.skill_id == "code_analysis"
    assert len(m.actions) == 8
    assert skill.check_credentials()


@pytest.mark.asyncio
async def test_unknown_action(skill):
    result = await skill.execute("nonexistent", {})
    assert not result.success
