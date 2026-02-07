"""Tests for WorkingMemorySkill."""
import pytest
from singularity.skills.working_memory import WorkingMemorySkill


@pytest.fixture
def skill():
    return WorkingMemorySkill()


@pytest.mark.asyncio
async def test_note_and_recall(skill):
    r = await skill.execute("note", {"key": "plan", "value": "Build API first"})
    assert r.success
    r = await skill.execute("recall", {"key": "plan"})
    assert r.success
    assert r.data["value"] == "Build API first"


@pytest.mark.asyncio
async def test_note_with_tags(skill):
    r = await skill.execute("note", {"key": "k1", "value": "v1", "tags": ["urgent", "dev"]})
    assert r.success
    r = await skill.execute("search", {"tag": "urgent"})
    assert r.success
    assert r.data["count"] == 1
    assert r.data["results"][0]["key"] == "k1"


@pytest.mark.asyncio
async def test_recall_not_found_with_suggestions(skill):
    await skill.execute("note", {"key": "my_plan", "value": "x"})
    r = await skill.execute("recall", {"key": "plan"})
    assert not r.success
    assert "my_plan" in r.data.get("suggestions", [])


@pytest.mark.asyncio
async def test_forget(skill):
    await skill.execute("note", {"key": "tmp", "value": "data"})
    r = await skill.execute("forget", {"key": "tmp"})
    assert r.success
    r = await skill.execute("recall", {"key": "tmp"})
    assert not r.success


@pytest.mark.asyncio
async def test_set_goal_and_complete_sub_goals(skill):
    r = await skill.execute("set_goal", {
        "goal": "Launch MVP",
        "sub_goals": ["Design API", "Write code", "Deploy"]
    })
    assert r.success
    assert r.data["goal"] == "Launch MVP"
    assert len(r.data["sub_goals"]) == 3

    r = await skill.execute("complete_sub_goal", {"index": 0})
    assert r.success
    assert r.data["progress"] == "1/3"
    assert not r.data["all_complete"]

    r = await skill.execute("complete_sub_goal", {"index": 1})
    r = await skill.execute("complete_sub_goal", {"index": 2})
    assert r.data["all_complete"]


@pytest.mark.asyncio
async def test_context_stack(skill):
    await skill.execute("set_goal", {"goal": "Main task"})
    r = await skill.execute("push_context", {"label": "subtask-1", "data": {"file": "a.py"}})
    assert r.success
    assert r.data["depth"] == 1

    await skill.execute("set_goal", {"goal": "Fix bug in a.py"})
    r = await skill.execute("pop_context", {})
    assert r.success
    assert r.data["restored_goal"] == "Main task"
    assert r.data["label"] == "subtask-1"
    assert r.data["data"]["file"] == "a.py"


@pytest.mark.asyncio
async def test_search_by_keyword(skill):
    await skill.execute("note", {"key": "api_design", "value": "REST endpoints"})
    await skill.execute("note", {"key": "db_schema", "value": "PostgreSQL tables"})
    r = await skill.execute("search", {"query": "rest"})
    assert r.success
    assert r.data["count"] == 1


@pytest.mark.asyncio
async def test_summary(skill):
    await skill.execute("set_goal", {"goal": "Test goal", "sub_goals": ["step1"]})
    await skill.execute("note", {"key": "n1", "value": "v1"})
    r = await skill.execute("summary", {})
    assert r.success
    assert "Test goal" in r.message
    assert r.data["note_count"] == 1


@pytest.mark.asyncio
async def test_clear(skill):
    await skill.execute("note", {"key": "k", "value": "v"})
    await skill.execute("set_goal", {"goal": "g"})
    r = await skill.execute("clear", {"confirm": False})
    assert not r.success  # Must confirm
    r = await skill.execute("clear", {"confirm": True})
    assert r.success
    assert not skill.has_content()


@pytest.mark.asyncio
async def test_context_summary_for_llm(skill):
    assert not skill.has_content()
    assert skill.get_context_summary() == "Working memory is empty."

    await skill.execute("set_goal", {"goal": "Build API", "sub_goals": ["Design", "Code"]})
    await skill.execute("note", {"key": "tech", "value": "FastAPI + SQLite", "tags": ["stack"]})
    summary = skill.get_context_summary()
    assert "CURRENT GOAL: Build API" in summary
    assert "WORKING NOTES" in summary
    assert "tech" in summary


@pytest.mark.asyncio
async def test_max_notes_eviction(skill):
    skill.MAX_NOTES = 3
    await skill.execute("note", {"key": "a", "value": "1"})
    await skill.execute("note", {"key": "b", "value": "2"})
    await skill.execute("note", {"key": "c", "value": "3"})
    await skill.execute("note", {"key": "d", "value": "4"})
    assert len(skill._notes) == 3
    assert "a" not in skill._notes  # oldest evicted
    assert "d" in skill._notes


@pytest.mark.asyncio
async def test_unknown_action(skill):
    r = await skill.execute("nonexistent", {})
    assert not r.success


@pytest.mark.asyncio
async def test_empty_context_stack_pop(skill):
    r = await skill.execute("pop_context", {})
    assert not r.success
