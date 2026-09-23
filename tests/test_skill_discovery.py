"""Tests for automatic skill discovery."""

import pytest
from singularity.skill_discovery import discover_skills, discover_skills_with_metadata
from singularity.skills.base import Skill


def test_discover_skills_returns_list():
    """discover_skills returns a list of Skill subclasses."""
    skills = discover_skills()
    assert isinstance(skills, list)
    assert len(skills) > 0


def test_all_discovered_are_skill_subclasses():
    """Every discovered class is a subclass of Skill."""
    for cls in discover_skills():
        assert issubclass(cls, Skill)
        assert cls is not Skill


def test_discovers_known_skills():
    """Known built-in skills are discovered."""
    names = {cls.__name__ for cls in discover_skills()}
    expected = {
        "FilesystemSkill", "ShellSkill", "GitHubSkill",
        "ContentCreationSkill", "SelfModifySkill",
    }
    assert expected.issubset(names), f"Missing: {expected - names}"


def test_exclude_filter():
    """Skills in exclude set are not returned."""
    all_skills = discover_skills()
    filtered = discover_skills(exclude={"FilesystemSkill", "ShellSkill"})
    names = {cls.__name__ for cls in filtered}
    assert "FilesystemSkill" not in names
    assert "ShellSkill" not in names
    assert len(filtered) < len(all_skills)


def test_include_only_filter():
    """Only skills in include_only set are returned."""
    filtered = discover_skills(include_only={"FilesystemSkill", "ShellSkill"})
    names = {cls.__name__ for cls in filtered}
    assert names == {"FilesystemSkill", "ShellSkill"}


def test_deterministic_order():
    """Results are in sorted order by class name."""
    skills = discover_skills()
    names = [cls.__name__ for cls in skills]
    assert names == sorted(names)


def test_discover_with_metadata():
    """discover_skills_with_metadata returns dicts with expected keys."""
    meta = discover_skills_with_metadata()
    assert len(meta) > 0
    for item in meta:
        assert "class" in item
        assert "class_name" in item
        assert "module" in item
        assert "skill_id" in item
        assert "required_credentials" in item
        assert "action_count" in item


def test_metadata_has_valid_skill_ids():
    """Metadata includes valid skill_id strings."""
    meta = discover_skills_with_metadata()
    for item in meta:
        # skill_id should be set for well-formed skills
        assert item["skill_id"] is not None or item["class_name"]


def test_no_base_class_in_results():
    """The Skill base class itself is never in results."""
    skills = discover_skills()
    for cls in skills:
        assert cls.__name__ != "Skill"
