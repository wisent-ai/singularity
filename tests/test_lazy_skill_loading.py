"""Tests for lazy skill loading in autonomous_agent.py and skills/__init__.py."""
import importlib
import sys
import pytest


def test_skills_init_imports_base_eagerly():
    """Base classes should always be available."""
    from singularity.skills import Skill, SkillRegistry, SkillManifest
    assert Skill is not None
    assert SkillRegistry is not None


def test_skills_init_lazy_getattr():
    """__getattr__ should lazily load skill classes."""
    from singularity.skills import FilesystemSkill
    assert FilesystemSkill is not None
    assert hasattr(FilesystemSkill, 'execute')


def test_skills_init_unknown_attr_raises():
    """Accessing a non-existent attribute should raise AttributeError."""
    import singularity.skills as skills_mod
    with pytest.raises(AttributeError):
        _ = skills_mod.NoSuchSkill


def test_skill_modules_registry_exists():
    """SKILL_MODULES registry should be importable."""
    from singularity.autonomous_agent import SKILL_MODULES
    assert isinstance(SKILL_MODULES, list)
    assert len(SKILL_MODULES) >= 10  # at least 10 skills registered


def test_skill_modules_entries_are_tuples():
    """Each entry should be (module_path, class_name)."""
    from singularity.autonomous_agent import SKILL_MODULES
    for entry in SKILL_MODULES:
        assert isinstance(entry, tuple)
        assert len(entry) == 2
        module_path, class_name = entry
        assert module_path.startswith("singularity.skills.")
        assert class_name.endswith("Skill")


def test_load_skill_class_valid():
    """_load_skill_class should successfully load a skill with no deps."""
    from singularity.autonomous_agent import AutonomousAgent
    agent = AutonomousAgent.__new__(AutonomousAgent)
    # Need _log for _load_skill_class
    agent._log = lambda tag, msg: None
    cls = agent._load_skill_class("singularity.skills.filesystem", "FilesystemSkill")
    assert cls is not None
    assert cls.__name__ == "FilesystemSkill"


def test_load_skill_class_invalid_module():
    """_load_skill_class should return None for non-existent modules."""
    from singularity.autonomous_agent import AutonomousAgent
    agent = AutonomousAgent.__new__(AutonomousAgent)
    agent._log = lambda tag, msg: None
    cls = agent._load_skill_class("singularity.skills.nonexistent", "FakeSkill")
    assert cls is None


def test_load_skill_class_invalid_class():
    """_load_skill_class should return None for non-existent class in valid module."""
    from singularity.autonomous_agent import AutonomousAgent
    agent = AutonomousAgent.__new__(AutonomousAgent)
    agent._log = lambda tag, msg: None
    cls = agent._load_skill_class("singularity.skills.filesystem", "NonexistentSkill")
    assert cls is None
