"""
Automatic Skill Discovery

Scans the skills package to find all Skill subclasses without requiring
hardcoded imports. This replaces the manual import list in autonomous_agent.py.

Usage:
    from singularity.skill_discovery import discover_skills
    skill_classes = discover_skills()
    # Returns list of Skill subclasses found in singularity/skills/
"""

import importlib
import inspect
import pkgutil
from pathlib import Path
from typing import Dict, List, Optional, Set, Type

from .skills.base import Skill


def discover_skills(
    package_path: Optional[str] = None,
    exclude: Optional[Set[str]] = None,
    include_only: Optional[Set[str]] = None,
) -> List[Type[Skill]]:
    """
    Discover all Skill subclasses in the skills package.

    Scans all .py files in the skills directory, imports them, and finds
    classes that inherit from Skill.

    Args:
        package_path: Path to skills package directory. Defaults to
                      singularity/skills/.
        exclude: Set of skill class names to exclude (e.g. {"BrowserSkill"}).
        include_only: If set, only include these skill class names.

    Returns:
        List of Skill subclass types, deduplicated and sorted by name.
    """
    if package_path is None:
        package_path = str(Path(__file__).parent / "skills")

    exclude = exclude or set()
    discovered: Dict[str, Type[Skill]] = {}

    skills_package = importlib.import_module("singularity.skills")
    package_dir = Path(package_path)

    for module_info in pkgutil.iter_modules([str(package_dir)]):
        module_name = module_info.name

        # Skip base, __init__, and private modules
        if module_name.startswith("_") or module_name == "base":
            continue

        try:
            module = importlib.import_module(f"singularity.skills.{module_name}")
        except Exception:
            continue

        for name, obj in inspect.getmembers(module, inspect.isclass):
            # Must be a Skill subclass but not Skill itself
            if not issubclass(obj, Skill) or obj is Skill:
                continue

            # Must be defined in this module (not re-imported)
            if obj.__module__ != module.__name__:
                continue

            # Apply filters
            if name in exclude:
                continue
            if include_only is not None and name not in include_only:
                continue

            discovered[name] = obj

    # Sort by class name for deterministic ordering
    return [discovered[name] for name in sorted(discovered)]


def discover_skills_with_metadata(
    package_path: Optional[str] = None,
    exclude: Optional[Set[str]] = None,
) -> List[Dict]:
    """
    Discover skills and return metadata about each.

    Returns:
        List of dicts with 'class', 'name', 'module', 'skill_id',
        'required_credentials' for each discovered skill.
    """
    skills = discover_skills(package_path=package_path, exclude=exclude)
    result = []

    for skill_class in skills:
        info = {
            "class": skill_class,
            "class_name": skill_class.__name__,
            "module": skill_class.__module__,
        }

        # Try to get manifest info without credentials
        try:
            instance = skill_class(credentials={})
            manifest = instance.manifest
            info["skill_id"] = manifest.skill_id
            info["skill_name"] = manifest.name
            info["category"] = manifest.category
            info["required_credentials"] = manifest.required_credentials
            info["action_count"] = len(manifest.actions)
        except Exception:
            info["skill_id"] = None
            info["skill_name"] = skill_class.__name__
            info["category"] = "unknown"
            info["required_credentials"] = []
            info["action_count"] = 0

        result.append(info)

    return result
