#!/usr/bin/env python3
"""
Skill Discovery & Plugin Loading

Automatically discovers and loads Skill subclasses from directories.
Enables dynamic skill loading without hardcoded imports, supporting:
- Auto-discovery of built-in skills
- Loading plugins from external directories
- Hot-reloading skills at runtime
- Dependency-aware loading order

This is foundational infrastructure for self-improvement (agent can extend
itself), replication (replicas can load different skill sets), and goal
setting (agent can assess and load skills it needs).
"""

import importlib
import importlib.util
import inspect
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Type

from .skills.base import Skill, SkillRegistry


class SkillLoadError:
    """Record of a failed skill load attempt."""
    
    def __init__(self, path: str, error: str, skill_name: str = ""):
        self.path = path
        self.error = error
        self.skill_name = skill_name
    
    def to_dict(self) -> Dict:
        return {
            "path": self.path,
            "error": self.error,
            "skill_name": self.skill_name,
        }


class DiscoveredSkill:
    """A skill class discovered from a module."""
    
    def __init__(self, skill_class: Type[Skill], module_path: str, source: str = "builtin"):
        self.skill_class = skill_class
        self.module_path = module_path
        self.source = source  # "builtin", "plugin", "dynamic"
        
        # Extract metadata by instantiating with empty credentials
        try:
            instance = skill_class({})
            manifest = instance.manifest
            self.skill_id = manifest.skill_id
            self.name = manifest.name
            self.category = manifest.category
            self.description = manifest.description
            self.required_credentials = manifest.required_credentials
        except Exception:
            self.skill_id = skill_class.__name__.lower()
            self.name = skill_class.__name__
            self.category = "unknown"
            self.description = ""
            self.required_credentials = []
    
    def to_dict(self) -> Dict:
        return {
            "skill_id": self.skill_id,
            "name": self.name,
            "class": self.skill_class.__name__,
            "module": self.module_path,
            "source": self.source,
            "category": self.category,
            "description": self.description,
            "required_credentials": self.required_credentials,
        }


class SkillLoader:
    """
    Discovers and loads Skill subclasses from Python modules.
    
    Supports:
    - Scanning directories for skill modules
    - Loading individual modules by path
    - Tracking loaded vs available skills
    - Error reporting for failed loads
    - Filtering by credential availability
    
    Usage:
        loader = SkillLoader()
        
        # Discover built-in skills
        skills = loader.discover_directory("singularity/skills")
        
        # Discover plugins
        plugins = loader.discover_directory("/path/to/plugins", source="plugin")
        
        # Load all into a registry
        loaded, errors = loader.load_into_registry(registry, credentials)
    """
    
    # Files to skip when scanning directories
    SKIP_FILES = {"__init__.py", "__pycache__", "base.py"}
    
    def __init__(self):
        self.discovered: Dict[str, DiscoveredSkill] = {}  # skill_id -> DiscoveredSkill
        self.errors: List[SkillLoadError] = []
        self._loaded_modules: Dict[str, object] = {}  # module_name -> module
    
    def discover_directory(
        self,
        directory: str,
        source: str = "builtin",
        package: str = "",
    ) -> List[DiscoveredSkill]:
        """
        Scan a directory for Python files containing Skill subclasses.
        
        Args:
            directory: Path to scan for .py files
            source: Label for where these skills came from ("builtin", "plugin", etc.)
            package: Python package name for relative imports (e.g., "singularity.skills")
            
        Returns:
            List of discovered skills
        """
        dir_path = Path(directory)
        if not dir_path.is_dir():
            self.errors.append(SkillLoadError(
                path=str(dir_path),
                error=f"Directory not found: {dir_path}",
            ))
            return []
        
        discovered = []
        
        for py_file in sorted(dir_path.glob("*.py")):
            if py_file.name in self.SKIP_FILES:
                continue
            if py_file.name.startswith("_"):
                continue
                
            skills = self.discover_module(str(py_file), source=source, package=package)
            discovered.extend(skills)
        
        return discovered
    
    def discover_module(
        self,
        module_path: str,
        source: str = "plugin",
        package: str = "",
    ) -> List[DiscoveredSkill]:
        """
        Load a Python module and find all Skill subclasses in it.
        
        Args:
            module_path: Path to the .py file
            source: Label for where this skill came from
            package: Python package for the module
            
        Returns:
            List of discovered skills from this module
        """
        path = Path(module_path)
        if not path.exists():
            self.errors.append(SkillLoadError(
                path=str(path),
                error=f"File not found: {path}",
            ))
            return []
        
        if not path.suffix == ".py":
            return []
        
        # Determine module name
        module_name = path.stem
        if package:
            full_module_name = f"{package}.{module_name}"
        else:
            full_module_name = f"_plugin_{module_name}_{id(path)}"
        
        # Check if already loaded
        if full_module_name in self._loaded_modules:
            module = self._loaded_modules[full_module_name]
        else:
            try:
                module = self._load_module(str(path), full_module_name)
                self._loaded_modules[full_module_name] = module
            except Exception as e:
                self.errors.append(SkillLoadError(
                    path=str(path),
                    error=f"Import error: {e}",
                ))
                return []
        
        # Find all Skill subclasses
        discovered = []
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if (
                issubclass(obj, Skill)
                and obj is not Skill
                and obj.__module__ == module.__name__
            ):
                try:
                    skill_info = DiscoveredSkill(obj, str(path), source=source)
                    
                    # Avoid duplicates - keep the first one found
                    if skill_info.skill_id not in self.discovered:
                        self.discovered[skill_info.skill_id] = skill_info
                        discovered.append(skill_info)
                    
                except Exception as e:
                    self.errors.append(SkillLoadError(
                        path=str(path),
                        error=f"Failed to inspect {name}: {e}",
                        skill_name=name,
                    ))
        
        return discovered
    
    def _load_module(self, file_path: str, module_name: str) -> object:
        """Load a Python module from a file path."""
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot create module spec for {file_path}")
        
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    
    def load_into_registry(
        self,
        registry: SkillRegistry,
        credentials: Dict[str, str],
        skill_ids: Optional[Set[str]] = None,
        require_credentials: bool = True,
        wiring_hooks: Optional[Dict[str, callable]] = None,
    ) -> Tuple[List[str], List[SkillLoadError]]:
        """
        Load discovered skills into a SkillRegistry.
        
        Args:
            registry: The SkillRegistry to load skills into
            credentials: Credentials to pass to skills
            skill_ids: Optional set of skill IDs to load (None = load all)
            require_credentials: If True, skip skills missing required credentials
            wiring_hooks: Optional dict of {skill_id: callable(skill)} for post-install wiring
            
        Returns:
            Tuple of (loaded_skill_ids, errors)
        """
        registry.set_credentials(credentials)
        loaded = []
        errors = []
        
        for skill_id, discovered in self.discovered.items():
            if skill_ids and skill_id not in skill_ids:
                continue
            
            try:
                # Install the skill
                registry.install(discovered.skill_class)
                skill = registry.get(skill_id)
                
                if skill is None:
                    errors.append(SkillLoadError(
                        path=discovered.module_path,
                        error=f"Skill {skill_id} not found after install",
                        skill_name=discovered.name,
                    ))
                    continue
                
                # Check credentials if required
                if require_credentials and not skill.check_credentials():
                    registry.uninstall(skill_id)
                    continue
                
                # Apply wiring hooks
                if wiring_hooks and skill_id in wiring_hooks:
                    try:
                        wiring_hooks[skill_id](skill)
                    except Exception as e:
                        errors.append(SkillLoadError(
                            path=discovered.module_path,
                            error=f"Wiring hook failed for {skill_id}: {e}",
                            skill_name=discovered.name,
                        ))
                
                loaded.append(skill_id)
                
            except Exception as e:
                errors.append(SkillLoadError(
                    path=discovered.module_path,
                    error=f"Failed to load {skill_id}: {e}",
                    skill_name=discovered.name,
                ))
        
        self.errors.extend(errors)
        return loaded, errors
    
    def get_available(self, credentials: Optional[Dict[str, str]] = None) -> List[Dict]:
        """
        Get list of all discovered skills with availability info.
        
        Args:
            credentials: If provided, check which skills have required credentials
            
        Returns:
            List of skill info dicts with availability status
        """
        result = []
        for skill_id, discovered in self.discovered.items():
            info = discovered.to_dict()
            
            if credentials is not None:
                missing = [
                    cred for cred in discovered.required_credentials
                    if not credentials.get(cred)
                ]
                info["credentials_available"] = len(missing) == 0
                info["missing_credentials"] = missing
            
            result.append(info)
        
        return result
    
    def get_by_category(self) -> Dict[str, List[Dict]]:
        """Get discovered skills grouped by category."""
        categories: Dict[str, List[Dict]] = {}
        for skill_id, discovered in self.discovered.items():
            cat = discovered.category
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(discovered.to_dict())
        return categories
    
    def get_errors(self) -> List[Dict]:
        """Get all load errors as dicts."""
        return [e.to_dict() for e in self.errors]
    
    def clear(self):
        """Clear all discovered skills and errors."""
        self.discovered.clear()
        self.errors.clear()
        self._loaded_modules.clear()
    
    def summary(self) -> Dict:
        """Get a summary of discovery results."""
        by_source = {}
        for d in self.discovered.values():
            by_source[d.source] = by_source.get(d.source, 0) + 1
        
        by_category = {}
        for d in self.discovered.values():
            by_category[d.category] = by_category.get(d.category, 0) + 1
        
        return {
            "total_discovered": len(self.discovered),
            "by_source": by_source,
            "by_category": by_category,
            "errors": len(self.errors),
            "skill_ids": list(self.discovered.keys()),
        }


def discover_builtin_skills() -> SkillLoader:
    """
    Discover all built-in skills from the singularity/skills directory.
    
    Returns:
        SkillLoader with all built-in skills discovered
    """
    loader = SkillLoader()
    skills_dir = Path(__file__).parent / "skills"
    loader.discover_directory(
        str(skills_dir),
        source="builtin",
        package="singularity.skills",
    )
    return loader


def discover_plugins(plugin_dir: str) -> SkillLoader:
    """
    Discover plugin skills from an external directory.
    
    Args:
        plugin_dir: Path to directory containing plugin .py files
        
    Returns:
        SkillLoader with discovered plugins
    """
    loader = SkillLoader()
    loader.discover_directory(str(plugin_dir), source="plugin")
    return loader
