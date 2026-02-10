"""
Main PluginLoader - lazy loading skill system with registry support.
"""

import json
import importlib
from pathlib import Path
from typing import Dict, List, Optional, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ..base import Skill

from .registry import SkillMetadata, WIRING_HOOKS
from .discovery import DiscoveryMixin
from .registry import ValidationMixin
from .marketplace import MarketplaceMixin


class PluginLoader(DiscoveryMixin, ValidationMixin, MarketplaceMixin):
    """
    Manages lazy loading of skills from a registry.

    Features: registry-based discovery, lazy loading, directory scanning,
    wiring hooks, MCP registry, marketplace installation.
    """

    def __init__(self, registry_path: Optional[str] = None):
        self._registry: Dict[str, SkillMetadata] = {}
        self._loaded_skills: Dict[str, 'Skill'] = {}
        self._default_registry_path = Path(__file__).parent.parent / "registry.json"
        if registry_path:
            self._load_registry(Path(registry_path))
        elif self._default_registry_path.exists():
            self._load_registry(self._default_registry_path)

    def _load_registry(self, path: Path) -> None:
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            for skill_id, sd in data.get("skills", {}).items():
                m = sd.get("manifest", {})
                self._registry[skill_id] = SkillMetadata(
                    skill_id=skill_id, module=sd.get("module", ""),
                    class_name=sd.get("class", ""), name=m.get("name", skill_id),
                    version=m.get("version", "1.0.0"), category=m.get("category", "general"),
                    description=m.get("description", ""),
                    required_credentials=m.get("required_credentials", []),
                    wiring=sd.get("wiring"), actions=m.get("actions", []),
                    install_cost=m.get("install_cost", 0), author=m.get("author", "system"))
        except Exception as e:
            print(f"Warning: Failed to load skill registry from {path}: {e}")

    def get_manifest(self, skill_id: str) -> Optional[SkillMetadata]:
        return self._registry.get(skill_id)

    def list_available(self, category: Optional[str] = None) -> List[SkillMetadata]:
        skills = list(self._registry.values())
        if category: skills = [s for s in skills if s.category == category]
        return sorted(skills, key=lambda s: s.skill_id)

    def load(self, skill_id: str, credentials: Dict[str, str] = None) -> Optional['Skill']:
        """Lazy load a skill by ID. Returns cached instance on subsequent calls."""
        if skill_id in self._loaded_skills:
            return self._loaded_skills[skill_id]
        metadata = self._registry.get(skill_id)
        if not metadata: return None
        try:
            module = importlib.import_module(metadata.module)
            skill_class = getattr(module, metadata.class_name)
            skill = skill_class(credentials=credentials or {})
            self._loaded_skills[skill_id] = skill
            return skill
        except Exception as e:
            print(f"Warning: Failed to load skill '{skill_id}': {e}")
            return None

    def load_with_wiring(self, skill_id: str, credentials: Dict[str, str], agent: Any) -> Optional['Skill']:
        """Load a skill and apply wiring hooks if needed."""
        skill = self.load(skill_id, credentials)
        if not skill: return None
        metadata = self._registry.get(skill_id)
        if metadata and metadata.wiring:
            wiring_fn = WIRING_HOOKS.get(metadata.wiring)
            if wiring_fn:
                try: wiring_fn(skill, agent)
                except Exception as e: print(f"Warning: Wiring failed for '{skill_id}': {e}")
        return skill

    def is_loaded(self, skill_id: str) -> bool: return skill_id in self._loaded_skills
    def unload(self, skill_id: str) -> bool:
        if skill_id in self._loaded_skills: del self._loaded_skills[skill_id]; return True
        return False
    def get_loaded(self, skill_id: str): return self._loaded_skills.get(skill_id)
    def list_loaded(self) -> List[str]: return list(self._loaded_skills.keys())

    def register(self, metadata: SkillMetadata):
        self._registry[metadata.skill_id] = metadata

    def save_registry(self, path: Optional[str] = None):
        out = Path(path) if path else self._default_registry_path
        data = {"version": "1.0", "skills": {}}
        for sid, m in self._registry.items():
            data["skills"][sid] = {
                "module": m.module, "class": m.class_name, "wiring": m.wiring,
                "manifest": {
                    "skill_id": m.skill_id, "name": m.name, "version": m.version,
                    "category": m.category, "description": m.description,
                    "required_credentials": m.required_credentials,
                    "actions": m.actions, "install_cost": m.install_cost, "author": m.author}}
        with open(out, 'w') as f: json.dump(data, f, indent=2)
