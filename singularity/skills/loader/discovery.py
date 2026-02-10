"""
Skill discovery - finding and scanning for skills (Python files and SKILL.md).
"""

import ast
import re
from pathlib import Path
from typing import Dict, List, Optional, Union

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

from .registry import SkillMetadata, SkillMdFile, SKILL_DIRECTORIES


class DiscoveryMixin:
    """Mixin providing skill discovery methods for PluginLoader."""

    def discover(self, directories: List[str]) -> None:
        """Scan directories for Python skill files and update registry."""
        for dir_path in directories:
            path = Path(dir_path)
            if not path.exists():
                continue
            for py_file in path.glob("*.py"):
                if py_file.name.startswith("_"):
                    continue
                try:
                    with open(py_file, 'r') as f:
                        tree = ast.parse(f.read())
                    for node in ast.walk(tree):
                        if isinstance(node, ast.ClassDef):
                            for base in node.bases:
                                base_name = ""
                                if isinstance(base, ast.Name): base_name = base.id
                                elif isinstance(base, ast.Attribute): base_name = base.attr
                                if base_name == "Skill":
                                    skill_id = self._class_to_skill_id(node.name)
                                    rel = py_file.relative_to(path.parent)
                                    module = str(rel).replace("/", ".").replace(".py", "")
                                    if skill_id not in self._registry:
                                        self._registry[skill_id] = SkillMetadata(
                                            skill_id=skill_id, module=module, class_name=node.name,
                                            name=node.name.replace("Skill", ""), version="1.0.0",
                                            category="discovered",
                                            description=f"Discovered from {py_file.name}",
                                            required_credentials=[])
                                    break
                except Exception:
                    continue

    def _class_to_skill_id(self, class_name: str) -> str:
        name = class_name[:-5] if class_name.endswith("Skill") else class_name
        name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()

    def discover_skill_md(self, directories: List[Union[str, Path]] = None) -> List[SkillMdFile]:
        """Discover SKILL.md files from standard directories."""
        dirs = [Path(d) for d in directories] + list(SKILL_DIRECTORIES) if directories else list(SKILL_DIRECTORIES)
        discovered, seen = [], set()
        for dir_path in dirs:
            if not dir_path.exists(): continue
            for skill_dir in dir_path.iterdir():
                if not skill_dir.is_dir(): continue
                skill_md_path = skill_dir / "SKILL.md"
                if not skill_md_path.exists(): continue
                try:
                    skill = self._parse_skill_md(skill_md_path)
                    if skill and skill.name not in seen:
                        discovered.append(skill)
                        seen.add(skill.name)
                        self._register_skill_md(skill)
                except Exception as e:
                    print(f"Warning: Failed to parse {skill_md_path}: {e}")
        return discovered

    def _parse_skill_md(self, path: Path) -> Optional[SkillMdFile]:
        content = path.read_text()
        frontmatter, instructions = {}, content
        if content.startswith("---") and HAS_YAML:
            parts = content.split("---", 2)
            if len(parts) >= 3:
                try:
                    frontmatter = yaml.safe_load(parts[1]) or {}
                    instructions = parts[2].strip()
                except Exception:
                    pass
        name = frontmatter.get("name", path.parent.name)
        if not name: return None
        meta = frontmatter.get("metadata", {})
        oc = meta.get("openclaw", {}) if isinstance(meta, dict) else {}
        req = oc.get("requires", {}) if isinstance(oc, dict) else {}
        return SkillMdFile(
            name=name, description=frontmatter.get("description", ""),
            instructions=instructions, path=path,
            homepage=frontmatter.get("homepage"),
            user_invocable=frontmatter.get("user-invocable", True),
            disable_model_invocation=frontmatter.get("disable-model-invocation", False),
            command_dispatch=frontmatter.get("command-dispatch"),
            command_tool=frontmatter.get("command-tool"),
            command_arg_mode=frontmatter.get("command-arg-mode", "raw"),
            requires_bins=req.get("bins", []) if isinstance(req, dict) else [],
            requires_env=req.get("env", []) if isinstance(req, dict) else [],
            requires_config=req.get("config", []) if isinstance(req, dict) else [],
            os_platforms=oc.get("os", []) if isinstance(oc, dict) else [])

    def _register_skill_md(self, skill: SkillMdFile) -> None:
        sid = skill.name.lower().replace(" ", "_").replace("-", "_")
        self._registry[sid] = SkillMetadata(
            skill_id=sid, module="", class_name="", name=skill.name,
            version="1.0.0", category="skill_md", description=skill.description,
            required_credentials=[], source_type="skill_md",
            source_path=str(skill.path), instructions=skill.instructions,
            homepage=skill.homepage, user_invocable=skill.user_invocable,
            requires_bins=skill.requires_bins, requires_env=skill.requires_env,
            os_platforms=skill.os_platforms)

    def get_skill_instructions(self, skill_id: str) -> Optional[str]:
        metadata = self._registry.get(skill_id)
        if metadata and metadata.source_type == "skill_md":
            return metadata.instructions
        return None
