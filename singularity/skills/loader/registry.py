"""
Skill registry data structures, constants, and validation.
"""

import os
import shutil
import platform
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field


SKILL_DIRECTORIES = [
    Path.home() / ".claude" / "skills",
    Path.home() / ".codex" / "skills",
    Path.home() / ".openclaw" / "skills",
]

MCP_REGISTRY_URL = "https://registry.modelcontextprotocol.io/v0"

MARKETPLACES = {
    "anthropic": "https://raw.githubusercontent.com/anthropics/skills/main",
    "skillsmp": "https://api.skillsmp.com/v1",
    "openclaw": "https://raw.githubusercontent.com/VoltAgent/awesome-openclaw-skills/main",
}


@dataclass
class SkillMetadata:
    """Metadata for a skill without loading its code."""
    skill_id: str
    module: str
    class_name: str
    name: str
    version: str
    category: str
    description: str
    required_credentials: List[str]
    wiring: Optional[str] = None
    actions: List[Dict] = field(default_factory=list)
    install_cost: float = 0
    author: str = "system"
    source_type: str = "python"
    source_path: Optional[str] = None
    instructions: Optional[str] = None
    homepage: Optional[str] = None
    user_invocable: bool = True
    requires_bins: List[str] = field(default_factory=list)
    requires_env: List[str] = field(default_factory=list)
    os_platforms: List[str] = field(default_factory=list)


@dataclass
class MCPServerInfo:
    """Metadata for an MCP server from the registry."""
    name: str
    description: str
    repository: Optional[str] = None
    homepage: Optional[str] = None
    transport: str = "stdio"
    command: Optional[str] = None
    args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    categories: List[str] = field(default_factory=list)
    author: Optional[str] = None
    version: Optional[str] = None


@dataclass
class SkillMdFile:
    """Parsed SKILL.md file following Anthropic/OpenClaw standard."""
    name: str
    description: str
    instructions: str
    path: Path
    homepage: Optional[str] = None
    user_invocable: bool = True
    disable_model_invocation: bool = False
    command_dispatch: Optional[str] = None
    command_tool: Optional[str] = None
    command_arg_mode: str = "raw"
    requires_bins: List[str] = field(default_factory=list)
    requires_env: List[str] = field(default_factory=list)
    requires_config: List[str] = field(default_factory=list)
    os_platforms: List[str] = field(default_factory=list)


# Wiring functions for special skills that need agent context
WIRING_HOOKS = {
    "cognition_hooks": lambda skill, agent: skill.set_cognition_hooks(
        get_prompt=agent.cognition.get_system_prompt,
        set_prompt=agent.cognition.set_system_prompt,
        append_prompt=agent.cognition.append_to_prompt,
        get_available_models=agent.cognition.get_available_models,
        get_current_model=agent.cognition.get_current_model,
        switch_model=agent.cognition.switch_model,
        record_example=agent.cognition.record_training_example,
        get_examples=agent.cognition.get_training_examples,
        clear_examples=agent.cognition.clear_training_examples,
        export_training=agent.cognition.export_training_data,
        start_finetune=agent.cognition.start_finetune,
        check_finetune=agent.cognition.check_finetune_status,
        use_finetuned=agent.cognition.use_finetuned_model,
    ),
    "llm": lambda skill, agent: skill.set_llm(
        agent.cognition.llm, agent.cognition.llm_type, agent.cognition.llm_model
    ),
    "agent_info": lambda skill, agent: skill.set_agent_info(agent.name),
}


class ValidationMixin:
    """Mixin providing validation methods for PluginLoader."""

    def check_credentials(self, skill_id: str, credentials: Dict[str, str]) -> bool:
        metadata = self._registry.get(skill_id)
        if not metadata: return False
        return all(credentials.get(c) for c in metadata.required_credentials)

    def get_missing_credentials(self, skill_id: str, credentials: Dict[str, str]) -> List[str]:
        metadata = self._registry.get(skill_id)
        if not metadata: return []
        return [c for c in metadata.required_credentials if not credentials.get(c)]

    def check_skill_requirements(self, skill_id: str) -> Dict[str, List[str]]:
        metadata = self._registry.get(skill_id)
        if not metadata: return {}
        missing = {"bins": [], "env": [], "os": []}
        for b in metadata.requires_bins:
            if not shutil.which(b): missing["bins"].append(b)
        for e in metadata.requires_env:
            if not os.environ.get(e): missing["env"].append(e)
        if metadata.os_platforms:
            cur = {"darwin": "darwin", "linux": "linux", "windows": "win32"}.get(platform.system().lower(), "")
            if cur not in metadata.os_platforms: missing["os"] = metadata.os_platforms
        return {k: v for k, v in missing.items() if v}
