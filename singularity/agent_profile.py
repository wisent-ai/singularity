#!/usr/bin/env python3
"""
Agent Profile System - Reusable agent configurations for spawning and deployment.

Profiles define complete agent configurations including identity, model,
skills, budget, and behavior. They enable:
- Template-based agent spawning via OrchestratorSkill
- Reproducible agent configurations
- Profile inheritance and overrides
- Loading profiles from JSON files
"""

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any, Union


@dataclass
class AgentProfile:
    """
    Complete configuration for an autonomous agent.

    Profiles can be:
    - Created programmatically
    - Loaded from JSON files
    - Registered in a ProfileRegistry for lookup by name
    - Used with AutonomousAgent.from_profile() to create agents
    """

    # Identity
    name: str = "Agent"
    ticker: str = "AGENT"
    agent_type: str = "general"
    specialty: str = ""

    # Economics
    starting_balance: float = 100.0
    instance_type: str = "local"
    cycle_interval_seconds: float = 5.0

    # LLM Configuration
    llm_provider: str = "anthropic"
    llm_model: str = "claude-sonnet-4-20250514"
    llm_base_url: str = "http://localhost:8000/v1"

    # Prompt
    system_prompt: Optional[str] = None
    system_prompt_file: Optional[str] = None

    # Skill configuration
    skills: Optional[List[str]] = None  # Whitelist: load only these skill IDs
    disabled_skills: Optional[List[str]] = None  # Blacklist: exclude these skill IDs

    # Environment variables to set when creating agent
    env_vars: Optional[Dict[str, str]] = None

    # Arbitrary metadata for custom use
    metadata: Optional[Dict[str, Any]] = None

    # Parent profile name (for inheritance)
    extends: Optional[str] = None

    @classmethod
    def from_dict(cls, data: dict) -> "AgentProfile":
        """Create profile from a dictionary, ignoring unknown keys."""
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered)

    @classmethod
    def from_file(cls, path: str) -> "AgentProfile":
        """Load profile from a JSON file."""
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Profile file not found: {path}")

        with open(file_path, "r") as f:
            data = json.load(f)

        return cls.from_dict(data)

    def to_dict(self) -> dict:
        """Export profile as a dictionary (excludes None values)."""
        result = {}
        for k, v in asdict(self).items():
            if v is not None:
                result[k] = v
        return result

    def save(self, path: str) -> None:
        """Save profile to a JSON file."""
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def merge(self, overrides: dict) -> "AgentProfile":
        """Create a new profile with overrides applied."""
        base = self.to_dict()
        # Deep merge metadata
        if "metadata" in overrides and "metadata" in base:
            merged_meta = {**base.get("metadata", {}), **overrides["metadata"]}
            overrides = {**overrides, "metadata": merged_meta}
        base.update(overrides)
        return AgentProfile.from_dict(base)

    def to_agent_kwargs(self) -> dict:
        """Convert profile to kwargs for AutonomousAgent constructor."""
        kwargs = {
            "name": self.name,
            "ticker": self.ticker,
            "agent_type": self.agent_type,
            "specialty": self.specialty,
            "starting_balance": self.starting_balance,
            "instance_type": self.instance_type,
            "cycle_interval_seconds": self.cycle_interval_seconds,
            "llm_provider": self.llm_provider,
            "llm_model": self.llm_model,
            "llm_base_url": self.llm_base_url,
        }
        if self.system_prompt is not None:
            kwargs["system_prompt"] = self.system_prompt
        if self.system_prompt_file is not None:
            kwargs["system_prompt_file"] = self.system_prompt_file

        # Set env vars if specified
        if self.env_vars:
            for key, value in self.env_vars.items():
                os.environ[key] = value

        return kwargs


class ProfileRegistry:
    """
    Registry of named agent profiles.

    Supports:
    - Registering profiles by name
    - Loading profiles from a directory
    - Profile inheritance (extends)
    - Listing available profiles
    """

    def __init__(self, profiles_dir: Optional[str] = None):
        self._profiles: Dict[str, AgentProfile] = {}
        self._load_builtins()
        if profiles_dir:
            self.load_directory(profiles_dir)

    def _load_builtins(self):
        """Load built-in agent profiles."""
        for name, profile in BUILTIN_PROFILES.items():
            self._profiles[name] = profile

    def register(self, name: str, profile: AgentProfile) -> None:
        """Register a profile by name."""
        self._profiles[name] = profile

    def get(self, name: str) -> Optional[AgentProfile]:
        """Get a profile by name, resolving inheritance."""
        profile = self._profiles.get(name)
        if profile is None:
            return None

        # Resolve inheritance chain
        if profile.extends:
            parent = self.get(profile.extends)
            if parent:
                # Child overrides parent
                child_overrides = {
                    k: v for k, v in profile.to_dict().items()
                    if k != "extends"
                }
                return parent.merge(child_overrides)

        return profile

    def list_profiles(self) -> List[str]:
        """List all registered profile names."""
        return sorted(self._profiles.keys())

    def load_directory(self, directory: str) -> int:
        """
        Load all .json profile files from a directory.

        Returns the number of profiles loaded.
        """
        dir_path = Path(directory)
        if not dir_path.is_dir():
            return 0

        count = 0
        for file_path in dir_path.glob("*.json"):
            try:
                profile = AgentProfile.from_file(str(file_path))
                name = file_path.stem  # filename without extension
                self._profiles[name] = profile
                count += 1
            except (json.JSONDecodeError, FileNotFoundError):
                continue

        return count

    def save_all(self, directory: str) -> int:
        """Save all profiles to a directory. Returns count saved."""
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)

        count = 0
        for name, profile in self._profiles.items():
            profile.save(str(dir_path / f"{name}.json"))
            count += 1

        return count

    def to_dict(self) -> Dict[str, dict]:
        """Export all profiles as a dictionary."""
        return {name: profile.to_dict() for name, profile in self._profiles.items()}


# ═══════════════════════════════════════════════════════════════════
# Built-in Profiles
# ═══════════════════════════════════════════════════════════════════

BUILTIN_PROFILES: Dict[str, AgentProfile] = {
    "coder": AgentProfile(
        name="Coder",
        ticker="CODE",
        agent_type="coder",
        specialty="Software development, debugging, code review, and system architecture",
        starting_balance=50.0,
        llm_model="claude-sonnet-4-20250514",
        system_prompt=(
            "You are an expert software developer. Focus on writing clean, "
            "well-tested code. Use filesystem and shell tools to read, write, "
            "and execute code. Prefer small, incremental changes with testing."
        ),
        disabled_skills=["twitter", "namecheap", "email", "browser", "crypto"],
        metadata={"category": "engineering", "risk_level": "low"},
    ),
    "researcher": AgentProfile(
        name="Researcher",
        ticker="RSCH",
        agent_type="researcher",
        specialty="Information gathering, analysis, and synthesis",
        starting_balance=30.0,
        llm_model="claude-sonnet-4-20250514",
        system_prompt=(
            "You are a thorough researcher. Use browser and request tools to "
            "gather information. Synthesize findings into clear, structured "
            "reports. Save important findings to files for persistence."
        ),
        disabled_skills=["namecheap", "crypto", "twitter"],
        metadata={"category": "research", "risk_level": "low"},
    ),
    "writer": AgentProfile(
        name="Writer",
        ticker="WRIT",
        agent_type="writer",
        specialty="Content creation, copywriting, and documentation",
        starting_balance=20.0,
        llm_model="claude-sonnet-4-20250514",
        system_prompt=(
            "You are a skilled writer. Create compelling content - articles, "
            "documentation, marketing copy, technical writing. Use content "
            "creation tools for generation and filesystem for saving work."
        ),
        disabled_skills=["namecheap", "crypto", "browser"],
        metadata={"category": "content", "risk_level": "low"},
    ),
    "devops": AgentProfile(
        name="DevOps",
        ticker="DEVOP",
        agent_type="devops",
        specialty="Infrastructure, deployment, monitoring, and automation",
        starting_balance=40.0,
        llm_model="claude-sonnet-4-20250514",
        system_prompt=(
            "You are a DevOps engineer. Manage infrastructure, deployments, "
            "and automation. Use shell, filesystem, and GitHub tools. Focus "
            "on reliability, monitoring, and CI/CD pipelines."
        ),
        disabled_skills=["twitter", "namecheap", "email", "crypto"],
        metadata={"category": "engineering", "risk_level": "medium"},
    ),
    "trader": AgentProfile(
        name="Trader",
        ticker="TRADE",
        agent_type="trader",
        specialty="Cryptocurrency trading and financial analysis",
        starting_balance=100.0,
        llm_model="claude-sonnet-4-20250514",
        system_prompt=(
            "You are a crypto trader. Analyze markets, execute trades, and "
            "manage risk. Use crypto tools for trading and request tools for "
            "market data. Never risk more than 10% of balance on one trade."
        ),
        disabled_skills=["namecheap", "browser", "twitter"],
        metadata={"category": "finance", "risk_level": "high"},
    ),
    "minimal": AgentProfile(
        name="Minimal",
        ticker="MIN",
        agent_type="general",
        specialty="Basic operations with minimal tools",
        starting_balance=10.0,
        llm_model="claude-sonnet-4-20250514",
        skills=["filesystem", "shell"],
        metadata={"category": "utility", "risk_level": "low"},
    ),
}


# Convenience: global default registry
_default_registry: Optional[ProfileRegistry] = None


def get_default_registry() -> ProfileRegistry:
    """Get or create the default global profile registry."""
    global _default_registry
    if _default_registry is None:
        _default_registry = ProfileRegistry()
    return _default_registry
