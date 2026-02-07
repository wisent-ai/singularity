"""
WisentBot Skills - Modular capabilities for autonomous agents.

Skills provide specific capabilities that agents can use to interact
with the world. Each skill has a manifest describing its actions.

Skills are lazily imported to avoid crashes from missing optional
dependencies. Use ``from singularity.skills.base import Skill`` for
the base class, or import individual skills directly.
"""

from .base import Skill, SkillRegistry, SkillManifest, SkillAction, SkillResult

# Lazy imports: individual skills are imported on first access so that
# a missing optional dependency (e.g. seleniumbase, web3) doesn't
# prevent the entire package from loading.

_LAZY_IMPORTS = {
    "BrowserSkill": ".browser",
    "ContentCreationSkill": ".content",
    "EmailSkill": ".email",
    "FilesystemSkill": ".filesystem",
    "GitHubSkill": ".github",
    "MCPClientSkill": ".mcp_client",
    "NamecheapSkill": ".namecheap",
    "RequestSkill": ".request",
    "ShellSkill": ".shell",
    "TwitterSkill": ".twitter",
    "VercelSkill": ".vercel",
    "SelfModifySkill": ".self_modify",
    "SteeringSkill": ".steering",
    "MemorySkill": ".memory",
    "OrchestratorSkill": ".orchestrator",
    "CryptoSkill": ".crypto",
}


def __getattr__(name):
    """Lazily import skill classes on first access."""
    if name in _LAZY_IMPORTS:
        import importlib
        module = importlib.import_module(_LAZY_IMPORTS[name], package=__name__)
        cls = getattr(module, name)
        # Cache in module namespace for subsequent accesses
        globals()[name] = cls
        return cls
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Base
    "Skill",
    "SkillRegistry",
    "SkillManifest",
    "SkillAction",
    "SkillResult",
    # Skills (lazily loaded)
    "BrowserSkill",
    "ContentCreationSkill",
    "EmailSkill",
    "FilesystemSkill",
    "GitHubSkill",
    "MCPClientSkill",
    "NamecheapSkill",
    "RequestSkill",
    "ShellSkill",
    "TwitterSkill",
    "VercelSkill",
    "SelfModifySkill",
    "SteeringSkill",
    "MemorySkill",
    "OrchestratorSkill",
    "CryptoSkill",
]
