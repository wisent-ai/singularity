"""
Plugin Loader - lazy loading skill system with registry, SKILL.md, MCP, marketplaces.
"""

from .registry import SkillMetadata, MCPServerInfo, SkillMdFile, SKILL_DIRECTORIES, MCP_REGISTRY_URL, MARKETPLACES, WIRING_HOOKS
from .loader import PluginLoader
from .marketplace import MarketplaceMixin

__all__ = [
    'PluginLoader', 'MarketplaceMixin',
    'SkillMetadata', 'MCPServerInfo', 'SkillMdFile',
    'SKILL_DIRECTORIES', 'MCP_REGISTRY_URL', 'MARKETPLACES', 'WIRING_HOOKS',
]
