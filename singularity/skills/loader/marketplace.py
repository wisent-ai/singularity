"""
Marketplace and MCP registry integration mixin for PluginLoader.
"""

import asyncio
import aiohttp
from pathlib import Path
from typing import Dict, List, Optional

from .registry import SkillMetadata, MCPServerInfo, MCP_REGISTRY_URL, MARKETPLACES


class MarketplaceMixin:
    """Mixin providing MCP registry and marketplace methods for PluginLoader."""

    async def search_mcp_registry(self, query="", category="", limit=50) -> List[MCPServerInfo]:
        try:
            params = {"limit": limit}
            if query: params["q"] = query
            if category: params["category"] = category
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{MCP_REGISTRY_URL}/servers", params=params) as resp:
                    if resp.status != 200: return []
                    data = await resp.json()
                    items = data.get("servers", data if isinstance(data, list) else [])
                    return [self._parse_mcp_server(item) for item in items]
        except Exception as e:
            print(f"Warning: MCP registry query failed: {e}")
            return []

    @staticmethod
    def _parse_mcp_server(item: dict) -> MCPServerInfo:
        sd = item.get("server", item) if isinstance(item, dict) else item
        repo = sd.get("repository", {})
        repo_url = repo.get("url") if isinstance(repo, dict) else repo
        transport = "stdio"
        pkgs = sd.get("packages", [])
        if pkgs and isinstance(pkgs, list):
            ti = pkgs[0].get("transport", {})
            transport = ti.get("type", "stdio") if isinstance(ti, dict) else "stdio"
        return MCPServerInfo(
            name=sd.get("name", ""), description=sd.get("description", ""),
            repository=repo_url, homepage=sd.get("homepage"), transport=transport,
            command=sd.get("command"), args=sd.get("args", []), env=sd.get("env", {}),
            categories=sd.get("categories", []), author=sd.get("author"), version=sd.get("version"))

    def search_mcp_registry_sync(self, query="", limit=50) -> List[MCPServerInfo]:
        return _run_sync(self.search_mcp_registry(query, limit=limit))

    def register_mcp_server(self, server: MCPServerInfo) -> str:
        sid = f"mcp_{server.name.lower().replace(' ', '_').replace('-', '_')}"
        self._registry[sid] = SkillMetadata(
            skill_id=sid, module="singularity.skills.builtin.mcp_client",
            class_name="MCPClientSkill", name=f"MCP: {server.name}",
            version=server.version or "1.0.0", category="mcp",
            description=server.description, required_credentials=[],
            source_type="mcp", source_path=server.repository or server.homepage,
            author=server.author or "community")
        return sid

    async def install_from_marketplace(self, skill_name: str, marketplace="anthropic") -> bool:
        if marketplace not in MARKETPLACES:
            print(f"Unknown marketplace: {marketplace}")
            return False
        base_url = MARKETPLACES[marketplace]
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{base_url}/skills/{skill_name}/SKILL.md") as resp:
                    if resp.status != 200:
                        print(f"Skill not found: {skill_name} in {marketplace}")
                        return False
                    content = await resp.text()
            skills_dir = Path.home() / ".claude" / "skills" / skill_name
            skills_dir.mkdir(parents=True, exist_ok=True)
            (skills_dir / "SKILL.md").write_text(content)
            skill = self._parse_skill_md(skills_dir / "SKILL.md")
            if skill:
                self._register_skill_md(skill)
                print(f"Installed skill: {skill_name}")
                return True
            return False
        except Exception as e:
            print(f"Failed to install {skill_name}: {e}")
            return False

    def install_from_marketplace_sync(self, skill_name: str, marketplace="anthropic") -> bool:
        return _run_sync(self.install_from_marketplace(skill_name, marketplace))

    async def search_marketplace(self, query: str, marketplace="skillsmp", limit=20) -> List[Dict]:
        if marketplace == "skillsmp":
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"{MARKETPLACES['skillsmp']}/skills/search", params={"q": query, "limit": limit}
                    ) as resp:
                        if resp.status == 200: return await resp.json()
            except Exception as e:
                print(f"Marketplace search failed: {e}")
        return []

    def add_marketplace(self, name: str, url: str):
        MARKETPLACES[name] = url


def _run_sync(coro):
    """Run an async coroutine synchronously."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)
