"""MCP Client Skill - Connect to Model Context Protocol servers."""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from singularity.skills.base import Skill, SkillResult, SkillManifest, SkillAction
from . import handlers

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    HAS_MCP = True
except ImportError:
    HAS_MCP = False

try:
    from mcp.client.streamable_http import streamablehttp_client
    HAS_MCP_HTTP = True
except ImportError:
    HAS_MCP_HTTP = False


@dataclass
class MCPServer:
    """Configuration for an MCP server"""
    name: str
    transport: str  # "stdio" or "http"
    command: Optional[str] = None
    args: List[str] = field(default_factory=list)
    url: Optional[str] = None
    env: Optional[Dict[str, str]] = None


def _a(n, d, p=None, prob=0.95):
    return SkillAction(name=n, description=d, parameters=p or {}, estimated_cost=0, success_probability=prob)


class MCPClientSkill(Skill):
    """MCP Client - Connect to MCP servers and use their tools."""

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self.servers: Dict[str, MCPServer] = {}
        self.sessions: Dict[str, Any] = {}
        self.server_tools: Dict[str, List[Dict]] = {}
        self._exit_stacks: Dict[str, Any] = {}

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="mcp", name="MCP Client", version="1.0.0", category="integration",
            description="Connect to MCP servers and use their tools",
            required_credentials=[], install_cost=0,
            actions=[
                _a("add_server", "Add an MCP server configuration", {
                    "name": "unique server name", "transport": "stdio or http",
                    "command": "command to run (stdio)", "args": "command arguments (stdio)",
                    "url": "server URL (http)"}),
                _a("connect", "Connect to a configured MCP server", {"name": "server name"}, 0.85),
                _a("disconnect", "Disconnect from an MCP server", {"name": "server name"}),
                _a("list_servers", "List all configured MCP servers", prob=1.0),
                _a("list_tools", "List tools from a connected server", {"name": "server name"}, 0.9),
                _a("call_tool", "Call a tool on an MCP server", {
                    "server": "server name", "tool": "tool name", "arguments": "tool arguments (dict)"}, 0.85),
                _a("list_resources", "List resources from a server", {"name": "server name"}, 0.9),
                _a("read_resource", "Read a resource from an MCP server", {
                    "server": "server name", "uri": "resource URI"}, 0.85),
            ])

    def check_credentials(self) -> bool:
        return HAS_MCP

    async def execute(self, action: str, params: Dict) -> SkillResult:
        if not HAS_MCP:
            return SkillResult(success=False, message="MCP SDK not installed. Run: pip install mcp")
        try:
            dispatch = {
                "add_server": lambda: handlers.add_server(self, params.get("name", ""),
                    params.get("transport", "stdio"), params.get("command"), params.get("args", []),
                    params.get("url"), params.get("env")),
                "connect": lambda: handlers.connect(self, params.get("name", "")),
                "disconnect": lambda: handlers.disconnect(self, params.get("name", "")),
                "list_servers": lambda: handlers.list_servers(self),
                "list_tools": lambda: handlers.list_tools(self, params.get("name", "")),
                "call_tool": lambda: handlers.call_tool(self, params.get("server", ""),
                    params.get("tool", ""), params.get("arguments", {})),
                "list_resources": lambda: handlers.list_resources(self, params.get("name", "")),
                "read_resource": lambda: handlers.read_resource(self, params.get("server", ""),
                    params.get("uri", "")),
            }
            handler = dispatch.get(action)
            if not handler:
                return SkillResult(success=False, message=f"Unknown action: {action}")
            return await handler()
        except Exception as e:
            return SkillResult(success=False, message=f"MCP error: {str(e)}")

    async def close(self):
        for name in list(self.sessions.keys()):
            await handlers.disconnect(self, name)


def get_mcp_tools_for_agent(mcp_skill: MCPClientSkill) -> List[Dict]:
    """Get all MCP tools formatted for agent use."""
    tools = []
    for server_name, server_tools in mcp_skill.server_tools.items():
        for tool in server_tools:
            tools.append({"name": f"mcp:{server_name}:{tool['name']}",
                          "description": tool.get('description', ''),
                          "parameters": tool.get('schema', {})})
    return tools
