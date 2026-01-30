#!/usr/bin/env python3
"""
MCP Client Skill - Connect to Model Context Protocol servers

Allows the agent to use tools from any MCP server.
"""

import asyncio
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from .base import Skill, SkillResult, SkillManifest, SkillAction

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
    command: Optional[str] = None  # For stdio
    args: List[str] = field(default_factory=list)
    url: Optional[str] = None  # For http
    env: Optional[Dict[str, str]] = None


class MCPClientSkill(Skill):
    """
    MCP Client - Connect to MCP servers and use their tools.

    Supports:
    - stdio transport (spawn local server process)
    - HTTP/SSE transport (connect to remote server)
    """

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self.servers: Dict[str, MCPServer] = {}
        self.sessions: Dict[str, Any] = {}
        self.server_tools: Dict[str, List[Dict]] = {}
        self._exit_stacks: Dict[str, Any] = {}

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="mcp",
            name="MCP Client",
            version="1.0.0",
            category="integration",
            description="Connect to MCP servers and use their tools",
            required_credentials=[],
            install_cost=0,
            actions=[
                SkillAction(
                    name="add_server",
                    description="Add an MCP server configuration",
                    parameters={
                        "name": "unique server name",
                        "transport": "stdio or http",
                        "command": "command to run (stdio)",
                        "args": "command arguments (stdio)",
                        "url": "server URL (http)"
                    },
                    estimated_cost=0,
                    success_probability=0.95
                ),
                SkillAction(
                    name="connect",
                    description="Connect to a configured MCP server",
                    parameters={"name": "server name to connect to"},
                    estimated_cost=0,
                    success_probability=0.85
                ),
                SkillAction(
                    name="disconnect",
                    description="Disconnect from an MCP server",
                    parameters={"name": "server name to disconnect from"},
                    estimated_cost=0,
                    success_probability=0.95
                ),
                SkillAction(
                    name="list_servers",
                    description="List all configured MCP servers",
                    parameters={},
                    estimated_cost=0,
                    success_probability=1.0
                ),
                SkillAction(
                    name="list_tools",
                    description="List tools available from a connected server",
                    parameters={"name": "server name"},
                    estimated_cost=0,
                    success_probability=0.9
                ),
                SkillAction(
                    name="call_tool",
                    description="Call a tool on an MCP server",
                    parameters={
                        "server": "server name",
                        "tool": "tool name",
                        "arguments": "tool arguments (dict)"
                    },
                    estimated_cost=0,
                    success_probability=0.85
                ),
                SkillAction(
                    name="list_resources",
                    description="List resources available from a server",
                    parameters={"name": "server name"},
                    estimated_cost=0,
                    success_probability=0.9
                ),
                SkillAction(
                    name="read_resource",
                    description="Read a resource from an MCP server",
                    parameters={
                        "server": "server name",
                        "uri": "resource URI"
                    },
                    estimated_cost=0,
                    success_probability=0.85
                ),
            ]
        )

    def check_credentials(self) -> bool:
        return HAS_MCP

    async def execute(self, action: str, params: Dict) -> SkillResult:
        if not HAS_MCP:
            return SkillResult(
                success=False,
                message="MCP SDK not installed. Run: pip install mcp"
            )

        try:
            if action == "add_server":
                return await self._add_server(
                    params.get("name", ""),
                    params.get("transport", "stdio"),
                    params.get("command"),
                    params.get("args", []),
                    params.get("url"),
                    params.get("env")
                )
            elif action == "connect":
                return await self._connect(params.get("name", ""))
            elif action == "disconnect":
                return await self._disconnect(params.get("name", ""))
            elif action == "list_servers":
                return await self._list_servers()
            elif action == "list_tools":
                return await self._list_tools(params.get("name", ""))
            elif action == "call_tool":
                return await self._call_tool(
                    params.get("server", ""),
                    params.get("tool", ""),
                    params.get("arguments", {})
                )
            elif action == "list_resources":
                return await self._list_resources(params.get("name", ""))
            elif action == "read_resource":
                return await self._read_resource(
                    params.get("server", ""),
                    params.get("uri", "")
                )
            else:
                return SkillResult(success=False, message=f"Unknown action: {action}")
        except Exception as e:
            return SkillResult(success=False, message=f"MCP error: {str(e)}")

    async def _add_server(
        self,
        name: str,
        transport: str,
        command: Optional[str],
        args: List[str],
        url: Optional[str],
        env: Optional[Dict[str, str]]
    ) -> SkillResult:
        """Add an MCP server configuration"""
        if not name:
            return SkillResult(success=False, message="Server name required")

        if transport == "stdio" and not command:
            return SkillResult(success=False, message="Command required for stdio transport")

        if transport == "http" and not url:
            return SkillResult(success=False, message="URL required for http transport")

        self.servers[name] = MCPServer(
            name=name,
            transport=transport,
            command=command,
            args=args if isinstance(args, list) else [args] if args else [],
            url=url,
            env=env
        )

        return SkillResult(
            success=True,
            message=f"Added MCP server: {name}",
            data={"name": name, "transport": transport}
        )

    async def _connect(self, name: str) -> SkillResult:
        """Connect to an MCP server"""
        if name not in self.servers:
            return SkillResult(success=False, message=f"Server not found: {name}")

        if name in self.sessions:
            return SkillResult(success=True, message=f"Already connected to {name}")

        server = self.servers[name]

        try:
            from contextlib import AsyncExitStack
            exit_stack = AsyncExitStack()
            self._exit_stacks[name] = exit_stack

            if server.transport == "stdio":
                server_params = StdioServerParameters(
                    command=server.command,
                    args=server.args,
                    env=server.env
                )

                stdio_transport = await exit_stack.enter_async_context(
                    stdio_client(server_params)
                )
                read_stream, write_stream = stdio_transport

                session = await exit_stack.enter_async_context(
                    ClientSession(read_stream, write_stream)
                )

            elif server.transport == "http":
                if not HAS_MCP_HTTP:
                    return SkillResult(
                        success=False,
                        message="HTTP transport not available"
                    )

                http_transport = await exit_stack.enter_async_context(
                    streamablehttp_client(server.url)
                )
                read_stream, write_stream, _ = http_transport

                session = await exit_stack.enter_async_context(
                    ClientSession(read_stream, write_stream)
                )
            else:
                return SkillResult(
                    success=False,
                    message=f"Unknown transport: {server.transport}"
                )

            await session.initialize()
            self.sessions[name] = session

            # Cache available tools
            tools_response = await session.list_tools()
            self.server_tools[name] = [
                {
                    "name": tool.name,
                    "description": getattr(tool, 'description', ''),
                    "schema": getattr(tool, 'inputSchema', {})
                }
                for tool in tools_response.tools
            ]

            return SkillResult(
                success=True,
                message=f"Connected to {name}",
                data={
                    "server": name,
                    "tools": [t["name"] for t in self.server_tools[name]]
                }
            )

        except Exception as e:
            if name in self._exit_stacks:
                await self._exit_stacks[name].aclose()
                del self._exit_stacks[name]
            return SkillResult(success=False, message=f"Connection failed: {e}")

    async def _disconnect(self, name: str) -> SkillResult:
        """Disconnect from an MCP server"""
        if name not in self.sessions:
            return SkillResult(success=False, message=f"Not connected to {name}")

        try:
            if name in self._exit_stacks:
                await self._exit_stacks[name].aclose()
                del self._exit_stacks[name]

            del self.sessions[name]
            if name in self.server_tools:
                del self.server_tools[name]

            return SkillResult(
                success=True,
                message=f"Disconnected from {name}"
            )
        except Exception as e:
            return SkillResult(success=False, message=f"Disconnect error: {e}")

    async def _list_servers(self) -> SkillResult:
        """List all configured servers"""
        servers = []
        for name, server in self.servers.items():
            servers.append({
                "name": name,
                "transport": server.transport,
                "connected": name in self.sessions,
                "tools_count": len(self.server_tools.get(name, []))
            })

        return SkillResult(
            success=True,
            message=f"Found {len(servers)} servers",
            data={"servers": servers}
        )

    async def _list_tools(self, name: str) -> SkillResult:
        """List tools from a connected server"""
        if name not in self.sessions:
            return SkillResult(success=False, message=f"Not connected to {name}")

        tools = self.server_tools.get(name, [])
        return SkillResult(
            success=True,
            message=f"Found {len(tools)} tools",
            data={"tools": tools}
        )

    async def _call_tool(
        self,
        server: str,
        tool: str,
        arguments: Dict
    ) -> SkillResult:
        """Call a tool on an MCP server"""
        if server not in self.sessions:
            return SkillResult(success=False, message=f"Not connected to {server}")

        if not tool:
            return SkillResult(success=False, message="Tool name required")

        session = self.sessions[server]

        try:
            result = await session.call_tool(tool, arguments=arguments)

            # Extract content from result
            content = []
            for item in result.content:
                if hasattr(item, 'text'):
                    content.append(item.text)
                elif hasattr(item, 'data'):
                    content.append(str(item.data))
                else:
                    content.append(str(item))

            return SkillResult(
                success=True,
                message=f"Called {tool}",
                data={
                    "tool": tool,
                    "result": content,
                    "is_error": getattr(result, 'isError', False)
                }
            )
        except Exception as e:
            return SkillResult(success=False, message=f"Tool call failed: {e}")

    async def _list_resources(self, name: str) -> SkillResult:
        """List resources from a connected server"""
        if name not in self.sessions:
            return SkillResult(success=False, message=f"Not connected to {name}")

        session = self.sessions[name]

        try:
            result = await session.list_resources()
            resources = [
                {
                    "uri": r.uri,
                    "name": getattr(r, 'name', ''),
                    "description": getattr(r, 'description', ''),
                    "mimeType": getattr(r, 'mimeType', '')
                }
                for r in result.resources
            ]

            return SkillResult(
                success=True,
                message=f"Found {len(resources)} resources",
                data={"resources": resources}
            )
        except Exception as e:
            return SkillResult(success=False, message=f"List resources failed: {e}")

    async def _read_resource(self, server: str, uri: str) -> SkillResult:
        """Read a resource from an MCP server"""
        if server not in self.sessions:
            return SkillResult(success=False, message=f"Not connected to {server}")

        if not uri:
            return SkillResult(success=False, message="Resource URI required")

        session = self.sessions[server]

        try:
            result = await session.read_resource(uri)

            contents = []
            for item in result.contents:
                if hasattr(item, 'text'):
                    contents.append({"type": "text", "text": item.text})
                elif hasattr(item, 'blob'):
                    contents.append({"type": "blob", "size": len(item.blob)})
                else:
                    contents.append({"type": "unknown"})

            return SkillResult(
                success=True,
                message=f"Read resource: {uri}",
                data={"uri": uri, "contents": contents}
            )
        except Exception as e:
            return SkillResult(success=False, message=f"Read resource failed: {e}")

    async def close(self):
        """Clean up all connections"""
        for name in list(self.sessions.keys()):
            await self._disconnect(name)


def get_mcp_tools_for_agent(mcp_skill: MCPClientSkill) -> List[Dict]:
    """
    Get all MCP tools formatted for agent use.

    Returns tools in format:
    {
        "name": "mcp:{server}:{tool}",
        "description": "...",
        "parameters": {...}
    }
    """
    tools = []
    for server_name, server_tools in mcp_skill.server_tools.items():
        for tool in server_tools:
            tools.append({
                "name": f"mcp:{server_name}:{tool['name']}",
                "description": tool.get('description', ''),
                "parameters": tool.get('schema', {})
            })
    return tools
