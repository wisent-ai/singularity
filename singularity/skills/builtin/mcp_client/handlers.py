"""Handler functions for MCPClientSkill actions."""

from typing import Optional, List, Dict
from singularity.skills.base import SkillResult
from . import MCPServer, HAS_MCP_HTTP


async def add_server(skill, name: str, transport: str, command: Optional[str],
                     args: List[str], url: Optional[str], env: Optional[Dict[str, str]]) -> SkillResult:
    if not name:
        return SkillResult(success=False, message="Server name required")
    if transport == "stdio" and not command:
        return SkillResult(success=False, message="Command required for stdio transport")
    if transport == "http" and not url:
        return SkillResult(success=False, message="URL required for http transport")
    skill.servers[name] = MCPServer(
        name=name, transport=transport, command=command,
        args=args if isinstance(args, list) else [args] if args else [],
        url=url, env=env)
    return SkillResult(success=True, message=f"Added MCP server: {name}",
                       data={"name": name, "transport": transport})


async def connect(skill, name: str) -> SkillResult:
    if name not in skill.servers:
        return SkillResult(success=False, message=f"Server not found: {name}")
    if name in skill.sessions:
        return SkillResult(success=True, message=f"Already connected to {name}")

    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    server = skill.servers[name]
    try:
        from contextlib import AsyncExitStack
        exit_stack = AsyncExitStack()
        skill._exit_stacks[name] = exit_stack

        if server.transport == "stdio":
            server_params = StdioServerParameters(
                command=server.command, args=server.args, env=server.env)
            stdio_transport = await exit_stack.enter_async_context(stdio_client(server_params))
            read_stream, write_stream = stdio_transport
            session = await exit_stack.enter_async_context(ClientSession(read_stream, write_stream))

        elif server.transport == "http":
            if not HAS_MCP_HTTP:
                return SkillResult(success=False, message="HTTP transport not available")
            from mcp.client.streamable_http import streamablehttp_client
            http_transport = await exit_stack.enter_async_context(streamablehttp_client(server.url))
            read_stream, write_stream, _ = http_transport
            session = await exit_stack.enter_async_context(ClientSession(read_stream, write_stream))
        else:
            return SkillResult(success=False, message=f"Unknown transport: {server.transport}")

        await session.initialize()
        skill.sessions[name] = session

        tools_response = await session.list_tools()
        skill.server_tools[name] = [
            {"name": tool.name, "description": getattr(tool, 'description', ''),
             "schema": getattr(tool, 'inputSchema', {})}
            for tool in tools_response.tools]

        return SkillResult(success=True, message=f"Connected to {name}",
            data={"server": name, "tools": [t["name"] for t in skill.server_tools[name]]})

    except Exception as e:
        if name in skill._exit_stacks:
            await skill._exit_stacks[name].aclose()
            del skill._exit_stacks[name]
        return SkillResult(success=False, message=f"Connection failed: {e}")


async def disconnect(skill, name: str) -> SkillResult:
    if name not in skill.sessions:
        return SkillResult(success=False, message=f"Not connected to {name}")
    try:
        if name in skill._exit_stacks:
            await skill._exit_stacks[name].aclose()
            del skill._exit_stacks[name]
        del skill.sessions[name]
        if name in skill.server_tools:
            del skill.server_tools[name]
        return SkillResult(success=True, message=f"Disconnected from {name}")
    except Exception as e:
        return SkillResult(success=False, message=f"Disconnect error: {e}")


async def list_servers(skill) -> SkillResult:
    servers = [{"name": name, "transport": server.transport,
                "connected": name in skill.sessions,
                "tools_count": len(skill.server_tools.get(name, []))}
               for name, server in skill.servers.items()]
    return SkillResult(success=True, message=f"Found {len(servers)} servers", data={"servers": servers})


async def list_tools(skill, name: str) -> SkillResult:
    if name not in skill.sessions:
        return SkillResult(success=False, message=f"Not connected to {name}")
    tools = skill.server_tools.get(name, [])
    return SkillResult(success=True, message=f"Found {len(tools)} tools", data={"tools": tools})


async def call_tool(skill, server: str, tool: str, arguments: Dict) -> SkillResult:
    if server not in skill.sessions:
        return SkillResult(success=False, message=f"Not connected to {server}")
    if not tool:
        return SkillResult(success=False, message="Tool name required")
    session = skill.sessions[server]
    try:
        result = await session.call_tool(tool, arguments=arguments)
        content = []
        for item in result.content:
            if hasattr(item, 'text'):
                content.append(item.text)
            elif hasattr(item, 'data'):
                content.append(str(item.data))
            else:
                content.append(str(item))
        return SkillResult(success=True, message=f"Called {tool}",
            data={"tool": tool, "result": content, "is_error": getattr(result, 'isError', False)})
    except Exception as e:
        return SkillResult(success=False, message=f"Tool call failed: {e}")


async def list_resources(skill, name: str) -> SkillResult:
    if name not in skill.sessions:
        return SkillResult(success=False, message=f"Not connected to {name}")
    session = skill.sessions[name]
    try:
        result = await session.list_resources()
        resources = [{"uri": r.uri, "name": getattr(r, 'name', ''),
                      "description": getattr(r, 'description', ''),
                      "mimeType": getattr(r, 'mimeType', '')}
                     for r in result.resources]
        return SkillResult(success=True, message=f"Found {len(resources)} resources",
                           data={"resources": resources})
    except Exception as e:
        return SkillResult(success=False, message=f"List resources failed: {e}")


async def read_resource(skill, server: str, uri: str) -> SkillResult:
    if server not in skill.sessions:
        return SkillResult(success=False, message=f"Not connected to {server}")
    if not uri:
        return SkillResult(success=False, message="Resource URI required")
    session = skill.sessions[server]
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
        return SkillResult(success=True, message=f"Read resource: {uri}",
                           data={"uri": uri, "contents": contents})
    except Exception as e:
        return SkillResult(success=False, message=f"Read resource failed: {e}")
