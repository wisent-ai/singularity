#!/usr/bin/env python3
"""
Serverless Function Skill - Deploy lightweight Python functions as HTTP endpoints.

Built in response to Feature Request #156 from agent Eve:
  "I need a way to make my services persistently available so users can
   actually call them and I can earn revenue."

This skill fills the critical gap between heavy Docker-based deployments
(PublicServiceDeployerSkill) and having no deployment option at all. Agents can:

1. **Deploy functions** - Write a Python handler, deploy it as an HTTP endpoint
2. **Route requests** - Map URL paths to specific function handlers
3. **Manage lifecycle** - Enable/disable/update functions without container overhead
4. **Track invocations** - Per-function call counts, latency, errors, revenue
5. **Generate server code** - Create self-contained ASGI server files ready for hosting
6. **Bundle functions** - Package multiple functions into a single deployable service

Architecture:
  Functions are stored as Python source with metadata. The skill can:
  - Generate a standalone FastAPI/ASGI server that routes to all active functions
  - Track invocation metrics for billing integration
  - Support multiple HTTP methods per function
  - Chain functions into pipelines

Why serverless over Docker?
  - No Docker daemon required (works in any Python environment)
  - Sub-second deploy time vs minutes for container builds
  - Zero infrastructure overhead for simple API endpoints
  - Lower cost for low-traffic services
  - Perfect for MVP/prototype services that can graduate to Docker later

Part of the Revenue Generation pillar - enables agents to rapidly deploy
paid services. Addresses Issue #156.
"""

import json
import os
import time
import uuid
import hashlib
import textwrap
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

from .base import Skill, SkillResult, SkillManifest, SkillAction

DATA_DIR = Path(__file__).parent.parent / "data"
FUNCTIONS_FILE = DATA_DIR / "serverless_functions.json"
FUNCTION_CODE_DIR = DATA_DIR / "function_code"

# Supported HTTP methods
SUPPORTED_METHODS = ["GET", "POST", "PUT", "DELETE", "PATCH"]

# Default function template
DEFAULT_HANDLER_TEMPLATE = '''async def handler(request):
    """
    Handle incoming HTTP request.

    Args:
        request: dict with keys: method, path, headers, query_params, body

    Returns:
        dict with keys: status_code (int), headers (dict), body (any JSON-serializable)
    """
    return {
        "status_code": 200,
        "headers": {"Content-Type": "application/json"},
        "body": {"message": "Hello from {function_name}!"}
    }
'''

# ASGI server template for code generation
ASGI_SERVER_TEMPLATE = '''#!/usr/bin/env python3
"""
Auto-generated ASGI server for {agent_name}'s serverless functions.
Generated at: {generated_at}
Functions: {function_count}

Run with: uvicorn server:app --host 0.0.0.0 --port 8080
Or:       python server.py
"""

import json
import time
import traceback
from typing import Any, Dict

# --- Function Handlers ---
{function_imports}

# --- Route Table ---
ROUTES = {routes_dict}

async def dispatch(scope, receive, send):
    """ASGI application entry point."""
    if scope["type"] == "lifespan":
        message = await receive()
        if message["type"] == "lifespan.startup":
            await send({{"type": "lifespan.startup.complete"}})
        return

    if scope["type"] != "http":
        return

    path = scope.get("path", "/")
    method = scope.get("method", "GET").upper()

    # Health check
    if path == "/health":
        body = json.dumps({{"status": "healthy", "functions": len(ROUTES), "agent": "{agent_name}"}}).encode()
        await send({{"type": "http.response.start", "status": 200, "headers": [[b"content-type", b"application/json"]]}})
        await send({{"type": "http.response.body", "body": body}})
        return

    # Function index
    if path == "/" and method == "GET":
        endpoints = []
        for route_path, info in ROUTES.items():
            endpoints.append({{"path": route_path, "methods": info["methods"], "description": info.get("description", "")}})
        body = json.dumps({{"agent": "{agent_name}", "endpoints": endpoints}}).encode()
        await send({{"type": "http.response.start", "status": 200, "headers": [[b"content-type", b"application/json"]]}})
        await send({{"type": "http.response.body", "body": body}})
        return

    # Route lookup
    route_key = path
    route = ROUTES.get(route_key)
    if not route:
        body = json.dumps({{"error": "not_found", "path": path}}).encode()
        await send({{"type": "http.response.start", "status": 404, "headers": [[b"content-type", b"application/json"]]}})
        await send({{"type": "http.response.body", "body": body}})
        return

    if method not in route["methods"]:
        body = json.dumps({{"error": "method_not_allowed", "allowed": route["methods"]}}).encode()
        await send({{"type": "http.response.start", "status": 405, "headers": [[b"content-type", b"application/json"]]}})
        await send({{"type": "http.response.body", "body": body}})
        return

    # Parse request
    request_body = b""
    while True:
        message = await receive()
        request_body += message.get("body", b"")
        if not message.get("more_body", False):
            break

    try:
        parsed_body = json.loads(request_body) if request_body else None
    except json.JSONDecodeError:
        parsed_body = request_body.decode("utf-8", errors="replace")

    query_string = scope.get("query_string", b"").decode()
    query_params = dict(pair.split("=", 1) for pair in query_string.split("&") if "=" in pair) if query_string else {{}}
    headers = {{k.decode(): v.decode() for k, v in scope.get("headers", [])}}

    request = {{
        "method": method,
        "path": path,
        "headers": headers,
        "query_params": query_params,
        "body": parsed_body,
    }}

    # Execute handler
    start_time = time.time()
    try:
        handler_fn = route["handler"]
        result = await handler_fn(request)
        status = result.get("status_code", 200)
        resp_headers = result.get("headers", {{"Content-Type": "application/json"}})
        resp_body = result.get("body", {{}})
    except Exception as e:
        status = 500
        resp_headers = {{"Content-Type": "application/json"}}
        resp_body = {{"error": "internal_error", "message": str(e)}}

    # Send response
    header_list = [[k.encode(), v.encode()] for k, v in resp_headers.items()]
    body_bytes = json.dumps(resp_body).encode() if not isinstance(resp_body, bytes) else resp_body
    await send({{"type": "http.response.start", "status": status, "headers": header_list}})
    await send({{"type": "http.response.body", "body": body_bytes}})

app = dispatch

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8080, log_level="info")
'''


def _load_functions() -> Dict:
    """Load function registry from disk."""
    if FUNCTIONS_FILE.exists():
        try:
            return json.loads(FUNCTIONS_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {
        "functions": {},
        "routes": {},  # path -> function_id
        "agents": {},  # agent_name -> [function_ids]
        "stats": {
            "total_deployments": 0,
            "total_invocations": 0,
            "total_errors": 0,
            "total_revenue": 0.0,
        },
    }


def _save_functions(data: Dict) -> None:
    """Persist function registry."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FUNCTIONS_FILE.write_text(json.dumps(data, indent=2, default=str))


def _save_function_code(function_id: str, code: str) -> Path:
    """Save function source code to a separate file."""
    FUNCTION_CODE_DIR.mkdir(parents=True, exist_ok=True)
    code_path = FUNCTION_CODE_DIR / f"{function_id}.py"
    code_path.write_text(code)
    return code_path


def _load_function_code(function_id: str) -> Optional[str]:
    """Load function source code."""
    code_path = FUNCTION_CODE_DIR / f"{function_id}.py"
    if code_path.exists():
        return code_path.read_text()
    return None


class ServerlessFunctionSkill(Skill):
    """
    Deploy lightweight Python functions as HTTP endpoints.

    This enables agents (like Eve) to deploy Python handlers as HTTP API
    endpoints without Docker, containers, or complex infrastructure.
    Functions are stored as source code with metadata and can be bundled
    into a standalone ASGI server for hosting.
    """

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="serverless_function",
            name="Serverless Functions",
            version="1.0.0",
            category="infrastructure",
            description=(
                "Deploy lightweight Python functions as HTTP endpoints. "
                "No Docker required - just write a handler and deploy. "
                "Built for rapid service deployment and revenue generation."
            ),
            actions=[
                SkillAction(
                    name="deploy",
                    description=(
                        "Deploy a Python function as an HTTP endpoint. "
                        "Provide the handler code, route path, and HTTP methods."
                    ),
                    parameters={
                        "agent_name": {
                            "type": "str", "required": True,
                            "description": "Name of the agent deploying the function",
                        },
                        "function_name": {
                            "type": "str", "required": True,
                            "description": "Unique name for this function",
                        },
                        "route": {
                            "type": "str", "required": True,
                            "description": "URL path (e.g. /api/review)",
                        },
                        "handler_code": {
                            "type": "str", "required": True,
                            "description": "Python async function source code",
                        },
                        "methods": {
                            "type": "list", "required": False,
                            "description": "HTTP methods (default: ['POST'])",
                        },
                        "description": {
                            "type": "str", "required": False,
                            "description": "Human-readable description",
                        },
                        "price_per_call": {
                            "type": "float", "required": False,
                            "description": "Price per invocation in USD (default: 0)",
                        },
                        "rate_limit": {
                            "type": "int", "required": False,
                            "description": "Max calls per minute (0=unlimited)",
                        },
                        "timeout_seconds": {
                            "type": "int", "required": False,
                            "description": "Max execution time (default: 30)",
                        },
                    },
                    estimated_cost=0.0,
                ),
                SkillAction(
                    name="update",
                    description="Update an existing function's code or configuration",
                    parameters={
                        "function_id": {
                            "type": "str", "required": True,
                            "description": "ID of the function to update",
                        },
                        "handler_code": {
                            "type": "str", "required": False,
                            "description": "New handler code",
                        },
                        "description": {
                            "type": "str", "required": False,
                            "description": "Updated description",
                        },
                        "price_per_call": {
                            "type": "float", "required": False,
                            "description": "Updated price per call",
                        },
                        "rate_limit": {
                            "type": "int", "required": False,
                            "description": "Updated rate limit",
                        },
                        "methods": {
                            "type": "list", "required": False,
                            "description": "Updated HTTP methods",
                        },
                    },
                    estimated_cost=0.0,
                ),
                SkillAction(
                    name="remove",
                    description="Remove a deployed function and its route",
                    parameters={
                        "function_id": {
                            "type": "str", "required": True,
                            "description": "ID of the function to remove",
                        },
                    },
                    estimated_cost=0.0,
                ),
                SkillAction(
                    name="enable",
                    description="Enable a disabled function",
                    parameters={
                        "function_id": {
                            "type": "str", "required": True,
                            "description": "ID of the function to enable",
                        },
                    },
                    estimated_cost=0.0,
                ),
                SkillAction(
                    name="disable",
                    description="Disable a function without removing it",
                    parameters={
                        "function_id": {
                            "type": "str", "required": True,
                            "description": "ID of the function to disable",
                        },
                    },
                    estimated_cost=0.0,
                ),
                SkillAction(
                    name="invoke",
                    description=(
                        "Locally invoke a function for testing. "
                        "Simulates an HTTP request and returns the response."
                    ),
                    parameters={
                        "function_id": {
                            "type": "str", "required": True,
                            "description": "ID of the function to invoke",
                        },
                        "method": {
                            "type": "str", "required": False,
                            "description": "HTTP method (default: POST)",
                        },
                        "body": {
                            "type": "dict", "required": False,
                            "description": "Request body",
                        },
                        "query_params": {
                            "type": "dict", "required": False,
                            "description": "Query parameters",
                        },
                        "headers": {
                            "type": "dict", "required": False,
                            "description": "Request headers",
                        },
                    },
                    estimated_cost=0.0,
                ),
                SkillAction(
                    name="list",
                    description="List all deployed functions, optionally filtered by agent",
                    parameters={
                        "agent_name": {
                            "type": "str", "required": False,
                            "description": "Filter by agent name",
                        },
                        "status": {
                            "type": "str", "required": False,
                            "description": "Filter by status: active, disabled, all (default: all)",
                        },
                    },
                    estimated_cost=0.0,
                ),
                SkillAction(
                    name="inspect",
                    description="Get detailed info about a function including its source code",
                    parameters={
                        "function_id": {
                            "type": "str", "required": True,
                            "description": "ID of the function to inspect",
                        },
                    },
                    estimated_cost=0.0,
                ),
                SkillAction(
                    name="generate_server",
                    description=(
                        "Generate a standalone ASGI server file that serves "
                        "all active functions for an agent. Ready for hosting."
                    ),
                    parameters={
                        "agent_name": {
                            "type": "str", "required": True,
                            "description": "Agent whose functions to include",
                        },
                        "output_path": {
                            "type": "str", "required": False,
                            "description": "Where to write the server file",
                        },
                    },
                    estimated_cost=0.0,
                ),
                SkillAction(
                    name="stats",
                    description="Get invocation statistics across all functions or per-agent",
                    parameters={
                        "agent_name": {
                            "type": "str", "required": False,
                            "description": "Filter stats by agent",
                        },
                    },
                    estimated_cost=0.0,
                ),
            ],
            required_credentials=[],
        )

    def _default_state(self) -> Dict:
        return _load_functions()

    def __init__(self):
        self._state: Optional[Dict] = None

    def _ensure_state(self) -> Dict:
        if self._state is None:
            self._state = self._default_state()
        return self._state

    def _persist(self) -> None:
        if self._state is not None:
            _save_functions(self._state)

    def execute(self, action: str, params: Optional[Dict] = None) -> SkillResult:
        params = params or {}
        handler = {
            "deploy": self._deploy,
            "update": self._update,
            "remove": self._remove,
            "enable": self._enable,
            "disable": self._disable,
            "invoke": self._invoke,
            "list": self._list,
            "inspect": self._inspect,
            "generate_server": self._generate_server,
            "stats": self._stats,
        }.get(action)

        if not handler:
            return SkillResult(
                success=False,
                message=f"Unknown action: {action}. Available: {list(self._get_handlers().keys())}",
            )

        return handler(params)

    def _get_handlers(self) -> Dict:
        return {
            "deploy": self._deploy,
            "update": self._update,
            "remove": self._remove,
            "enable": self._enable,
            "disable": self._disable,
            "invoke": self._invoke,
            "list": self._list,
            "inspect": self._inspect,
            "generate_server": self._generate_server,
            "stats": self._stats,
        }

    def _deploy(self, params: Dict) -> SkillResult:
        """Deploy a new serverless function."""
        agent_name = params.get("agent_name")
        function_name = params.get("function_name")
        route = params.get("route")
        handler_code = params.get("handler_code")

        if not all([agent_name, function_name, route, handler_code]):
            return SkillResult(
                success=False,
                message="Required: agent_name, function_name, route, handler_code",
            )

        # Validate route format
        if not route.startswith("/"):
            route = "/" + route

        state = self._ensure_state()

        # Check for route conflict
        if route in state["routes"]:
            existing_id = state["routes"][route]
            existing = state["functions"].get(existing_id, {})
            return SkillResult(
                success=False,
                message=f"Route {route} already assigned to function '{existing.get('name', existing_id)}'",
                data={"conflicting_function_id": existing_id},
            )

        # Validate handler code has an async handler function
        if "async def handler" not in handler_code and "async def " not in handler_code:
            return SkillResult(
                success=False,
                message="Handler code must contain an async function. Expected 'async def handler(request):'",
            )

        # Validate methods
        methods = params.get("methods", ["POST"])
        if isinstance(methods, str):
            methods = [methods]
        methods = [m.upper() for m in methods]
        invalid_methods = [m for m in methods if m not in SUPPORTED_METHODS]
        if invalid_methods:
            return SkillResult(
                success=False,
                message=f"Invalid methods: {invalid_methods}. Supported: {SUPPORTED_METHODS}",
            )

        # Generate function ID
        function_id = f"fn-{uuid.uuid4().hex[:12]}"

        # Save function code
        code_hash = hashlib.sha256(handler_code.encode()).hexdigest()[:16]
        _save_function_code(function_id, handler_code)

        # Create function record
        now = datetime.utcnow().isoformat()
        function_record = {
            "id": function_id,
            "name": function_name,
            "agent_name": agent_name,
            "route": route,
            "methods": methods,
            "description": params.get("description", ""),
            "price_per_call": params.get("price_per_call", 0.0),
            "rate_limit": params.get("rate_limit", 0),
            "timeout_seconds": params.get("timeout_seconds", 30),
            "status": "active",
            "code_hash": code_hash,
            "version": 1,
            "created_at": now,
            "updated_at": now,
            "invocations": 0,
            "errors": 0,
            "total_revenue": 0.0,
            "avg_latency_ms": 0.0,
        }

        # Register
        state["functions"][function_id] = function_record
        state["routes"][route] = function_id
        if agent_name not in state["agents"]:
            state["agents"][agent_name] = []
        state["agents"][agent_name].append(function_id)
        state["stats"]["total_deployments"] += 1

        self._persist()

        return SkillResult(
            success=True,
            message=f"Function '{function_name}' deployed at {route}",
            data={
                "function_id": function_id,
                "route": route,
                "methods": methods,
                "agent_name": agent_name,
                "status": "active",
                "price_per_call": function_record["price_per_call"],
            },
        )

    def _update(self, params: Dict) -> SkillResult:
        """Update an existing function's code or config."""
        function_id = params.get("function_id")
        if not function_id:
            return SkillResult(success=False, message="Required: function_id")

        state = self._ensure_state()
        func = state["functions"].get(function_id)
        if not func:
            return SkillResult(success=False, message=f"Function not found: {function_id}")

        updated_fields = []

        # Update handler code
        if "handler_code" in params:
            code = params["handler_code"]
            if "async def handler" not in code and "async def " not in code:
                return SkillResult(
                    success=False,
                    message="Handler code must contain an async function",
                )
            _save_function_code(function_id, code)
            func["code_hash"] = hashlib.sha256(code.encode()).hexdigest()[:16]
            func["version"] += 1
            updated_fields.append("handler_code")

        # Update description
        if "description" in params:
            func["description"] = params["description"]
            updated_fields.append("description")

        # Update price
        if "price_per_call" in params:
            func["price_per_call"] = float(params["price_per_call"])
            updated_fields.append("price_per_call")

        # Update rate limit
        if "rate_limit" in params:
            func["rate_limit"] = int(params["rate_limit"])
            updated_fields.append("rate_limit")

        # Update methods
        if "methods" in params:
            methods = params["methods"]
            if isinstance(methods, str):
                methods = [methods]
            methods = [m.upper() for m in methods]
            invalid = [m for m in methods if m not in SUPPORTED_METHODS]
            if invalid:
                return SkillResult(
                    success=False,
                    message=f"Invalid methods: {invalid}",
                )
            func["methods"] = methods
            updated_fields.append("methods")

        if not updated_fields:
            return SkillResult(
                success=False,
                message="No fields to update. Provide at least one of: handler_code, description, price_per_call, rate_limit, methods",
            )

        func["updated_at"] = datetime.utcnow().isoformat()
        self._persist()

        return SkillResult(
            success=True,
            message=f"Function '{func['name']}' updated: {', '.join(updated_fields)}",
            data={
                "function_id": function_id,
                "updated_fields": updated_fields,
                "version": func["version"],
            },
        )

    def _remove(self, params: Dict) -> SkillResult:
        """Remove a deployed function."""
        function_id = params.get("function_id")
        if not function_id:
            return SkillResult(success=False, message="Required: function_id")

        state = self._ensure_state()
        func = state["functions"].get(function_id)
        if not func:
            return SkillResult(success=False, message=f"Function not found: {function_id}")

        # Remove route mapping
        route = func["route"]
        if route in state["routes"] and state["routes"][route] == function_id:
            del state["routes"][route]

        # Remove from agent's function list
        agent_name = func["agent_name"]
        if agent_name in state["agents"]:
            state["agents"][agent_name] = [
                fid for fid in state["agents"][agent_name] if fid != function_id
            ]
            if not state["agents"][agent_name]:
                del state["agents"][agent_name]

        # Remove function record
        del state["functions"][function_id]

        # Remove code file
        code_path = FUNCTION_CODE_DIR / f"{function_id}.py"
        if code_path.exists():
            code_path.unlink()

        self._persist()

        return SkillResult(
            success=True,
            message=f"Function '{func['name']}' removed from {route}",
            data={
                "function_id": function_id,
                "route": route,
                "agent_name": agent_name,
            },
        )

    def _enable(self, params: Dict) -> SkillResult:
        """Enable a disabled function."""
        function_id = params.get("function_id")
        if not function_id:
            return SkillResult(success=False, message="Required: function_id")

        state = self._ensure_state()
        func = state["functions"].get(function_id)
        if not func:
            return SkillResult(success=False, message=f"Function not found: {function_id}")

        if func["status"] == "active":
            return SkillResult(
                success=True,
                message=f"Function '{func['name']}' is already active",
                data={"function_id": function_id, "status": "active"},
            )

        func["status"] = "active"
        func["updated_at"] = datetime.utcnow().isoformat()
        self._persist()

        return SkillResult(
            success=True,
            message=f"Function '{func['name']}' enabled",
            data={"function_id": function_id, "status": "active"},
        )

    def _disable(self, params: Dict) -> SkillResult:
        """Disable a function without removing it."""
        function_id = params.get("function_id")
        if not function_id:
            return SkillResult(success=False, message="Required: function_id")

        state = self._ensure_state()
        func = state["functions"].get(function_id)
        if not func:
            return SkillResult(success=False, message=f"Function not found: {function_id}")

        if func["status"] == "disabled":
            return SkillResult(
                success=True,
                message=f"Function '{func['name']}' is already disabled",
                data={"function_id": function_id, "status": "disabled"},
            )

        func["status"] = "disabled"
        func["updated_at"] = datetime.utcnow().isoformat()
        self._persist()

        return SkillResult(
            success=True,
            message=f"Function '{func['name']}' disabled",
            data={"function_id": function_id, "status": "disabled"},
        )

    def _invoke(self, params: Dict) -> SkillResult:
        """Locally invoke a function for testing."""
        function_id = params.get("function_id")
        if not function_id:
            return SkillResult(success=False, message="Required: function_id")

        state = self._ensure_state()
        func = state["functions"].get(function_id)
        if not func:
            return SkillResult(success=False, message=f"Function not found: {function_id}")

        if func["status"] != "active":
            return SkillResult(
                success=False,
                message=f"Function '{func['name']}' is {func['status']}. Enable it first.",
            )

        # Load and compile handler code
        code = _load_function_code(function_id)
        if not code:
            return SkillResult(
                success=False,
                message=f"Function code not found for {function_id}",
            )

        # Build request dict
        method = params.get("method", "POST").upper()
        if method not in func["methods"]:
            return SkillResult(
                success=False,
                message=f"Method {method} not allowed. Allowed: {func['methods']}",
            )

        request = {
            "method": method,
            "path": func["route"],
            "headers": params.get("headers", {}),
            "query_params": params.get("query_params", {}),
            "body": params.get("body"),
        }

        # Execute handler
        start_time = time.time()
        try:
            # Compile and extract handler
            local_ns = {}
            exec(code, {"__builtins__": __builtins__}, local_ns)
            handler_fn = local_ns.get("handler")
            if not handler_fn:
                return SkillResult(
                    success=False,
                    message="No 'handler' function found in function code",
                )

            # Run async handler synchronously for testing
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If we're already in an async context, create a new thread
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as pool:
                        result = pool.submit(
                            asyncio.run, handler_fn(request)
                        ).result(timeout=func["timeout_seconds"])
                else:
                    result = loop.run_until_complete(handler_fn(request))
            except RuntimeError:
                result = asyncio.run(handler_fn(request))

            latency_ms = (time.time() - start_time) * 1000

            # Track invocation
            func["invocations"] += 1
            state["stats"]["total_invocations"] += 1
            revenue = func["price_per_call"]
            func["total_revenue"] += revenue
            state["stats"]["total_revenue"] += revenue

            # Update average latency
            n = func["invocations"]
            func["avg_latency_ms"] = (
                (func["avg_latency_ms"] * (n - 1) + latency_ms) / n
            )

            self._persist()

            return SkillResult(
                success=True,
                message=f"Function '{func['name']}' invoked successfully ({latency_ms:.1f}ms)",
                data={
                    "response": result,
                    "latency_ms": round(latency_ms, 1),
                    "invocation_count": func["invocations"],
                    "revenue_earned": revenue,
                },
                revenue=revenue,
            )

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            func["invocations"] += 1
            func["errors"] += 1
            state["stats"]["total_invocations"] += 1
            state["stats"]["total_errors"] += 1
            self._persist()

            return SkillResult(
                success=False,
                message=f"Function error: {str(e)}",
                data={
                    "error": str(e),
                    "latency_ms": round(latency_ms, 1),
                    "invocation_count": func["invocations"],
                    "error_count": func["errors"],
                },
            )

    def _list(self, params: Dict) -> SkillResult:
        """List all deployed functions."""
        state = self._ensure_state()
        agent_name = params.get("agent_name")
        status_filter = params.get("status", "all")

        functions = []
        for fid, func in state["functions"].items():
            if agent_name and func["agent_name"] != agent_name:
                continue
            if status_filter != "all" and func["status"] != status_filter:
                continue
            functions.append({
                "id": func["id"],
                "name": func["name"],
                "agent_name": func["agent_name"],
                "route": func["route"],
                "methods": func["methods"],
                "status": func["status"],
                "price_per_call": func["price_per_call"],
                "invocations": func["invocations"],
                "errors": func["errors"],
                "total_revenue": func["total_revenue"],
                "version": func["version"],
            })

        return SkillResult(
            success=True,
            message=f"Found {len(functions)} function(s)",
            data={
                "functions": functions,
                "total": len(functions),
                "filter_agent": agent_name,
                "filter_status": status_filter,
            },
        )

    def _inspect(self, params: Dict) -> SkillResult:
        """Get detailed info about a function."""
        function_id = params.get("function_id")
        if not function_id:
            return SkillResult(success=False, message="Required: function_id")

        state = self._ensure_state()
        func = state["functions"].get(function_id)
        if not func:
            return SkillResult(success=False, message=f"Function not found: {function_id}")

        code = _load_function_code(function_id)

        error_rate = (
            (func["errors"] / func["invocations"] * 100)
            if func["invocations"] > 0
            else 0.0
        )

        return SkillResult(
            success=True,
            message=f"Function '{func['name']}' details",
            data={
                "function": func,
                "source_code": code,
                "error_rate_pct": round(error_rate, 1),
                "estimated_monthly_revenue": func["total_revenue"] * 30 if func["invocations"] > 0 else 0,
            },
        )

    def _generate_server(self, params: Dict) -> SkillResult:
        """Generate a standalone ASGI server for an agent's functions."""
        agent_name = params.get("agent_name")
        if not agent_name:
            return SkillResult(success=False, message="Required: agent_name")

        state = self._ensure_state()
        agent_functions = state["agents"].get(agent_name, [])

        if not agent_functions:
            return SkillResult(
                success=False,
                message=f"No functions found for agent '{agent_name}'",
            )

        # Collect active functions
        active_functions = []
        for fid in agent_functions:
            func = state["functions"].get(fid)
            if func and func["status"] == "active":
                code = _load_function_code(fid)
                if code:
                    active_functions.append((fid, func, code))

        if not active_functions:
            return SkillResult(
                success=False,
                message=f"No active functions for agent '{agent_name}'",
            )

        # Generate function import blocks
        function_imports = []
        routes_entries = []

        for fid, func, code in active_functions:
            safe_name = func["name"].replace("-", "_").replace(" ", "_").lower()
            # Indent the code inside its own namespace
            function_imports.append(
                f"# --- {func['name']} ({func['route']}) ---\n{code}\n"
                f"_handler_{safe_name} = handler  # capture reference\n"
            )
            routes_entries.append(
                f'    "{func["route"]}": {{"handler": _handler_{safe_name}, '
                f'"methods": {func["methods"]}, '
                f'"description": {json.dumps(func["description"])}}}'
            )

        function_imports_str = "\n\n".join(function_imports)
        routes_str = "{\n" + ",\n".join(routes_entries) + "\n}"

        server_code = ASGI_SERVER_TEMPLATE.format(
            agent_name=agent_name,
            generated_at=datetime.utcnow().isoformat(),
            function_count=len(active_functions),
            function_imports=function_imports_str,
            routes_dict=routes_str,
        )

        # Write to output path
        output_path = params.get("output_path")
        if output_path:
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(server_code)
            wrote_to = str(out)
        else:
            out_dir = DATA_DIR / "generated_servers"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{agent_name}_server.py"
            out_path.write_text(server_code)
            wrote_to = str(out_path)

        return SkillResult(
            success=True,
            message=f"Generated ASGI server with {len(active_functions)} function(s) for '{agent_name}'",
            data={
                "server_path": wrote_to,
                "function_count": len(active_functions),
                "functions": [
                    {"name": f["name"], "route": f["route"], "methods": f["methods"]}
                    for _, f, _ in active_functions
                ],
                "run_command": f"uvicorn {agent_name}_server:app --host 0.0.0.0 --port 8080",
            },
        )

    def _stats(self, params: Dict) -> SkillResult:
        """Get invocation statistics."""
        state = self._ensure_state()
        agent_name = params.get("agent_name")

        if agent_name:
            agent_funcs = state["agents"].get(agent_name, [])
            functions = [
                state["functions"][fid]
                for fid in agent_funcs
                if fid in state["functions"]
            ]
        else:
            functions = list(state["functions"].values())

        total_invocations = sum(f["invocations"] for f in functions)
        total_errors = sum(f["errors"] for f in functions)
        total_revenue = sum(f["total_revenue"] for f in functions)
        active_count = sum(1 for f in functions if f["status"] == "active")
        disabled_count = sum(1 for f in functions if f["status"] == "disabled")

        # Top functions by invocations
        top_by_invocations = sorted(functions, key=lambda f: f["invocations"], reverse=True)[:5]
        # Top by revenue
        top_by_revenue = sorted(functions, key=lambda f: f["total_revenue"], reverse=True)[:5]

        error_rate = (total_errors / total_invocations * 100) if total_invocations > 0 else 0

        return SkillResult(
            success=True,
            message=f"Stats for {len(functions)} function(s)" + (f" ({agent_name})" if agent_name else ""),
            data={
                "total_functions": len(functions),
                "active_functions": active_count,
                "disabled_functions": disabled_count,
                "total_invocations": total_invocations,
                "total_errors": total_errors,
                "error_rate_pct": round(error_rate, 1),
                "total_revenue": round(total_revenue, 4),
                "top_by_invocations": [
                    {"name": f["name"], "invocations": f["invocations"]}
                    for f in top_by_invocations
                ],
                "top_by_revenue": [
                    {"name": f["name"], "revenue": f["total_revenue"]}
                    for f in top_by_revenue
                ],
                "global_stats": state["stats"],
            },
        )

    def get_actions(self) -> list:
        return self.manifest.actions

    def estimate_cost(self, action: str, params: Optional[Dict] = None) -> float:
        return 0.0
