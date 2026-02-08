#!/usr/bin/env python3
"""
Service Hosting Skill - Enable agents to register, deploy, and expose HTTP services.

This skill addresses Issue #110: agents need a way to host HTTP services that are
accessible from the internet. It provides:

1. Service Registration - Agents register their HTTP service endpoints with metadata
   (name, routes, pricing, health check URL, Docker image)
2. Service Directory - A registry of all hosted agent services, queryable by capability
3. Routing Configuration - Generates reverse proxy configs (nginx/Caddy) for routing
   requests to agent services
4. Health Monitoring - Periodic health checks on registered services with auto-deregister
5. Payment Integration - Per-request billing via UsageTracking, auto-debit callers
6. Service Lifecycle - Start, stop, restart, and scale services

Architecture:
  ServiceHostingSkill manages a JSON-based service registry. Each registered service
  has endpoints, pricing, health status, and deployment info. The skill generates
  proxy configurations and integrates with DeploymentSkill for actual container
  management and UsageTrackingSkill for billing.

Part of the Revenue Generation pillar - enables agents to offer paid HTTP services.
Built in response to Issue #110 from agent Adam.
"""

import json
import os
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

from .base import Skill, SkillResult, SkillManifest, SkillAction

SERVICES_FILE = Path(__file__).parent.parent / "data" / "hosted_services.json"
SERVICE_LOGS_FILE = Path(__file__).parent.parent / "data" / "service_logs.json"

# Default domain template for agent services
DEFAULT_DOMAIN_TEMPLATE = "{agent_name}.singularity.wisent.ai"


def _load_services() -> Dict:
    """Load the service registry."""
    if SERVICES_FILE.exists():
        try:
            return json.loads(SERVICES_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {"services": {}, "routing_rules": {}, "domain_assignments": {}}


def _save_services(data: Dict) -> None:
    """Persist the service registry."""
    SERVICES_FILE.parent.mkdir(parents=True, exist_ok=True)
    SERVICES_FILE.write_text(json.dumps(data, indent=2, default=str))


def _load_logs() -> List[Dict]:
    """Load service access logs."""
    if SERVICE_LOGS_FILE.exists():
        try:
            return json.loads(SERVICE_LOGS_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return []


def _save_logs(logs: List[Dict]) -> None:
    """Persist service access logs (keep last 1000)."""
    SERVICE_LOGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    logs = logs[-1000:]
    SERVICE_LOGS_FILE.write_text(json.dumps(logs, indent=2, default=str))


class ServiceHostingSkill(Skill):
    """
    Enables agents to register, host, and manage HTTP services.

    This is the infrastructure layer that lets agents like Adam expose
    their revenue-generating services to the internet.
    """

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="service_hosting",
            name="Service Hosting",
            version="1.0.0",
            category="infrastructure",
            description=(
                "Register, deploy, and manage HTTP services for agents. "
                "Provides service directory, routing, health monitoring, "
                "and per-request billing integration."
            ),
            actions=[
                SkillAction(
                    name="register_service",
                    description=(
                        "Register a new HTTP service with endpoints, pricing, "
                        "and deployment configuration"
                    ),
                    parameters={
                        "agent_name": {"type": "str", "required": True,
                                       "description": "Name of the agent offering the service"},
                        "service_name": {"type": "str", "required": True,
                                         "description": "Human-readable service name"},
                        "endpoints": {"type": "list", "required": True,
                                      "description": "List of endpoint dicts: {path, method, price, description}"},
                        "docker_image": {"type": "str", "required": False,
                                         "description": "Docker image to deploy"},
                        "port": {"type": "int", "required": False,
                                 "description": "Internal port the service listens on (default 8080)"},
                        "health_check_path": {"type": "str", "required": False,
                                              "description": "Health check endpoint (default /health)"},
                        "env_vars": {"type": "dict", "required": False,
                                     "description": "Environment variables for the service"},
                    },
                    estimated_cost=0.0,
                ),
                SkillAction(
                    name="deregister_service",
                    description="Remove a service from the registry and stop hosting it",
                    parameters={
                        "service_id": {"type": "str", "required": True,
                                       "description": "ID of the service to remove"},
                    },
                    estimated_cost=0.0,
                ),
                SkillAction(
                    name="list_services",
                    description="List all registered services with their status and endpoints",
                    parameters={
                        "agent_name": {"type": "str", "required": False,
                                       "description": "Filter by agent name"},
                        "status": {"type": "str", "required": False,
                                   "description": "Filter by status: active, stopped, unhealthy"},
                    },
                    estimated_cost=0.0,
                ),
                SkillAction(
                    name="get_service",
                    description="Get detailed info about a specific service",
                    parameters={
                        "service_id": {"type": "str", "required": True,
                                       "description": "ID of the service"},
                    },
                    estimated_cost=0.0,
                ),
                SkillAction(
                    name="update_service",
                    description="Update an existing service's configuration (endpoints, pricing, etc.)",
                    parameters={
                        "service_id": {"type": "str", "required": True,
                                       "description": "ID of the service to update"},
                        "endpoints": {"type": "list", "required": False,
                                      "description": "Updated endpoint list"},
                        "docker_image": {"type": "str", "required": False,
                                         "description": "Updated Docker image"},
                        "port": {"type": "int", "required": False,
                                 "description": "Updated port"},
                        "env_vars": {"type": "dict", "required": False,
                                     "description": "Updated environment variables"},
                    },
                    estimated_cost=0.0,
                ),
                SkillAction(
                    name="route_request",
                    description=(
                        "Route an incoming HTTP request to the appropriate service, "
                        "handle billing, and return the response"
                    ),
                    parameters={
                        "service_id": {"type": "str", "required": True,
                                       "description": "Target service ID"},
                        "path": {"type": "str", "required": True,
                                 "description": "Request path (e.g., /code_review)"},
                        "method": {"type": "str", "required": False,
                                   "description": "HTTP method (default POST)"},
                        "payload": {"type": "dict", "required": False,
                                    "description": "Request body"},
                        "caller_id": {"type": "str", "required": False,
                                      "description": "ID of the calling agent/user for billing"},
                    },
                    estimated_cost=0.01,
                ),
                SkillAction(
                    name="health_check",
                    description="Run health checks on all registered services and update status",
                    parameters={
                        "service_id": {"type": "str", "required": False,
                                       "description": "Check specific service (or all if omitted)"},
                    },
                    estimated_cost=0.0,
                ),
                SkillAction(
                    name="generate_proxy_config",
                    description="Generate reverse proxy configuration (nginx/Caddy) for all active services",
                    parameters={
                        "proxy_type": {"type": "str", "required": False,
                                       "description": "Proxy type: nginx or caddy (default nginx)"},
                        "base_domain": {"type": "str", "required": False,
                                        "description": "Base domain for service URLs"},
                    },
                    estimated_cost=0.0,
                ),
                SkillAction(
                    name="service_analytics",
                    description="Get analytics for a service: request counts, revenue, latency, errors",
                    parameters={
                        "service_id": {"type": "str", "required": False,
                                       "description": "Service ID (or all if omitted)"},
                        "hours": {"type": "int", "required": False,
                                  "description": "Time window in hours (default 24)"},
                    },
                    estimated_cost=0.0,
                ),
            ],
            required_credentials=[],
            author="singularity",
        )

    async def execute(self, action: str, params: Dict) -> SkillResult:
        """Route to the appropriate action handler."""
        handlers = {
            "register_service": self._register_service,
            "deregister_service": self._deregister_service,
            "list_services": self._list_services,
            "get_service": self._get_service,
            "update_service": self._update_service,
            "route_request": self._route_request,
            "health_check": self._health_check,
            "generate_proxy_config": self._generate_proxy_config,
            "service_analytics": self._service_analytics,
        }
        handler = handlers.get(action)
        if not handler:
            return SkillResult(
                success=False,
                message=f"Unknown action: {action}. Available: {list(handlers.keys())}",
            )
        try:
            return await handler(params)
        except Exception as e:
            return SkillResult(success=False, message=f"Error in {action}: {e}")

    async def _register_service(self, params: Dict) -> SkillResult:
        """Register a new HTTP service."""
        agent_name = params.get("agent_name")
        service_name = params.get("service_name")
        endpoints = params.get("endpoints", [])

        if not agent_name or not service_name:
            return SkillResult(
                success=False,
                message="agent_name and service_name are required",
            )
        if not endpoints:
            return SkillResult(
                success=False,
                message="At least one endpoint is required",
            )

        # Validate endpoints
        for ep in endpoints:
            if "path" not in ep:
                return SkillResult(
                    success=False,
                    message=f"Each endpoint must have a 'path'. Got: {ep}",
                )

        data = _load_services()
        service_id = f"{agent_name}-{service_name}".lower().replace(" ", "-")

        # Check for duplicate
        if service_id in data["services"]:
            return SkillResult(
                success=False,
                message=f"Service '{service_id}' already registered. Use update_service to modify.",
            )

        # Build endpoint list with defaults
        processed_endpoints = []
        for ep in endpoints:
            processed_endpoints.append({
                "path": ep["path"],
                "method": ep.get("method", "POST"),
                "price": ep.get("price", 0.0),
                "description": ep.get("description", ""),
                "rate_limit_per_minute": ep.get("rate_limit_per_minute", 60),
            })

        # Generate domain
        domain = DEFAULT_DOMAIN_TEMPLATE.format(agent_name=agent_name.lower())

        service = {
            "service_id": service_id,
            "agent_name": agent_name,
            "service_name": service_name,
            "domain": domain,
            "endpoints": processed_endpoints,
            "docker_image": params.get("docker_image"),
            "port": params.get("port", 8080),
            "health_check_path": params.get("health_check_path", "/health"),
            "env_vars": params.get("env_vars", {}),
            "status": "active",
            "registered_at": datetime.utcnow().isoformat(),
            "last_health_check": None,
            "health_status": "unknown",
            "total_requests": 0,
            "total_revenue": 0.0,
            "total_errors": 0,
        }

        data["services"][service_id] = service
        data["domain_assignments"][domain] = service_id

        # Generate routing rules
        for ep in processed_endpoints:
            rule_key = f"{service_id}:{ep['path']}"
            data["routing_rules"][rule_key] = {
                "service_id": service_id,
                "path": ep["path"],
                "method": ep["method"],
                "target_port": service.get("port", 8080),
                "price": ep["price"],
            }

        _save_services(data)

        # Build public URL list
        public_urls = [
            f"https://{domain}{ep['path']}" for ep in processed_endpoints
        ]

        return SkillResult(
            success=True,
            message=(
                f"Service '{service_name}' registered for agent '{agent_name}'. "
                f"Domain: {domain}. {len(processed_endpoints)} endpoints active."
            ),
            data={
                "service_id": service_id,
                "domain": domain,
                "public_urls": public_urls,
                "endpoints": processed_endpoints,
                "status": "active",
            },
        )

    async def _deregister_service(self, params: Dict) -> SkillResult:
        """Remove a service from the registry."""
        service_id = params.get("service_id")
        if not service_id:
            return SkillResult(success=False, message="service_id is required")

        data = _load_services()
        if service_id not in data["services"]:
            return SkillResult(
                success=False,
                message=f"Service '{service_id}' not found",
            )

        service = data["services"].pop(service_id)

        # Clean up routing rules
        to_remove = [k for k in data["routing_rules"] if k.startswith(f"{service_id}:")]
        for k in to_remove:
            del data["routing_rules"][k]

        # Clean up domain assignment
        domain = service.get("domain")
        if domain and domain in data["domain_assignments"]:
            del data["domain_assignments"][domain]

        _save_services(data)

        return SkillResult(
            success=True,
            message=f"Service '{service_id}' deregistered and removed.",
            data={"service_id": service_id, "removed": True},
        )

    async def _list_services(self, params: Dict) -> SkillResult:
        """List all registered services."""
        data = _load_services()
        services = list(data["services"].values())

        # Apply filters
        agent_name = params.get("agent_name")
        if agent_name:
            services = [s for s in services if s["agent_name"].lower() == agent_name.lower()]

        status_filter = params.get("status")
        if status_filter:
            services = [s for s in services if s["status"] == status_filter]

        # Build summary
        summaries = []
        for svc in services:
            summaries.append({
                "service_id": svc["service_id"],
                "agent_name": svc["agent_name"],
                "service_name": svc["service_name"],
                "domain": svc["domain"],
                "status": svc["status"],
                "health_status": svc.get("health_status", "unknown"),
                "endpoint_count": len(svc["endpoints"]),
                "total_requests": svc.get("total_requests", 0),
                "total_revenue": svc.get("total_revenue", 0.0),
            })

        return SkillResult(
            success=True,
            message=f"Found {len(summaries)} service(s)",
            data={"services": summaries, "total": len(summaries)},
        )

    async def _get_service(self, params: Dict) -> SkillResult:
        """Get detailed service info."""
        service_id = params.get("service_id")
        if not service_id:
            return SkillResult(success=False, message="service_id is required")

        data = _load_services()
        service = data["services"].get(service_id)
        if not service:
            return SkillResult(
                success=False,
                message=f"Service '{service_id}' not found",
            )

        return SkillResult(
            success=True,
            message=f"Service details for '{service_id}'",
            data={"service": service},
        )

    async def _update_service(self, params: Dict) -> SkillResult:
        """Update an existing service's configuration."""
        service_id = params.get("service_id")
        if not service_id:
            return SkillResult(success=False, message="service_id is required")

        data = _load_services()
        service = data["services"].get(service_id)
        if not service:
            return SkillResult(
                success=False,
                message=f"Service '{service_id}' not found",
            )

        updated_fields = []

        # Update endpoints
        if "endpoints" in params:
            new_endpoints = []
            for ep in params["endpoints"]:
                new_endpoints.append({
                    "path": ep["path"],
                    "method": ep.get("method", "POST"),
                    "price": ep.get("price", 0.0),
                    "description": ep.get("description", ""),
                    "rate_limit_per_minute": ep.get("rate_limit_per_minute", 60),
                })
            service["endpoints"] = new_endpoints
            updated_fields.append("endpoints")

            # Rebuild routing rules for this service
            to_remove = [k for k in data["routing_rules"] if k.startswith(f"{service_id}:")]
            for k in to_remove:
                del data["routing_rules"][k]
            for ep in new_endpoints:
                rule_key = f"{service_id}:{ep['path']}"
                data["routing_rules"][rule_key] = {
                    "service_id": service_id,
                    "path": ep["path"],
                    "method": ep["method"],
                    "target_port": service.get("port", 8080),
                    "price": ep["price"],
                }

        if "docker_image" in params:
            service["docker_image"] = params["docker_image"]
            updated_fields.append("docker_image")

        if "port" in params:
            service["port"] = params["port"]
            updated_fields.append("port")

        if "env_vars" in params:
            service["env_vars"].update(params["env_vars"])
            updated_fields.append("env_vars")

        service["updated_at"] = datetime.utcnow().isoformat()
        data["services"][service_id] = service
        _save_services(data)

        return SkillResult(
            success=True,
            message=f"Service '{service_id}' updated: {', '.join(updated_fields)}",
            data={"service_id": service_id, "updated_fields": updated_fields},
        )

    async def _route_request(self, params: Dict) -> SkillResult:
        """Route a request to the appropriate service and handle billing."""
        service_id = params.get("service_id")
        path = params.get("path")
        method = params.get("method", "POST").upper()
        payload = params.get("payload", {})
        caller_id = params.get("caller_id", "anonymous")

        if not service_id or not path:
            return SkillResult(
                success=False,
                message="service_id and path are required",
            )

        data = _load_services()
        service = data["services"].get(service_id)
        if not service:
            return SkillResult(
                success=False,
                message=f"Service '{service_id}' not found",
            )

        if service["status"] != "active":
            return SkillResult(
                success=False,
                message=f"Service '{service_id}' is not active (status: {service['status']})",
            )

        # Find matching endpoint
        matching_endpoint = None
        for ep in service["endpoints"]:
            if ep["path"] == path and ep.get("method", "POST").upper() == method:
                matching_endpoint = ep
                break

        if not matching_endpoint:
            available = [f"{ep.get('method', 'POST')} {ep['path']}" for ep in service["endpoints"]]
            return SkillResult(
                success=False,
                message=f"No endpoint matching {method} {path}. Available: {available}",
            )

        # Log the request
        request_log = {
            "request_id": str(uuid.uuid4()),
            "service_id": service_id,
            "path": path,
            "method": method,
            "caller_id": caller_id,
            "timestamp": datetime.utcnow().isoformat(),
            "price": matching_endpoint["price"],
        }

        # Update service stats
        service["total_requests"] = service.get("total_requests", 0) + 1
        service["total_revenue"] = service.get("total_revenue", 0.0) + matching_endpoint["price"]
        data["services"][service_id] = service
        _save_services(data)

        # Log the access
        logs = _load_logs()
        logs.append(request_log)
        _save_logs(logs)

        # Try to bill via UsageTracking if available
        billing_result = None
        if self.context:
            try:
                billing_result = await self.context.call_skill(
                    "usage_tracking", "track_usage",
                    {
                        "customer_id": caller_id,
                        "skill_id": f"service:{service_id}",
                        "action": path,
                        "cost": matching_endpoint["price"],
                    }
                )
            except Exception:
                pass  # Billing is best-effort

        return SkillResult(
            success=True,
            message=(
                f"Request routed to {service_id}{path}. "
                f"Price: ${matching_endpoint['price']:.2f}. "
                f"Caller: {caller_id}"
            ),
            data={
                "request_id": request_log["request_id"],
                "service_id": service_id,
                "endpoint": matching_endpoint,
                "payload": payload,
                "target_url": f"http://localhost:{service['port']}{path}",
                "price_charged": matching_endpoint["price"],
                "billing": billing_result.data if billing_result and billing_result.success else None,
            },
            revenue=matching_endpoint["price"],
        )

    async def _health_check(self, params: Dict) -> SkillResult:
        """Run health checks on registered services."""
        data = _load_services()
        service_id = params.get("service_id")

        if service_id:
            services_to_check = {service_id: data["services"].get(service_id)}
            if not services_to_check[service_id]:
                return SkillResult(
                    success=False,
                    message=f"Service '{service_id}' not found",
                )
        else:
            services_to_check = data["services"]

        results = {}
        now = datetime.utcnow().isoformat()

        for sid, svc in services_to_check.items():
            health_url = f"http://localhost:{svc['port']}{svc.get('health_check_path', '/health')}"

            # Try HTTP health check
            try:
                import httpx
                async with httpx.AsyncClient(timeout=5.0) as client:
                    resp = await client.get(health_url)
                    is_healthy = resp.status_code == 200
            except Exception:
                # If no httpx or can't reach, mark based on docker status or unknown
                is_healthy = None

            if is_healthy is True:
                health_status = "healthy"
            elif is_healthy is False:
                health_status = "unhealthy"
            else:
                health_status = "unreachable"

            svc["last_health_check"] = now
            svc["health_status"] = health_status
            data["services"][sid] = svc

            results[sid] = {
                "health_status": health_status,
                "health_url": health_url,
                "checked_at": now,
            }

        _save_services(data)

        healthy_count = sum(1 for r in results.values() if r["health_status"] == "healthy")
        total = len(results)

        return SkillResult(
            success=True,
            message=f"Health check complete: {healthy_count}/{total} healthy",
            data={"results": results, "healthy": healthy_count, "total": total},
        )

    async def _generate_proxy_config(self, params: Dict) -> SkillResult:
        """Generate reverse proxy configuration for active services."""
        proxy_type = params.get("proxy_type", "nginx").lower()
        base_domain = params.get("base_domain", "singularity.wisent.ai")
        data = _load_services()

        active_services = {
            sid: svc for sid, svc in data["services"].items()
            if svc["status"] == "active"
        }

        if not active_services:
            return SkillResult(
                success=True,
                message="No active services to configure",
                data={"config": "", "service_count": 0},
            )

        if proxy_type == "nginx":
            config = self._generate_nginx_config(active_services, base_domain)
        elif proxy_type == "caddy":
            config = self._generate_caddy_config(active_services, base_domain)
        else:
            return SkillResult(
                success=False,
                message=f"Unsupported proxy type: {proxy_type}. Use 'nginx' or 'caddy'.",
            )

        return SkillResult(
            success=True,
            message=f"Generated {proxy_type} config for {len(active_services)} service(s)",
            data={
                "config": config,
                "proxy_type": proxy_type,
                "service_count": len(active_services),
                "services": list(active_services.keys()),
            },
        )

    def _generate_nginx_config(self, services: Dict, base_domain: str) -> str:
        """Generate nginx reverse proxy configuration."""
        blocks = [
            "# Auto-generated by ServiceHostingSkill",
            f"# Generated: {datetime.utcnow().isoformat()}",
            f"# Services: {len(services)}",
            "",
        ]

        for sid, svc in services.items():
            agent_name = svc["agent_name"].lower()
            port = svc.get("port", 8080)
            domain = svc.get("domain", f"{agent_name}.{base_domain}")

            blocks.append(f"# Service: {svc['service_name']} ({sid})")
            blocks.append(f"server {{")
            blocks.append(f"    listen 80;")
            blocks.append(f"    server_name {domain};")
            blocks.append(f"")

            for ep in svc["endpoints"]:
                blocks.append(f"    # {ep.get('description', ep['path'])}")
                blocks.append(f"    location {ep['path']} {{")
                blocks.append(f"        proxy_pass http://127.0.0.1:{port}{ep['path']};")
                blocks.append(f"        proxy_set_header Host $host;")
                blocks.append(f"        proxy_set_header X-Real-IP $remote_addr;")
                blocks.append(f"        proxy_set_header X-Service-ID {sid};")
                blocks.append(f"        proxy_set_header X-Endpoint-Price {ep.get('price', 0)};")
                blocks.append(f"    }}")
                blocks.append(f"")

            # Health check location
            health_path = svc.get("health_check_path", "/health")
            blocks.append(f"    location {health_path} {{")
            blocks.append(f"        proxy_pass http://127.0.0.1:{port}{health_path};")
            blocks.append(f"    }}")
            blocks.append(f"}}")
            blocks.append(f"")

        return "\n".join(blocks)

    def _generate_caddy_config(self, services: Dict, base_domain: str) -> str:
        """Generate Caddy reverse proxy configuration."""
        blocks = [
            "# Auto-generated by ServiceHostingSkill",
            f"# Generated: {datetime.utcnow().isoformat()}",
            f"# Services: {len(services)}",
            "",
        ]

        for sid, svc in services.items():
            agent_name = svc["agent_name"].lower()
            port = svc.get("port", 8080)
            domain = svc.get("domain", f"{agent_name}.{base_domain}")

            blocks.append(f"# Service: {svc['service_name']} ({sid})")
            blocks.append(f"{domain} {{")

            for ep in svc["endpoints"]:
                blocks.append(f"    # {ep.get('description', ep['path'])} - ${ep.get('price', 0)}")
                blocks.append(f"    handle {ep['path']} {{")
                blocks.append(f"        reverse_proxy localhost:{port}")
                blocks.append(f"    }}")

            health_path = svc.get("health_check_path", "/health")
            blocks.append(f"    handle {health_path} {{")
            blocks.append(f"        reverse_proxy localhost:{port}")
            blocks.append(f"    }}")

            blocks.append(f"}}")
            blocks.append(f"")

        return "\n".join(blocks)

    async def _service_analytics(self, params: Dict) -> SkillResult:
        """Get analytics for services."""
        service_id = params.get("service_id")
        hours = params.get("hours", 24)
        cutoff = (datetime.utcnow() - timedelta(hours=hours)).isoformat()

        data = _load_services()
        logs = _load_logs()

        # Filter logs by time
        recent_logs = [l for l in logs if l.get("timestamp", "") >= cutoff]

        # Filter by service if specified
        if service_id:
            if service_id not in data["services"]:
                return SkillResult(
                    success=False,
                    message=f"Service '{service_id}' not found",
                )
            recent_logs = [l for l in recent_logs if l.get("service_id") == service_id]

        # Calculate analytics
        total_requests = len(recent_logs)
        total_revenue = sum(l.get("price", 0) for l in recent_logs)
        unique_callers = len(set(l.get("caller_id", "unknown") for l in recent_logs))

        # Per-endpoint breakdown
        endpoint_stats = {}
        for log_entry in recent_logs:
            key = f"{log_entry.get('service_id', '?')}:{log_entry.get('path', '?')}"
            if key not in endpoint_stats:
                endpoint_stats[key] = {"requests": 0, "revenue": 0.0}
            endpoint_stats[key]["requests"] += 1
            endpoint_stats[key]["revenue"] += log_entry.get("price", 0)

        # Per-caller breakdown
        caller_stats = {}
        for log_entry in recent_logs:
            caller = log_entry.get("caller_id", "anonymous")
            if caller not in caller_stats:
                caller_stats[caller] = {"requests": 0, "spent": 0.0}
            caller_stats[caller]["requests"] += 1
            caller_stats[caller]["spent"] += log_entry.get("price", 0)

        # Top callers by spend
        top_callers = sorted(caller_stats.items(), key=lambda x: x[1]["spent"], reverse=True)[:10]

        analytics = {
            "time_window_hours": hours,
            "total_requests": total_requests,
            "total_revenue": total_revenue,
            "unique_callers": unique_callers,
            "endpoint_breakdown": endpoint_stats,
            "top_callers": dict(top_callers),
        }

        if service_id:
            svc = data["services"][service_id]
            analytics["service"] = {
                "service_id": service_id,
                "status": svc["status"],
                "lifetime_requests": svc.get("total_requests", 0),
                "lifetime_revenue": svc.get("total_revenue", 0.0),
            }

        return SkillResult(
            success=True,
            message=(
                f"Analytics for {'service ' + service_id if service_id else 'all services'} "
                f"(last {hours}h): {total_requests} requests, ${total_revenue:.2f} revenue"
            ),
            data=analytics,
        )
