#!/usr/bin/env python3
"""
WebhookSkill - Inbound webhook endpoint management for external integrations.

Enables external systems (GitHub, Stripe, Zapier, custom apps) to trigger
agent actions via HTTP webhooks. This is the critical bridge between external
events and agent capabilities — a key Revenue Generation enabler.

Features:
- Register named webhook endpoints mapped to skill actions
- HMAC-SHA256 signature verification for security
- Payload transformation with JSONPath-like field mapping
- Event routing to any registered skill action
- Webhook delivery history and replay
- Rate limiting per endpoint
- Filter rules to conditionally process webhooks

Example flow:
1. Register: "github-push" webhook → maps to "revenue_services:code_review"
2. External: GitHub sends POST to /webhooks/github-push
3. Agent: Validates signature, transforms payload, executes code_review
4. Result: Stored in history, optional callback fired

This makes the agent "callable" from any external system, turning it into
a real integration point that customers can wire into their workflows.
"""

import hashlib
import hmac
import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field, asdict

from .base import Skill, SkillManifest, SkillAction, SkillResult


@dataclass
class WebhookEndpoint:
    """A registered webhook endpoint configuration."""
    endpoint_id: str
    name: str
    description: str
    # Target skill action to invoke
    target_skill_id: str
    target_action: str
    # Security
    secret: Optional[str] = None  # HMAC signing secret
    # Payload transformation: maps incoming fields to skill params
    # e.g., {"repository.full_name": "repo", "head_commit.message": "description"}
    field_mapping: Dict[str, str] = field(default_factory=dict)
    # Static params always passed to the target action
    static_params: Dict[str, Any] = field(default_factory=dict)
    # Filter: only process if these conditions match
    # e.g., {"action": "opened", "pull_request.state": "open"}
    filters: Dict[str, Any] = field(default_factory=dict)
    # Rate limiting
    max_calls_per_minute: int = 60
    # State
    enabled: bool = True
    created_at: str = ""
    total_calls: int = 0
    total_successes: int = 0
    total_failures: int = 0
    last_called_at: Optional[str] = None

    def to_dict(self) -> Dict:
        d = asdict(self)
        # Don't expose the secret in list responses
        if d.get("secret"):
            d["secret"] = "***"
        return d


@dataclass
class WebhookDelivery:
    """Record of a webhook delivery (incoming call)."""
    delivery_id: str
    endpoint_id: str
    endpoint_name: str
    received_at: str
    # Incoming data
    headers: Dict[str, str]
    payload: Dict[str, Any]
    source_ip: Optional[str] = None
    # Processing
    signature_valid: Optional[bool] = None
    filters_matched: bool = True
    transformed_params: Optional[Dict[str, Any]] = None
    # Result
    target_skill_id: Optional[str] = None
    target_action: Optional[str] = None
    result_success: Optional[bool] = None
    result_message: Optional[str] = None
    processing_time_ms: Optional[float] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict:
        return asdict(self)


class WebhookSkill(Skill):
    """Skill for managing inbound webhook endpoints and processing deliveries."""

    def __init__(self, credentials: Dict[str, str] = None, data_dir: str = None):
        super().__init__(credentials)
        self._data_dir = Path(data_dir) if data_dir else Path("singularity/data")
        self._endpoints: Dict[str, WebhookEndpoint] = {}
        self._deliveries: List[WebhookDelivery] = []
        self._rate_tracker: Dict[str, List[float]] = {}  # endpoint_id -> list of timestamps
        self._max_deliveries = 500  # Keep last N deliveries
        self._load_data()

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="webhook",
            name="Webhook Manager",
            version="1.0.0",
            category="integration",
            description=(
                "Manage inbound webhook endpoints for external integrations. "
                "Register webhooks that map external events (GitHub, Stripe, etc.) "
                "to agent skill actions. Includes HMAC signature verification, "
                "payload transformation, filtering, and delivery history."
            ),
            actions=[
                SkillAction(
                    name="register",
                    description=(
                        "Register a new webhook endpoint. Maps an external event source "
                        "to a target skill action with optional payload transformation."
                    ),
                    parameters={
                        "name": {"type": "string", "required": True,
                                 "description": "Unique name for the webhook (used in URL path)"},
                        "description": {"type": "string", "required": False,
                                        "description": "Human-readable description"},
                        "target_skill_id": {"type": "string", "required": True,
                                            "description": "Skill to invoke when webhook fires"},
                        "target_action": {"type": "string", "required": True,
                                          "description": "Action within the target skill"},
                        "secret": {"type": "string", "required": False,
                                   "description": "HMAC-SHA256 signing secret for verification"},
                        "field_mapping": {"type": "object", "required": False,
                                          "description": "Map payload fields to action params"},
                        "static_params": {"type": "object", "required": False,
                                          "description": "Static params always sent to action"},
                        "filters": {"type": "object", "required": False,
                                    "description": "Only process if payload matches these values"},
                        "max_calls_per_minute": {"type": "integer", "required": False,
                                                 "description": "Rate limit (default 60)"},
                    },
                ),
                SkillAction(
                    name="receive",
                    description=(
                        "Process an incoming webhook delivery. Validates signature, "
                        "applies filters, transforms payload, and routes to target skill."
                    ),
                    parameters={
                        "endpoint_name": {"type": "string", "required": True,
                                          "description": "Name of the registered webhook endpoint"},
                        "payload": {"type": "object", "required": True,
                                    "description": "The incoming webhook payload"},
                        "headers": {"type": "object", "required": False,
                                    "description": "HTTP headers from the request"},
                        "signature": {"type": "string", "required": False,
                                      "description": "HMAC signature from the request"},
                        "source_ip": {"type": "string", "required": False,
                                      "description": "Source IP of the request"},
                    },
                ),
                SkillAction(
                    name="list_endpoints",
                    description="List all registered webhook endpoints with their stats.",
                    parameters={
                        "enabled_only": {"type": "boolean", "required": False,
                                         "description": "Only show enabled endpoints"},
                    },
                ),
                SkillAction(
                    name="get_endpoint",
                    description="Get details of a specific webhook endpoint.",
                    parameters={
                        "name": {"type": "string", "required": True,
                                 "description": "Endpoint name"},
                    },
                ),
                SkillAction(
                    name="update_endpoint",
                    description="Update an existing webhook endpoint configuration.",
                    parameters={
                        "name": {"type": "string", "required": True,
                                 "description": "Endpoint name to update"},
                        "enabled": {"type": "boolean", "required": False,
                                    "description": "Enable/disable the endpoint"},
                        "target_skill_id": {"type": "string", "required": False,
                                            "description": "New target skill"},
                        "target_action": {"type": "string", "required": False,
                                          "description": "New target action"},
                        "secret": {"type": "string", "required": False,
                                   "description": "New signing secret"},
                        "field_mapping": {"type": "object", "required": False,
                                          "description": "New field mapping"},
                        "static_params": {"type": "object", "required": False,
                                          "description": "New static params"},
                        "filters": {"type": "object", "required": False,
                                    "description": "New filter rules"},
                        "max_calls_per_minute": {"type": "integer", "required": False,
                                                 "description": "New rate limit"},
                    },
                ),
                SkillAction(
                    name="delete_endpoint",
                    description="Delete a webhook endpoint.",
                    parameters={
                        "name": {"type": "string", "required": True,
                                 "description": "Endpoint name to delete"},
                    },
                ),
                SkillAction(
                    name="get_deliveries",
                    description="Get webhook delivery history for debugging and auditing.",
                    parameters={
                        "endpoint_name": {"type": "string", "required": False,
                                          "description": "Filter by endpoint name"},
                        "success_only": {"type": "boolean", "required": False,
                                         "description": "Only show successful deliveries"},
                        "limit": {"type": "integer", "required": False,
                                  "description": "Max results (default 20)"},
                    },
                ),
                SkillAction(
                    name="replay",
                    description="Replay a past webhook delivery (re-process with current config).",
                    parameters={
                        "delivery_id": {"type": "string", "required": True,
                                        "description": "ID of the delivery to replay"},
                    },
                ),
                SkillAction(
                    name="test_endpoint",
                    description="Send a test payload to a webhook endpoint to verify configuration.",
                    parameters={
                        "name": {"type": "string", "required": True,
                                 "description": "Endpoint name to test"},
                        "payload": {"type": "object", "required": False,
                                    "description": "Test payload (uses sample if omitted)"},
                    },
                ),
            ],
            required_credentials=[],
        )

    async def execute(self, action: str, params: Dict) -> SkillResult:
        actions = {
            "register": self._register,
            "receive": self._receive,
            "list_endpoints": self._list_endpoints,
            "get_endpoint": self._get_endpoint,
            "update_endpoint": self._update_endpoint,
            "delete_endpoint": self._delete_endpoint,
            "get_deliveries": self._get_deliveries,
            "replay": self._replay,
            "test_endpoint": self._test_endpoint,
        }
        handler = actions.get(action)
        if not handler:
            return SkillResult(
                success=False,
                message=f"Unknown action: {action}. Available: {list(actions.keys())}"
            )
        return await handler(params)

    # --- Core Actions ---

    async def _register(self, params: Dict) -> SkillResult:
        """Register a new webhook endpoint."""
        name = params.get("name", "").strip()
        if not name:
            return SkillResult(success=False, message="Endpoint name is required")

        # Check for duplicate names
        for ep in self._endpoints.values():
            if ep.name == name:
                return SkillResult(
                    success=False,
                    message=f"Endpoint '{name}' already exists. Use update_endpoint to modify."
                )

        target_skill = params.get("target_skill_id", "").strip()
        target_action = params.get("target_action", "").strip()
        if not target_skill or not target_action:
            return SkillResult(
                success=False,
                message="target_skill_id and target_action are required"
            )

        endpoint = WebhookEndpoint(
            endpoint_id=str(uuid.uuid4()),
            name=name,
            description=params.get("description", ""),
            target_skill_id=target_skill,
            target_action=target_action,
            secret=params.get("secret"),
            field_mapping=params.get("field_mapping", {}),
            static_params=params.get("static_params", {}),
            filters=params.get("filters", {}),
            max_calls_per_minute=params.get("max_calls_per_minute", 60),
            enabled=True,
            created_at=datetime.utcnow().isoformat(),
        )

        self._endpoints[endpoint.endpoint_id] = endpoint
        self._save_data()

        return SkillResult(
            success=True,
            message=f"Webhook endpoint '{name}' registered. URL path: /webhooks/{name}",
            data={
                "endpoint_id": endpoint.endpoint_id,
                "name": name,
                "url_path": f"/webhooks/{name}",
                "target": f"{target_skill}:{target_action}",
                "has_secret": bool(endpoint.secret),
            }
        )

    async def _receive(self, params: Dict) -> SkillResult:
        """Process an incoming webhook delivery."""
        endpoint_name = params.get("endpoint_name", "").strip()
        payload = params.get("payload", {})
        headers = params.get("headers", {})
        signature = params.get("signature")
        source_ip = params.get("source_ip")

        # Find the endpoint
        endpoint = self._find_endpoint_by_name(endpoint_name)
        if not endpoint:
            return SkillResult(
                success=False,
                message=f"Webhook endpoint '{endpoint_name}' not found"
            )

        if not endpoint.enabled:
            return SkillResult(
                success=False,
                message=f"Webhook endpoint '{endpoint_name}' is disabled"
            )

        start_time = time.monotonic()

        # Create delivery record
        delivery = WebhookDelivery(
            delivery_id=str(uuid.uuid4()),
            endpoint_id=endpoint.endpoint_id,
            endpoint_name=endpoint_name,
            received_at=datetime.utcnow().isoformat(),
            headers=headers,
            payload=payload,
            source_ip=source_ip,
            target_skill_id=endpoint.target_skill_id,
            target_action=endpoint.target_action,
        )

        # Rate limit check
        if not self._check_rate_limit(endpoint):
            delivery.error = "Rate limit exceeded"
            delivery.processing_time_ms = (time.monotonic() - start_time) * 1000
            self._record_delivery(delivery)
            endpoint.total_calls += 1
            endpoint.total_failures += 1
            self._save_data()
            return SkillResult(
                success=False,
                message=f"Rate limit exceeded for '{endpoint_name}' "
                        f"(max {endpoint.max_calls_per_minute}/min)"
            )

        # Signature verification
        if endpoint.secret:
            is_valid = self._verify_signature(payload, endpoint.secret, signature)
            delivery.signature_valid = is_valid
            if not is_valid:
                delivery.error = "Invalid signature"
                delivery.processing_time_ms = (time.monotonic() - start_time) * 1000
                self._record_delivery(delivery)
                endpoint.total_calls += 1
                endpoint.total_failures += 1
                endpoint.last_called_at = datetime.utcnow().isoformat()
                self._save_data()
                return SkillResult(
                    success=False,
                    message="Webhook signature verification failed"
                )

        # Apply filters
        if endpoint.filters and not self._check_filters(payload, endpoint.filters):
            delivery.filters_matched = False
            delivery.processing_time_ms = (time.monotonic() - start_time) * 1000
            self._record_delivery(delivery)
            endpoint.total_calls += 1
            endpoint.last_called_at = datetime.utcnow().isoformat()
            self._save_data()
            return SkillResult(
                success=True,
                message=f"Webhook received but filtered out (filters didn't match)",
                data={"filtered": True, "delivery_id": delivery.delivery_id}
            )

        # Transform payload to action params
        action_params = self._transform_payload(
            payload, endpoint.field_mapping, endpoint.static_params
        )
        delivery.transformed_params = action_params

        # Execute the target skill action
        result = await self._route_to_skill(
            endpoint.target_skill_id, endpoint.target_action, action_params
        )

        elapsed_ms = (time.monotonic() - start_time) * 1000
        delivery.result_success = result.success
        delivery.result_message = result.message
        delivery.processing_time_ms = elapsed_ms

        # Update endpoint stats
        endpoint.total_calls += 1
        if result.success:
            endpoint.total_successes += 1
        else:
            endpoint.total_failures += 1
        endpoint.last_called_at = datetime.utcnow().isoformat()

        self._record_delivery(delivery)
        self._save_data()

        return SkillResult(
            success=result.success,
            message=f"Webhook '{endpoint_name}' → {endpoint.target_skill_id}:{endpoint.target_action}: {result.message}",
            data={
                "delivery_id": delivery.delivery_id,
                "target": f"{endpoint.target_skill_id}:{endpoint.target_action}",
                "params_sent": action_params,
                "result": result.data,
                "processing_time_ms": elapsed_ms,
            }
        )

    async def _list_endpoints(self, params: Dict) -> SkillResult:
        """List all registered webhook endpoints."""
        enabled_only = params.get("enabled_only", False)

        endpoints = list(self._endpoints.values())
        if enabled_only:
            endpoints = [ep for ep in endpoints if ep.enabled]

        return SkillResult(
            success=True,
            message=f"Found {len(endpoints)} webhook endpoints",
            data={
                "endpoints": [ep.to_dict() for ep in endpoints],
                "total": len(endpoints),
            }
        )

    async def _get_endpoint(self, params: Dict) -> SkillResult:
        """Get details of a specific endpoint."""
        name = params.get("name", "").strip()
        endpoint = self._find_endpoint_by_name(name)
        if not endpoint:
            return SkillResult(success=False, message=f"Endpoint '{name}' not found")

        # Include recent deliveries for this endpoint
        recent = [
            d.to_dict() for d in self._deliveries
            if d.endpoint_name == name
        ][-10:]

        data = endpoint.to_dict()
        data["recent_deliveries"] = recent

        return SkillResult(success=True, message=f"Endpoint '{name}' details", data=data)

    async def _update_endpoint(self, params: Dict) -> SkillResult:
        """Update an existing endpoint."""
        name = params.get("name", "").strip()
        endpoint = self._find_endpoint_by_name(name)
        if not endpoint:
            return SkillResult(success=False, message=f"Endpoint '{name}' not found")

        updated_fields = []
        for field_name in ["enabled", "target_skill_id", "target_action", "secret",
                           "field_mapping", "static_params", "filters", "max_calls_per_minute"]:
            if field_name in params:
                setattr(endpoint, field_name, params[field_name])
                updated_fields.append(field_name)

        self._save_data()

        return SkillResult(
            success=True,
            message=f"Updated endpoint '{name}': {', '.join(updated_fields)}",
            data={"updated_fields": updated_fields, "endpoint": endpoint.to_dict()}
        )

    async def _delete_endpoint(self, params: Dict) -> SkillResult:
        """Delete a webhook endpoint."""
        name = params.get("name", "").strip()
        endpoint = self._find_endpoint_by_name(name)
        if not endpoint:
            return SkillResult(success=False, message=f"Endpoint '{name}' not found")

        del self._endpoints[endpoint.endpoint_id]
        self._save_data()

        return SkillResult(
            success=True,
            message=f"Deleted webhook endpoint '{name}'",
            data={"deleted_endpoint_id": endpoint.endpoint_id, "name": name}
        )

    async def _get_deliveries(self, params: Dict) -> SkillResult:
        """Get webhook delivery history."""
        endpoint_name = params.get("endpoint_name")
        success_only = params.get("success_only", False)
        limit = params.get("limit", 20)

        deliveries = list(self._deliveries)

        if endpoint_name:
            deliveries = [d for d in deliveries if d.endpoint_name == endpoint_name]
        if success_only:
            deliveries = [d for d in deliveries if d.result_success is True]

        # Most recent first
        deliveries = deliveries[-limit:]
        deliveries.reverse()

        return SkillResult(
            success=True,
            message=f"Found {len(deliveries)} deliveries",
            data={
                "deliveries": [d.to_dict() for d in deliveries],
                "total": len(deliveries),
            }
        )

    async def _replay(self, params: Dict) -> SkillResult:
        """Replay a past webhook delivery."""
        delivery_id = params.get("delivery_id", "").strip()

        original = None
        for d in self._deliveries:
            if d.delivery_id == delivery_id:
                original = d
                break

        if not original:
            return SkillResult(
                success=False,
                message=f"Delivery '{delivery_id}' not found"
            )

        # Re-process with current endpoint config
        return await self._receive({
            "endpoint_name": original.endpoint_name,
            "payload": original.payload,
            "headers": original.headers,
            "signature": None,  # Skip signature check on replay
            "source_ip": original.source_ip,
        })

    async def _test_endpoint(self, params: Dict) -> SkillResult:
        """Send a test payload to an endpoint."""
        name = params.get("name", "").strip()
        endpoint = self._find_endpoint_by_name(name)
        if not endpoint:
            return SkillResult(success=False, message=f"Endpoint '{name}' not found")

        test_payload = params.get("payload", {
            "_test": True,
            "message": "Test webhook delivery",
            "timestamp": datetime.utcnow().isoformat(),
        })

        # Generate valid signature for the test
        signature = None
        if endpoint.secret:
            payload_bytes = json.dumps(test_payload, sort_keys=True).encode()
            signature = hmac.new(
                endpoint.secret.encode(), payload_bytes, hashlib.sha256
            ).hexdigest()

        return await self._receive({
            "endpoint_name": name,
            "payload": test_payload,
            "headers": {"X-Test": "true"},
            "signature": signature,
            "source_ip": "127.0.0.1",
        })

    # --- Helper Methods ---

    def _find_endpoint_by_name(self, name: str) -> Optional[WebhookEndpoint]:
        """Find an endpoint by its name."""
        for ep in self._endpoints.values():
            if ep.name == name:
                return ep
        return None

    def _verify_signature(self, payload: Dict, secret: str, signature: Optional[str]) -> bool:
        """Verify HMAC-SHA256 signature of the payload."""
        if not signature:
            return False
        try:
            payload_bytes = json.dumps(payload, sort_keys=True).encode()
            expected = hmac.new(
                secret.encode(), payload_bytes, hashlib.sha256
            ).hexdigest()
            return hmac.compare_digest(expected, signature)
        except Exception:
            return False

    def _check_filters(self, payload: Dict, filters: Dict[str, Any]) -> bool:
        """Check if payload matches all filter conditions."""
        for path, expected_value in filters.items():
            actual = self._extract_field(payload, path)
            if actual != expected_value:
                return False
        return True

    def _transform_payload(
        self, payload: Dict, field_mapping: Dict[str, str],
        static_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Transform incoming payload to target action params."""
        result = dict(static_params)  # Start with static params

        if field_mapping:
            for source_path, target_param in field_mapping.items():
                value = self._extract_field(payload, source_path)
                if value is not None:
                    result[target_param] = value
        elif not static_params:
            # No mapping defined — pass the whole payload as 'payload'
            result["payload"] = payload

        return result

    def _extract_field(self, data: Any, path: str) -> Any:
        """Extract a nested field using dot notation (e.g., 'repository.full_name')."""
        parts = path.split(".")
        current = data
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None
        return current

    def _check_rate_limit(self, endpoint: WebhookEndpoint) -> bool:
        """Check if the endpoint is within its rate limit."""
        now = time.monotonic()
        window = 60.0  # 1 minute window

        if endpoint.endpoint_id not in self._rate_tracker:
            self._rate_tracker[endpoint.endpoint_id] = []

        # Clean old entries
        timestamps = self._rate_tracker[endpoint.endpoint_id]
        timestamps[:] = [t for t in timestamps if now - t < window]

        if len(timestamps) >= endpoint.max_calls_per_minute:
            return False

        timestamps.append(now)
        return True

    async def _route_to_skill(
        self, skill_id: str, action: str, params: Dict
    ) -> SkillResult:
        """Route the transformed payload to the target skill action."""
        if self.context:
            return await self.context.call_skill(skill_id, action, params)
        else:
            # No context available — return a simulated success for testing
            return SkillResult(
                success=True,
                message=f"Routed to {skill_id}:{action} (no context - dry run)",
                data={"params": params, "dry_run": True}
            )

    def _record_delivery(self, delivery: WebhookDelivery):
        """Record a delivery and trim history."""
        self._deliveries.append(delivery)
        if len(self._deliveries) > self._max_deliveries:
            self._deliveries = self._deliveries[-self._max_deliveries:]

    # --- Persistence ---

    def _get_data_path(self) -> Path:
        return self._data_dir / "webhooks.json"

    def _save_data(self):
        """Save endpoints to disk."""
        try:
            self._data_dir.mkdir(parents=True, exist_ok=True)
            data = {
                "endpoints": {
                    eid: asdict(ep) for eid, ep in self._endpoints.items()
                },
            }
            with open(self._get_data_path(), "w") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    def _load_data(self):
        """Load endpoints from disk."""
        path = self._get_data_path()
        if not path.exists():
            return
        try:
            with open(path) as f:
                data = json.load(f)
            for eid, ep_data in data.get("endpoints", {}).items():
                self._endpoints[eid] = WebhookEndpoint(**ep_data)
        except Exception:
            pass
