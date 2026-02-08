#!/usr/bin/env python3
"""
API Gateway Skill - API key management, rate limiting, and per-key usage tracking.

This skill is the critical infrastructure for monetizing the agent's services.
It manages the full lifecycle of API keys: creation, revocation, scope-based
permissions, per-key rate limits, and detailed usage tracking for billing.

Pillar: Revenue Generation
- Without proper API key management, you can't give customers access
- Without rate limiting, a single user can exhaust all resources
- Without per-key usage tracking, you can't bill customers accurately
"""

import hashlib
import json
import os
import secrets
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from singularity.skills.base import (
    Skill,
    SkillAction,
    SkillManifest,
    SkillResult,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


@dataclass
class APIKey:
    """Represents a managed API key with scopes and limits."""
    key_id: str
    key_hash: str  # SHA-256 hash of the actual key (never store plaintext)
    name: str
    owner: str
    scopes: List[str]  # e.g. ["skills:read", "tasks:write", "admin"]
    rate_limit: int  # requests per minute (0 = unlimited)
    daily_limit: int  # requests per day (0 = unlimited)
    created_at: str
    expires_at: Optional[str] = None
    revoked: bool = False
    revoked_at: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UsageRecord:
    """Tracks API usage per key."""
    key_id: str
    total_requests: int = 0
    total_errors: int = 0
    total_cost: float = 0.0
    total_revenue: float = 0.0
    requests_today: int = 0
    today_date: str = ""
    last_request_at: Optional[str] = None
    endpoint_counts: Dict[str, int] = field(default_factory=dict)
    daily_history: List[Dict[str, Any]] = field(default_factory=list)


class RateLimiter:
    """Token bucket rate limiter per API key."""

    def __init__(self):
        # key_id -> list of request timestamps (within window)
        self._windows: Dict[str, List[float]] = {}
        self._window_seconds: int = 60  # 1-minute window

    def check(self, key_id: str, rate_limit: int) -> tuple:
        """
        Check if request is allowed under rate limit.
        Returns (allowed: bool, remaining: int, reset_at: float).
        """
        if rate_limit <= 0:
            return (True, -1, 0)  # unlimited

        now = time.time()
        cutoff = now - self._window_seconds

        # Clean old entries
        if key_id not in self._windows:
            self._windows[key_id] = []
        self._windows[key_id] = [t for t in self._windows[key_id] if t > cutoff]

        current_count = len(self._windows[key_id])
        remaining = max(0, rate_limit - current_count)

        if current_count >= rate_limit:
            # Find when the oldest request in window expires
            reset_at = self._windows[key_id][0] + self._window_seconds
            return (False, 0, reset_at)

        return (True, remaining - 1, now + self._window_seconds)

    def record(self, key_id: str):
        """Record a request for rate limiting."""
        if key_id not in self._windows:
            self._windows[key_id] = []
        self._windows[key_id].append(time.time())

    def reset(self, key_id: str):
        """Reset rate limit window for a key."""
        self._windows.pop(key_id, None)


class APIGatewaySkill(Skill):
    """
    API Gateway for managing keys, rate limiting, and usage tracking.

    Actions:
    - create_key: Generate a new API key with scopes and limits
    - revoke_key: Revoke an existing API key
    - list_keys: List all API keys (metadata only, not the actual keys)
    - get_key: Get details for a specific key
    - update_key: Update key settings (scopes, limits, metadata)
    - check_access: Validate a key and check if action is permitted
    - record_usage: Record an API call for billing/tracking
    - get_usage: Get usage statistics for a key
    - get_billing: Get billing summary across all keys
    - rotate_key: Generate a new key value while keeping the same key_id
    """

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self._keys: Dict[str, APIKey] = {}
        self._usage: Dict[str, UsageRecord] = {}
        self._rate_limiter = RateLimiter()
        self._persist_path = os.path.join(DATA_DIR, "api_gateway.json")
        self._load()

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="api_gateway",
            name="API Gateway",
            version="1.0.0",
            category="revenue",
            description="API key management, rate limiting, and per-key usage tracking for monetizing agent services",
            actions=[
                SkillAction(
                    name="create_key",
                    description="Create a new API key with scopes and rate limits",
                    parameters={
                        "name": {"type": "string", "required": True, "description": "Human-readable name for the key"},
                        "owner": {"type": "string", "required": True, "description": "Owner/customer identifier"},
                        "scopes": {"type": "list", "required": False, "description": "Permission scopes (default: ['skills:read', 'tasks:write'])"},
                        "rate_limit": {"type": "int", "required": False, "description": "Max requests per minute (0=unlimited, default=60)"},
                        "daily_limit": {"type": "int", "required": False, "description": "Max requests per day (0=unlimited, default=1000)"},
                        "expires_in_days": {"type": "int", "required": False, "description": "Days until expiry (0=never, default=0)"},
                        "metadata": {"type": "dict", "required": False, "description": "Custom metadata (plan tier, etc.)"},
                    },
                ),
                SkillAction(
                    name="revoke_key",
                    description="Revoke an API key immediately",
                    parameters={
                        "key_id": {"type": "string", "required": True, "description": "ID of the key to revoke"},
                    },
                ),
                SkillAction(
                    name="list_keys",
                    description="List all API keys with their metadata (not the actual key values)",
                    parameters={
                        "include_revoked": {"type": "bool", "required": False, "description": "Include revoked keys (default: False)"},
                        "owner": {"type": "string", "required": False, "description": "Filter by owner"},
                    },
                ),
                SkillAction(
                    name="get_key",
                    description="Get details for a specific API key by ID",
                    parameters={
                        "key_id": {"type": "string", "required": True, "description": "ID of the key"},
                    },
                ),
                SkillAction(
                    name="update_key",
                    description="Update API key settings (scopes, rate limits, metadata)",
                    parameters={
                        "key_id": {"type": "string", "required": True, "description": "ID of the key to update"},
                        "scopes": {"type": "list", "required": False, "description": "New scopes"},
                        "rate_limit": {"type": "int", "required": False, "description": "New rate limit"},
                        "daily_limit": {"type": "int", "required": False, "description": "New daily limit"},
                        "metadata": {"type": "dict", "required": False, "description": "Metadata to merge"},
                    },
                ),
                SkillAction(
                    name="check_access",
                    description="Validate an API key and check if a specific action is permitted",
                    parameters={
                        "api_key": {"type": "string", "required": True, "description": "The raw API key to validate"},
                        "required_scope": {"type": "string", "required": False, "description": "Scope required for this action"},
                    },
                ),
                SkillAction(
                    name="record_usage",
                    description="Record an API call for usage tracking and billing",
                    parameters={
                        "key_id": {"type": "string", "required": True, "description": "ID of the API key"},
                        "endpoint": {"type": "string", "required": False, "description": "Endpoint/action called"},
                        "cost": {"type": "float", "required": False, "description": "Cost of this call"},
                        "revenue": {"type": "float", "required": False, "description": "Revenue from this call"},
                        "error": {"type": "bool", "required": False, "description": "Whether the call resulted in an error"},
                    },
                ),
                SkillAction(
                    name="get_usage",
                    description="Get usage statistics for a specific API key",
                    parameters={
                        "key_id": {"type": "string", "required": True, "description": "ID of the API key"},
                    },
                ),
                SkillAction(
                    name="get_billing",
                    description="Get billing summary across all keys with revenue and cost breakdown",
                    parameters={
                        "owner": {"type": "string", "required": False, "description": "Filter by owner"},
                    },
                ),
                SkillAction(
                    name="rotate_key",
                    description="Rotate an API key - generates new key value, keeps same key_id and settings",
                    parameters={
                        "key_id": {"type": "string", "required": True, "description": "ID of the key to rotate"},
                    },
                ),
            ],
            required_credentials=[],
        )

    async def execute(self, action: str, params: Dict) -> SkillResult:
        handlers = {
            "create_key": self._create_key,
            "revoke_key": self._revoke_key,
            "list_keys": self._list_keys,
            "get_key": self._get_key,
            "update_key": self._update_key,
            "check_access": self._check_access,
            "record_usage": self._record_usage,
            "get_usage": self._get_usage,
            "get_billing": self._get_billing,
            "rotate_key": self._rotate_key,
        }
        handler = handlers.get(action)
        if not handler:
            return SkillResult(success=False, message=f"Unknown action: {action}")
        try:
            return await handler(params)
        except Exception as e:
            return SkillResult(success=False, message=f"Error in {action}: {e}")

    # ── Key Management ──────────────────────────────────────────

    async def _create_key(self, params: Dict) -> SkillResult:
        name = params.get("name")
        owner = params.get("owner")
        if not name or not owner:
            return SkillResult(success=False, message="'name' and 'owner' are required")

        scopes = params.get("scopes", ["skills:read", "tasks:write"])
        rate_limit = int(params.get("rate_limit", 60))
        daily_limit = int(params.get("daily_limit", 1000))
        expires_in_days = int(params.get("expires_in_days", 0))
        metadata = params.get("metadata", {})

        # Generate key
        raw_key = f"sg_{secrets.token_urlsafe(32)}"
        key_id = f"key_{secrets.token_hex(8)}"
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()

        now = datetime.utcnow()
        expires_at = None
        if expires_in_days > 0:
            expires_at = (now + timedelta(days=expires_in_days)).isoformat()

        api_key = APIKey(
            key_id=key_id,
            key_hash=key_hash,
            name=name,
            owner=owner,
            scopes=scopes,
            rate_limit=rate_limit,
            daily_limit=daily_limit,
            created_at=now.isoformat(),
            expires_at=expires_at,
            metadata=metadata,
        )

        self._keys[key_id] = api_key
        self._usage[key_id] = UsageRecord(key_id=key_id)
        self._save()

        return SkillResult(
            success=True,
            message=f"API key created: {key_id}",
            data={
                "key_id": key_id,
                "api_key": raw_key,  # Only returned once at creation time!
                "name": name,
                "owner": owner,
                "scopes": scopes,
                "rate_limit": rate_limit,
                "daily_limit": daily_limit,
                "expires_at": expires_at,
            },
        )

    async def _revoke_key(self, params: Dict) -> SkillResult:
        key_id = params.get("key_id")
        if not key_id or key_id not in self._keys:
            return SkillResult(success=False, message=f"Key not found: {key_id}")

        key = self._keys[key_id]
        if key.revoked:
            return SkillResult(success=False, message=f"Key already revoked: {key_id}")

        key.revoked = True
        key.revoked_at = datetime.utcnow().isoformat()
        self._rate_limiter.reset(key_id)
        self._save()

        return SkillResult(
            success=True,
            message=f"Key {key_id} revoked",
            data={"key_id": key_id, "revoked_at": key.revoked_at},
        )

    async def _list_keys(self, params: Dict) -> SkillResult:
        include_revoked = params.get("include_revoked", False)
        owner_filter = params.get("owner")

        keys = []
        for key in self._keys.values():
            if not include_revoked and key.revoked:
                continue
            if owner_filter and key.owner != owner_filter:
                continue
            keys.append({
                "key_id": key.key_id,
                "name": key.name,
                "owner": key.owner,
                "scopes": key.scopes,
                "rate_limit": key.rate_limit,
                "daily_limit": key.daily_limit,
                "created_at": key.created_at,
                "expires_at": key.expires_at,
                "revoked": key.revoked,
            })

        return SkillResult(
            success=True,
            message=f"Found {len(keys)} keys",
            data={"keys": keys, "total": len(keys)},
        )

    async def _get_key(self, params: Dict) -> SkillResult:
        key_id = params.get("key_id")
        if not key_id or key_id not in self._keys:
            return SkillResult(success=False, message=f"Key not found: {key_id}")

        key = self._keys[key_id]
        usage = self._usage.get(key_id, UsageRecord(key_id=key_id))

        return SkillResult(
            success=True,
            message=f"Key: {key.name}",
            data={
                "key_id": key.key_id,
                "name": key.name,
                "owner": key.owner,
                "scopes": key.scopes,
                "rate_limit": key.rate_limit,
                "daily_limit": key.daily_limit,
                "created_at": key.created_at,
                "expires_at": key.expires_at,
                "revoked": key.revoked,
                "usage": {
                    "total_requests": usage.total_requests,
                    "total_errors": usage.total_errors,
                    "total_cost": usage.total_cost,
                    "total_revenue": usage.total_revenue,
                    "requests_today": usage.requests_today,
                },
            },
        )

    async def _update_key(self, params: Dict) -> SkillResult:
        key_id = params.get("key_id")
        if not key_id or key_id not in self._keys:
            return SkillResult(success=False, message=f"Key not found: {key_id}")

        key = self._keys[key_id]
        if key.revoked:
            return SkillResult(success=False, message=f"Cannot update revoked key: {key_id}")

        updated = []
        if "scopes" in params:
            key.scopes = params["scopes"]
            updated.append("scopes")
        if "rate_limit" in params:
            key.rate_limit = int(params["rate_limit"])
            updated.append("rate_limit")
        if "daily_limit" in params:
            key.daily_limit = int(params["daily_limit"])
            updated.append("daily_limit")
        if "metadata" in params:
            key.metadata.update(params["metadata"])
            updated.append("metadata")

        self._save()
        return SkillResult(
            success=True,
            message=f"Updated key {key_id}: {', '.join(updated)}",
            data={"key_id": key_id, "updated_fields": updated},
        )

    async def _rotate_key(self, params: Dict) -> SkillResult:
        key_id = params.get("key_id")
        if not key_id or key_id not in self._keys:
            return SkillResult(success=False, message=f"Key not found: {key_id}")

        key = self._keys[key_id]
        if key.revoked:
            return SkillResult(success=False, message=f"Cannot rotate revoked key: {key_id}")

        # Generate new key value
        raw_key = f"sg_{secrets.token_urlsafe(32)}"
        key.key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        self._rate_limiter.reset(key_id)
        self._save()

        return SkillResult(
            success=True,
            message=f"Key {key_id} rotated. New key issued.",
            data={
                "key_id": key_id,
                "api_key": raw_key,  # Only returned once!
                "name": key.name,
            },
        )

    # ── Access Control ──────────────────────────────────────────

    async def _check_access(self, params: Dict) -> SkillResult:
        raw_key = params.get("api_key")
        required_scope = params.get("required_scope")

        if not raw_key:
            return SkillResult(
                success=False,
                message="No API key provided",
                data={"allowed": False, "reason": "missing_key"},
            )

        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()

        # Find the key
        matched_key = None
        for key in self._keys.values():
            if key.key_hash == key_hash:
                matched_key = key
                break

        if not matched_key:
            return SkillResult(
                success=False,
                message="Invalid API key",
                data={"allowed": False, "reason": "invalid_key"},
            )

        # Check revocation
        if matched_key.revoked:
            return SkillResult(
                success=False,
                message="API key has been revoked",
                data={"allowed": False, "reason": "revoked", "key_id": matched_key.key_id},
            )

        # Check expiration
        if matched_key.expires_at:
            expires = datetime.fromisoformat(matched_key.expires_at)
            if datetime.utcnow() > expires:
                return SkillResult(
                    success=False,
                    message="API key has expired",
                    data={"allowed": False, "reason": "expired", "key_id": matched_key.key_id},
                )

        # Check scope
        if required_scope:
            # Support wildcard scopes: "admin" grants everything
            has_scope = (
                "admin" in matched_key.scopes
                or required_scope in matched_key.scopes
                or self._scope_matches(required_scope, matched_key.scopes)
            )
            if not has_scope:
                return SkillResult(
                    success=False,
                    message=f"Key lacks required scope: {required_scope}",
                    data={
                        "allowed": False,
                        "reason": "insufficient_scope",
                        "key_id": matched_key.key_id,
                        "required": required_scope,
                        "available": matched_key.scopes,
                    },
                )

        # Check rate limit
        allowed, remaining, reset_at = self._rate_limiter.check(
            matched_key.key_id, matched_key.rate_limit
        )
        if not allowed:
            return SkillResult(
                success=False,
                message="Rate limit exceeded",
                data={
                    "allowed": False,
                    "reason": "rate_limited",
                    "key_id": matched_key.key_id,
                    "reset_at": reset_at,
                },
            )

        # Check daily limit
        usage = self._usage.get(matched_key.key_id, UsageRecord(key_id=matched_key.key_id))
        today = datetime.utcnow().strftime("%Y-%m-%d")
        if usage.today_date != today:
            usage.requests_today = 0
            usage.today_date = today

        if matched_key.daily_limit > 0 and usage.requests_today >= matched_key.daily_limit:
            return SkillResult(
                success=False,
                message="Daily limit exceeded",
                data={
                    "allowed": False,
                    "reason": "daily_limit_exceeded",
                    "key_id": matched_key.key_id,
                    "daily_limit": matched_key.daily_limit,
                    "used_today": usage.requests_today,
                },
            )

        # Access granted - record in rate limiter
        self._rate_limiter.record(matched_key.key_id)

        return SkillResult(
            success=True,
            message="Access granted",
            data={
                "allowed": True,
                "key_id": matched_key.key_id,
                "owner": matched_key.owner,
                "scopes": matched_key.scopes,
                "remaining_rate": remaining,
                "remaining_daily": (
                    matched_key.daily_limit - usage.requests_today - 1
                    if matched_key.daily_limit > 0
                    else -1
                ),
            },
        )

    def _scope_matches(self, required: str, available: List[str]) -> bool:
        """Check if any available scope covers the required scope.

        Supports hierarchical scopes: 'skills:*' matches 'skills:read'.
        """
        req_parts = required.split(":")
        for scope in available:
            scope_parts = scope.split(":")
            if len(scope_parts) >= 1 and scope_parts[-1] == "*":
                # Wildcard: check prefix match
                if req_parts[: len(scope_parts) - 1] == scope_parts[:-1]:
                    return True
        return False

    # ── Usage Tracking ──────────────────────────────────────────

    async def _record_usage(self, params: Dict) -> SkillResult:
        key_id = params.get("key_id")
        if not key_id or key_id not in self._keys:
            return SkillResult(success=False, message=f"Key not found: {key_id}")

        if key_id not in self._usage:
            self._usage[key_id] = UsageRecord(key_id=key_id)

        usage = self._usage[key_id]
        endpoint = params.get("endpoint", "unknown")
        cost = float(params.get("cost", 0))
        revenue = float(params.get("revenue", 0))
        is_error = params.get("error", False)

        # Roll over daily counter if needed
        today = datetime.utcnow().strftime("%Y-%m-%d")
        if usage.today_date != today:
            # Save yesterday's stats
            if usage.today_date:
                usage.daily_history.append({
                    "date": usage.today_date,
                    "requests": usage.requests_today,
                })
                # Keep last 30 days
                usage.daily_history = usage.daily_history[-30:]
            usage.requests_today = 0
            usage.today_date = today

        usage.total_requests += 1
        usage.requests_today += 1
        if is_error:
            usage.total_errors += 1
        usage.total_cost += cost
        usage.total_revenue += revenue
        usage.last_request_at = datetime.utcnow().isoformat()

        # Track per-endpoint counts
        usage.endpoint_counts[endpoint] = usage.endpoint_counts.get(endpoint, 0) + 1

        self._save()

        return SkillResult(
            success=True,
            message=f"Usage recorded for {key_id}",
            data={
                "key_id": key_id,
                "total_requests": usage.total_requests,
                "requests_today": usage.requests_today,
            },
        )

    async def _get_usage(self, params: Dict) -> SkillResult:
        key_id = params.get("key_id")
        if not key_id or key_id not in self._keys:
            return SkillResult(success=False, message=f"Key not found: {key_id}")

        usage = self._usage.get(key_id, UsageRecord(key_id=key_id))

        return SkillResult(
            success=True,
            message=f"Usage for {key_id}",
            data={
                "key_id": key_id,
                "total_requests": usage.total_requests,
                "total_errors": usage.total_errors,
                "error_rate": (
                    usage.total_errors / usage.total_requests
                    if usage.total_requests > 0
                    else 0
                ),
                "total_cost": usage.total_cost,
                "total_revenue": usage.total_revenue,
                "profit": usage.total_revenue - usage.total_cost,
                "requests_today": usage.requests_today,
                "last_request_at": usage.last_request_at,
                "endpoint_counts": usage.endpoint_counts,
                "daily_history": usage.daily_history,
            },
        )

    async def _get_billing(self, params: Dict) -> SkillResult:
        owner_filter = params.get("owner")

        billing = {
            "total_revenue": 0.0,
            "total_cost": 0.0,
            "total_profit": 0.0,
            "total_requests": 0,
            "active_keys": 0,
            "revoked_keys": 0,
            "per_owner": {},
        }

        for key_id, key in self._keys.items():
            if owner_filter and key.owner != owner_filter:
                continue

            usage = self._usage.get(key_id, UsageRecord(key_id=key_id))

            if key.revoked:
                billing["revoked_keys"] += 1
            else:
                billing["active_keys"] += 1

            billing["total_revenue"] += usage.total_revenue
            billing["total_cost"] += usage.total_cost
            billing["total_requests"] += usage.total_requests

            # Aggregate per owner
            if key.owner not in billing["per_owner"]:
                billing["per_owner"][key.owner] = {
                    "owner": key.owner,
                    "keys": 0,
                    "total_requests": 0,
                    "total_revenue": 0.0,
                    "total_cost": 0.0,
                }
            owner_data = billing["per_owner"][key.owner]
            owner_data["keys"] += 1
            owner_data["total_requests"] += usage.total_requests
            owner_data["total_revenue"] += usage.total_revenue
            owner_data["total_cost"] += usage.total_cost

        billing["total_profit"] = billing["total_revenue"] - billing["total_cost"]

        # Convert per_owner to list
        billing["per_owner"] = list(billing["per_owner"].values())

        return SkillResult(
            success=True,
            message=f"Billing summary: {billing['active_keys']} active keys, ${billing['total_revenue']:.2f} revenue",
            data=billing,
        )

    # ── Persistence ─────────────────────────────────────────────

    def _save(self):
        os.makedirs(DATA_DIR, exist_ok=True)
        data = {
            "keys": {kid: asdict(k) for kid, k in self._keys.items()},
            "usage": {kid: asdict(u) for kid, u in self._usage.items()},
        }
        try:
            with open(self._persist_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    def _load(self):
        if not os.path.exists(self._persist_path):
            return
        try:
            with open(self._persist_path) as f:
                data = json.load(f)

            for kid, kdata in data.get("keys", {}).items():
                self._keys[kid] = APIKey(**kdata)

            for kid, udata in data.get("usage", {}).items():
                self._usage[kid] = UsageRecord(**udata)
        except Exception:
            pass
